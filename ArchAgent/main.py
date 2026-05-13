import argparse
import csv
import dotenv
import json
import os
import re
import shutil
import subprocess
import uuid

from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import TypedDict, Tuple, Dict, List, Any
import xml.etree.ElementTree as ET

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

from util.Fqn import Fqn # check if FQN exists and return the PATH for it
from util.FileSystem import FileSystem
from util.Dependencies import Dependencies

from State import State #langgraph state class

from tools.context_builder import extract_observed_external_calls

## Helper functions

# execute a prompt command
def _run(cmd: list[str], cwd: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )

# function to extract JSON from the plan
def _extract_json_object_only(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s

    if s.startswith("{") and s.endswith("}"):
        return s

    i = s.find("{")
    j = s.rfind("}")
    if i != -1 and j != -1 and j > i:
        return s[i : j + 1].strip()

    return s

# function to validate the path
def _is_safe_repo_rel_path(p: str) -> bool:

    # if path is empty or null
    if not p or "\x00" in p:
        return False

    # if p is an absolute path
    pp = Path(p)
    if pp.is_absolute():
        return False

    # to avoid DOS path
    if ":" in p.split("/")[0]:  # ex: C:\
        return False
    
    # if use return in the path
    parts = pp.as_posix().split("/")
    if any(part == ".." for part in parts):
        return False
    
    # if valid path
    return True

def _validate_allowed_paths(rel_paths: list[str], allowed: set[str]) -> tuple[bool, str]:
    for rp in rel_paths:
        if not _is_safe_repo_rel_path(rp):
            return False, f"unsafe path: {rp}"
        if rp not in allowed:
            return False, f"path not in allowed_paths: {rp}"
    return True, ""

# load meta,json file
def _load_meta_or_init(meta_path: Path, repo_path: Path, base_commit: str | None) -> dict:
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"repo_path": str(repo_path), "base_commit": base_commit}

# ex. receive src/main/java/org/jsoup/Jsoup.java, return src/main/java
def _infer_source_root_from_target(repo_path: Path, target_rel: str, target_fqn: str) -> str:
    # target package path from FQN
    pkg_parts = target_fqn.split(".")[:-1]          # ['org','jsoup']
    pkg_path = "/".join(pkg_parts)                  # 'org/jsoup'

    tr = Path(target_rel).as_posix()

    # remove trailing '/org/jsoup/Jsoup.java' from the target path
    suffix = f"{pkg_path}/{target_fqn.split('.')[-1]}.java"
    if tr.endswith(suffix):
        src_root = tr[: -len(suffix)].rstrip("/")
        if src_root:
            return src_root

    # fallback: go up N dirs (package depth + 1 file)
    p = Path(tr)
    up = len(pkg_parts) + 1
    for _ in range(up):
        p = p.parent
    return p.as_posix()

def _java_fqn_to_path(repo_path: Path, class_fqn: str, source_root_rel: str) -> str:
    rel = Path(source_root_rel) / Path(class_fqn.replace(".", "/") + ".java")
    return str((repo_path / rel).resolve())

def _extract_fqn_from_java(code: str, filename: str) -> str:
    m = re.search(r'^\s*package\s+([a-zA-Z0-9_.]+)\s*;', code, re.MULTILINE)
    pkg = m.group(1) if m else ""
    cls = Path(filename).stem
    return f"{pkg}.{cls}" if pkg else cls

def _tail(s: str, n: int = 40) -> str:
    lines = (s or "").splitlines()
    return "\n".join(lines[-n:]).strip()

def _git_current_commit(repo_path: Path) -> str:
    p = _run(["git", "rev-parse", "HEAD"], cwd=repo_path)
    if p.returncode != 0:
        raise RuntimeError("git rev-parse HEAD failed:\n" + _tail(p.stderr))
    return p.stdout.strip()

def _to_repo_rel(repo_path: Path, p: str) -> str:
    pp = Path(p)
    if pp.is_absolute():
        try:
            return str(pp.resolve().relative_to(repo_path))
        except Exception:
            return str(pp.as_posix()).lstrip("/")
    return pp.as_posix()

## get target file/class
def _read_target_file(repo_path: Path, target_file: str) -> tuple[str, str]:

    ### check if target file is empty
    if not target_file or "\x00" in target_file:
        raise RuntimeError("target_file is empty/invalid")

    tf = Path(target_file)

    # treat if the path don't have data/repositories/...
    if tf.is_absolute():
        abs_p = tf.resolve()
    else:
        abs_p = (repo_path / tf).resolve()

    # must be inside repo
    if repo_path != abs_p and repo_path not in abs_p.parents:
        raise RuntimeError(f"target_file is outside repo: {abs_p}")

    # convert to repo-relative (for planner and logging)
    rel = str(abs_p.relative_to(repo_path)).replace("\\", "/")

    if not abs_p.exists() or not abs_p.is_file():
        raise RuntimeError(f"target_file does not exist or is not a file: {rel}")

    code = abs_p.read_text(encoding="utf-8", errors="replace") # get file content
    return rel, code # return relative path and file content

def _get_plan_dir(state: State) -> Path:
    run_dir = Path(state["run_dir"])
    plan_idx = int(state.get("plan_idx", 0))
    plan_dir = run_dir / f"plan_{plan_idx:02d}"
    plan_dir.mkdir(parents=True, exist_ok=True)
    return plan_dir

def _designite_smell_present(
    designite_dir: Path,
    target_class_fqn: str,
    smell_name: str,
    csv_name: str = "DesignSmells.csv"
) -> bool:

    csv_path = designite_dir / csv_name
    if not csv_path.exists():
        return False

    target_fqn = target_class_fqn.strip()

    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pkg = (row.get("Package") or "").strip()
            cls = (row.get("Class") or "").strip()

            if not pkg or not cls:
                continue

            row_fqn = f"{pkg}.{cls}"

            #print(row_fqn, target_fqn, row.get("Smell"), smell_name)

            if row_fqn == target_fqn and (row.get("Smell") or "").strip() == smell_name:
                return True

    return False

# return target type (class or package) on state
def _get_target_type(state: State) -> str:
    target_type = (state.get("target_type") or "class").strip().lower()
    if target_type not in {"class", "package"}:
        return "class"
    return target_type

# set target name in State
def _get_target_identity(state: State) -> str:
    return (state.get("target_name") or "").strip()

# return target scope
def _resolve_target_scope(state: State) -> dict:
    target_type = _get_target_type(state)

    return {
        "target_type": target_type,
        "target_name": (state.get("target_name") or "").strip(),
        "target_identity": _get_target_identity(state),
        "target_file": (state.get("target_file") or "").strip(),
        "target_files": list(state.get("target_files") or []),
        "target_source_root": (state.get("target_source_root") or "").strip(),
    }

# args: read target argument and define if it is a package or class
def _infer_target_type_from_name(target_name: str) -> str:
    name = (target_name or "").strip()
    if not name:
        raise ValueError("target_name is empty")

    last = name.split(".")[-1]
    if last[:1].isupper():
        return "class"
    return "package"

def _run_build(repo_path: Path, tmp_dir: Path) -> subprocess.CompletedProcess:
    cmd = ["mvn", "-q",
       "-DskipTests",
       "-Drat.skip=true",
       "-Dcheckstyle.skip=true",
       "-Dspotbugs.skip=true",
       "-Dpmd.skip=true",
       "-DskipITs",
       "clean", "verify"]

    env = os.environ.copy()
    env["MAVEN_OPTS"] = f"-Xshare:off -Djava.io.tmpdir={tmp_dir}"

    return _run(cmd, cwd=repo_path, env=env)

def _run_designite(
    repo_path: Path,
    out_dir: Path,
    jar_path: Path,
) -> Tuple[Path, list[str]]:

    if out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    java22 = "/usr/lib/jvm/jdk-22.0.2-oracle-x64/bin/java"

    #cmd = [
    #    "java", "-jar", str(jar_path),
    #    "-i", str(repo_path),
    #    "-o", str(out_dir),
    #]

    cmd = [java22, "-jar", str(jar_path), "-g", "-i", str(repo_path), "-o", str(out_dir)]

    p = _run(cmd, cwd=repo_path)

    log = (p.stdout or "") + ("\n" if p.stdout else "") + (p.stderr or "")
    (out_dir / "designite.log").write_text(log, encoding="utf-8")

    if p.returncode != 0:
        raise RuntimeError(
            f"Designite failed (rc={p.returncode}). See log at {out_dir / 'designite.log'}"
        )

    return out_dir, cmd

def get_package_dependencies(graphml_path: str, target_name: str):
    deps = Dependencies(target_name)
    return deps._get_package_dependencies(graphml_path)

## Function to add visibility update ops to the plan for related classes in the same package that are not moved but have internal dependencies with the moved classes (to keep compilation working after the move)
def enrich_plan_with_visibility_ops(plan: dict, state: State) -> dict:
    internal_deps = state.get("internal_deps") or []
    target_files = state.get("target_files") or []
    target_source_root = state.get("target_source_root") or ""
    target_name = state.get("target_name") or ""
    selected_cluster = set(plan.get("selected_cluster") or [])

    # FQN -> file path
    class_to_file = {}
    for f in target_files:
        fqn = Fqn(target_name)._java_file_to_fqn(f, target_source_root)
        class_to_file[fqn] = f

    # Build relation map in both directions:
    # moved class -> related internal classes
    related_by_class: dict[str, set[str]] = {}

    for src, dst in internal_deps:
        related_by_class.setdefault(src, set()).add(dst)
        related_by_class.setdefault(dst, set()).add(src)

    original_blocks = plan.get("blocks") or []
    enriched_blocks = []

    prepared_visibility_classes: set[str] = set()
    next_id = 1

    for block in original_blocks:
        ops = block.get("ops") or []

        move_op = None
        moved_old_fqn = None

        for op in ops:
            if op.get("op") == "MOVE_CLASS":
                inputs = op.get("inputs") or []
                if inputs:
                    move_op = op
                    moved_old_fqn = inputs[0]
                    break

        if not move_op or not moved_old_fqn:
            block["id"] = next_id
            enriched_blocks.append(block)
            next_id += 1
            continue

        related_classes = sorted(
            cls
            for cls in related_by_class.get(moved_old_fqn, set())
            if cls in class_to_file
            and cls not in selected_cluster
        )

        visibility_classes = [moved_old_fqn] + related_classes

        visibility_classes = [
            cls
            for cls in visibility_classes
            if cls in class_to_file
            and cls not in prepared_visibility_classes
        ]

        if visibility_classes:
            visibility_files = sorted({
                class_to_file[cls]
                for cls in visibility_classes
                if cls in class_to_file
            })

            enriched_blocks.append({
                "id": next_id,
                "goal": f"Prepare visibility before moving {moved_old_fqn}.",
                "files": visibility_files,
                "ops": [
                    {
                        "op": "UPDATE_VISIBILITY",
                        "inputs": visibility_classes,
                        "outputs": [],
                        "details": (
                            "Update only the minimum required visibility in these "
                            "target-package classes so moved cluster classes can "
                            "compile from the new package."
                        ),
                        "risk": "medium",
                        "api_change": True,
                    }
                ],
            })

            next_id += 1
            prepared_visibility_classes.update(visibility_classes)

        # Keep only CREATE_PACKAGE and MOVE_CLASS in the move block.
        move_block_ops = [
            op for op in ops
            if op.get("op") in {"CREATE_PACKAGE", "MOVE_CLASS"}
        ]

        move_block_files = list(block.get("files") or [])

        # Guarantee the moved source file is present.
        if moved_old_fqn in class_to_file:
            move_block_files.append(class_to_file[moved_old_fqn])

        block["id"] = next_id
        block["files"] = sorted(set(move_block_files))
        block["ops"] = move_block_ops

        enriched_blocks.append(block)
        next_id += 1

    plan["blocks"] = enriched_blocks
    return plan

# Nodes
def route_node(state: State) -> State:
    state["msg"] = (
        f"route ok: repo_path={state.get('repo_path')}"
        f"target_type={_get_target_type(state)}"
        f"target={_get_target_identity(state)}"
    )
    return state

## Node to initiate the graph run
def init_run_node(state: State) -> State:
    repo_path = Path(state["repo_path"]).resolve()

    ### defines start/base commit if not set
    head = _git_current_commit(repo_path) # get the start commit

    if not state.get("start_commit"):
        state["start_commit"] = head # set current commit before refactoring
    
    if not state.get("base_commit"):
        state["base_commit"] = head # set checkpoint commit

    ### create log directory in repo_path/agent_runs/
    # TODO: create a function
    runs_root = repo_path / "agent_runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    ### create run directory repo_path/agent_runs/run_timestamp
    # TODO: create a function
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = uuid.uuid4().hex[:8]
    run_dir = runs_root / f"{ts}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    state["run_dir"] = str(run_dir)

    ### load smell type
    smell_type = ""
    try:
        inp = json.loads(state.get("planner_input_json", "") or "{}")
        smell_type = str(inp.get("smell", "")).strip()
    except Exception:
        smell_type = ""
    state["smell_type"] = smell_type
    state.setdefault("designite_smell_name", state["smell_type"])

    # TODO: improve this defaulting logic
    if state["smell_type"] in {"Insufficient Modularization", "Hub-like Modularization"}:
        state.setdefault("designite_smells_csv", "DesignSmells.csv")
    else:
        state.setdefault("designite_smells_csv", "ArchitectureSmells.csv")

    meta = {
        "repo_path": str(repo_path),
        "target_type": _get_target_type(state),
        "target_name": _get_target_identity(state),
        "start_commit": state["start_commit"],
        "base_commit": state["base_commit"],
        "smell_type": state.get("smell_type", ""),
        "designite_smell_name": state.get("designite_smell_name", ""),
        "designite_smells_csv": state.get("designite_smells_csv", "DesignSmells.csv"),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    state.setdefault("executor_feedback", "")

    # plan lifecycle init
    state.setdefault("plan_idx", 0)
    
    # designite analysis states
    state.setdefault("smell_persist_replans", 0)
    state.setdefault("smell_quality_analysis", "")
    state.setdefault("smell_quality_ok", False)
    state.setdefault("smell_quality_error", "")
    
    # prompts states
    if state["smell_type"] == "Insufficient Modularization":
        state.setdefault("smell_quality_prompt_path", "data/prompts/quality_IM.prompt")
    elif state["smell_type"] == "Hub-like Modularization":
        state.setdefault("smell_quality_prompt_path", "data/prompts/quality_HM.prompt")

    ## give a workfull commit to the plan
    state["plan_base_commit"] = state.get("base_commit") or head

    state["designite_smells_csv"] = state.get("designite_smells_csv", "DesignSmells.csv")

    # Por padrão, assume que o nome do smell no Designite é igual ao "smell"
    # (no futuro você pode mapear aqui, se os nomes divergirem)
    state["designite_smell_name"] = state.get("designite_smell_name", state["smell_type"])

    plan_dir = _get_plan_dir(state)

    ## update meta.json
    meta.update({
        "plan_idx": state["plan_idx"],
        "plan_base_commit": state["plan_base_commit"],
        "plan_dir": str(plan_dir),
        "smell_persist_replans": state["smell_persist_replans"],
    })
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    ### Update state message
    state["msg"] = (
        state.get("msg", "")
        + f" | init_run ok type={_get_target_type(state)} run_dir={run_dir.name}"
    )

    return state

## function to resolve class data for planner input
def resolve_target_class_node(state: State) -> State:
    ### initialize node
    repo_path = Path(state["repo_path"]).resolve()
    run_dir = Path(state["run_dir"])
    plan_dir = _get_plan_dir(state)
    meta_path = run_dir / "meta.json"
    meta = _load_meta_or_init(meta_path, repo_path, state.get("base_commit"))

    if not state["target_file"]: # check if target file is provided
        raise RuntimeError("target_file is required for target_type=class")

    target_rel, code = _read_target_file(repo_path, state["target_file"])
    state["target_file"] = target_rel
    
    target_fqn = _extract_fqn_from_java(code, target_rel)
    state["target_class_fqn"] = target_fqn
    state["target_source_root"] = _infer_source_root_from_target(repo_path, target_rel, target_fqn)

    ### update meta.json
    meta.update({
        "target_type": state["target_type"],
        "target_name": state["target_name"],
        "target_file": state["target_file"],
        "target_source_root": state["target_source_root"]
    })
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    ### Update state message
    state["msg"] = (
        state.get("msg", "")
        + f" | file content loaded for {state['target_file']}"
    )

    return state

## filter in the package only the classes that can be moved without breaking incoming dependencies from outside the package
'''def _filter_movable_package_scope(
    state: State,
    target_name: str,
    incoming_deps: list[Tuple[str, str]],
    allowed_classes: set[str],
    internal_deps: list[Tuple[str, str]],
) -> tuple[list[Tuple[str, str]], list[str]]:
    incoming_targets = {dst for _, dst in incoming_deps}

    movable_classes = {
        cls
        for cls in allowed_classes
        if cls not in incoming_targets
    }

    movable_internal_deps = [
        (src, dst)
        for src, dst in internal_deps
        if src in movable_classes and dst in movable_classes
    ]

    movable_files = [
        f for f in state["target_files"]
        if Fqn(target_name)._java_file_to_fqn(
            f,
            state["target_source_root"]
        ) in movable_classes
    ]

    if not movable_files:
        raise RuntimeError(f"No movable classes found for package: {target_name}")

    return movable_internal_deps, movable_files
'''

def _filter_closed_movable_clusters(
    state: State,
    target_name: str,
    incoming_deps: list[Tuple[str, str]],
    allowed_classes: set[str],
    internal_deps: list[Tuple[str, str]],
    max_cluster_size: int = 3,
) -> tuple[list[Tuple[str, str]], list[str], list[list[str]]]:
    incoming_targets = {dst for _, dst in incoming_deps}

    # classes sem dependência de entrada externa
    candidate_classes = {
        cls
        for cls in allowed_classes
        if cls not in incoming_targets
    }

    # mapa de outgoing deps internas
    outgoing_map: dict[str, set[str]] = {}
    for src, dst in internal_deps:
        if src in candidate_classes and dst in allowed_classes:
            outgoing_map.setdefault(src, set()).add(dst)

    closed_clusters: list[list[str]] = []

    for cls in sorted(candidate_classes):
        cluster = {cls}
        changed = True

        while changed:
            changed = False
            for current in list(cluster):
                for dep in outgoing_map.get(current, set()):
                    if dep not in candidate_classes:
                        cluster = set()
                        changed = False
                        break

                    if dep not in cluster:
                        cluster.add(dep)
                        changed = True

                if not cluster:
                    break

        if not cluster:
            continue

        if len(cluster) <= max_cluster_size:
            closed_clusters.append(sorted(cluster))

    # remover duplicatas
    unique_clusters = []
    seen = set()

    for cluster in closed_clusters:
        key = tuple(cluster)
        if key not in seen:
            seen.add(key)
            unique_clusters.append(cluster)

    movable_classes = {
        cls
        for cluster in unique_clusters
        for cls in cluster
    }

    movable_internal_deps = [
        (src, dst)
        for src, dst in internal_deps
        if src in movable_classes and dst in movable_classes
    ]

    movable_files = [
        f for f in state["target_files"]
        if Fqn(target_name)._java_file_to_fqn(
            f,
            state["target_source_root"]
        ) in movable_classes
    ]

    return movable_internal_deps, movable_files, unique_clusters

## function to resolve package data for planner input
def resolve_target_package_node(state: State) -> State:
    ### initialize node
    repo_path = Path(state["repo_path"]).resolve()
    run_dir = Path(state["run_dir"])
    meta_path = run_dir / "meta.json"
    meta = _load_meta_or_init(meta_path, repo_path, state.get("base_commit"))

    ### get and check target_name
    target_name = _get_target_identity(state)

    if not target_name:
        raise RuntimeError("target_name is required for target_type=package")
    
    ### get and check target path
    target_path = Fqn(target_name).find_in_repo(repo_path)

    if target_path is None:
        raise RuntimeError(f"package target not found: {target_name}")

    if not target_path.is_dir():
        raise RuntimeError(f"package target must resolve to a directory: {target_path}")
    
    ### get package files, check them and save in state
    target_files = FileSystem(str(repo_path), str(target_path)).list_java_files_in_dir()
    
    if not target_files:
        raise RuntimeError(f"package target has no .java files: {target_name}")

    state["target_files"] = target_files

    ### get source root and save on state
    source_root_path = target_path.resolve()

    for _ in target_name.split("."):
        source_root_path = source_root_path.parent

    state["target_source_root"] = str(
        source_root_path.relative_to(repo_path)
    ).replace("\\", "/")

    ### list classes inside the directory
    allowed_classes = Fqn(target_name)._java_files_to_fqns(
        state["target_files"],
        state["target_source_root"]
    )
    
    ### run Designite for the current package resolution
    jar_env = os.getenv("DESIGNITE_JAR_PATH")
    if not jar_env:
        raise RuntimeError("DESIGNITE_JAR_PATH is not set")

    designite_jar = Path(jar_env).expanduser().resolve()
    if not designite_jar.exists() or not designite_jar.is_file():
        raise RuntimeError(f"Designite JAR not found at {designite_jar}")

    plan_dir = _get_plan_dir(state)
    designite_scope_dir = plan_dir / "package_scope_designite"

    out_dir, cmd = _run_designite(repo_path, designite_scope_dir, designite_jar)

    graphml_path = out_dir / "DependencyGraph.graphml"
    if not graphml_path.exists():
        raise RuntimeError(f"DependencyGraph.graphml not found at {graphml_path}")

    ### get package dependencies
    internal_deps, outgoing_deps, incoming_deps = get_package_dependencies(
        graphml_path,
        target_name
    )
    
    ### resolve internal deps
    internal_deps = [ # remove not allowed classes
        (src, dst)
        for src, dst in internal_deps
        if src in allowed_classes and dst in allowed_classes
    ]

    state["internal_deps"] = internal_deps

    ### resolve incoming deps
    incoming_deps = [
        (src, dst)
        for src, dst in incoming_deps
        if dst in allowed_classes
    ]

    state["incoming_deps"] = incoming_deps

    ### resolve outgoing deps
    outgoing_deps = [
        (src, dst)
        for src, dst in outgoing_deps
        if src in allowed_classes
    ]
    state["outgoing_deps"] = outgoing_deps

    ### update input

    planner_input = {
        "smell": state.get("smell_type"),
        "target_type": "package",
        "target_name": target_name,
        "target_source_root": state["target_source_root"],
        "target_files": state["target_files"],
        "internal_deps": state["internal_deps"],
        "incoming_deps": state["incoming_deps"],
        "outgoing_deps": state["outgoing_deps"],
    }
    state["planner_input_json"] = json.dumps(planner_input, indent=2)

    plan_dir = _get_plan_dir(state)
    (plan_dir / "planner.input.package.json").write_text(
        state["planner_input_json"],
        encoding="utf-8"
    )

    ### update meta data
    meta.update({
        "target_type": state["target_type"],
        "target_name": state["target_name"],
        "target_files": state["target_files"],
        "internal_deps": state["internal_deps"],
        "target_files_count": len(state["target_files"]),
        "target_source_root": state["target_source_root"],
        "incoming_deps": state["incoming_deps"],
        "incoming_deps_count": len(state["incoming_deps"]),
        "outgoing_deps": state["outgoing_deps"],
        "outgoing_deps_count": len(state["outgoing_deps"]),
    })
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    ### update state message
    state["msg"] = (
        state.get("msg", "")
        + f" | package target resolved files={len(state['target_files'])}"
    )

    return state

def after_init_run(state: State) -> str:
    target_type = _get_target_type(state)

    if target_type == "class":
        return "resolve_target_class"
    elif target_type == "package":
        return "resolve_target_package"
    else:
        raise RuntimeError(f"unsupported target_type: {target_type}")

def planner_node(state: State) -> State:
    repo_path = Path(state["repo_path"]).resolve()
    run_dir = Path(state["run_dir"])
    plan_dir = _get_plan_dir(state)

    meta_path = run_dir / "meta.json"
    meta = _load_meta_or_init(meta_path, repo_path, state.get("base_commit"))

    prompt = state.get("planner_prompt", "").strip()
    planner_input = state.get("planner_input_json", "").strip()

    if not prompt:
        state["plan_ok"] = False
        state["plan_error"] = "planner_prompt missing"
        meta.update({"plan_ok": False, "plan_error": state["plan_error"]})
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return state

    if not planner_input:
        state["plan_ok"] = False
        state["plan_error"] = "planner_input_json missing"
        meta.update({"plan_ok": False, "plan_error": state["plan_error"]})
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return state

    (plan_dir / "planner.prompt.md").write_text(prompt, encoding="utf-8")
    (plan_dir / "planner.input.json").write_text(planner_input, encoding="utf-8")

    llm = ChatOpenAI(
        model=os.getenv("PLANNER_MODEL", "gpt-5-mini"),
        temperature=0.0,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    rendered = prompt.replace("{input}", planner_input)

    try:
        res = llm.invoke(
            [
                SystemMessage(content="Return ONLY valid JSON. No extra text."),
                HumanMessage(content=rendered),
            ]
        )

        raw = (res.content or "").strip()
        state["plan_json_text"] = raw
        (plan_dir / "planner.raw.txt").write_text(raw, encoding="utf-8")

        json_text = _extract_json_object_only(raw)

        plan = json.loads(json_text)

        # increase Plan with deps for God Component
        if (
            _get_target_type(state) == "package"
            and state.get("smell_type") == "God Component"
        ):
            plan = enrich_plan_with_visibility_ops(plan, state)

        if not isinstance(plan, dict) or "blocks" not in plan:
            raise ValueError("plan JSON missing required top-level keys (expected dict with 'blocks').")
        if not isinstance(plan["blocks"], list):
            raise ValueError("'blocks' must be a list.")

        state["plan"] = plan
        state["plan_ok"] = True
        state["plan_error"] = ""

        (plan_dir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

        meta.update({"plan_ok": True, "plan_error": ""})
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        state["msg"] = state.get("msg", "") + " | planner ok"
        return state

    except Exception as e:
        err = str(e)
        state["plan_ok"] = False
        state["plan_error"] = err
        (plan_dir / "planner.error.txt").write_text(err + "\n", encoding="utf-8")

        meta.update({"plan_ok": False, "plan_error": err})
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        state["msg"] = state.get("msg", "") + " | planner FAIL"
        return state

def after_planner(state: State) -> str:
    if state.get("plan_ok"):
        return "stage_block"
    else:
        return END


def stage_block_node(state: State) -> State:
    run_dir = Path(state["run_dir"])
    plan = state.get("plan")
    
    if not plan or not isinstance(plan, dict):
        raise RuntimeError("stage_block_node: missing/invalid state['plan']")

    blocks = plan.get("blocks", [])
    if not isinstance(blocks, list):
        raise RuntimeError("stage_block_node: plan['blocks'] must be a list")

    idx = state.get("block_idx", 0)

    if idx >= len(blocks):
        state["done"] = True
        state["staged_block"] = {}
        state["staged_block_ops"] = []
        state["staged_block_files"] = []
        state["msg"] = state.get("msg", "") + " | stage_block: done"
        return state

    blk = blocks[idx]
    ops = blk.get("ops", []) or []
    files = blk.get("files", []) or []

    # IMPORTANT: reset attempt/feedback for THIS block
    state["executor_feedback"] = ""

    state["block_attempt"] = 0
    state.setdefault("max_block_attempts", 5)

    state["done"] = False
    state["staged_block"] = blk
    state["staged_block_id"] = blk.get("id")
    state["staged_block_ops"] = ops
    state["staged_block_files"] = files

    plan_dir = _get_plan_dir(state)
    (plan_dir / f"staged.block.{idx}.json").write_text(json.dumps(blk, indent=2), encoding="utf-8")

    state["msg"] = state.get("msg", "") + f" | staged block_idx={idx} id={blk.get('id')} ops={len(ops)}"
    return state

# based on the plan, define what files can be affected by the LLM
def resolve_files_for_block_node(state: State) -> State:
    repo_path = Path(state["repo_path"]).resolve()
    run_dir = Path(state["run_dir"])

    # get dic of blocks and list of operations
    blk = state.get("staged_block") or {}
    ops = blk.get("ops", []) or []

    '''existing: set[str] = set()
    for f in (state.get("staged_block_files") or []):
        p = Path(f)
        if p.is_absolute():
            existing.add(str(p))
        else:
            existing.add(str((repo_path / p).resolve()))'''

    allowed_scope = set(state.get("target_files") or [])
    allowed_scope.update(state.get("external_files") or [])

    existing: set[str] = set()
    rejected: list[str] = []

    for f in (state.get("staged_block_files") or []):
        rel = _to_repo_rel(repo_path, f)

        if rel not in allowed_scope:
            rejected.append(rel)
            continue

        existing.add(str((repo_path / rel).resolve()))

    new_files: set[str] = set()
    for op in ops:
        op_name = (op.get("op") or "").strip()

        if op_name in {"EXTRACT_CLASS", "EXTRACT_INTERFACE"}:
            for out in (op.get("outputs") or []):
                if isinstance(out, str) and "." in out and not out.endswith("/"):
                    if out.split(".")[-1][:1].isupper():
                        new_files.add(
                            _java_fqn_to_path(repo_path, out, state["target_source_root"])
                        )

        if op_name == "MOVE_CLASS":
            for out in (op.get("outputs") or []):
                if isinstance(out, str) and "." in out and out.split(".")[-1][:1].isupper():
                    new_files.add(
                        _java_fqn_to_path(repo_path, out, state["target_source_root"])
                    )

    all_files = sorted(existing.union(new_files))

    state["executor_existing_files"] = sorted(existing)
    state["executor_new_files"] = sorted(new_files)
    state["executor_files"] = all_files
    state["executor_rejected_files"] = rejected

    plan_dir = _get_plan_dir(state)
    (plan_dir / "executor.files.json").write_text(json.dumps(all_files, indent=2), encoding="utf-8")

    state["msg"] = state.get("msg", "") + f" | files={len(all_files)} (new={len(new_files)})"
    return state


def lock_workspace_node(state: State) -> State:
    repo_path = Path(state["repo_path"]).resolve()
    run_dir = Path(state["run_dir"])
    plan_dir = _get_plan_dir(state)

    if not state.get("base_commit"):
        state["base_commit"] = _git_current_commit(repo_path)
    base = state["base_commit"]

    _run(["git", "reset", "--hard", base], cwd=repo_path)
    _run(["git", "clean", "-fd", "-e", "agent_runs/", "-e", "tmp/"], cwd=repo_path)

    state["workspace_commit"] = _git_current_commit(repo_path)

    (plan_dir / "workspace.lock.txt").write_text(
        f"base_commit={base}\nworkspace_commit={state['workspace_commit']}\n",
        encoding="utf-8",
    )

    # clean old tries state
    state["files_to_write"] = []
    state["files_to_delete"] = []
    state["apply_ok"] = False
    state["apply_error"] = ""
    state["rollback_reason"] = ""

    state["msg"] = state.get("msg", "") + f" | workspace locked @{state['workspace_commit'][:8]}"
    return state

def executor_node(state: State) -> State:

    # load log files
    repo_path = Path(state["repo_path"]).resolve()
    run_dir = Path(state["run_dir"])
    plan_dir = _get_plan_dir(state)
    
    meta_path = run_dir / "meta.json"
    meta = _load_meta_or_init(meta_path, repo_path, state.get("base_commit"))

    # get plan blocks and envolved files
    blk = state.get("staged_block") or {}
    files = state.get("executor_files") or []

    # Read current file contents (for context)
    file_blobs: list[dict] = []
    for fp in files:
        p = Path(fp)
        rel_path = _to_repo_rel(repo_path, str(p))

        if p.exists() and p.is_file():
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                content = f"<<ERROR reading file: {e}>>"
        else:
            content = "<<NEW FILE (does not exist yet)>>"

        file_blobs.append({"path": rel_path, "content": content})

    allowed_paths = [f["path"] for f in file_blobs]

    SYSTEM = """Return ONLY valid JSON (no markdown, no explanations, no backticks).
Your job: produce the FINAL full contents of files after applying this refactoring block.

Rules (hard):
- Output MUST be a single JSON object.
- You may only write/delete files listed in allowed_paths.
- Paths MUST be repository-relative (e.g., src/main/java/...).
- For files_to_write: provide full file content (complete file text).
- If you cannot comply perfectly, output: {} (an empty JSON object).

JSON schema:
{
  "files_to_write": [
    {"path": "src/....java", "content": "FULL FILE CONTENT HERE"}
  ],
  "files_to_delete": ["src/.../Old.java"]
}
"""

    executor_prompt = {
        "task": "Generate full-code file outputs for the staged refactoring block.",
        "staged_block": blk,
        "allowed_paths": allowed_paths,
        "files_context": file_blobs,
        "feedback": state.get("executor_feedback", "")
    }

    state["executor_prompt"] = json.dumps(executor_prompt, indent=2)

    llm = ChatOpenAI(
        model=os.getenv("EXECUTOR_MODEL", "gpt-5-mini"),
        temperature=0.0,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    res = llm.invoke([
        SystemMessage(content=SYSTEM),
        HumanMessage(content=state["executor_prompt"]),
    ])

    raw = (res.content or "").strip()
    state["executor_raw"] = raw

    (plan_dir / "executor.prompt.json").write_text(state["executor_prompt"], encoding="utf-8")
    (plan_dir / "executor.raw.txt").write_text(raw, encoding="utf-8")

    # ---- Parse JSON strictly ----
    try:
        json_text = _extract_json_object_only(raw)
        data = json.loads(json_text) if json_text else {}

        if not isinstance(data, dict):
            raise ValueError("executor output is not a JSON object")

        writes = data.get("files_to_write", [])
        deletes = data.get("files_to_delete", [])

        if writes is None:
            writes = []
        if deletes is None:
            deletes = []

        if not isinstance(writes, list) or not isinstance(deletes, list):
            raise ValueError("files_to_write/files_to_delete must be lists")

        # Validate each write entry
        cleaned_writes: list[dict] = []
        for item in writes:
            if not isinstance(item, dict):
                raise ValueError("files_to_write entries must be objects")
            path = (item.get("path") or "").strip()
            content = item.get("content")

            if not path:
                raise ValueError("files_to_write entry missing path")
            if path not in allowed_paths:
                raise ValueError(f"write path not allowed: {path}")
            if content is None or not isinstance(content, str):
                raise ValueError(f"write content must be string for: {path}")

            cleaned_writes.append({"path": path, "content": content})

        # Validate deletes
        cleaned_deletes: list[str] = []
        for p in deletes:
            if not isinstance(p, str):
                raise ValueError("files_to_delete entries must be strings")
            rp = p.strip()
            if not rp:
                continue
            if rp not in allowed_paths:
                raise ValueError(f"delete path not allowed: {rp}")
            cleaned_deletes.append(rp)

        state["executor_result"] = data
        state["files_to_write"] = cleaned_writes
        state["files_to_delete"] = cleaned_deletes

        (plan_dir / "executor.result.json").write_text(
            json.dumps(
                {
                    "files_to_write": [{"path": x["path"], "content_len": len(x["content"])} for x in cleaned_writes],
                    "files_to_delete": cleaned_deletes,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        state["msg"] = state.get("msg", "") + f" | executor ok (writes={len(cleaned_writes)} deletes={len(cleaned_deletes)})"

        state["rollback_reason"] = ""

        #update meta file
        meta.update({
            "executor_ok": True,
            "executor_writes": len(state.get("files_to_write") or []),
            "executor_deletes": len(state.get("files_to_delete") or []),
        })
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return state

    except Exception as e:
        err = str(e)
        state["executor_result"] = {}
        state["files_to_write"] = []
        state["files_to_delete"] = []
        state["rollback_reason"] = "invalid_executor_json"
        state["executor_feedback"] = f"EXECUTOR_INVALID_JSON: {err}"

        # add block tries
        state["block_attempt"] = state.get("block_attempt", 0) + 1

        # if block tries exceed
        if state["block_attempt"] >= state.get("max_block_attempts", 5):
            state["rollback_reason"] = "block_attempt_exhausted"
            state["rollback_commit"] = state["plan_base_commit"]
            state["replan_trigger"] = "block_attempt_exhausted"

        (plan_dir / "executor.parse_error.txt").write_text(err + "\n", encoding="utf-8")
        state["msg"] = state.get("msg", "") + f" | executor FAIL(parse): {err}"

        #update meta file
        meta.update({
            "executor_ok": False,
            "executor_error": err,
            "rollback_reason": state.get("rollback_reason"),
        })
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return state

def after_executor(state: State) -> str:
    # if executor not failed parsing JSON, and produced files to write/delete
    if (state.get("rollback_reason") in (None, "", "unknown")
        and ((state.get("files_to_write") or []) or (state.get("files_to_delete") or []))):
        return "apply_files"

    # block attempts are reached
    if state.get("rollback_reason") == "block_attempt_exhausted":
        return "rollback"

    # if failed, retry
    return "retry_executor"

def retry_executor_node(state: State) -> State:
    run_dir = Path(state["run_dir"])
    plan_dir = _get_plan_dir(state)

    # create a unique retry log file
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    rid = uuid.uuid4().hex[:8]

    reason = state.get("rollback_reason", "")
    
    attempt = state.get("block_attempt", 0)
    fname = f"retry.block_attempt{attempt}.{ts}_{rid}.txt"

    content = (
        f"attempt={attempt}\n"
        f"reason={reason}\n"
        f"feedback={state.get('executor_feedback','')}\n"
    )
    (plan_dir / fname).write_text(content, encoding="utf-8")

    index_line = json.dumps(
        {
            "file": fname,
            "attempt": attempt,
            "reason": reason,
            "ts": ts,
        },
        ensure_ascii=False,
    )
    with open(plan_dir / "retry.index.jsonl", "a", encoding="utf-8") as f:
        f.write(index_line + "\n")

    state["msg"] = state.get("msg", "") + f" | retry logged={fname}"
    return state

def apply_files_node(state: State) -> State:
    # load log files
    repo_path = Path(state["repo_path"]).resolve()
    run_dir = Path(state["run_dir"])
    plan_dir = _get_plan_dir(state)

    meta_path = run_dir / "meta.json"
    meta = _load_meta_or_init(meta_path, repo_path, state.get("base_commit"))

    # load executor suggestions
    files_to_write = state.get("files_to_write") or []
    files_to_delete = state.get("files_to_delete") or []

    # allowed paths
    allowed = set()
    for fp in (state.get("executor_files") or []):
        allowed.add(_to_repo_rel(repo_path, fp))

    write_paths = [f.get("path", "") for f in files_to_write]

    # validate paths before applying (create or edit files)
    ok, err = _validate_allowed_paths(write_paths, allowed)
    if not ok:
        state["apply_ok"] = False
        state["apply_error"] = err
        state["rollback_reason"] = "apply_files_invalid_paths"
        state["executor_feedback"] = f"APPLY_FILES_INVALID_PATHS: {err}"
        (plan_dir / "apply_files.error.txt").write_text(err + "\n", encoding="utf-8")
        meta.update({"apply_ok": False, "apply_error": err, "rollback_reason": state["rollback_reason"]})
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        state["msg"] = state.get("msg","") + f" | apply_files FAIL: {err}"

        state["block_attempt"] = state.get("block_attempt", 0) + 1

        if state["block_attempt"] >= state.get("max_block_attempts", 5):
            state["rollback_reason"] = "block_attempt_exhausted"
            state["rollback_commit"] = state["plan_base_commit"]
            state["replan_trigger"] = "block_attempt_exhausted"

        return state

    # validate paths before applying (delete files)
    ok, err = _validate_allowed_paths(files_to_delete, allowed)
    if not ok:
        state["apply_ok"] = False
        state["apply_error"] = err
        state["rollback_reason"] = "apply_files_invalid_delete_paths"
        state["executor_feedback"] = f"APPLY_FILES_INVALID_DELETE_PATHS: {err}"
        (plan_dir / "apply_files.error.txt").write_text(err + "\n", encoding="utf-8")
        meta.update({"apply_ok": False, "apply_error": err, "rollback_reason": state["rollback_reason"]})
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        state["msg"] = state.get("msg","") + f" | apply_files FAIL: {err}"
        return state

    # apply deletions to avoid conflicts
    deleted = []
    for rp in files_to_delete:
        abs_p = (repo_path / rp).resolve()
        if repo_path not in abs_p.parents and abs_p != repo_path:
            continue
        if abs_p.exists() and abs_p.is_file():
            abs_p.unlink()
            deleted.append(rp)

    # write or edit files
    written = [] # create a list of written files
    for f in files_to_write: # each file dict
        rp = f["path"] # get (new?) path
        content = f.get("content", "") # get new content
        abs_p = (repo_path / rp).resolve() # get absolute path
        if repo_path not in abs_p.parents and abs_p != repo_path: # guarantee inside repo
            continue
        abs_p.parent.mkdir(parents=True, exist_ok=True) # create new packages if needed
        abs_p.write_text(content, encoding="utf-8") # create or edit file
        written.append(rp) # register written file

    # log do que foi aplicado
    (plan_dir / "apply_files.written.json").write_text(json.dumps(written, indent=2), encoding="utf-8")
    (plan_dir / "apply_files.deleted.json").write_text(json.dumps(deleted, indent=2), encoding="utf-8")

    state["apply_ok"] = True
    state["apply_error"] = ""
    state["msg"] = state.get("msg","") + f" | apply_files ok (write={len(written)} del={len(deleted)})"

    meta.update({
        "apply_ok": True,
        "apply_error": "",
        "apply_written": len(written),
        "apply_deleted": len(deleted),
    })
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return state

def after_apply_files(state: State) -> str:
    if state.get("apply_ok"):
        return "compile"

    # decide based on current block_attempt
    if state.get("block_attempt", 0) < state.get("max_block_attempts", 5):
        return "retry_executor"

    return "rollback"

def rollback_node(state: State) -> State:
    repo_path = Path(state["repo_path"]).resolve()

    if state.get("rollback_commit"):
        base = state.get("rollback_commit")
    else:
        base = state.get("base_commit")


    run_dir_str = state.get("run_dir")
    run_dir = Path(run_dir_str) if run_dir_str else None

    plan_dir = _get_plan_dir(state)

    if not base:
        state["msg"] = state.get("msg", "") + " | rollback skipped (no base_commit)"
        return state

    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        statusp = _run(["git", "status", "--porcelain=v1"], cwd=repo_path)
        (plan_dir / "git_status_before_rollback.txt").write_text(
            (statusp.stdout or "") + ("\n" if statusp.stdout else "") + (statusp.stderr or ""),
            encoding="utf-8",
        )

    _run(["git", "reset", "--hard", base], cwd=repo_path)
    _run(["git", "clean", "-fd", "-e", "agent_runs/", "-e", "tmp/"], cwd=repo_path)

    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)

    state["msg"] = state.get("msg", "") + " | rollback done"

    if run_dir:
        meta_path = run_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            meta = {"repo_path": str(repo_path), "base_commit": base, "note": "meta.json was missing; created during rollback"}

        meta["rolled_back"] = True
        meta["rollback_reason"] = state.get("rollback_reason", "unknown")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        (plan_dir / "rollback.reason.txt").write_text(meta["rollback_reason"] + "\n", encoding="utf-8")

    return state

def after_rollback(state: State) -> str:
    if state.get("replan_trigger") or state.get("rollback_reason") in {
        "block_attempt_exhausted",
        "smell_persist_force_rollback",
        "compile_failed",
    }:
        return "prepare_replan"
    return END

def compile_node(state: State) -> State:
    repo_path = Path(state["repo_path"]).resolve()

    run_dir = Path(state["run_dir"])
    plan_dir = _get_plan_dir(state)

    tmp_dir = repo_path / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    '''cmd = ["mvn", "-q",
       #"-DskipTests",
       "-Drat.skip=true",
       "-Dcheckstyle.skip=true",
       "-Dspotbugs.skip=true",
       "-Dpmd.skip=true",
       "-DskipITs",
       "clean", "verify"]

    env = os.environ.copy()
    env["MAVEN_OPTS"] = f"-Xshare:off -Djava.io.tmpdir={tmp_dir}"

    p = _run(cmd, cwd=repo_path, env=env)'''

    p = _run_build(repo_path, tmp_dir)

    state["compile_returncode"] = p.returncode
    state["compile_ok"] = (p.returncode == 0)
    state["maven_tmp_dir"] = str(tmp_dir)

    combined = (p.stdout or "") + "\n" + (p.stderr or "")

    (plan_dir / "compile.log").write_text(combined, encoding="utf-8")

    state["compile_log_tail"] = _tail(combined, 40)

    if state["compile_ok"]:
        state["msg"] = state.get("msg", "") + " | compile ok"
    else:
        state["msg"] = state.get("msg", "") + " | compile FAIL"
        state["rollback_reason"] = "compile_failed"
        state["executor_feedback"] = "COMPILE_FAILED:\n" + state.get("compile_log_tail", "")

    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(
        {
            "compile_ok": state["compile_ok"],
            "compile_returncode": state["compile_returncode"],
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return state

def after_compile(state: State) -> str:
    if state.get("compile_ok"):
        return "promote_baseline"
    return "rollback"

def after_stage_block(state: State) -> str:
    if state.get("done"):
        return "designite"
    return "resolve_files"

def advance_block_node(state: State) -> State:
    # get next planned block
    state["block_idx"] = state.get("block_idx", 0) + 1

    # clean state
    state["staged_block"] = {}
    state["staged_block_ops"] = []
    state["staged_block_files"] = []
    state["executor_files"] = []
    state["executor_existing_files"] = []
    state["executor_new_files"] = []
    state["files_to_write"] = []
    state["files_to_delete"] = []
    state["apply_ok"] = False
    state["apply_error"] = ""
    state["rollback_reason"] = ""

    state["block_attempt"] = 0

    state["msg"] = state.get("msg", "") + f" | advance_block -> {state['block_idx']}"
    return state

# promote baseline node, to maintain updated baseline after successful run
def promote_baseline_node(state: State) -> State:
    repo_path = Path(state["repo_path"]).resolve()
    run_dir = Path(state["run_dir"])
    plan_dir = _get_plan_dir(state)

    # if compile failed, skip
    if not state.get("compile_ok"):
        state["msg"] = state.get("msg", "") + " | promote_baseline skipped (compile not ok)"
        return state

    # get the current block idx and id
    idx = state.get("block_idx", 0)
    block_id = state.get("staged_block_id", idx)

    # check if any change
    diff = _run(["git", "diff", "--name-only"], cwd=repo_path)
    dirty = bool((diff.stdout or "").strip())

    promoted = False

    # if had uncommitted changes, commit them as new baseline
    if dirty:
        _run(
            ["git", "add", "-A", "--", ".", ":!agent_runs/", ":!tmp/"],
            cwd=repo_path,
        )
        msg = f"agent: apply block {block_id} (idx={idx})"
        c = _run(["git", "commit", "-m", msg], cwd=repo_path)
        if c.returncode != 0:
            err = _tail((c.stdout or "") + "\n" + (c.stderr or ""), 80)
            state["rollback_reason"] = "baseline_commit_failed"
            state["msg"] = state.get("msg", "") + " | promote_baseline FAIL(commit)"
            (plan_dir / "baseline.commit.error.txt").write_text(err + "\n", encoding="utf-8")
            raise RuntimeError(f"Baseline commit failed:\n{err}")
        promoted = True

    # define new baseline commit
    new_base = _git_current_commit(repo_path)
    state["base_commit"] = new_base

    # upload meta.json and logs
    meta_path = run_dir / "meta.json"
    meta = _load_meta_or_init(meta_path, repo_path, new_base)
    meta.update(
        {
            "base_commit": new_base,
            "baseline_promoted": promoted,
            "baseline_promoted_at": datetime.now().isoformat(),
            "baseline_block_idx": idx,
            "baseline_block_id": block_id,
            "baseline_dirty_before_commit": dirty,
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    (plan_dir / "baseline.promoted.txt").write_text(
        f"base_commit={new_base}\npromoted={promoted}\n", encoding="utf-8"
    )

    state["msg"] = state.get("msg", "") + f" | baseline promoted={promoted} @{new_base[:8]}"
    return state

def designite_node(state: State) -> State:
    repo_path = Path(state["repo_path"]).resolve()
    run_dir = Path(state["run_dir"])
    plan_dir = _get_plan_dir(state)

    # meta.json
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {
            "repo_path": str(repo_path),
            "base_commit": state.get("base_commit"),
        }

    # define early so except can reference it safely
    analysis_out = plan_dir / "designite_analysis"

    try:
        jar_env = os.getenv("DESIGNITE_JAR_PATH")
        if not jar_env:
            raise RuntimeError("DESIGNITE_JAR_PATH is not set")

        designite_jar = Path(jar_env).expanduser().resolve()
        if not designite_jar.exists() or not designite_jar.is_file():
            raise RuntimeError(f"Designite JAR not found at {designite_jar}")

        analysis_out.mkdir(parents=True, exist_ok=True)

        out_dir, cmd = _run_designite(repo_path, analysis_out, designite_jar)

        # check if smell was removed
        present = _designite_smell_present(
            designite_dir=out_dir,
            target_class_fqn=state.get("target_class_fqn", ""),
            smell_name=state.get("designite_smell_name", state.get("smell_type", "")),
            csv_name=state.get("designite_smells_csv", "DesignSmells.csv"),
        )
        state["smell_still_present"] = bool(present)

        state["msg"] = state.get("msg", "") + f" | smell_present={state['smell_still_present']}"

        meta.update({
            "smell_still_present": state["smell_still_present"],
            "smell_type": state.get("smell_type", ""),
            "designite_smell_name": state.get("designite_smell_name", ""),
        })
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # update state
        state["designite_ok"] = True
        state["msg"] = state.get("msg", "") + " | designite done"

        # persist command
        (plan_dir / "designite.cmd.txt").write_text(" ".join(cmd), encoding="utf-8")

        # update meta.json
        meta.update(
            {
                "designite_ok": True,
            }
        )
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return state

    except Exception as e:
        state["designite_ok"] = False
        state["rollback_reason"] = "designite_failed"
        state["msg"] = state.get("msg", "") + f" | designite FAIL: {e}"

        meta.update(
            {
                "designite_ok": False,
                "designite_error": str(e),
                "rollback_reason": "designite_failed",
            }
        )
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        raise

# case smell removed END
# case smell persists and plans <= 4, replan
# case smell persists and plans > 4, roolback
def after_designite(state: State) -> str:

    if state.get("smell_still_present"):
        
        # quantity plan tries
        if state["plan_idx"] <= 4:
            state["replan_trigger"] = "smell_persist_keep_progress"
            return "smell_quality_check"
        else:
            state["rollback_commit"] = state["plan_base_commit"]
            state["replan_trigger"] = "smell_persist_force_rollback"
            return "rollback"
    
    if not state.get("designite_ok"):
        return "rollback"

    return END

# agent responsible to explain why smell wasn't removed
def smell_quality_check_node(state: State) -> State:
    # load log paths
    repo_path = Path(state["repo_path"]).resolve()
    run_dir = Path(state["run_dir"])
    plan_dir = _get_plan_dir(state)

    # load meta.json
    meta_path = run_dir / "meta.json"
    meta = _load_meta_or_init(meta_path, repo_path, state.get("base_commit"))

    # get the prompt
    with open(state["smell_quality_prompt_path"], "r", encoding="utf-8") as f:
        PROMPT_TEMPLATE = f.read()

    if not PROMPT_TEMPLATE:
        state["smell_quality_ok"] = False
        state["smell_quality_error"] = "smell quality prompt missing"
        meta.update({"smell_quality_ok": False, "smell_quality_error": state["smell_quality_error"]})
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return state

    # get plan
    refactoring_plan = json.dumps(state.get("plan") or {}, indent=2)

    # get target class (code)
    target_file = state.get("target_file", "")
    _, refactoring_code = _read_target_file(repo_path, target_file)

    # reder the prompt
    rendered = (
        PROMPT_TEMPLATE
        .replace("{refactoring_plan}", refactoring_plan)
        .replace("{refactoring_code}", refactoring_code)
    )

    # define the LLM
    llm = ChatOpenAI(
        model=os.getenv("QUALITY_MODEL", "gpt-5-mini"),
        temperature=0.0,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    try:
        # run the inference
        res = llm.invoke([
            SystemMessage(content="Be concise. Follow the instructions exactly."),
            HumanMessage(content=rendered),
        ])

        # set the state
        analysis = (res.content or "").strip()
        state["smell_quality_ok"] = True
        state["smell_quality_analysis"] = analysis

        # generate the log files
        (plan_dir / "smell_quality.prompt.md").write_text(PROMPT_TEMPLATE, encoding="utf-8")
        (plan_dir / "smell_quality.input.md").write_text(rendered, encoding="utf-8")
        (plan_dir / "smell_quality.output.txt").write_text(analysis, encoding="utf-8")

        # set meta.json
        meta.update({
            "smell_quality_ok": True,
            "smell_quality_error": "",
        })
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        state["msg"] = state.get("msg", "") + " | smell_quality_check done"
        return state

    except Exception as e:
        err = str(e)
        state["smell_quality_ok"] = False
        state["smell_quality_error"] = err
        meta.update({"smell_quality_ok": False, "smell_quality_error": err})
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        state["msg"] = state.get("msg", "") + f" | smell_quality_check FAIL: {err}"
        return state

def prepare_replan_node(state: State) -> State:
    repo_path = Path(state["repo_path"]).resolve()
    run_dir = Path(state["run_dir"])

    # add plan counter
    state["plan_idx"] = state.get("plan_idx", 0) + 1
    state["smell_persist_replans"] += 1

    # check if reach plan threshold
    if state.get("plan_idx", 0) > 4:
        state["stop_reason"] = "max_plans_reached"
        state["msg"] = state.get("msg", "") + " | stop: max plans reached"
        return state

    old_rollback_reason = state.get("rollback_reason", "")
    old_replan_trigger = state.get("replan_trigger", "")
    last_error = state.get("executor_feedback", "")

    # new plan must start from block 0
    state["block_idx"] = 0
    state["block_attempt"] = 0

    # clean per-block state (same idea as advance_block_node)
    state["staged_block"] = {}
    state["staged_block_ops"] = []
    state["staged_block_files"] = []
    state["executor_files"] = []
    state["executor_existing_files"] = []
    state["executor_new_files"] = []
    state["files_to_write"] = []
    state["files_to_delete"] = []
    state["apply_ok"] = False
    state["apply_error"] = ""

    # start-of-plan bookkeeping
    state["plan_base_commit"] = state.get("base_commit")  # base_commit is the current baseline commit
    state["rollback_commit"] = ""                         # prevent leaking rollback target
    state["rollback_reason"] = old_rollback_reason        # optional: avoid leaking into planner_input
    state["replan_trigger"] = old_replan_trigger          # very important: avoid infinite replan loop

    # create new plan folder
    plan_dir = _get_plan_dir(state)

    '''
    # get target code
    target_rel, target_code = _read_target_file(repo_path, state["target_file"])

    # get old plan to pass for new prompt
    previous_plan = state.get("plan") or {}

    # input for the new plan
    planner_input = {
        "smell": state.get("smell_type"),
        "target_file": target_rel,
        "target_code": target_code,
        "previous_plan": previous_plan,
        "replan_reason": state.get("rollback_reason") or state.get("replan_trigger") or "",
        "last_error": state.get("executor_feedback", ""),
        "smell_persist_analysis": state.get("smell_quality_analysis", ""),
    }
    '''

    previous_plan = state.get("plan") or {}
    target_type = _get_target_type(state)

    if target_type == "class":
        target_rel, target_code = _read_target_file(repo_path, state["target_file"])

        planner_input = {
            "smell": state.get("smell_type"),
            "target_type": "class",
            "target_name": state.get("target_name", ""),
            "target_file": target_rel,
            "target_code": target_code,
            "previous_plan": previous_plan,
            "replan_reason": old_rollback_reason or old_replan_trigger,
            "last_error": last_error,
            "smell_persist_analysis": state.get("smell_quality_analysis", ""),
        }

    elif target_type == "package":

        state = resolve_target_package_node(state)

        planner_input = json.loads(state["planner_input_json"])
        planner_input.update({
            "previous_plan": previous_plan,
            "replan_reason": old_rollback_reason or old_replan_trigger,
            "last_error": last_error,
            "smell_persist_analysis": state.get("smell_quality_analysis", ""),
        })

        state["planner_input_json"] = json.dumps(planner_input, indent=2)

        '''planner_input = {
            "smell": state.get("smell_type"),
            "target_type": "package",
            "target_name": state.get("target_name", ""),
            "target_source_root": state.get("target_source_root", ""),
            "target_files": state.get("target_files", []),
            "internal_deps": state.get("internal_deps", []),
            "previous_plan": previous_plan,
            "replan_reason": old_rollback_reason or old_replan_trigger,
            "last_error": last_error,
            "smell_persist_analysis": state.get("smell_quality_analysis", ""),
        }'''

    else:
        raise RuntimeError(f"unsupported target_type: {target_type}")

    state["planner_input_json"] = json.dumps(planner_input, indent=2)
    (plan_dir / "planner.replan.input.json").write_text(
        state["planner_input_json"],
        encoding="utf-8"
    )

    # meta.json update
    meta_path = run_dir / "meta.json"
    meta = _load_meta_or_init(meta_path, repo_path, state.get("base_commit"))

    meta.update({
        "plan_idx": state["plan_idx"],
        "plan_dir": str(plan_dir),
        "smell_persist_replans": state.get("smell_persist_replans", 0),
        "replan_trigger": state.get("replan_trigger", ""),
    })
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return state

# Is the max plans reached? STOP! Else, replan
def after_prepare_replan(state: State) -> str:
    # plan_00..plan_04
    #if state.get("plan_idx", 0) > 4:
    #    return END
    if "stop: max plans reached" in (state.get("msg") or ""):
        return END
    return "planner"
    
def build_graph():
    g = StateGraph(State)

    g.add_node("route", route_node)
    g.add_node("init_run", init_run_node)
    g.add_node("resolve_target_class", resolve_target_class_node)
    g.add_node("resolve_target_package", resolve_target_package_node)
    g.add_node("planner", planner_node)
    g.add_node("stage_block", stage_block_node)
    g.add_node("resolve_files", resolve_files_for_block_node)
    g.add_node("lock_workspace", lock_workspace_node)
    g.add_node("executor", executor_node)
    g.add_node("apply_files", apply_files_node)
    g.add_node("retry_executor", retry_executor_node)
    g.add_node("compile", compile_node)
    g.add_node("promote_baseline", promote_baseline_node)
    g.add_node("designite", designite_node)
    g.add_node("smell_quality_check", smell_quality_check_node)
    g.add_node("prepare_replan", prepare_replan_node)
    g.add_node("advance_block", advance_block_node)
    g.add_node("rollback", rollback_node)

    g.set_entry_point("route")
    g.add_edge("route", "init_run")

    g.add_conditional_edges(
        "init_run",
        after_init_run,
        {
            "resolve_target_class": "resolve_target_class",
            "resolve_target_package": "resolve_target_package"
        },
    )

    g.add_edge("resolve_target_class", "planner")
    g.add_edge("resolve_target_package", "planner")

    g.add_edge("planner", END)

    '''g.add_conditional_edges(
        "planner",
        after_planner,
        {
            "stage_block": "stage_block",
            END: END,
        },
    )'''

    g.add_conditional_edges(
        "stage_block",
        after_stage_block,
        {
            "resolve_files": "resolve_files",
            "designite": "designite",
        },
    )
    
    g.add_edge("resolve_files", "lock_workspace")
    g.add_edge("lock_workspace", "executor")

    g.add_conditional_edges(
        "executor",
        after_executor,
        {
            "apply_files": "apply_files",
            "retry_executor": "retry_executor",
            "rollback": "rollback",
        },
    )

    g.add_edge("retry_executor", "lock_workspace")

    g.add_conditional_edges(
        "apply_files",
        after_apply_files,
        {
            "compile": "compile",
            "retry_executor": "retry_executor",
            "rollback": "rollback",
        },
    )

    g.add_conditional_edges(
        "compile",
        after_compile,
        {
            "promote_baseline": "promote_baseline",   # compile ok
            "rollback": "rollback",    # compile fail
        },
    )

    g.add_edge("promote_baseline", "advance_block")
    g.add_edge("advance_block", "stage_block")

    g.add_conditional_edges(
        "designite",
        after_designite,
        {
            "smell_quality_check": "smell_quality_check",
            "rollback": "rollback",
            END: END,
        },
    )

    g.add_edge("smell_quality_check", "prepare_replan")

    g.add_conditional_edges(
        "prepare_replan",
        after_prepare_replan,
        {
            "planner": "planner",
            END: END
        },
    )


    g.add_conditional_edges(
        "rollback",
        after_rollback,
        {
            "prepare_replan": "prepare_replan",
            END: END,
        },
    )

    return g.compile()

if __name__ == "__main__":

    # TODO: check if Java version pass by project build
    # TODO: check if target has designite smell, if not, exit early

    # load environment variables from .env file
    dotenv.load_dotenv()
    repo_path = Path(os.getenv("REPO_PATH")).resolve()

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--smell", required=True, help="Type of smell (GC, HM or IM)")
    parser.add_argument("--target", required=True, help="FQN of target class or package")
    args = parser.parse_args()
    smell = (args.smell or "").strip() # --smell
    target_name = (args.target or "").strip() # --target

    # check valid smells
    if smell not in {"GC", "HM", "IM"}:
        raise ValueError("Invalid smell type. Must be 'GC', 'HM' or 'IM'.")

    # return target data
    target_type = _infer_target_type_from_name(target_name)
    target_path = Fqn(target_name).find_in_repo(repo_path)

    # check valid target FQN
    if target_path is None:
        raise ValueError(f"Target {target_name} not found in {repo_path}.")

    # load planner prompt template
    # TODO: read template inside planner node
    with open(f"data/prompts/planner_{smell}.prompt", "r", encoding="utf-8") as f:
        PROMPT_TEMPLATE = f.read()
    
    planner_input = {}

    # prepare planner input based on smell type and target type
    if target_type == "class":
        # check if target path is a file
        if not target_path.is_file():
            raise ValueError(f"Class target must resolve to a file, got: {target_path}")

        target_file = str(target_path.relative_to(repo_path)).replace("\\", "/")
        target_rel, target_code = _read_target_file(repo_path, target_file)

        # insufficient modularization
        if smell == "IM":
            planner_input = {
                "smell": "Insufficient Modularization",
                "target_file": target_rel,
                "target_code": target_code,
            }
        # hub-like modularization
        elif smell == "HM":
            observed_external_calls = extract_observed_external_calls(target_code)
            planner_input = {
                "smell": "Hub-like Modularization", # smell type
                "target_file": target_rel, # path to the target class
                "target_code": target_code, # raw code of target class
                "observed_external_calls": observed_external_calls, # list of external calls
            }
        else:
            raise ValueError("Invalid smell type for class target. Must be 'IM' or 'HM'.")
        
        invoke_input = {
            "repo_path": str(repo_path),
            "target_name": target_name,
            "target_type": target_type,
            "target_file": target_rel,
            "planner_prompt": PROMPT_TEMPLATE,
            "planner_input_json": json.dumps(planner_input, indent=2),
        }
        
    elif target_type == "package":
        # check if target path is a directory
        if not target_path.is_dir():
            raise ValueError(f"Package target must resolve to a directory, got: {target_path}")
        
        # god component
        if smell == "GC":
            planner_input = {
                "smell": "God Component", # smell type
                "target_name": target_name,
            }
        else:
            raise ValueError("Invalid smell type for package target. Must be 'GC'.")
        
        invoke_input = {
            "repo_path": str(repo_path),
            "target_name": target_name,
            "target_type": target_type,
            "planner_prompt": PROMPT_TEMPLATE,
            "planner_input_json": json.dumps(planner_input, indent=2),
        }

    app = build_graph()
    out = app.invoke(invoke_input)

    open(f"{repo_path}/state.json", "w", encoding="utf-8").write(json.dumps(out, indent=2))