import csv
import dotenv
import json
import os
import re
import shutil
import subprocess
import uuid

from datetime import datetime
from pathlib import Path
from typing import TypedDict, Tuple

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END

class State(TypedDict, total=False):
    repo_path: str
    msg: str
    run_dir: str

    target_file: str
    target_class_fqn: str
    target_source_root: str

    start_commit: str
    base_commit: str

    # plan lifecycle
    plan_idx: int # no of plan tries
    plan_base_commit: str # commit before apply the plan
    plan_dir: str # log dir of current plan
    smell_persist_replans: int # replan counter
    replan_trigger: str

    # planning data
    planner_prompt: str # plan prompt template
    planner_input_json: str # input data
    plan_json_text: str 
    plan: dict # the plan
    plan_ok: bool # status after try to generate plan
    plan_error: str # tail when plan generation failed

    # blocks of plan
    block_idx: int
    staged_block: dict
    staged_block_id: int
    staged_block_files: list[str]
    staged_block_ops: list[dict]
    done: bool

    block_attempt: int # counter of tries per block
    max_block_attempts: int # threshold tries

    # executor data
    executor_files: list[str]
    executor_new_files: list[str]
    executor_existing_files: list[str]
    workspace_commit: str
    executor_prompt: str
    executor_raw: str
    executor_feedback: str

    executor_result: dict                 # executor result data
    files_to_write: list[dict]            # each: {"path": "...", "content": "..."}
    files_to_delete: list[str]            # repo-relative paths

    # apply files node
    apply_ok: bool
    apply_error: str

    # compilation data
    compile_ok: bool
    compile_returncode: int
    compile_log_tail: str
    maven_tmp_dir: str

    # designite/smell data
    designite_ok: bool                 # if designite run successfully
    smell_type: str                    # eg: "Insufficient Modularization"
    designite_smells_csv: str          # designite smell file eg: "DesignSmells.csv"
    designite_smell_name: str          # smell label on designite output
    smell_still_present: bool          # smell remove evaluation

    rollback_reason: str
    rollback_commit: str

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

# get target file/class
def _read_target_file(repo_path: Path, target_file: str) -> tuple[str, str]:

    # if target file is invalid or null
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

    code = abs_p.read_text(encoding="utf-8", errors="replace")
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

    #smell_name = "Insufficient Modularization" # TODO: remove after tests

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

def route_node(state: State) -> State:
    state["msg"] = f"route ok: repo_path={state.get('repo_path')}"
    return state

def init_run_node(state: State) -> State:
    repo_path = Path(state["repo_path"]).resolve()

    # get the start commit
    head = _git_current_commit(repo_path)

    # defines start/base commit if not set
    if not state.get("start_commit"):
        state["start_commit"] = head

    if not state.get("base_commit"):
        state["base_commit"] = head

    # create framework folder, if it not exists
    runs_root = repo_path / "agent_runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    # smell config
    smell_type = ""
    try:
        inp = json.loads(state.get("planner_input_json", "") or "{}")
        smell_type = str(inp.get("smell", "")).strip()
    except Exception:
        smell_type = ""
    state["smell_type"] = smell_type
    
    target_rel, code = _read_target_file(repo_path, state["target_file"])
    state["target_file"] = target_rel
    
    target_fqn = _extract_fqn_from_java(code, target_rel)
    state["target_class_fqn"] = target_fqn
    state["target_source_root"] = _infer_source_root_from_target(repo_path, target_rel, target_fqn)

    # create log folder for a run
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = uuid.uuid4().hex[:8]
    run_dir = runs_root / f"{ts}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    state["run_dir"] = str(run_dir)

    state.setdefault("designite_smell_name", state["smell_type"])
    state.setdefault("designite_smells_csv", "DesignSmells.csv")  # ou outro no futuro

    meta = {
        "repo_path": str(repo_path),
        "start_commit": state["start_commit"],
        "base_commit": state["base_commit"],
        "smell_type": state.get("smell_type", ""),
        "designite_smell_name": state.get("designite_smell_name", ""),
        "designite_smells_csv": state.get("designite_smells_csv", "DesignSmells.csv"),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    state["msg"] = state.get("msg", "") + f" | run_dir={run_dir.name}"

    state.setdefault("executor_feedback", "")

    # plan lifecycle init
    state.setdefault("plan_idx", 0)
    state.setdefault("smell_persist_replans", 0)

    ## give a workfull commit to the plan
    state["plan_base_commit"] = state.get("base_commit") or head

    plan_dir = _get_plan_dir(state)
    #state["plan_dir"] = str(plan_dir)

    ## update meta.json
    meta.update({
        "plan_idx": state["plan_idx"],
        "plan_base_commit": state["plan_base_commit"],
        "plan_dir": str(plan_dir),
        "smell_persist_replans": state["smell_persist_replans"],
    })
    (Path(state["run_dir"]) / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    state["smell_type"] = smell_type or state.get("smell_type", "Insufficient Modularization")
    state["designite_smells_csv"] = state.get("designite_smells_csv", "DesignSmells.csv")

    # Por padrão, assume que o nome do smell no Designite é igual ao "smell"
    # (no futuro você pode mapear aqui, se os nomes divergirem)
    state["designite_smell_name"] = state.get("designite_smell_name", state["smell_type"])

    return state

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

    existing: set[str] = set()
    for f in (state.get("staged_block_files") or []):
        p = Path(f)
        if p.is_absolute():
            existing.add(str(p))
        else:
            existing.add(str((repo_path / p).resolve()))

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

'''def after_executor(state: State) -> str:
    # if executor not failed parsing JSON, and produced files to write/delete
    if (state.get("rollback_reason") in (None, "", "unknown")
        and ((state.get("files_to_write") or []) or (state.get("files_to_delete") or []))):
        return "apply_files"

    # if executor failed, retry
    state["block_attempt"] = state.get("block_attempt", 0) + 1 # add attempt
    
    if state["block_attempt"] < state.get("max_block_attempts", 5): # if not exceed attempts
        return "retry_executor"
    
    state["rollback_reason"] = "block_attempt_exhausted"
    state["rollback_commit"] = state["plan_base_commit"]
    state["replan_trigger"] = "block_attempt_exhausted"
    
    return "rollback" # if all attempts exhausted, replan'''

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

    cmd = ["mvn", "-q",
       #"-DskipTests",
       "-Drat.skip=true",
       "-Dcheckstyle.skip=true",
       "-Dspotbugs.skip=true",
       "-Dpmd.skip=true",
       "-DskipITs",
       "clean", "verify"]

    env = os.environ.copy()
    env["MAVEN_OPTS"] = f"-Xshare:off -Djava.io.tmpdir={tmp_dir}"

    p = _run(cmd, cwd=repo_path, env=env)

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

def _run_designite(
    repo_path: Path,
    out_dir: Path,
    jar_path: Path,
) -> Tuple[Path, list[str]]:

    if out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "java", "-jar", str(jar_path),
        "-i", str(repo_path),
        "-o", str(out_dir),
    ]

    p = _run(cmd, cwd=repo_path)

    log = (p.stdout or "") + ("\n" if p.stdout else "") + (p.stderr or "")
    (out_dir / "designite.log").write_text(log, encoding="utf-8")

    if p.returncode != 0:
        raise RuntimeError(
            f"Designite failed (rc={p.returncode}). See log at {out_dir / 'designite.log'}"
        )

    return out_dir, cmd

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

def after_designite(state: State) -> str:

    if state.get("smell_still_present"):
        run_dir = Path(state["run_dir"])
        repo_path = Path(state["repo_path"]).resolve()

        meta_path = run_dir / "meta.json"
        meta = _load_meta_or_init(meta_path, repo_path, state.get("base_commit"))
        
        state["smell_persist_replans"] += 1
        
        meta.update(
            {
                "smell_persist_replans": state["smell_persist_replans"],
            }
        )
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        
        # quantity plan tries
        if state["plan_idx"] <= 4:
            state["replan_trigger"] = "smell_persist_keep_progress"
            return "prepare_replan"
        else:
            state["rollback_commit"] = state["plan_base_commit"]
            state["replan_trigger"] = "smell_persist_force_rollback"
            return "rollback"
    
    if not state.get("designite_ok"):
        return "rollback"

    return END

def prepare_replan_node(state: State) -> State:
    repo_path = Path(state["repo_path"]).resolve()
    run_dir = Path(state["run_dir"])

    if state.get("plan_idx", 0) > 4:
        state["msg"] = state.get("msg", "") + " | stop: max plans reached"
        return state

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

    # add plan counter
    state["plan_idx"] = state.get("plan_idx", 0) + 1

    # start-of-plan bookkeeping
    state["plan_base_commit"] = state.get("base_commit")  # base_commit is the current baseline commit
    state["rollback_commit"] = ""                         # prevent leaking rollback target
    state["rollback_reason"] = ""                         # optional: avoid leaking into planner_input
    state["replan_trigger"] = ""                          # very important: avoid infinite replan loop

    # create new plan folder
    plan_dir = _get_plan_dir(state)

    # get target code
    target_rel, target_code = _read_target_file(repo_path, state["target_file"])

    # get old plan to pass for new prompt
    previous_plan = state.get("plan") or {}

    # input for the new plan
    planner_input = {
        "smell": state.get("smell_type", "Insufficient Modularization"),
        "target_file": target_rel,
        "target_code": target_code,
        "previous_plan": previous_plan,
        "replan_reason": state.get("rollback_reason") or state.get("replan_trigger") or "",
        "last_error": state.get("executor_feedback", ""),
    }

    state["planner_input_json"] = json.dumps(planner_input, indent=2)

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

def after_prepare_replan(state: State) -> str:
    # max 5 plans total: plan_00..plan_04
    if state.get("plan_idx", 0) > 4:
        return END
    return "planner"
    
def build_graph():
    g = StateGraph(State)

    g.add_node("route", route_node)
    g.add_node("init_run", init_run_node)
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
    g.add_node("prepare_replan", prepare_replan_node)
    g.add_node("advance_block", advance_block_node)
    g.add_node("rollback", rollback_node)

    g.set_entry_point("route")
    g.add_edge("route", "init_run")
    g.add_edge("init_run", "planner")

    g.add_conditional_edges(
        "planner",
        after_planner,
        {
            "stage_block": "stage_block",
            END: END,
        },
    )

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

    g.add_conditional_edges(
        "designite",
        after_designite,
        {
            "prepare_replan": "prepare_replan",
            "rollback": "rollback",
            END: END,
        },
    )

    g.add_edge("advance_block", "stage_block")

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
    dotenv.load_dotenv()

    app = build_graph()

    with open("data/prompts/planner_agent.prompt", "r", encoding="utf-8") as f:
        PROMPT_TEMPLATE = f.read()

    REPO_PATH = "data/repositories/commons-lang"
    TARGET_FILE = "src/main/java/org/apache/commons/lang3/time/FastDatePrinter.java"

    repo_path = Path(REPO_PATH).resolve()
    target_rel, target_code = _read_target_file(repo_path, TARGET_FILE)

    planner_input = {
        "smell": "Insufficient Modularization",
        "target_file": target_rel,
        "target_code": target_code,
    }

    out = app.invoke(
        {
            "repo_path": str(repo_path),
            "target_file": target_rel,
            "planner_prompt": PROMPT_TEMPLATE,
            "planner_input_json": json.dumps(planner_input, indent=2)
        }
    )