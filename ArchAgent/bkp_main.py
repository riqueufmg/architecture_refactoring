import json
import os
import dotenv
import subprocess
import uuid

from datetime import datetime
from pathlib import Path
from pprint import pprint
from typing import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END


class State(TypedDict, total=False):
    repo_path: str
    msg: str
    run_dir: str

    # planning data
    planner_prompt: str
    planner_input_json: str
    plan_json_text: str
    plan: dict
    plan_ok: bool
    plan_error: str

    # blocks of plan
    block_idx: int
    staged_block: dict
    staged_block_id: int
    staged_block_files: list[str]
    staged_block_ops: list[dict]
    done: bool

    # executor data
    executor_files: list[str]
    executor_new_files: list[str]
    executor_existing_files: list[str]
    workspace_commit: str
    executor_prompt: str
    executor_raw: str

    executor_attempt: int
    executor_feedback: str
    max_attempts: int

    # patch application data
    patch_text: str
    patch_applied: bool
    patch_apply_log_tail: str
    git_diff_tail: str
    base_commit: str

    patch_valid: bool
    patch_validation_error: str

    # compilation data
    compile_ok: bool
    compile_returncode: int
    compile_log_tail: str
    maven_tmp_dir: str

    rollback_reason: str


def _run(cmd: list[str], cwd: Path, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
    )


def _extract_json_only(s: str) -> str:
    s = (s or "").strip()

    # Extract from ```json ... ```
    if "```" in s:
        parts = s.split("```")
        candidates: list[str] = []
        for p in parts:
            t = p.strip()
            if t.startswith("json"):
                t = t[4:].strip()
            if t.startswith("{") and t.endswith("}"):
                candidates.append(t)
        if candidates:
            return max(candidates, key=len)

    return s


def _load_meta_or_init(meta_path: Path, repo_path: Path, base_commit: str | None) -> dict:
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"repo_path": str(repo_path), "base_commit": base_commit}


def _java_fqn_to_path(repo_path: Path, class_fqn: str) -> str:
    # Ex: org.jsoup.internal.JsoupParser -> src/main/java/org/jsoup/internal/JsoupParser.java
    rel = Path("src/main/java") / Path(class_fqn.replace(".", "/") + ".java")
    return str((repo_path / rel).resolve())


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


def _touched_paths_from_diff(patch_text: str) -> set[str]:
    touched: set[str] = set()
    for line in (patch_text or "").splitlines():
        if line.startswith("diff --git "):
            # diff --git a/xxx b/yyy
            parts = line.split()
            if len(parts) >= 4:
                a_path = parts[2]
                b_path = parts[3]
                if a_path.startswith("a/"):
                    touched.add(a_path[2:])
                if b_path.startswith("b/"):
                    touched.add(b_path[2:])
    return touched


def _validate_patch_basic(patch_text: str, allowed_paths: set[str]) -> tuple[bool, str]:
    if not (patch_text or "").strip():
        return False, "empty patch"

    if not patch_text.endswith("\n"):
        return False, "patch must end with trailing newline"

    bad_markers = [
        "```",
        "*** Begin Patch",
        "*** End Patch",
        "*** Add File:",
        "Index:",
        "Explanation:",
        "Here is",
        "Patch:",
    ]

    if any(m in patch_text for m in bad_markers):
        return False, "patch contains extra text/markdown"

    if "diff --git " not in patch_text:
        return False, "missing 'diff --git' header"

    if "data/repositories/" in patch_text:
        return False, "diff contains forbidden 'data/repositories/' prefix"

    # proíbe paths absolutos nos headers (bem comum quando o modelo erra)
    for line in patch_text.splitlines():
        if line.startswith("diff --git "):
            if " /" in line:  # casos tipo a//data/... ou a//abs
                return False, "diff header seems to contain absolute path"

    touched = _touched_paths_from_diff(patch_text)
    if not touched:
        return False, "could not parse touched file paths from diff headers"

    # Só permitir tocar em arquivos do executor_files (relativos)
    not_allowed = sorted([p for p in touched if p not in allowed_paths])
    if not_allowed:
        return False, "patch touches files not in allowed list: " + ", ".join(not_allowed[:10])

    for line in patch_text.splitlines():
        if line.startswith("--- ") or line.startswith("+++ "):
            path = line.split(maxsplit=1)[1].strip()
            # allow /dev/null
            if path == "/dev/null":
                continue
            # must be a/<rel> or b/<rel>
            if not (path.startswith("a/") or path.startswith("b/")):
                return False, "invalid ---/+++ header (must start with a/ or b/)"
            rel = path[2:]
            if rel.startswith("/") or "data/repositories/" in rel:
                return False, "forbidden path in ---/+++ header"
            if rel not in allowed_paths:
                return False, f"---/+++ touches file not allowed: {rel}"

    return True, ""


def route_node(state: State) -> State:
    state["msg"] = f"route ok: repo_path={state.get('repo_path')}"
    return state


def init_run_node(state: State) -> State:
    repo_path = Path(state["repo_path"]).resolve()

    runs_root = repo_path / "agent_runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = uuid.uuid4().hex[:8]
    run_dir = runs_root / f"{ts}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    state["run_dir"] = str(run_dir)

    meta = {
        "repo_path": str(repo_path),
        "base_commit": state.get("base_commit"),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    state["msg"] = state.get("msg", "") + f" | run_dir={run_dir.name}"

    state.setdefault("executor_attempt", 0)
    state.setdefault("max_attempts", 3)
    state.setdefault("executor_feedback", "")

    return state


def planner_node(state: State) -> State:
    repo_path = Path(state["repo_path"]).resolve()
    run_dir = Path(state["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

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

    (run_dir / "planner.prompt.md").write_text(prompt, encoding="utf-8")
    (run_dir / "planner.input.json").write_text(planner_input, encoding="utf-8")

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
        (run_dir / "planner.raw.txt").write_text(raw, encoding="utf-8")

        json_text = _extract_json_only(raw)
        plan = json.loads(json_text)

        if not isinstance(plan, dict) or "blocks" not in plan:
            raise ValueError("plan JSON missing required top-level keys (expected dict with 'blocks').")
        if not isinstance(plan["blocks"], list):
            raise ValueError("'blocks' must be a list.")

        state["plan"] = plan
        state["plan_ok"] = True
        state["plan_error"] = ""

        (run_dir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

        meta.update({"plan_ok": True, "plan_error": ""})
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        state["msg"] = state.get("msg", "") + " | planner ok"
        return state

    except Exception as e:
        err = str(e)
        state["plan_ok"] = False
        state["plan_error"] = err
        (run_dir / "planner.error.txt").write_text(err + "\n", encoding="utf-8")

        meta.update({"plan_ok": False, "plan_error": err})
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        state["msg"] = state.get("msg", "") + " | planner FAIL"
        return state


def after_planner(state: State) -> str:
    return "stage_block" if state.get("plan_ok") else END


def stage_block_node(state: State) -> State:
    run_dir = Path(state["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

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
    state["executor_attempt"] = 0
    state["executor_feedback"] = ""
    state.setdefault("max_attempts", 3)

    state["done"] = False
    state["staged_block"] = blk
    state["staged_block_id"] = blk.get("id")
    state["staged_block_ops"] = ops
    state["staged_block_files"] = files

    (run_dir / f"staged.block.{idx}.json").write_text(json.dumps(blk, indent=2), encoding="utf-8")

    state["msg"] = state.get("msg", "") + f" | staged block_idx={idx} id={blk.get('id')} ops={len(ops)}"
    return state


def resolve_files_for_block_node(state: State) -> State:
    repo_path = Path(state["repo_path"]).resolve()
    run_dir = Path(state["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    blk = state.get("staged_block") or {}
    ops = blk.get("ops", []) or []

    existing: set[str] = set()
    for f in (state.get("staged_block_files") or []):
        p = Path(f)
        if p.is_absolute():
            existing.add(str(p))
        elif str(p).startswith("src/"):
            existing.add(str((repo_path / p).resolve()))
        else:
            existing.add(str(p.resolve()))

    new_files: set[str] = set()
    for op in ops:
        op_name = (op.get("op") or "").strip()

        if op_name in {"EXTRACT_CLASS", "EXTRACT_INTERFACE"}:
            for out in (op.get("outputs") or []):
                if isinstance(out, str) and "." in out and not out.endswith("/"):
                    if out.split(".")[-1][:1].isupper():
                        new_files.add(_java_fqn_to_path(repo_path, out))

        if op_name == "MOVE_CLASS":
            for out in (op.get("outputs") or []):
                if isinstance(out, str) and "." in out and out.split(".")[-1][:1].isupper():
                    new_files.add(_java_fqn_to_path(repo_path, out))

    all_files = sorted(existing.union(new_files))

    state["executor_existing_files"] = sorted(existing)
    state["executor_new_files"] = sorted(new_files)
    state["executor_files"] = all_files

    (run_dir / "executor.files.json").write_text(json.dumps(all_files, indent=2), encoding="utf-8")

    state["msg"] = state.get("msg", "") + f" | files={len(all_files)} (new={len(new_files)})"
    return state


def lock_workspace_node(state: State) -> State:
    repo_path = Path(state["repo_path"]).resolve()
    run_dir = Path(state["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    if not state.get("base_commit"):
        state["base_commit"] = _git_current_commit(repo_path)
    base = state["base_commit"]

    _run(["git", "reset", "--hard", base], cwd=repo_path)
    _run(["git", "clean", "-fd", "-e", "agent_runs/", "-e", "tmp/"], cwd=repo_path)

    state["workspace_commit"] = _git_current_commit(repo_path)

    (run_dir / "workspace.lock.txt").write_text(
        f"base_commit={base}\nworkspace_commit={state['workspace_commit']}\n",
        encoding="utf-8",
    )

    state["msg"] = state.get("msg", "") + f" | workspace locked @{state['workspace_commit'][:8]}"
    return state


def executor_node(state: State) -> State:
    repo_path = Path(state["repo_path"]).resolve()
    run_dir = Path(state["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    blk = state.get("staged_block") or {}
    files = state.get("executor_files") or []

    file_blobs: list[dict] = []
    for fp in files:
        p = Path(fp)
        if p.exists() and p.is_file():
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
            except Exception as e:
                content = f"<<ERROR reading file: {e}>>"
        else:
            content = "<<NEW FILE (does not exist yet)>>"

        rel_path = _to_repo_rel(repo_path, str(p))
        file_blobs.append({"path": rel_path, "content": content})

    SYSTEM = """You are a tool that outputs ONLY a git-apply compatible unified diff.
Rules (hard):
- Output ONLY the patch text. No code fences, no explanations, no headers like "*** Begin Patch".
- Every file must start with: diff --git a/<path> b/<path>
- Must include: --- a/<path> and +++ b/<path>
- Paths MUST be repository-relative and MUST be one of the allowed paths provided.
- For new files: use --- /dev/null and +++ b/<path>
- Patch must end with a trailing newline.
If you cannot comply perfectly, output an empty string."""

    executor_prompt = {
        "task": "Generate a git patch for the staged refactoring block.",
        "staged_block": blk,
        "files": file_blobs,
        "note": "You MUST only touch paths in allowed_paths. If you need a new file, it must also be in allowed_paths.",
        "allowed_paths": [f["path"] for f in file_blobs],
        "feedback": state.get("executor_feedback", ""),
        "attempt": state.get("executor_attempt", 0),
        "max_attempts": state.get("max_attempts", 3),
    }

    state["executor_prompt"] = json.dumps(executor_prompt, indent=2)

    llm = ChatOpenAI(
        model=os.getenv("EXECUTOR_MODEL", "gpt-5-mini"),
        temperature=0.0,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    res = llm.invoke(
        [
            SystemMessage(content=SYSTEM),
            HumanMessage(content=state["executor_prompt"]),
        ]
    )

    raw = (res.content or "").strip()
    state["executor_raw"] = raw

    (run_dir / "executor.prompt.json").write_text(state["executor_prompt"], encoding="utf-8")
    (run_dir / "executor.raw.txt").write_text(raw, encoding="utf-8")

    state["msg"] = state.get("msg", "") + " | executor ok"
    return state


def use_executor_patch_node(state: State) -> State:
    raw = (state.get("executor_raw") or "")
    if raw and not raw.endswith("\n"):
        raw += "\n"

    state["patch_text"] = raw
    state["msg"] = state.get("msg", "") + f" | patch loaded (len={len(raw)})"
    return state


def validate_patch_node(state: State) -> State:
    repo_path = Path(state["repo_path"]).resolve()
    patch_text = state.get("patch_text", "") or ""

    allowed_abs = state.get("executor_files") or []
    allowed_rel = set(_to_repo_rel(repo_path, p) for p in allowed_abs)

    ok, err = _validate_patch_basic(patch_text, allowed_rel)

    state["patch_valid"] = ok
    state["patch_validation_error"] = err

    run_dir = Path(state["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "patch.validation.json").write_text(
        json.dumps(
            {
                "patch_valid": ok,
                "error": err,
                "allowed_rel_count": len(allowed_rel),
                "touched": sorted(list(_touched_paths_from_diff(patch_text)))[:50],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    if ok:
        state["msg"] = state.get("msg", "") + " | patch valid"
    else:
        state["msg"] = state.get("msg", "") + f" | patch INVALID: {err}"
        state["rollback_reason"] = "invalid_patch"
        state["executor_feedback"] = f"PATCH_INVALID: {err}"

    return state


def after_validate_patch(state: State) -> str:
    if state.get("patch_valid"):
        return "apply_patch"

    state["executor_attempt"] = state.get("executor_attempt", 0) + 1

    if state["executor_attempt"] < state.get("max_attempts", 3):
        return "retry_executor"

    return "rollback"

def retry_executor_node(state: State) -> State:
    run_dir = Path(state["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / f"retry.{state.get('executor_attempt', 0)}.txt").write_text(
        f"reason={state.get('rollback_reason','')}\n\n{state.get('executor_feedback','')}\n",
        encoding="utf-8",
    )
    state["msg"] = state.get("msg", "") + f" | retry_executor attempt={state.get('executor_attempt', 0)}"
    return state


def apply_patch_node(state: State) -> State:
    repo_path = Path(state["repo_path"]).resolve()

    if not state.get("base_commit"):
        state["base_commit"] = _git_current_commit(repo_path)

    run_dir = Path(state["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {"repo_path": str(repo_path), "base_commit": state.get("base_commit")}
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    meta["base_commit"] = state["base_commit"]
    patch_text = state.get("patch_text", "") or ""

    if not patch_text.strip():
        state["patch_applied"] = False
        state["patch_apply_log_tail"] = "No patch_text provided."
        state["msg"] = state.get("msg", "") + " | apply SKIP(no patch)"
        state["rollback_reason"] = "no_patch"

        (run_dir / "patch.diff").write_text("", encoding="utf-8")
        (run_dir / "apply.log").write_text("No patch_text provided.\n", encoding="utf-8")
        (run_dir / "git_diff.patch").write_text("", encoding="utf-8")
        (run_dir / "git_diff.stat").write_text("", encoding="utf-8")

        meta.update(
            {
                "patch_applied": False,
                "apply_skipped": True,
                "apply_returncode": None,
                "rollback_reason": "no_patch",
            }
        )
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return state

    patch_file = run_dir / "patch.diff"
    patch_file.write_text(patch_text, encoding="utf-8")

    p = _run(
        ["git", "apply", "--recount", "--verbose", "--whitespace=nowarn", str(patch_file)],
        cwd=repo_path,
    )

    combined = (p.stdout or "") + ("\n" if p.stdout else "") + (p.stderr or "")
    (run_dir / "apply.log").write_text(combined, encoding="utf-8")
    state["patch_apply_log_tail"] = _tail(combined, 60)

    meta.update({"apply_skipped": False, "apply_returncode": p.returncode})

    if p.returncode != 0:
        state["patch_applied"] = False
        state["msg"] = state.get("msg", "") + " | apply FAIL"
        state["rollback_reason"] = "apply_failed"
        state["executor_feedback"] = "APPLY_FAILED:\n" + state.get("patch_apply_log_tail", "")

        diffp = _run(["git", "diff"], cwd=repo_path)
        (run_dir / "git_diff.patch").write_text(diffp.stdout or "", encoding="utf-8")

        statp = _run(["git", "diff", "--stat"], cwd=repo_path)
        (run_dir / "git_diff.stat").write_text(statp.stdout or "", encoding="utf-8")
        state["git_diff_tail"] = _tail(statp.stdout or "", 40)

        if "patch does not apply" in (p.stderr or "").lower():
            meta["apply_error_hint"] = "patch does not apply (context mismatch or already applied)"
        elif "already exists" in (p.stderr or "").lower():
            meta["apply_error_hint"] = "possible already applied / file state mismatch"

        meta["patch_applied"] = False
        meta["rollback_reason"] = state["rollback_reason"]
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return state

    state["patch_applied"] = True
    state["msg"] = state.get("msg", "") + " | apply ok"

    diffp = _run(["git", "diff"], cwd=repo_path)
    (run_dir / "git_diff.patch").write_text(diffp.stdout or "", encoding="utf-8")

    statp = _run(["git", "diff", "--stat"], cwd=repo_path)
    (run_dir / "git_diff.stat").write_text(statp.stdout or "", encoding="utf-8")
    state["git_diff_tail"] = _tail(statp.stdout or "", 40)

    meta["patch_applied"] = True
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return state


def after_apply(state: State) -> str:
    if not state.get("patch_text", "").strip():
        return END
    if state.get("patch_applied"):
        return "compile"
    return "rollback"


def rollback_node(state: State) -> State:
    repo_path = Path(state["repo_path"]).resolve()
    base = state.get("base_commit")

    run_dir_str = state.get("run_dir")
    run_dir = Path(run_dir_str) if run_dir_str else None

    if not base:
        state["msg"] = state.get("msg", "") + " | rollback skipped (no base_commit)"
        return state

    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        statusp = _run(["git", "status", "--porcelain=v1"], cwd=repo_path)
        (run_dir / "git_status_before_rollback.txt").write_text(
            (statusp.stdout or "") + ("\n" if statusp.stdout else "") + (statusp.stderr or ""),
            encoding="utf-8",
        )

    _run(["git", "reset", "--hard", base], cwd=repo_path)
    _run(["git", "clean", "-fd", "-e", "agent_runs/", "-e", "tmp/"], cwd=repo_path)

    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)

    state["patch_applied"] = False
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

        (run_dir / "rollback.reason.txt").write_text(meta["rollback_reason"] + "\n", encoding="utf-8")

    return state


def after_rollback(state: State) -> str:
    reason = state.get("rollback_reason", "")

    # retry only for these reasons
    if reason in {"invalid_patch", "apply_failed"}:
        # executor_attempt already incremented in after_validate_patch for invalid_patch;
        # but apply_failed comes from apply_patch, so increment here.
        if reason == "apply_failed":
            state["executor_attempt"] = state.get("executor_attempt", 0) + 1

        if state["executor_attempt"] < state.get("max_attempts", 3):
            return "retry_executor"

    return END


def compile_node(state: State) -> State:
    repo_path = Path(state["repo_path"]).resolve()

    tmp_dir = repo_path / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["mvn", "-q", "-DskipTests", "-Drat.skip=true", "compile"]

    env = os.environ.copy()
    env["MAVEN_OPTS"] = f"-Xshare:off -Djava.io.tmpdir={tmp_dir}"

    p = _run(cmd, cwd=repo_path, env=env)

    state["compile_returncode"] = p.returncode
    state["compile_ok"] = (p.returncode == 0)
    state["maven_tmp_dir"] = str(tmp_dir)

    combined = (p.stdout or "") + "\n" + (p.stderr or "")

    run_dir = Path(state["run_dir"])
    (run_dir / "compile.log").write_text(combined, encoding="utf-8")

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
    return END if state.get("compile_ok") else "rollback"


def build_graph():
    g = StateGraph(State)

    g.add_node("route", route_node)
    g.add_node("init_run", init_run_node)
    g.add_node("planner", planner_node)
    g.add_node("stage_block", stage_block_node)
    g.add_node("resolve_files", resolve_files_for_block_node)
    g.add_node("lock_workspace", lock_workspace_node)
    g.add_node("executor", executor_node)
    g.add_node("use_executor_patch", use_executor_patch_node)
    g.add_node("validate_patch", validate_patch_node)
    g.add_node("retry_executor", retry_executor_node)

    g.add_node("apply_patch", apply_patch_node)
    g.add_node("compile", compile_node)
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

    g.add_edge("stage_block", "resolve_files")
    g.add_edge("resolve_files", "lock_workspace")
    g.add_edge("lock_workspace", "executor")
    g.add_edge("executor", "use_executor_patch")
    g.add_edge("use_executor_patch", "validate_patch")

    g.add_conditional_edges(
        "validate_patch",
        after_validate_patch,
        {
            "apply_patch": "apply_patch",
            "retry_executor": "retry_executor",
            "rollback": "rollback",
        },
    )

    g.add_edge("retry_executor", "lock_workspace")

    g.add_conditional_edges(
        "apply_patch",
        after_apply,
        {
            "compile": "compile",
            "rollback": "rollback",
            END: END,
        },
    )

    g.add_conditional_edges(
        "compile",
        after_compile,
        {
            END: END,
            "rollback": "rollback",
        },
    )

    g.add_conditional_edges(
        "rollback",
        after_rollback,
        {
            "retry_executor": "retry_executor",
            END: END,
        },
    )

    return g.compile()


if __name__ == "__main__":
    dotenv.load_dotenv()

    app = build_graph()

    # Example: run planner->executor->apply->compile pipeline
    with open("data/prompts/planner_agent.prompt", "r", encoding="utf-8") as f:
        YOUR_PROMPT_TEMPLATE_STRING = f.read()

    with open("data/examples/input.json", "r", encoding="utf-8") as f:
        YOUR_INPUT_DICT = json.loads(f.read())

    out = app.invoke(
        {
            "repo_path": "data/repositories/jsoup",
            "planner_prompt": YOUR_PROMPT_TEMPLATE_STRING,
            "planner_input_json": json.dumps(YOUR_INPUT_DICT, indent=2),
            "max_attempts": 5,
        }
    )
    pprint(out)