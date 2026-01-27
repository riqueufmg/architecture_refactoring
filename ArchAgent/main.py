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

    executor_plan_json_text: str          # raw code
    executor_plan: dict                   # plan for executor
    executor_plan_ok: bool                # plan validation
    executor_plan_error: str              # executor error data
    executor_result: dict                 # executor result data
    files_to_write: list[dict]            # each: {"path": "...", "content": "..."}
    files_to_delete: list[str]            # repo-relative paths

    # apply files node
    apply_ok: bool
    apply_error: str

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

    if not p or "\x00" in p:
        return False
    pp = Path(p)
    if pp.is_absolute():
        return False
    if ":" in p.split("/")[0]:  # ex: C:\
        return False
    
    parts = pp.as_posix().split("/")
    if any(part == ".." for part in parts):
        return False
    return True


def _validate_allowed_paths(rel_paths: list[str], allowed: set[str]) -> tuple[bool, str]:
    for rp in rel_paths:
        if not _is_safe_repo_rel_path(rp):
            return False, f"unsafe path: {rp}"
        if rp not in allowed:
            return False, f"path not in allowed_paths: {rp}"
    return True, ""

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

def route_node(state: State) -> State:
    state["msg"] = f"route ok: repo_path={state.get('repo_path')}"
    return state


def init_run_node(state: State) -> State:
    repo_path = Path(state["repo_path"]).resolve()

    # defines base commit if not set
    if not state.get("base_commit"):
        state["base_commit"] = _git_current_commit(repo_path)

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

        #json_text = _extract_json_only(raw) # TODO: remove this function?
        json_text = _extract_json_object_only(raw)

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
    run_dir.mkdir(parents=True, exist_ok=True)
    
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

    res = llm.invoke([
        SystemMessage(content=SYSTEM),
        HumanMessage(content=state["executor_prompt"]),
    ])

    raw = (res.content or "").strip()
    state["executor_raw"] = raw

    (run_dir / "executor.prompt.json").write_text(state["executor_prompt"], encoding="utf-8")
    (run_dir / "executor.raw.txt").write_text(raw, encoding="utf-8")

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

        (run_dir / "executor.result.json").write_text(
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
            "executor_attempt": state.get("executor_attempt", 0),
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

        (run_dir / "executor.parse_error.txt").write_text(err + "\n", encoding="utf-8")
        state["msg"] = state.get("msg", "") + f" | executor FAIL(parse): {err}"

        #update meta file
        meta.update({
            "executor_ok": False,
            "executor_attempt": state.get("executor_attempt", 0),
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

    # if executor failed, retry
    state["executor_attempt"] = state.get("executor_attempt", 0) + 1 # add attempt
    if state["executor_attempt"] < state.get("max_attempts", 3): # check attempts
        return "retry_executor"
    return "rollback" # if all attempts exhausted

def retry_executor_node(state: State) -> State:
    run_dir = Path(state["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    # create a unique retry log file
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    rid = uuid.uuid4().hex[:8]
    attempt = state.get("executor_attempt", 0)
    reason = state.get("rollback_reason", "")
    fname = f"retry.attempt{attempt}.{ts}_{rid}.txt"

    content = (
        f"attempt={attempt}\n"
        f"reason={reason}\n"
        f"feedback={state.get('executor_feedback','')}\n"
    )
    (run_dir / fname).write_text(content, encoding="utf-8")

    index_line = json.dumps(
        {
            "file": fname,
            "attempt": attempt,
            "reason": reason,
            "ts": ts,
        },
        ensure_ascii=False,
    )
    with open(run_dir / "retry.index.jsonl", "a", encoding="utf-8") as f:
        f.write(index_line + "\n")

    state["msg"] = state.get("msg", "") + f" | retry logged={fname}"
    return state

def apply_files_node(state: State) -> State:
    # load log files
    repo_path = Path(state["repo_path"]).resolve()
    run_dir = Path(state["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

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
        (run_dir / "apply_files.error.txt").write_text(err + "\n", encoding="utf-8")
        meta.update({"apply_ok": False, "apply_error": err, "rollback_reason": state["rollback_reason"]})
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        state["msg"] = state.get("msg","") + f" | apply_files FAIL: {err}"
        return state

    # validate paths before applying (delete files)
    ok, err = _validate_allowed_paths(files_to_delete, allowed)
    if not ok:
        state["apply_ok"] = False
        state["apply_error"] = err
        state["rollback_reason"] = "apply_files_invalid_delete_paths"
        state["executor_feedback"] = f"APPLY_FILES_INVALID_DELETE_PATHS: {err}"
        (run_dir / "apply_files.error.txt").write_text(err + "\n", encoding="utf-8")
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
    (run_dir / "apply_files.written.json").write_text(json.dumps(written, indent=2), encoding="utf-8")
    (run_dir / "apply_files.deleted.json").write_text(json.dumps(deleted, indent=2), encoding="utf-8")

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
    
    # if files changed successfully, proceed to compile
    if state.get("apply_ok"):
        return "compile"

    # failed to apply files; retry executor
    state["executor_attempt"] = state.get("executor_attempt", 0) + 1 # add attempt
    if state["executor_attempt"] < state.get("max_attempts", 3): # check attempts
        return "retry_executor"
    
     # if all attempts exhausted
    return "rollback"

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

    # consider quantity of attempts
    if state.get("executor_attempt", 0) < state.get("max_attempts", 3):
        # if executor, apply or compile failed, retry
        if state.get("rollback_reason") in {
            "invalid_executor_json",
            "apply_files_invalid_paths",
            "apply_files_invalid_delete_paths",
            "compile_failed",
        }:
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
    if state.get("compile_ok"):
        return END
    return "rollback"

def after_stage_block(state: State) -> str:
    if state.get("done"):
        return END
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

    state["msg"] = state.get("msg", "") + f" | advance_block -> {state['block_idx']}"
    return state

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
            END: END,
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
            END: "advance_block",       # compile_ok
            "rollback": "rollback",     # compile_fail 
        },
    )

    g.add_edge("advance_block", "stage_block")

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
            "max_attempts": 20,
        }
    )
    pprint(out)