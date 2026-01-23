import json
import os
import dotenv
import subprocess
import textwrap
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

    # patch application data
    patch_text: str
    patch_applied: bool
    patch_apply_log_tail: str
    git_diff_tail: str
    base_commit: str

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
        candidates = []
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

def _tail(s: str, n: int = 40) -> str:
    lines = (s or "").splitlines()
    return "\n".join(lines[-n:]).strip()

def _git_current_commit(repo_path: Path) -> str:
    p = _run(["git", "rev-parse", "HEAD"], cwd=repo_path)
    if p.returncode != 0:
        raise RuntimeError("git rev-parse HEAD failed:\n" + _tail(p.stderr))
    return p.stdout.strip()

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
    return state

def route_node(state: State) -> State:
    state["msg"] = f"route ok: repo_path={state.get('repo_path')}"
    return state

def planner_node(state: State) -> State:
    # get repo and run (log) dir
    repo_path = Path(state["repo_path"]).resolve()
    run_dir = Path(state["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)

    # load or init metrics and dependencies json
    meta_path = run_dir / "meta.json"
    meta = _load_meta_or_init(meta_path, repo_path, state.get("base_commit"))

    # load prompt template and input json (packages/classes w/ metrics and dependencies)
    prompt = state.get("planner_prompt", "").strip()
    planner_input = state.get("planner_input_json", "").strip()

    # log for absence of prompt
    if not prompt:
        state["plan_ok"] = False
        state["plan_error"] = "planner_prompt missing"
        meta.update({"plan_ok": False, "plan_error": state["plan_error"]})
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return state

    # log for absence of input json
    if not planner_input:
        state["plan_ok"] = False
        state["plan_error"] = "planner_input_json missing"
        meta.update({"plan_ok": False, "plan_error": state["plan_error"]})
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return state

    # save prompt and input json to run_dir
    (run_dir / "planner.prompt.md").write_text(prompt, encoding="utf-8")
    (run_dir / "planner.input.json").write_text(planner_input, encoding="utf-8")

    # LLM setup
    llm = ChatOpenAI(
        model=os.getenv("PLANNER_MODEL", "gpt-5-mini"),
        temperature=0.0,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    # Add the input JSON into the prompt template
    rendered = prompt.replace("{input}", planner_input)

    try:
        # invoke LLM
        res = llm.invoke([
            SystemMessage(content="Return ONLY valid JSON. No extra text."),
            HumanMessage(content=rendered),
        ])

        # process response
        raw = (res.content or "").strip()
        
        # save raw response in the state and run_dir
        state["plan_json_text"] = raw
        (run_dir / "planner.raw.txt").write_text(raw, encoding="utf-8")

        # extract JSON only
        json_text = _extract_json_only(raw)
        plan = json.loads(json_text)

        # validate plan structure
        if not isinstance(plan, dict) or "blocks" not in plan:
            raise ValueError("plan JSON missing required top-level keys (expected dict with 'blocks').")
        if not isinstance(plan["blocks"], list):
            raise ValueError("'blocks' must be a list.")

        state["plan"] = plan
        state["plan_ok"] = True
        state["plan_error"] = ""

        # save “oficial” plan.json
        (run_dir / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

        # update meta.json
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
    return END

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

    patch_text = state.get("patch_text", "")

    # --- SKIP (sem patch) ---
    if not patch_text.strip():
        state["patch_applied"] = False
        state["patch_apply_log_tail"] = "No patch_text provided."
        state["msg"] = state.get("msg", "") + " | apply SKIP(no patch)"
        state["rollback_reason"] = "no_patch"  # <-- FIX: reason no próprio node

        (run_dir / "patch.diff").write_text("", encoding="utf-8")
        (run_dir / "apply.log").write_text("No patch_text provided.\n", encoding="utf-8")
        (run_dir / "git_diff.patch").write_text("", encoding="utf-8")
        (run_dir / "git_diff.stat").write_text("", encoding="utf-8")

        meta.update({
            "patch_applied": False,
            "apply_skipped": True,
            "apply_returncode": None,
            "rollback_reason": "no_patch",
        })
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        return state

    # --- write patch to run_dir ---
    patch_file = run_dir / "patch.diff"
    patch_file.write_text(patch_text, encoding="utf-8")

    # --- apply patch (single execution) ---
    p = _run(
        ["git", "apply", "--verbose", "--whitespace=nowarn", str(patch_file)],
        cwd=repo_path
    )

    combined = (p.stdout or "") + ("\n" if p.stdout else "") + (p.stderr or "")
    (run_dir / "apply.log").write_text(combined, encoding="utf-8")
    state["patch_apply_log_tail"] = _tail(combined, 60)

    meta.update({
        "apply_skipped": False,
        "apply_returncode": p.returncode,
    })

    # --- if apply failed ---
    if p.returncode != 0:
        state["patch_applied"] = False
        state["msg"] = state.get("msg", "") + " | apply FAIL"
        state["rollback_reason"] = "apply_failed"  # <-- FIX: reason no próprio node

        # Mesmo no FAIL, salvar diff/stat ajuda a depurar o que ficou no working tree
        diffp = _run(["git", "diff"], cwd=repo_path)
        (run_dir / "git_diff.patch").write_text(diffp.stdout or "", encoding="utf-8")

        statp = _run(["git", "diff", "--stat"], cwd=repo_path)
        (run_dir / "git_diff.stat").write_text(statp.stdout or "", encoding="utf-8")
        state["git_diff_tail"] = _tail(statp.stdout or "", 40)

        # mensagem extra útil quando patch já está aplicado
        if "patch does not apply" in (p.stderr or "").lower():
            meta["apply_error_hint"] = "patch does not apply (context mismatch or already applied)"
        elif "already exists" in (p.stderr or "").lower():
            meta["apply_error_hint"] = "possible already applied / file state mismatch"

        meta["patch_applied"] = False
        meta["rollback_reason"] = state["rollback_reason"]
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        return state

    # --- apply OK ---
    state["patch_applied"] = True
    state["msg"] = state.get("msg", "") + " | apply ok"

    # salvar diff completo + stat
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
    elif state.get("patch_applied"):
        return "compile"
    else:
        return "rollback"

def rollback_node(state: State) -> State:
    repo_path = Path(state["repo_path"]).resolve()
    base = state.get("base_commit")

    run_dir_str = state.get("run_dir")
    run_dir = Path(run_dir_str) if run_dir_str else None

    # Se não temos commit base, não dá rollback
    if not base:
        state["msg"] = state.get("msg", "") + " | rollback skipped (no base_commit)"
        return state

    # Snapshot antes do reset (ajuda MUITO a debugar)
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        statusp = _run(["git", "status", "--porcelain=v1"], cwd=repo_path)
        (run_dir / "git_status_before_rollback.txt").write_text(
            (statusp.stdout or "") + ("\n" if statusp.stdout else "") + (statusp.stderr or ""),
            encoding="utf-8",
        )

    # Faz o rollback de forma determinística
    _run(["git", "reset", "--hard", base], cwd=repo_path)

    # <-- FIX: não apagar agent_runs/ e tmp/ (senão some meta.json e logs)
    _run(["git", "clean", "-fd", "-e", "agent_runs/", "-e", "tmp/"], cwd=repo_path)

    # <-- FIX: garantir que run_dir exista mesmo após clean
    if run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)

    state["patch_applied"] = False
    state["msg"] = state.get("msg", "") + " | rollback done"

    # Atualiza (ou cria) meta.json com segurança
    if run_dir:
        meta_path = run_dir / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        else:
            # fallback seguro: cria um meta mínimo
            meta = {
                "repo_path": str(repo_path),
                "base_commit": base,
                "note": "meta.json was missing; created during rollback",
            }

        meta["rolled_back"] = True
        meta["rollback_reason"] = state.get("rollback_reason", "unknown")
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        (run_dir / "rollback.reason.txt").write_text(meta["rollback_reason"] + "\n", encoding="utf-8")

    return state

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
        state["rollback_reason"] = "compile_failed"  # <-- FIX: reason no próprio node

    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update({
        "compile_ok": state["compile_ok"],
        "compile_returncode": state["compile_returncode"],
    })
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return state

def after_compile(state: State) -> str:
    return END if state.get("compile_ok") else "rollback"

def build_graph():
    g = StateGraph(State)

    g.add_node("route", route_node)
    g.add_node("init_run", init_run_node)
    g.add_node("planner", planner_node)
    g.add_node("apply_patch", apply_patch_node)
    g.add_node("compile", compile_node)
    g.add_node("rollback", rollback_node)

    g.set_entry_point("route")
    g.add_edge("route", "init_run")
    g.add_edge("init_run", "planner")
    #g.add_edge("init_run", "apply_patch")

    g.add_edge("planner", END) ## TODO: remove this temporary shortcut

    g.add_conditional_edges("apply_patch", after_apply, {
        "compile": "compile",
        "rollback": "rollback",
        END: END,
    })

    g.add_conditional_edges("compile", after_compile, {
        END: END,
        "rollback": "rollback"
    })

    g.add_edge("rollback", END)

    return g.compile()

if __name__ == "__main__":
    dotenv.load_dotenv()

    app = build_graph()

    '''file = open("test.patch", "r", encoding="utf-8")

    patch = file.read()
    out = app.invoke({
        "repo_path": "data/repositories/commons-lang",
        "patch_text": patch,
    })'''

    file = open("data/prompts/planner_agent.prompt", "r", encoding="utf-8")
    YOUR_PROMPT_TEMPLATE_STRING = file.read()

    file = open("data/examples/input.json", "r", encoding="utf-8")
    YOUR_INPUT_DICT = json.loads(file.read())

    out = app.invoke({
        "repo_path": "data/repositories/jsoup",
        "planner_prompt": YOUR_PROMPT_TEMPLATE_STRING,
        "planner_input_json": json.dumps(YOUR_INPUT_DICT, indent=2),
    })
    pprint(out)