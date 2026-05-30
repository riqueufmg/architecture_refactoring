import json
import shutil
import uuid

from datetime import datetime
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from mvp.source_refactor.state import SourceRefactorState

from mvp.source_refactor.lib.config_utils import (
    load_config,
    require_config_value,
    get_config_value,
)
from mvp.source_refactor.lib.json_utils import (
    read_json,
    write_json,
    extract_json_object_only,
)
from mvp.source_refactor.lib.path_utils import (
    require_absolute_path,
    ensure_file_exists,
    ensure_dir_exists,
)
from mvp.source_refactor.lib.git_utils import (
    ensure_clean_git_workspace,
    git_current_commit,
    git_status_porcelain,
    git_commit_all,
    git_reset_hard,
    git_clean_workspace,
)
from mvp.source_refactor.lib.file_ops import (
    load_files_context,
    ensure_path_inside_repo,
    write_text_file,
    delete_file,
)
from mvp.source_refactor.lib.maven_utils import run_compile_command

def load_config_node(state: SourceRefactorState) -> SourceRefactorState:
    config_path = require_absolute_path(state["config_path"], "config_path")
    ensure_file_exists(config_path, "config_path")

    cfg = load_config(config_path)

    planner_contract_path = require_absolute_path(
        str(require_config_value(cfg, "input.planner_contract")),
        "input.planner_contract",
    )
    ensure_file_exists(planner_contract_path, "input.planner_contract")

    runs_dir = require_absolute_path(
        str(require_config_value(cfg, "output.runs_dir")),
        "output.runs_dir",
    )
    runs_dir.mkdir(parents=True, exist_ok=True)

    state["config_path"] = str(config_path)
    state["config"] = cfg
    state["planner_contract_path"] = str(planner_contract_path)

    return state


def init_run_node(state: SourceRefactorState) -> SourceRefactorState:
    cfg = state["config"]

    runs_dir = require_absolute_path(
        str(require_config_value(cfg, "output.runs_dir")),
        "output.runs_dir",
    )

    planner_contract_path = Path(state["planner_contract_path"]).resolve()

    # Preferimos usar o mesmo run_id do Planner, quando existir.
    planner_run_dir = planner_contract_path.parent.parent

    run_id_cfg = cfg.get("mvp", {}).get("run_id", "auto")

    if run_id_cfg and run_id_cfg != "auto":
        run_id = str(run_id_cfg)
        run_dir = runs_dir / run_id
    else:
        # Se o contrato está em data/runs/<run_id>/planner/contract.json,
        # reutilizamos <run_id>.
        if planner_contract_path.parent.name == "planner":
            run_id = planner_run_dir.name
            run_dir = planner_run_dir
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            rid = uuid.uuid4().hex[:8]
            run_id = f"{ts}_{rid}"
            run_dir = runs_dir / run_id

    source_refactor_dir = run_dir / "source_refactor"
    source_refactor_dir.mkdir(parents=True, exist_ok=True)

    state["run_id"] = run_id
    state["run_dir"] = str(run_dir)
    state["source_refactor_dir"] = str(source_refactor_dir)

    shutil.copyfile(state["config_path"], source_refactor_dir / "config.snapshot.yml")

    return state


def load_planner_contract_node(state: SourceRefactorState) -> SourceRefactorState:
    source_refactor_dir = Path(state["source_refactor_dir"])
    contract_path = Path(state["planner_contract_path"]).resolve()

    contract = read_json(contract_path)

    if contract.get("producer") != "planner":
        raise ValueError("Input contract was not produced by planner")

    if not contract.get("ok", False):
        raise ValueError("Planner contract is not ok")

    artifacts = contract.get("artifacts", {})
    project = contract.get("project", {})
    target = contract.get("target", {})

    plan_path = require_absolute_path(str(artifacts.get("plan", "")), "planner.artifacts.plan")
    planner_input_path = require_absolute_path(
        str(artifacts.get("planner_input", "")),
        "planner.artifacts.planner_input",
    )

    repo_path = require_absolute_path(str(project.get("repo_path", "")), "planner.project.repo_path")

    ensure_file_exists(plan_path, "planner.artifacts.plan")
    ensure_file_exists(planner_input_path, "planner.artifacts.planner_input")
    ensure_dir_exists(repo_path, "planner.project.repo_path")

    planner_input = read_json(planner_input_path)
    state["planner_input"] = planner_input

    target_source_root = str(
        target.get("target_source_root")
        or planner_input.get("target_source_root")
        or ""
    ).strip()

    state["target_source_root"] = target_source_root

    if "target_file" in planner_input:
        state["target_file"] = str(planner_input.get("target_file", ""))

    if "target_files" in planner_input:
        target_files = planner_input.get("target_files", [])
        if isinstance(target_files, list):
            state["target_files"] = [str(f) for f in target_files]
        else:
            state["target_files"] = []
        
    shutil.copyfile(planner_input_path, source_refactor_dir / "planner_input.json")

    state["planner_contract"] = contract
    state["planner_dir"] = str(contract_path.parent)
    state["planner_plan_path"] = str(plan_path)
    state["planner_input_path"] = str(planner_input_path)

    state["repo_path"] = str(repo_path)
    state["project_name"] = str(project.get("name", ""))

    state["smell"] = str(target.get("smell", ""))
    state["smell_name"] = str(target.get("smell_name", ""))
    state["target_type"] = str(target.get("target_type", ""))
    state["target_name"] = str(target.get("target_name", ""))

    shutil.copyfile(contract_path, source_refactor_dir / "input_contract.json")

    return state


def load_plan_node(state: SourceRefactorState) -> SourceRefactorState:
    source_refactor_dir = Path(state["source_refactor_dir"])

    plan = read_json(state["planner_plan_path"])

    if "blocks" not in plan or not isinstance(plan["blocks"], list):
        raise ValueError("Planner plan must contain a blocks list")

    state["input_plan"] = plan

    write_json(source_refactor_dir / "input_plan.json", plan)

    return state

def ensure_clean_workspace_node(state: SourceRefactorState) -> SourceRefactorState:
    repo_path = Path(state["repo_path"]).resolve()
    source_refactor_dir = Path(state["source_refactor_dir"])

    ensure_clean_git_workspace(repo_path)

    state["workspace_clean"] = True

    write_json(
        source_refactor_dir / "git" / "workspace_status.json",
        {
            "workspace_clean": True,
            "repo_path": str(repo_path),
        },
    )

    return state


def record_initial_commit_node(state: SourceRefactorState) -> SourceRefactorState:
    repo_path = Path(state["repo_path"]).resolve()
    source_refactor_dir = Path(state["source_refactor_dir"])

    initial_commit = git_current_commit(repo_path)

    state["initial_commit"] = initial_commit
    state["last_good_commit"] = initial_commit
    state["block_commits"] = []
    state["repair_commits"] = []
    state["block_summaries"] = []

    git_dir = source_refactor_dir / "git"
    git_dir.mkdir(parents=True, exist_ok=True)

    (git_dir / "initial_commit.txt").write_text(
        initial_commit + "\n",
        encoding="utf-8",
    )

    (git_dir / "last_good_commit.txt").write_text(
        initial_commit + "\n",
        encoding="utf-8",
    )

    write_json(git_dir / "block_commits.json", [])
    write_json(git_dir / "repair_commits.json", [])
    write_json(source_refactor_dir / "block_summaries.json", [])

    return state

def prepare_executable_plan_node(state: SourceRefactorState) -> SourceRefactorState:
    source_refactor_dir = Path(state["source_refactor_dir"])

    # MVP inicial: sem enrichment preventivo.
    executable_plan = dict(state["input_plan"])

    state["executable_plan"] = executable_plan
    state["current_block_index"] = 0

    write_json(source_refactor_dir / "executable_plan.json", executable_plan)

    return state

def stage_block_node(state: SourceRefactorState) -> SourceRefactorState:
    source_refactor_dir = Path(state["source_refactor_dir"])
    executable_plan = state["executable_plan"]
    blocks = executable_plan.get("blocks", [])

    block_index = int(state.get("current_block_index", 0))

    if block_index >= len(blocks):
        state["stop_reason"] = "no_more_blocks"
        return state

    block = blocks[block_index]
    block_id = str(block.get("id", block_index + 1))

    block_dir = source_refactor_dir / f"block_{int(block_id):03d}"
    block_dir.mkdir(parents=True, exist_ok=True)

    state["current_block"] = block
    state["current_block_id"] = block_id
    state["current_block_dir"] = str(block_dir)

    write_json(block_dir / "staged.block.json", block)

    cfg = state["config"]

    state["block_attempt"] = 0
    state["max_block_attempts"] = int(
        get_config_value(cfg, "executor.max_block_attempts", 5)
    )

    state["executor_feedback"] = ""
    state["executor_ok"] = False
    state["executor_error"] = ""
    state["apply_ok"] = False
    state["apply_error"] = ""
    state["rollback_reason"] = ""

    return state

def lock_workspace_node(state: SourceRefactorState) -> SourceRefactorState:
    repo_path = Path(state["repo_path"]).resolve()
    block_dir = Path(state["current_block_dir"])

    base = state.get("last_good_commit") or state.get("initial_commit")

    if not base:
        raise RuntimeError("lock_workspace_node: missing last_good_commit/initial_commit")

    before_commit = git_current_commit(repo_path)

    reset_output = git_reset_hard(repo_path, base)
    git_clean_workspace(repo_path)

    after_commit = git_current_commit(repo_path)

    write_json(
        block_dir / "workspace.lock.json",
        {
            "base_commit": base,
            "before_commit": before_commit,
            "after_commit": after_commit,
            "reset_output": reset_output,
        },
    )

    state["workspace_clean"] = True
    state["last_good_commit"] = base

    # Limpa estado transitório da tentativa atual.
    state["allowed_files"] = []
    state["files_context"] = []

    state["executor_ok"] = False
    state["executor_error"] = ""
    state["execute_plan_raw"] = ""
    state["execute_plan_result"] = {}

    state["apply_ok"] = False
    state["apply_error"] = ""
    state["applied_files"] = []

    state["compile_ok"] = False
    state["compile_return_code"] = -1
    state["compile_log_path"] = ""

    return state

def resolve_files_context_node(state: SourceRefactorState) -> SourceRefactorState:
    repo_path = Path(state["repo_path"]).resolve()
    block_dir = Path(state["current_block_dir"])
    block = state["current_block"]

    from mvp.source_refactor.lib.java_path_utils import (
        resolve_java_fqn_to_path,
        looks_like_java_class_fqn,
        normalize_repo_relative_path,
    )

    block_files = block.get("files", [])

    if not isinstance(block_files, list):
        raise ValueError("current block files must be a list")

    ops = block.get("ops", [])

    if not isinstance(ops, list):
        raise ValueError("current block ops must be a list")

    existing_files: set[str] = set()
    new_files: set[str] = set()
    rejected_files: list[str] = []

    # 1. Primeiro, aceitar arquivos declarados diretamente em block.files.
    #    Se o arquivo existe, ele é existing_file.
    #    Se não existe, mas é .java e está dentro do repo, ele é new_file.
    for f in block_files:
        rel = normalize_repo_relative_path(str(f))

        if not rel:
            continue

        if not rel.endswith(".java"):
            rejected_files.append(rel)
            continue

        abs_path = (repo_path / rel).resolve()

        try:
            abs_path.relative_to(repo_path)
        except ValueError:
            rejected_files.append(rel)
            continue

        if abs_path.exists():
            existing_files.add(rel)
        else:
            new_files.add(rel)

    # 2. Resolver source root.
    #    Idealmente vem do Planner. Se estiver ausente, usamos fallback.
    source_root = normalize_repo_relative_path(
        str(state.get("target_source_root", "") or "src/main/java")
    )

    # 3. Descobrir pacote alvo para quando outputs tiverem só nome simples.
    #    Exemplo: "CharUtilsConstants" em vez de
    #    "org.apache.commons.lang3.CharUtilsConstants".
    target_package = ""
    target_name = str(state.get("target_name", "") or "")

    if state.get("target_type") == "class" and "." in target_name:
        target_package = ".".join(target_name.split(".")[:-1])
    elif state.get("target_type") == "package":
        target_package = target_name

    # 4. Além de block.files, também aceitar arquivos novos derivados das ops.
    #    Isso cobre planos que colocam o novo arquivo em op.outputs.
    for op in ops:
        if not isinstance(op, dict):
            continue

        op_name = str(op.get("op", "")).strip()

        if op_name not in {"EXTRACT_CLASS", "EXTRACT_INTERFACE", "MOVE_CLASS"}:
            continue

        outputs = op.get("outputs") or []

        if not isinstance(outputs, list):
            continue

        for out in outputs:
            if not isinstance(out, str):
                continue

            out = out.strip()

            if not out:
                continue

            # Caso 1: output já é caminho repo-relativo.
            # Exemplo: src/main/java/org/apache/commons/lang3/CharUtilsConstants.java
            if out.endswith(".java"):
                rel = normalize_repo_relative_path(out)
                abs_path = (repo_path / rel).resolve()

                try:
                    abs_path.relative_to(repo_path)
                except ValueError:
                    rejected_files.append(rel)
                    continue

                if abs_path.exists():
                    existing_files.add(rel)
                else:
                    new_files.add(rel)

                continue

            # Caso 2: output é FQN completo.
            # Exemplo: org.apache.commons.lang3.CharUtilsConstants
            if looks_like_java_class_fqn(out):
                rel = resolve_java_fqn_to_path(repo_path, out, source_root)
                rel = normalize_repo_relative_path(rel)

                abs_path = (repo_path / rel).resolve()

                try:
                    abs_path.relative_to(repo_path)
                except ValueError:
                    rejected_files.append(rel)
                    continue

                if abs_path.exists():
                    existing_files.add(rel)
                else:
                    new_files.add(rel)

                continue

            # Caso 3: output é apenas nome simples de classe.
            # Exemplo: CharUtilsConstants
            if out[0].isupper() and "." not in out and target_package:
                new_fqn = f"{target_package}.{out}"
                rel = resolve_java_fqn_to_path(repo_path, new_fqn, source_root)
                rel = normalize_repo_relative_path(rel)

                abs_path = (repo_path / rel).resolve()

                try:
                    abs_path.relative_to(repo_path)
                except ValueError:
                    rejected_files.append(rel)
                    continue

                if abs_path.exists():
                    existing_files.add(rel)
                else:
                    new_files.add(rel)

                continue

    # 5. Se algum arquivo foi classificado como existing, ele não deve ficar em new.
    new_files = new_files - existing_files

    all_files = sorted(existing_files.union(new_files))

    files_context = load_files_context(repo_path, all_files)

    state["executor_existing_files"] = sorted(existing_files)
    state["executor_new_files"] = sorted(new_files)
    state["executor_files"] = all_files
    state["executor_rejected_files"] = rejected_files

    state["allowed_files"] = all_files
    state["files_context"] = files_context

    write_json(
        block_dir / "executor.files.json",
        {
            "executor_existing_files": sorted(existing_files),
            "executor_new_files": sorted(new_files),
            "executor_files": all_files,
            "executor_rejected_files": rejected_files,
        },
    )

    write_json(
        block_dir / "files_context.json",
        {
            "allowed_files": all_files,
            "files_context": files_context,
        },
    )

    return state

def execute_plan_node(state: SourceRefactorState) -> SourceRefactorState:
    cfg = state["config"]
    block_dir = Path(state["current_block_dir"])

    system_prompt_path = require_absolute_path(
        str(require_config_value(cfg, "prompts.system")),
        "prompts.system",
    )
    execute_prompt_path = require_absolute_path(
        str(require_config_value(cfg, "prompts.execute_plan")),
        "prompts.execute_plan",
    )

    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    execute_prompt = execute_prompt_path.read_text(encoding="utf-8")

    executor_input = {
        "repo_path": state["repo_path"],
        "target": {
            "smell": state["smell"],
            "smell_name": state["smell_name"],
            "target_type": state["target_type"],
            "target_name": state["target_name"],
        },
        "block": state["current_block"],
        "allowed_files": state["allowed_files"],
        "executor_existing_files": state.get("executor_existing_files", []),
        "executor_new_files": state.get("executor_new_files", []),
        "executor_rejected_files": state.get("executor_rejected_files", []),
        "files_context": state["files_context"],
        "feedback": state.get("executor_feedback", ""),
        "attempt": state.get("block_attempt", 0),
    }

    rendered = execute_prompt.replace(
        "{input}",
        json.dumps(executor_input, indent=2, ensure_ascii=False),
    )

    state["executor_system_prompt_path"] = str(system_prompt_path)
    state["execute_plan_prompt_path"] = str(execute_prompt_path)
    state["executor_system_prompt"] = system_prompt
    state["execute_plan_prompt"] = execute_prompt
    state["execute_plan_rendered"] = rendered

    attempt = int(state.get("block_attempt", 0))

    (block_dir / f"execute_plan.attempt_{attempt}.rendered.md").write_text(
        rendered,
        encoding="utf-8",
    )

    model = str(get_config_value(cfg, "models.executor", "gpt-5-mini"))
    temperature = float(get_config_value(cfg, "executor.temperature", 0.0))
    timeout = int(get_config_value(cfg, "executor.timeout", 180))
    max_retries = int(get_config_value(cfg, "executor.max_retries", 2))

    llm = ChatOpenAI(
        model=model,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
    )

    try:
        res = llm.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=rendered),
            ]
        )

        raw = (res.content or "").strip()
        state["execute_plan_raw"] = raw

        (block_dir / f"execute_plan.attempt_{attempt}.raw.txt").write_text(
            raw,
            encoding="utf-8",
        )

        json_text = extract_json_object_only(raw)
        result = json.loads(json_text)

        if not isinstance(result, dict):
            raise ValueError("Executor result must be a JSON object")

        result.setdefault("files_to_write", [])
        result.setdefault("files_to_delete", [])

        if not isinstance(result["files_to_write"], list):
            raise ValueError("files_to_write must be a list")

        if not isinstance(result["files_to_delete"], list):
            raise ValueError("files_to_delete must be a list")

        files_to_write = result.get("files_to_write", [])
        files_to_delete = result.get("files_to_delete", [])

        state["files_to_write"] = files_to_write
        state["files_to_delete"] = files_to_delete

        if not files_to_write and not files_to_delete:
            state["execute_plan_result"] = result
            state["files_to_write"] = []
            state["files_to_delete"] = []

            state["executor_ok"] = False
            state["executor_error"] = "executor_no_changes"
            state["rollback_reason"] = "executor_no_changes"
            state["executor_feedback"] = (
                "EXECUTOR_NO_CHANGES: The executor returned valid JSON, "
                "but produced no files_to_write or files_to_delete. "
                "Revise the same block and generate concrete file changes."
            )

            write_json(
                block_dir / f"execute_plan.attempt_{attempt}.result.json",
                result,
            )

            return state

        state["execute_plan_result"] = result
        state["executor_ok"] = True
        state["executor_error"] = ""
        state["rollback_reason"] = ""

        write_json(
            block_dir / f"execute_plan.attempt_{attempt}.result.json",
            result,
        )

        # Mantém também o nome antigo apontando para a última tentativa.
        write_json(block_dir / "execute_plan.result.json", result)
        (block_dir / "execute_plan.raw.txt").write_text(raw, encoding="utf-8")

        return state

    except Exception as e:
        err = str(e)

        state["execute_plan_result"] = {}
        state["executor_ok"] = False
        state["executor_error"] = err
        state["rollback_reason"] = "invalid_executor_json"
        state["executor_feedback"] = f"EXECUTOR_INVALID_JSON: {err}"

        (block_dir / f"execute_plan.attempt_{attempt}.parse_error.txt").write_text(
            err + "\n",
            encoding="utf-8",
        )

        return state
    
def after_execute_plan(state: SourceRefactorState) -> str:
    if state.get("executor_ok", False):
        return "apply_changes"

    current_attempt = int(state.get("block_attempt", 0))
    max_attempts = int(state.get("max_block_attempts", 5))

    if current_attempt + 1 < max_attempts:
        return "retry_executor"

    state["rollback_reason"] = "block_attempt_exhausted"
    state["stop_reason"] = "executor_failed_after_retries"
    return "rollback_final"

def retry_executor_node(state: SourceRefactorState) -> SourceRefactorState:
    block_dir = Path(state["current_block_dir"])

    current_attempt = int(state.get("block_attempt", 0))
    next_attempt = current_attempt + 1

    reason = state.get("rollback_reason", "")
    feedback = state.get("executor_feedback", "")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    rid = uuid.uuid4().hex[:8]
    fname = f"retry.block_attempt{next_attempt}.{ts}_{rid}.txt"

    content = (
        f"previous_attempt={current_attempt}\n"
        f"next_attempt={next_attempt}\n"
        f"reason={reason}\n"
        f"feedback={feedback}\n"
    )

    (block_dir / fname).write_text(content, encoding="utf-8")

    with (block_dir / "retry.index.jsonl").open("a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                {
                    "file": fname,
                    "previous_attempt": current_attempt,
                    "next_attempt": next_attempt,
                    "reason": reason,
                    "ts": ts,
                },
                ensure_ascii=False,
            )
            + "\n"
        )

    state["block_attempt"] = next_attempt

    # A próxima tentativa deve começar do baseline limpo.
    return state

def rollback_final_node(state: SourceRefactorState) -> SourceRefactorState:
    repo_path = Path(state["repo_path"]).resolve()
    source_refactor_dir = Path(state["source_refactor_dir"])

    current_block_dir = state.get("current_block_dir", "")
    block_dir = Path(current_block_dir) if current_block_dir else source_refactor_dir

    rollback_to = state.get("last_good_commit") or state.get("initial_commit")

    if not rollback_to:
        raise RuntimeError("rollback_final_node: missing last_good_commit/initial_commit")

    before_commit = git_current_commit(repo_path)
    before_status = git_status_porcelain(repo_path)

    reset_output = git_reset_hard(repo_path, rollback_to)
    git_clean_workspace(repo_path)

    after_commit = git_current_commit(repo_path)
    after_status = git_status_porcelain(repo_path)

    rollback_info = {
        "ok": True,
        "rollback_to": rollback_to,
        "before_commit": before_commit,
        "after_commit": after_commit,
        "before_status": before_status,
        "after_status": after_status,
        "reason": state.get("rollback_reason", ""),
        "current_block_id": state.get("current_block_id", ""),
        "compile_ok": state.get("compile_ok", False),
        "compile_return_code": state.get("compile_return_code", None),
        "executor_ok": state.get("executor_ok", False),
        "executor_error": state.get("executor_error", ""),
        "apply_ok": state.get("apply_ok", False),
        "apply_error": state.get("apply_error", ""),
        "reset_output": reset_output,
    }

    write_json(block_dir / "rollback.final.json", rollback_info)
    write_json(source_refactor_dir / "rollback.final.json", rollback_info)

    state["workspace_clean"] = True
    state["final_commit"] = after_commit
    state["stop_reason"] = state.get("stop_reason") or "rolled_back_after_failure"

    return state

def apply_changes_node(state: SourceRefactorState) -> SourceRefactorState:
    repo_path = Path(state["repo_path"]).resolve()
    block_dir = Path(state["current_block_dir"])

    result = state.get("execute_plan_result", {})
    allowed_files = set(state.get("allowed_files", []))

    attempt = int(state.get("block_attempt", 0))

    applied_files: list[str] = []

    try:
        for item in result.get("files_to_write", []):
            if not isinstance(item, dict):
                raise ValueError("Each files_to_write item must be an object")

            rel_path = str(item.get("path", "")).strip()

            if rel_path not in allowed_files:
                raise ValueError(
                    f"Executor attempted to write file outside allowed_files: {rel_path}"
                )

            content = item.get("content", None)

            if not isinstance(content, str):
                raise ValueError(f"Missing content for file: {rel_path}")

            abs_path = ensure_path_inside_repo(repo_path, rel_path)
            write_text_file(abs_path, content)
            applied_files.append(rel_path)

        for rel_path in result.get("files_to_delete", []):
            rel_path = str(rel_path).strip()

            if rel_path not in allowed_files:
                raise ValueError(
                    f"Executor attempted to delete file outside allowed_files: {rel_path}"
                )

            abs_path = ensure_path_inside_repo(repo_path, rel_path)
            delete_file(abs_path)
            applied_files.append(rel_path)

        git_status = git_status_porcelain(repo_path)

        state["applied_files"] = applied_files
        state["apply_ok"] = True
        state["apply_error"] = ""
        state["rollback_reason"] = ""

        write_json(
            block_dir / f"apply.attempt_{attempt}.result.json",
            {
                "ok": True,
                "applied_files": applied_files,
                "git_status": git_status,
            },
        )

        write_json(
            block_dir / "apply.result.json",
            {
                "ok": True,
                "applied_files": applied_files,
                "git_status": git_status,
            },
        )

        return state

    except Exception as e:
        err = str(e)

        state["applied_files"] = []
        state["apply_ok"] = False
        state["apply_error"] = err
        state["rollback_reason"] = "apply_changes_failed"
        state["executor_feedback"] = f"APPLY_CHANGES_FAILED: {err}"

        write_json(
            block_dir / f"apply.attempt_{attempt}.error.json",
            {
                "ok": False,
                "error": err,
                "rollback_reason": state["rollback_reason"],
            },
        )

        return state

def after_apply_changes(state: SourceRefactorState) -> str:
    if state.get("apply_ok", False):
        return "compile_source"

    current_attempt = int(state.get("block_attempt", 0))
    max_attempts = int(state.get("max_block_attempts", 5))

    if current_attempt + 1 < max_attempts:
        return "retry_executor"

    state["rollback_reason"] = "block_attempt_exhausted"
    state["stop_reason"] = "apply_failed_after_retries"
    return "rollback_final"

def compile_source_node(state: SourceRefactorState) -> SourceRefactorState:
    cfg = state["config"]
    repo_path = Path(state["repo_path"]).resolve()
    block_dir = Path(state["current_block_dir"])

    compile_command = get_config_value(cfg, "compile.command", [])

    if not isinstance(compile_command, list):
        raise ValueError("compile.command must be a list")

    compile_command = [str(part) for part in compile_command]

    timeout = int(get_config_value(cfg, "compile.timeout", 300))

    return_code, output = run_compile_command(
        repo_path=repo_path,
        command=compile_command,
        timeout=timeout,
    )

    compile_log_path = block_dir / "compile.log"
    compile_log_path.write_text(output, encoding="utf-8")

    compile_ok = return_code == 0

    state["compile_return_code"] = return_code
    state["compile_ok"] = compile_ok
    state["compile_log_path"] = str(compile_log_path)
    state["compile_log"] = output

    if not compile_ok:
        state["rollback_reason"] = "compile_failed"
        state["stop_reason"] = "compile_failed"

    write_json(
        block_dir / "compile.status.json",
        {
            "ok": compile_ok,
            "return_code": return_code,
            "command": compile_command,
            "compile_log": str(compile_log_path),
        },
    )

    return state

def after_compile_source(state: SourceRefactorState) -> str:
    if state.get("compile_ok", False):
        return "promote_block"

    return "rollback_final"

def promote_block_node(state: SourceRefactorState) -> SourceRefactorState:
    repo_path = Path(state["repo_path"]).resolve()
    source_refactor_dir = Path(state["source_refactor_dir"])

    block_id = state.get("current_block_id", "")
    block_index = int(state.get("current_block_index", 0))
    block_dir = Path(state["current_block_dir"])

    compile_ok = bool(state.get("compile_ok", False))

    if not compile_ok:
        state["stop_reason"] = "compile_failed"
        return state

    commit = git_commit_all(
        repo_path,
        f"source_refactor: apply block {block_id}",
    )

    state["current_block_commit"] = commit
    state["last_good_commit"] = commit

    block_commit = {
        "block_id": block_id,
        "block_index": block_index,
        "commit": commit,
        "message": f"source_refactor: apply block {block_id}",
    }

    block_commits = list(state.get("block_commits", []))
    block_commits.append(block_commit)
    state["block_commits"] = block_commits

    git_dir = source_refactor_dir / "git"
    git_dir.mkdir(parents=True, exist_ok=True)

    (git_dir / "last_good_commit.txt").write_text(
        commit + "\n",
        encoding="utf-8",
    )

    write_json(git_dir / "block_commits.json", block_commits)

    write_json(
        block_dir / "block.commit.json",
        block_commit,
    )

    block_summary = {
        "block_id": block_id,
        "block_index": block_index,
        "attempt": int(state.get("block_attempt", 0)),
        "max_block_attempts": int(state.get("max_block_attempts", 0)),

        "executor_ok": bool(state.get("executor_ok", False)),
        "executor_error": state.get("executor_error", ""),

        "apply_ok": bool(state.get("apply_ok", False)),
        "apply_error": state.get("apply_error", ""),
        "applied_files": state.get("applied_files", []),

        "compile_ok": bool(state.get("compile_ok", False)),
        "compile_return_code": state.get("compile_return_code", None),
        "compile_log_path": state.get("compile_log_path", ""),

        "commit": commit,
        "rollback_reason": state.get("rollback_reason", ""),
        "stop_reason": "block_applied",

        "files": {
            "executor_existing_files": state.get("executor_existing_files", []),
            "executor_new_files": state.get("executor_new_files", []),
            "executor_files": state.get("executor_files", []),
            "executor_rejected_files": state.get("executor_rejected_files", []),
        },
    }

    write_json(block_dir / "block.summary.json", block_summary)

    block_summaries = list(state.get("block_summaries", []))
    block_summaries.append(block_summary)
    state["block_summaries"] = block_summaries

    write_json(
        source_refactor_dir / "block_summaries.json",
        block_summaries,
    )

    state["stop_reason"] = "block_applied"

    return state

def advance_block_node(state: SourceRefactorState) -> SourceRefactorState:
    blocks = state.get("executable_plan", {}).get("blocks", [])

    current_index = int(state.get("current_block_index", 0))
    next_index = current_index + 1

    state["current_block_index"] = next_index

    if next_index >= len(blocks):
        state["stop_reason"] = "all_blocks_applied"
    else:
        state["stop_reason"] = "block_applied"

    return state

def after_advance_block(state: SourceRefactorState) -> str:
    blocks = state.get("executable_plan", {}).get("blocks", [])
    current_index = int(state.get("current_block_index", 0))

    if current_index >= len(blocks):
        return "save_status"

    return "stage_block"

def save_status_node(state: SourceRefactorState) -> SourceRefactorState:
    source_refactor_dir = Path(state["source_refactor_dir"])

    blocks = state.get("executable_plan", {}).get("blocks", [])

    final_ok = (
        state.get("stop_reason") == "all_blocks_applied"
        and bool(state.get("compile_ok", False))
        and bool(state.get("executor_ok", False))
        and bool(state.get("apply_ok", False))
    )

    status = {
        "mvp": "source_refactor",
        "ok": final_ok,
        "run_id": state.get("run_id", ""),
        "project": state.get("project_name", ""),
        "repo_path": state.get("repo_path", ""),
        "smell": state.get("smell", ""),
        "smell_name": state.get("smell_name", ""),
        "target_type": state.get("target_type", ""),
        "target_name": state.get("target_name", ""),
        "blocks_count": len(blocks),
        "blocks_applied": len(state.get("block_summaries", [])),
        "block_summaries": state.get("block_summaries", []),
        "current_block_id": state.get("current_block_id", ""),
        "applied_files": state.get("applied_files", []),
        "compile_ok": state.get("compile_ok", False),
        "compile_return_code": state.get("compile_return_code", None),
        "compile_log_path": state.get("compile_log_path", ""),
        "initial_commit": state.get("initial_commit", ""),
        "last_good_commit": state.get("last_good_commit", ""),
        "current_block_commit": state.get("current_block_commit", ""),
        "final_commit": state.get("final_commit", ""),
        "workspace_clean": state.get("workspace_clean", False),
        "block_attempt": state.get("block_attempt", 0),
        "max_block_attempts": state.get("max_block_attempts", 0),
        "executor_ok": state.get("executor_ok", False),
        "executor_error": state.get("executor_error", ""),
        "apply_ok": state.get("apply_ok", False),
        "apply_error": state.get("apply_error", ""),
        "rollback_reason": state.get("rollback_reason", ""),
        "stop_reason": state.get("stop_reason", "applied_first_block_only"),
    }

    contract = {
        "producer": "source_refactor",
        "version": "1.0",
        "ok": final_ok,
        "run_id": state.get("run_id", ""),
        "input": {
            "planner_contract": state.get("planner_contract_path", ""),
            "planner_plan": state.get("planner_plan_path", ""),
        },
        "project": {
            "name": state.get("project_name", ""),
            "repo_path": state.get("repo_path", ""),
        },
        "target": {
            "smell": state.get("smell", ""),
            "smell_name": state.get("smell_name", ""),
            "target_type": state.get("target_type", ""),
            "target_name": state.get("target_name", ""),
        },
        "artifacts": {
            "source_refactor_dir": str(source_refactor_dir),
            "input_contract": str(source_refactor_dir / "input_contract.json"),
            "input_plan": str(source_refactor_dir / "input_plan.json"),
            "executable_plan": str(source_refactor_dir / "executable_plan.json"),
            "status": str(source_refactor_dir / "status.json"),
            "git_dir": str(source_refactor_dir / "git"),
            "initial_commit": str(source_refactor_dir / "git" / "initial_commit.txt"),
            "last_good_commit": str(source_refactor_dir / "git" / "last_good_commit.txt"),
            "current_block_dir": state.get("current_block_dir", ""),
            "compile_log": state.get("compile_log_path", ""),
            "current_block_dir": state.get("current_block_dir", ""),
            "block_summaries": str(source_refactor_dir / "block_summaries.json"),
        },
        "commits": {
            "initial_commit": state.get("initial_commit", ""),
            "last_good_commit": state.get("last_good_commit", ""),
            "final_commit": state.get("final_commit", ""),
            "current_block_commit": state.get("current_block_commit", ""),
            "block_commits": state.get("block_commits", []),
        },
    }

    state["status"] = status
    state["contract"] = contract

    write_json(source_refactor_dir / "status.json", status)
    write_json(source_refactor_dir / "contract.json", contract)

    return state