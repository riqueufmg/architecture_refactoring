import shutil
import uuid
from datetime import datetime
from pathlib import Path

from mvp.source_refactor.state import SourceRefactorState
from mvp.source_refactor.lib.config_utils import load_config, require_config_value
from mvp.source_refactor.lib.json_utils import read_json, write_json
from mvp.source_refactor.lib.path_utils import (
    require_absolute_path,
    ensure_file_exists,
    ensure_dir_exists,
)
from mvp.source_refactor.lib.git_utils import (
    ensure_clean_git_workspace,
    git_current_commit,
)


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

    return state

def prepare_executable_plan_node(state: SourceRefactorState) -> SourceRefactorState:
    source_refactor_dir = Path(state["source_refactor_dir"])

    # MVP inicial: sem enrichment preventivo.
    executable_plan = dict(state["input_plan"])

    state["executable_plan"] = executable_plan
    state["current_block_index"] = 0

    write_json(source_refactor_dir / "executable_plan.json", executable_plan)

    return state


def save_status_node(state: SourceRefactorState) -> SourceRefactorState:
    source_refactor_dir = Path(state["source_refactor_dir"])

    blocks = state.get("executable_plan", {}).get("blocks", [])

    status = {
        "mvp": "source_refactor",
        "ok": True,
        "run_id": state.get("run_id", ""),
        "project": state.get("project_name", ""),
        "repo_path": state.get("repo_path", ""),
        "smell": state.get("smell", ""),
        "smell_name": state.get("smell_name", ""),
        "target_type": state.get("target_type", ""),
        "target_name": state.get("target_name", ""),
        "blocks_count": len(blocks),
        "initial_commit": state.get("initial_commit", ""),
        "last_good_commit": state.get("last_good_commit", ""),
        "workspace_clean": state.get("workspace_clean", False),
        "stop_reason": state.get("stop_reason", "loaded_plan_only"),
    }

    contract = {
        "producer": "source_refactor",
        "version": "1.0",
        "ok": True,
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
        },
    }

    state["status"] = status
    state["contract"] = contract

    write_json(source_refactor_dir / "status.json", status)
    write_json(source_refactor_dir / "contract.json", contract)

    return state