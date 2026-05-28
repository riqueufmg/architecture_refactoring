from typing import Any, TypedDict

class SourceRefactorState(TypedDict, total=False):
    config_path: str
    config: dict[str, Any]

    run_id: str
    run_dir: str
    source_refactor_dir: str

    planner_contract_path: str
    planner_contract: dict[str, Any]
    planner_dir: str
    planner_plan_path: str
    planner_input_path: str

    input_plan: dict[str, Any]
    executable_plan: dict[str, Any]

    repo_path: str
    project_name: str

    smell: str
    smell_name: str
    target_type: str
    target_name: str

    current_block_index: int
    current_block: dict[str, Any]

    status: dict[str, Any]
    contract: dict[str, Any]
    stop_reason: str

    initial_commit: str
    last_good_commit: str
    final_commit: str

    workspace_clean: bool
    block_commits: list[dict[str, str]]
    repair_commits: list[dict[str, str]]

    current_block_dir: str
    current_block_id: str

    allowed_files: list[str]
    files_context: list[dict[str, str]]

    executor_system_prompt_path: str
    execute_plan_prompt_path: str
    executor_system_prompt: str
    execute_plan_prompt: str
    execute_plan_rendered: str
    execute_plan_raw: str
    execute_plan_result: dict[str, Any]

    applied_files: list[str]