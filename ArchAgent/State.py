from typing import TypedDict, Any

class State(TypedDict, total=False):
    repo_path: str              # path to the repo to refactor
    msg: str
    run_dir: str                # path for log files

    project_root: str

    # configuration
    config: dict[str, Any]

    max_plans: int
    max_block_attempts: int
    max_compile_repair_attempts: int
    max_test_repair_attempts: int

    target_type: str            # class or package

    # state for classes
    target_file: str            # primary target file for class mode
    target_class_fqn: str       # class target FQN

    # state for package
    target_package_fqn: str     # package target FQN
    target_files: list[str]     # package target files (repo-relative)
    external_files: list[str]   # files outside the target package but depend on or are depended on by the target package (repo-relative)
    internal_deps: list[str]    # list of dependencies between files in the target package
    incoming_deps: list[str]    # list of dependencies from outside the target package to inside
    outgoing_deps: list[str]    # list of dependencies from inside the target package to outside
    movable_internal_deps: list[str] # list of internal dependencies that can be moved together with the target package
    movable_files: list[str]    # list of files that can be moved together with the

    target_name: str            # class or package FQN
    
    target_source_root: str     

    start_commit: str
    base_commit: str

    # prompts path
    executor_prompt_path: str          # prompt for LLM executor
    smell_quality_prompt_path: str     # prompt for LLM quality analysis

    # plan lifecycle
    plan_idx: int # num of plan tries
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
    executor_rejected_files: list[str]

    executor_result: dict                 # executor result data
    files_to_write: list[dict]            # each: {"path": "...", "content": "..."}
    files_to_delete: list[str]            # repo-relative paths

    # apply files node
    apply_ok: bool
    apply_error: str

    # openrewrite node
    openrewrite_ok: bool
    openrewrite_returncode: int
    openrewrite_error: str

    # compilation data
    compile_ok: bool
    compile_returncode: int
    compile_log_tail: str
    maven_tmp_dir: str

    # smell analysis data
    smell_quality_ok: bool             # check if the LLM define why LLM wasn't remove
    smell_quality_error: str           # check error in LLM inference
    smell_quality_analysis: str        # LLM quality analysis answer

    # designite/smell data
    designite_ok: bool                 # if designite run successfully
    smell_type: str                    # eg: "Insufficient Modularization"
    designite_smells_csv: str          # designite smell file eg: "DesignSmells.csv"
    designite_smell_name: str          # smell label on designite output
    smell_still_present: bool          # smell remove evaluation
    
    rollback_reason: str
    rollback_commit: str