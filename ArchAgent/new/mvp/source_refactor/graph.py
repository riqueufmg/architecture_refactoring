from langgraph.graph import StateGraph, END

from mvp.source_refactor.state import SourceRefactorState
from mvp.source_refactor.nodes import (
    load_config_node,
    init_run_node,
    load_planner_contract_node,
    load_plan_node,
    prepare_executable_plan_node,
    ensure_clean_workspace_node,
    record_initial_commit_node,
    save_status_node,
)


def build_source_refactor_graph():
    g = StateGraph(SourceRefactorState)

    g.add_node("load_config", load_config_node)
    g.add_node("init_run", init_run_node)
    g.add_node("load_planner_contract", load_planner_contract_node)
    g.add_node("load_plan", load_plan_node)
    g.add_node("prepare_executable_plan", prepare_executable_plan_node)
    g.add_node("ensure_clean_workspace", ensure_clean_workspace_node)
    g.add_node("record_initial_commit", record_initial_commit_node)
    g.add_node("save_status", save_status_node)

    g.set_entry_point("load_config")

    g.add_edge("load_config", "init_run")
    g.add_edge("init_run", "load_planner_contract")
    g.add_edge("load_planner_contract", "load_plan")
    g.add_edge("load_plan", "prepare_executable_plan")
    g.add_edge("prepare_executable_plan", "ensure_clean_workspace")
    g.add_edge("ensure_clean_workspace", "record_initial_commit")
    g.add_edge("record_initial_commit", "save_status")
    g.add_edge("save_status", END)

    return g.compile()