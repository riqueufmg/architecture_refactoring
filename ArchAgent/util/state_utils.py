from pathlib import Path

from State import State


def _get_plan_dir(state: State) -> Path:
    run_dir = Path(state["run_dir"])
    plan_idx = int(state.get("plan_idx", 0))
    plan_dir = run_dir / f"plan_{plan_idx:02d}"
    plan_dir.mkdir(parents=True, exist_ok=True)
    return plan_dir


def _get_target_type(state: State) -> str:
    target_type = (state.get("target_type") or "class").strip().lower()
    if target_type not in {"class", "package"}:
        return "class"
    return target_type


def _get_target_identity(state: State) -> str:
    return (state.get("target_name") or "").strip()