from pathlib import Path
from typing import Any

from State import State


def get_config(state: State) -> dict[str, Any]:
    cfg = state.get("config") or {}

    if not isinstance(cfg, dict):
        return {}

    return cfg


def get_config_value(state: State, dotted_key: str, default: Any = None) -> Any:
    current: Any = get_config(state)

    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]

    return current


def require_config_value(state: State, dotted_key: str) -> Any:
    value = get_config_value(state, dotted_key, default=None)

    if value is None:
        raise ValueError(f"Missing required config value: {dotted_key}")

    return value

def resolve_path_from_base(base_path: Path, maybe_relative: str | Path) -> Path:
    path = Path(maybe_relative).expanduser()

    if path.is_absolute():
        return path.resolve()

    return (base_path / path).resolve()