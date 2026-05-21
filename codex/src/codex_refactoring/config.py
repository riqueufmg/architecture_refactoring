from pathlib import Path
from typing import Any

import yaml

# load configuration file with experiment attributes
def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError("Config file must contain a YAML object.")

    required_fields = [
        "project_name",
        "repo_path",
        "smell",
        "smell_name",
        "target_type",
        "target_name",
        "maven_command",
        "designite",
        "codex",
    ]

    missing = [field for field in required_fields if field not in config]

    if missing:
        raise ValueError(f"Missing required config fields: {missing}")

    return config
