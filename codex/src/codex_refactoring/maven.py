import subprocess
import time
from pathlib import Path
from typing import Any


def run_maven(
    repo_path: str | Path,
    maven_command: list[str],
    log_path: str | Path,
) -> dict[str, Any]:
    repo = Path(repo_path).resolve()
    log_file = Path(log_path)

    start = time.time()

    with log_file.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            maven_command,
            cwd=repo,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    elapsed = round(time.time() - start, 2)

    return {
        "command": maven_command,
        "returncode": result.returncode,
        "seconds": elapsed,
        "ok": result.returncode == 0,
    }