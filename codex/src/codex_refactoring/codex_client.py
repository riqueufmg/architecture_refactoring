import subprocess
import time
from pathlib import Path
from typing import Any

# This module provides utilities for running Codex commands based on a given configuration.
def build_codex_command(config: dict[str, Any], repo_path: str | Path) -> list[str]:
    codex_config = config["codex"]

    base_command = codex_config.get("command", ["codex"])
    model = codex_config.get("model")
    sandbox = codex_config.get("sandbox", "workspace-write")

    command = [
        *base_command,
        "exec",
        "--cd",
        str(repo_path),
        "--sandbox",
        sandbox,
        "--json",
    ]

    if model:
        command.extend(["--model", str(model)])

    command.append("-")

    return command

# This function runs the Codex command with the specified configuration and paths for the repository, prompt, stdout, and stderr. It returns a dictionary containing the command executed, the return code, elapsed time, and whether the execution was successful.
def run_codex(
    config: dict[str, Any],
    repo_path: str | Path,
    prompt_path: str | Path,
    stdout_path: str | Path,
    stderr_path: str | Path,
) -> dict[str, Any]:
    command = build_codex_command(config, repo_path)

    start = time.time()

    with Path(prompt_path).open("r", encoding="utf-8") as prompt_file, \
        Path(stdout_path).open("w", encoding="utf-8") as stdout_file, \
        Path(stderr_path).open("w", encoding="utf-8") as stderr_file:

        result = subprocess.run(
            command,
            stdin=prompt_file,
            stdout=stdout_file,
            stderr=stderr_file,
            text=True,
            check=False,
        )

    elapsed = round(time.time() - start, 2)

    return {
        "command": command,
        "returncode": result.returncode,
        "seconds": elapsed,
        "ok": result.returncode == 0,
    }