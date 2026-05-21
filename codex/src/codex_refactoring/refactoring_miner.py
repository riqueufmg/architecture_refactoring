import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any


def run_git_command(repo_path: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
    )


def has_uncommitted_changes(repo_path: str | Path) -> bool:
    repo = Path(repo_path).resolve()

    result = run_git_command(repo, ["status", "--porcelain"])

    if result.returncode != 0:
        raise RuntimeError(
            "Could not check Git status before Refactoring Miner.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    return result.stdout.strip() != ""


def get_head_commit(repo_path: str | Path) -> str:
    repo = Path(repo_path).resolve()

    result = run_git_command(repo, ["rev-parse", "HEAD"])

    if result.returncode != 0:
        raise RuntimeError(
            "Could not get HEAD commit.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    return result.stdout.strip()


def create_temporary_final_commit(repo_path: str | Path, run_name: str) -> dict[str, Any]:
    repo = Path(repo_path).resolve()

    if not has_uncommitted_changes(repo):
        return {
            "created": False,
            "commit": get_head_commit(repo),
            "message": "No uncommitted changes found. Using current HEAD.",
        }

    add_result = run_git_command(repo, ["add", "-A"])

    if add_result.returncode != 0:
        raise RuntimeError(
            "Could not stage files for temporary final commit.\n"
            f"STDOUT:\n{add_result.stdout}\n"
            f"STDERR:\n{add_result.stderr}"
        )

    commit_message = f"Temporary Codex refactoring result for {run_name}"

    commit_result = run_git_command(
        repo,
        [
            "-c",
            "user.name=Codex Experiment",
            "-c",
            "user.email=codex-experiment@example.com",
            "commit",
            "-m",
            commit_message,
        ],
    )

    if commit_result.returncode != 0:
        raise RuntimeError(
            "Could not create temporary final commit.\n"
            f"STDOUT:\n{commit_result.stdout}\n"
            f"STDERR:\n{commit_result.stderr}"
        )

    return {
        "created": True,
        "commit": get_head_commit(repo),
        "message": commit_message,
    }


def summarize_refactoring_miner_json(json_path: str | Path) -> dict[str, Any]:
    path = Path(json_path)

    if not path.exists():
        return {
            "refactorings_count": 0,
            "refactoring_types": {},
        }

    data = json.loads(path.read_text(encoding="utf-8", errors="replace"))

    refactorings: list[dict[str, Any]] = []

    for commit in data.get("commits", []):
        refactorings.extend(commit.get("refactorings", []))

    type_counter = Counter(
        refactoring.get("type", "UNKNOWN")
        for refactoring in refactorings
    )

    return {
        "refactorings_count": len(refactorings),
        "refactoring_types": dict(type_counter),
    }


def run_refactoring_miner(
    repo_path: str | Path,
    command: list[str],
    start_commit: str,
    end_commit: str,
    output_json_path: str | Path,
    log_path: str | Path,
) -> dict[str, Any]:
    repo = Path(repo_path).resolve()
    output_json = Path(output_json_path).resolve()
    log_file = Path(log_path).resolve()

    output_json.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    resolved_command = command.copy()

    command_path = Path(resolved_command[0])

    if not command_path.is_absolute():
        resolved_command[0] = str(command_path.resolve())

    cmd = [
        *resolved_command,
        "-bc",
        str(repo),
        start_commit,
        end_commit,
        "-json",
        str(output_json),
    ]

    start = time.time()

    with log_file.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            cmd,
            cwd=repo,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    elapsed = round(time.time() - start, 2)

    summary = summarize_refactoring_miner_json(output_json)

    return {
        "command": cmd,
        "returncode": result.returncode,
        "seconds": elapsed,
        "ok": result.returncode == 0,
        "output_json": str(output_json),
        "log_path": str(log_file),
        **summary,
    }