import json
from datetime import datetime
from pathlib import Path
from typing import Any

from codex_refactoring.codex_client import run_codex
from codex_refactoring.git_utils import (
    ensure_clean_repo,
    ensure_git_repository,
    get_current_commit,
    get_diff_stats,
    save_git_diff,
)
from codex_refactoring.prompt_builder import build_prompt, save_prompt


def sanitize_for_path(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
    )


def create_run_dir(config: dict[str, Any], runs_root: str | Path = "runs") -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    project_name = sanitize_for_path(str(config["project_name"]))
    smell = sanitize_for_path(str(config["smell"]))
    target = sanitize_for_path(str(config["target_name"]))

    run_name = f"{timestamp}_{project_name}_{smell}_{target}"

    run_dir = Path(runs_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    return run_dir


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    output_path = Path(path)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def build_initial_status(
    config: dict[str, Any],
    repo_path: Path,
    initial_commit: str,
    run_dir: Path,
) -> dict[str, Any]:
    return {
        "project_name": config["project_name"],
        "smell": config["smell"],
        "smell_name": config["smell_name"],
        "target_type": config["target_type"],
        "target_name": config["target_name"],
        "repo_path": str(repo_path),
        "run_dir": str(run_dir),
        "initial_commit": initial_commit,
        "status": "initialized",
        "codex_success": None,
        "codex_returncode": None,
        "codex_wall_time_seconds": None,
        "compile_success": None,
        "smell_present_before": None,
        "smell_present_after": None,
        "smell_removed": None,
        "changed_files_count": None,
        "lines_added": None,
        "lines_deleted": None,
    }


def run_experiment(config: dict[str, Any]) -> Path:
    repo_path = ensure_git_repository(config["repo_path"])

    ensure_clean_repo(repo_path)

    initial_commit = get_current_commit(repo_path)

    run_dir = create_run_dir(config)

    input_data = {
        "config": config,
        "initial_commit": initial_commit,
    }

    status_data = build_initial_status(
        config=config,
        repo_path=repo_path,
        initial_commit=initial_commit,
        run_dir=run_dir,
    )

    write_json(run_dir / "input.json", input_data)
    write_json(run_dir / "status.json", status_data)

    prompt = build_prompt(config)
    prompt_path = run_dir / "prompt.md"
    save_prompt(prompt, prompt_path)

    codex_result = run_codex(
        config=config,
        repo_path=repo_path,
        prompt_path=prompt_path,
        stdout_path=run_dir / "codex.stdout.log",
        stderr_path=run_dir / "codex.stderr.log",
    )

    save_git_diff(repo_path, run_dir / "patch.diff")
    diff_stats = get_diff_stats(repo_path)

    status_data.update(
        {
            "status": "codex_finished" if codex_result["ok"] else "codex_failed",
            "codex_success": codex_result["ok"],
            "codex_returncode": codex_result["returncode"],
            "codex_wall_time_seconds": codex_result["seconds"],
            **diff_stats,
        }
    )

    write_json(run_dir / "status.json", status_data)

    metrics_data = {
        "project_name": config["project_name"],
        "smell": config["smell"],
        "target_type": config["target_type"],
        "target_name": config["target_name"],
        "initial_commit": initial_commit,
        "codex": codex_result,
        "diff": diff_stats,
    }

    write_json(run_dir / "metrics.json", metrics_data)

    return run_dir