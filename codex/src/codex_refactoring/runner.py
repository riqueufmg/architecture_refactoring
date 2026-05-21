import json
from datetime import datetime
from pathlib import Path
from typing import Any

from codex_refactoring.designite import (
    find_target_smell,
    run_designite
)

from codex_refactoring.refactoring_miner import (
    create_temporary_final_commit,
    run_refactoring_miner,
)

from codex_refactoring.prompt_builder import (
    build_compile_repair_prompt,
    build_continue_smell_prompt,
    build_prompt,
    save_prompt,
)

from codex_refactoring.codex_client import run_codex

from codex_refactoring.git_utils import (
    ensure_clean_repo,
    ensure_git_repository,
    get_current_commit,
    get_diff_stats,
    save_git_diff,
)
from codex_refactoring.maven import run_maven

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

def run_codex_attempt(
    config: dict[str, Any],
    repo_path: Path,
    run_dir: Path,
    smell_attempt_number: int,
    compile_attempt_number: int,
    prompt: str,
) -> dict[str, Any]:
    attempt_dir = (
        run_dir
        / f"smell_attempt_{smell_attempt_number}"
        / f"compile_attempt_{compile_attempt_number}"
    )

    attempt_dir.mkdir(parents=True, exist_ok=False)

    prompt_path = attempt_dir / "prompt.md"
    save_prompt(prompt, prompt_path)

    codex_result = run_codex(
        config=config,
        repo_path=repo_path,
        prompt_path=prompt_path,
        stdout_path=attempt_dir / "codex.stdout.log",
        stderr_path=attempt_dir / "codex.stderr.log",
    )

    save_git_diff(repo_path, attempt_dir / "patch.diff")
    save_git_diff(repo_path, run_dir / "patch.diff")

    diff_stats = get_diff_stats(repo_path)

    if not codex_result["ok"]:
        return {
            "smell_attempt": smell_attempt_number,
            "compile_attempt": compile_attempt_number,
            "attempt_dir": str(attempt_dir),
            "codex": codex_result,
            "diff": diff_stats,
            "maven": None,
            "codex_success": False,
            "compile_success": False,
        }

    maven_result = run_maven(
        repo_path=repo_path,
        maven_command=config["maven_command"],
        log_path=attempt_dir / "maven.log",
    )

    return {
        "smell_attempt": smell_attempt_number,
        "compile_attempt": compile_attempt_number,
        "attempt_dir": str(attempt_dir),
        "codex": codex_result,
        "diff": diff_stats,
        "maven": maven_result,
        "codex_success": True,
        "compile_success": maven_result["ok"],
    }

def run_compile_validated_attempts(
    config: dict[str, Any],
    repo_path: Path,
    run_dir: Path,
    smell_attempt_number: int,
    initial_prompt: str,
) -> dict[str, Any]:
    max_compile_attempts = int(config.get("codex", {}).get("max_attempts", 1))
    max_compile_attempts = max(1, max_compile_attempts)

    attempts: list[dict[str, Any]] = []

    current_prompt = initial_prompt
    successful_attempt: dict[str, Any] | None = None
    last_attempt: dict[str, Any] | None = None

    for compile_attempt_number in range(1, max_compile_attempts + 1):
        attempt_result = run_codex_attempt(
            config=config,
            repo_path=repo_path,
            run_dir=run_dir,
            smell_attempt_number=smell_attempt_number,
            compile_attempt_number=compile_attempt_number,
            prompt=current_prompt,
        )

        attempts.append(attempt_result)
        last_attempt = attempt_result

        if not attempt_result["codex_success"]:
            return {
                "status": "codex_failed",
                "compile_success": False,
                "successful_attempt": None,
                "last_attempt": attempt_result,
                "attempts": attempts,
            }

        if attempt_result["compile_success"]:
            successful_attempt = attempt_result
            break

        if compile_attempt_number < max_compile_attempts:
            current_prompt = build_compile_repair_prompt(
                config=config,
                maven_log_path=Path(attempt_result["attempt_dir"]) / "maven.log",
            )

    if successful_attempt is None:
        return {
            "status": "compile_failed",
            "compile_success": False,
            "successful_attempt": None,
            "last_attempt": last_attempt,
            "attempts": attempts,
        }

    return {
        "status": "compile_success",
        "compile_success": True,
        "successful_attempt": successful_attempt,
        "last_attempt": successful_attempt,
        "attempts": attempts,
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

    # ------------------------------------------------------------
    # 1. Run Designite before Codex
    # ------------------------------------------------------------
    designite_before_dir = run_dir / "designite_before"

    designite_before_result = run_designite(
        repo_path=repo_path,
        jar_path=config["designite"]["jar_path"],
        output_dir=designite_before_dir,
        log_path=run_dir / "designite_before.log",
        java_path=config["designite"].get("java_path"),
    )

    if not designite_before_result["ok"]:
        status_data.update(
            {
                "status": "designite_before_failed",
                "designite_before_returncode": designite_before_result["returncode"],
                "designite_before_wall_time_seconds": designite_before_result["seconds"],
            }
        )

        metrics_data = {
            "project_name": config["project_name"],
            "smell": config["smell"],
            "target_type": config["target_type"],
            "target_name": config["target_name"],
            "initial_commit": initial_commit,
            "designite_before": designite_before_result,
            "smell_before": None,
            "attempts": [],
            "codex": None,
            "diff": None,
            "maven": None,
            "designite_after": None,
            "smell_after": None,
            "smell_removed": None,
        }

        write_json(run_dir / "status.json", status_data)
        write_json(run_dir / "metrics.json", metrics_data)

        return run_dir

    smell_detection_config = config.get("smell_detection", {})

    smell_before = find_target_smell(
        designite_output_dir=designite_before_dir,
        smell_name=config["smell_name"],
        smell_code=config["smell"],
        target_name=config["target_name"],
        csv_file=smell_detection_config.get("csv_file"),
        smell_column=smell_detection_config.get("smell_column"),
        package_column=smell_detection_config.get("package_column"),
        class_column=smell_detection_config.get("class_column"),
        target_column=smell_detection_config.get("target_column"),
    )

    status_data.update(
        {
            "smell_present_before": smell_before["present"],
            "smell_matches_before": smell_before["matches_count"],
            "designite_before_returncode": designite_before_result["returncode"],
            "designite_before_wall_time_seconds": designite_before_result["seconds"],
        }
    )

    write_json(run_dir / "status.json", status_data)

    if not smell_before["present"]:
        status_data.update(
            {
                "status": "invalid_initial_state",
                "smell_removed": False,
            }
        )

        metrics_data = {
            "project_name": config["project_name"],
            "smell": config["smell"],
            "target_type": config["target_type"],
            "target_name": config["target_name"],
            "initial_commit": initial_commit,
            "designite_before": designite_before_result,
            "smell_before": smell_before,
            "attempts": [],
            "codex": None,
            "diff": None,
            "maven": None,
            "designite_after": None,
            "smell_after": None,
            "smell_removed": False,
        }

        write_json(run_dir / "status.json", status_data)
        write_json(run_dir / "metrics.json", metrics_data)

        return run_dir

    # ------------------------------------------------------------
    # 2. Run smell attempts
    # ------------------------------------------------------------
    max_smell_attempts = int(config.get("codex", {}).get("max_smell_attempts", 1))
    max_smell_attempts = max(1, max_smell_attempts)

    all_attempts: list[dict[str, Any]] = []

    current_prompt = build_prompt(config)

    final_codex_result: dict[str, Any] | None = None
    final_diff_stats: dict[str, Any] | None = None
    final_maven_result: dict[str, Any] | None = None
    final_designite_after_result: dict[str, Any] | None = None
    final_smell_after: dict[str, Any] | None = None
    final_smell_attempt: int | None = None
    final_compile_attempt: int | None = None

    smell_removed = False
    final_status = "smell_not_removed"

    for smell_attempt_number in range(1, max_smell_attempts + 1):
        compile_result = run_compile_validated_attempts(
            config=config,
            repo_path=repo_path,
            run_dir=run_dir,
            smell_attempt_number=smell_attempt_number,
            initial_prompt=current_prompt,
        )

        all_attempts.extend(compile_result["attempts"])

        last_attempt = compile_result["last_attempt"]

        if last_attempt is None:
            final_status = "compile_failed"
            break

        final_codex_result = last_attempt["codex"]
        final_diff_stats = last_attempt["diff"]
        final_maven_result = last_attempt["maven"]
        final_smell_attempt = last_attempt["smell_attempt"]
        final_compile_attempt = last_attempt["compile_attempt"]

        # Keep a copy of the latest cumulative patch at the run root.
        save_git_diff(repo_path, run_dir / "patch.diff")

        if compile_result["status"] == "codex_failed":
            final_status = "codex_failed"

            status_data.update(
                {
                    "status": final_status,
                    "codex_success": False,
                    "codex_returncode": final_codex_result["returncode"],
                    "codex_wall_time_seconds": final_codex_result["seconds"],
                    "compile_success": False,
                    "attempts_count": len(all_attempts),
                    "final_smell_attempt": final_smell_attempt,
                    "final_compile_attempt": final_compile_attempt,
                    **final_diff_stats,
                }
            )

            metrics_data = {
                "project_name": config["project_name"],
                "smell": config["smell"],
                "target_type": config["target_type"],
                "target_name": config["target_name"],
                "initial_commit": initial_commit,
                "designite_before": designite_before_result,
                "smell_before": smell_before,
                "attempts": all_attempts,
                "codex": final_codex_result,
                "diff": final_diff_stats,
                "maven": None,
                "designite_after": None,
                "smell_after": None,
                "smell_removed": None,
            }

            write_json(run_dir / "status.json", status_data)
            write_json(run_dir / "metrics.json", metrics_data)

            return run_dir

        if not compile_result["compile_success"]:
            final_status = "compile_failed"

            status_data.update(
                {
                    "status": final_status,
                    "codex_success": True,
                    "codex_returncode": final_codex_result["returncode"],
                    "codex_wall_time_seconds": final_codex_result["seconds"],
                    "compile_success": False,
                    "maven_returncode": None
                    if final_maven_result is None
                    else final_maven_result["returncode"],
                    "maven_wall_time_seconds": None
                    if final_maven_result is None
                    else final_maven_result["seconds"],
                    "attempts_count": len(all_attempts),
                    "final_smell_attempt": final_smell_attempt,
                    "final_compile_attempt": final_compile_attempt,
                    **final_diff_stats,
                }
            )

            metrics_data = {
                "project_name": config["project_name"],
                "smell": config["smell"],
                "target_type": config["target_type"],
                "target_name": config["target_name"],
                "initial_commit": initial_commit,
                "designite_before": designite_before_result,
                "smell_before": smell_before,
                "attempts": all_attempts,
                "codex": final_codex_result,
                "diff": final_diff_stats,
                "maven": final_maven_result,
                "designite_after": None,
                "smell_after": None,
                "smell_removed": False,
            }

            write_json(run_dir / "status.json", status_data)
            write_json(run_dir / "metrics.json", metrics_data)

            return run_dir

        # ------------------------------------------------------------
        # 3. Run Designite after successful compilation
        # ------------------------------------------------------------
        designite_after_dir = (
            run_dir
            / f"smell_attempt_{smell_attempt_number}"
            / "designite_after"
        )

        final_designite_after_result = run_designite(
            repo_path=repo_path,
            jar_path=config["designite"]["jar_path"],
            output_dir=designite_after_dir,
            log_path=run_dir / f"designite_after_smell_attempt_{smell_attempt_number}.log",
            java_path=config["designite"].get("java_path"),
        )

        if not final_designite_after_result["ok"]:
            final_status = "designite_after_failed"

            status_data.update(
                {
                    "status": final_status,
                    "codex_success": True,
                    "codex_returncode": final_codex_result["returncode"],
                    "codex_wall_time_seconds": final_codex_result["seconds"],
                    "compile_success": True,
                    "maven_returncode": final_maven_result["returncode"],
                    "maven_wall_time_seconds": final_maven_result["seconds"],
                    "designite_after_returncode": final_designite_after_result["returncode"],
                    "designite_after_wall_time_seconds": final_designite_after_result["seconds"],
                    "attempts_count": len(all_attempts),
                    "final_smell_attempt": final_smell_attempt,
                    "final_compile_attempt": final_compile_attempt,
                    **final_diff_stats,
                }
            )

            metrics_data = {
                "project_name": config["project_name"],
                "smell": config["smell"],
                "target_type": config["target_type"],
                "target_name": config["target_name"],
                "initial_commit": initial_commit,
                "designite_before": designite_before_result,
                "smell_before": smell_before,
                "attempts": all_attempts,
                "codex": final_codex_result,
                "diff": final_diff_stats,
                "maven": final_maven_result,
                "designite_after": final_designite_after_result,
                "smell_after": None,
                "smell_removed": None,
            }

            write_json(run_dir / "status.json", status_data)
            write_json(run_dir / "metrics.json", metrics_data)

            return run_dir

        final_smell_after = find_target_smell(
            designite_output_dir=designite_after_dir,
            smell_name=config["smell_name"],
            smell_code=config["smell"],
            target_name=config["target_name"],
            csv_file=smell_detection_config.get("csv_file"),
            smell_column=smell_detection_config.get("smell_column"),
            package_column=smell_detection_config.get("package_column"),
            class_column=smell_detection_config.get("class_column"),
            target_column=smell_detection_config.get("target_column"),
        )

        smell_removed = smell_before["present"] and not final_smell_after["present"]

        if smell_removed:
            final_status = "success"
            break

        final_status = "smell_not_removed"

        if smell_attempt_number < max_smell_attempts:
            current_prompt = build_continue_smell_prompt(config)
    
    # ------------------------------------------------------------
    # 4. Run Refactoring Miner on final valid state
    # ------------------------------------------------------------
    refactoring_miner_result = None
    final_commit_info = None

    refactoring_miner_config = config.get("refactoring_miner", {})
    refactoring_miner_enabled = bool(refactoring_miner_config.get("enabled", False))

    if refactoring_miner_enabled and final_maven_result is not None and final_maven_result["ok"]:
        try:
            final_commit_info = create_temporary_final_commit(
                repo_path=repo_path,
                run_name=run_dir.name,
            )

            refactoring_miner_result = run_refactoring_miner(
                repo_path=repo_path,
                command=refactoring_miner_config["command"],
                start_commit=initial_commit,
                end_commit=final_commit_info["commit"],
                output_json_path=run_dir / "refactoring_miner.json",
                log_path=run_dir / "refactoring_miner.log",
            )
        except Exception as error:
            refactoring_miner_result = {
                "ok": False,
                "error": str(error),
            }

    # ------------------------------------------------------------
    # 5. Save final status and metrics
    # ------------------------------------------------------------
    if final_codex_result is None or final_diff_stats is None or final_maven_result is None:
        status_data.update(
            {
                "status": final_status,
                "codex_success": None,
                "compile_success": False,
                "smell_removed": False,
                "attempts_count": len(all_attempts),
                
                "final_commit": None
                if final_commit_info is None
                else final_commit_info["commit"],
                "temporary_final_commit_created": None
                if final_commit_info is None
                else final_commit_info["created"],
                "refactoring_miner_success": None
                if refactoring_miner_result is None
                else refactoring_miner_result.get("ok"),
                "refactorings_count": None
                if refactoring_miner_result is None
                else refactoring_miner_result.get("refactorings_count"),
                "refactoring_types": None
                if refactoring_miner_result is None
                else refactoring_miner_result.get("refactoring_types"),
            }
        )

        metrics_data = {
            "project_name": config["project_name"],
            "smell": config["smell"],
            "target_type": config["target_type"],
            "target_name": config["target_name"],
            "initial_commit": initial_commit,
            "designite_before": designite_before_result,
            "smell_before": smell_before,
            "attempts": all_attempts,
            "codex": final_codex_result,
            "diff": final_diff_stats,
            "maven": final_maven_result,
            "designite_after": final_designite_after_result,
            "smell_after": final_smell_after,
            "smell_removed": False,
            "final_commit": final_commit_info,
            "refactoring_miner": refactoring_miner_result,
        }

        write_json(run_dir / "status.json", status_data)
        write_json(run_dir / "metrics.json", metrics_data)

        return run_dir

    status_data.update(
        {
            "status": final_status,
            "codex_success": True,
            "codex_returncode": final_codex_result["returncode"],
            "codex_wall_time_seconds": final_codex_result["seconds"],
            "compile_success": final_maven_result["ok"],
            "maven_returncode": final_maven_result["returncode"],
            "maven_wall_time_seconds": final_maven_result["seconds"],
            "designite_after_returncode": None
            if final_designite_after_result is None
            else final_designite_after_result["returncode"],
            "designite_after_wall_time_seconds": None
            if final_designite_after_result is None
            else final_designite_after_result["seconds"],
            "smell_present_after": None
            if final_smell_after is None
            else final_smell_after["present"],
            "smell_matches_after": None
            if final_smell_after is None
            else final_smell_after["matches_count"],
            "smell_removed": smell_removed,
            "attempts_count": len(all_attempts),
            "final_smell_attempt": final_smell_attempt,
            "final_compile_attempt": final_compile_attempt,

            "final_commit": None
            if final_commit_info is None
            else final_commit_info["commit"],
            "temporary_final_commit_created": None
            if final_commit_info is None
            else final_commit_info["created"],
            "refactoring_miner_success": None
            if refactoring_miner_result is None
            else refactoring_miner_result.get("ok"),
            "refactorings_count": None
            if refactoring_miner_result is None
            else refactoring_miner_result.get("refactorings_count"),
            "refactoring_types": None
            if refactoring_miner_result is None
            else refactoring_miner_result.get("refactoring_types"),
            "refactoring_miner_error": None
            if refactoring_miner_result is None
            else refactoring_miner_result.get("error"),

            **final_diff_stats,
        }
    )

    metrics_data = {
        "project_name": config["project_name"],
        "smell": config["smell"],
        "target_type": config["target_type"],
        "target_name": config["target_name"],
        "initial_commit": initial_commit,
        "designite_before": designite_before_result,
        "smell_before": smell_before,
        "attempts": all_attempts,
        "codex": final_codex_result,
        "diff": final_diff_stats,
        "maven": final_maven_result,
        "designite_after": final_designite_after_result,
        "smell_after": final_smell_after,
        "smell_removed": smell_removed,
        "final_commit": final_commit_info,
        "refactoring_miner": refactoring_miner_result,
    }

    write_json(run_dir / "status.json", status_data)
    write_json(run_dir / "metrics.json", metrics_data)

    return run_dir