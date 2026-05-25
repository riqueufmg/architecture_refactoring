import csv
import subprocess
import sys
import time
from pathlib import Path

import yaml


BASE_CONFIG = Path("configs/experiment_im.yaml")
CASES_FILE = Path("data/im_cases.csv")
GENERATED_CONFIG_DIR = Path("runs/generated_im_configs")
BATCH_RESULTS = Path("runs/im_batch_results.csv")


def run_command(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )


def git(repo_path: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return run_command(["git", *args], cwd=repo_path)


def get_head(repo_path: Path) -> str:
    result = git(repo_path, ["rev-parse", "HEAD"])

    if result.returncode != 0:
        raise RuntimeError(
            f"Could not get HEAD for {repo_path}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    return result.stdout.strip()


def reset_repo(repo_path: Path, commit: str) -> None:
    result = git(repo_path, ["reset", "--hard", commit])
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not reset {repo_path}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    result = git(repo_path, ["clean", "-fd"])
    if result.returncode != 0:
        raise RuntimeError(
            f"Could not clean {repo_path}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )


def safe_name(value: str) -> str:
    return (
        value.replace("/", "_")
        .replace("\\", "_")
        .replace(":", "_")
        .replace(" ", "_")
        .replace(".", "_")
    )


def load_cases() -> list[dict[str, str]]:
    with CASES_FILE.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def create_case_config(case: dict[str, str], index: int) -> Path:
    with BASE_CONFIG.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config["project_name"] = case["project_name"]
    config["repo_path"] = case["repo_path"]

    config["smell"] = "IM"
    config["smell_name"] = "Insufficient Modularization"
    config["target_type"] = "class"
    config["target_name"] = case["target_name"]

    config["smell_detection"] = {
        "csv_file": "DesignSmells.csv",
        "smell_column": "Smell",
        "package_column": "Package",
        "class_column": "Class",
    }

    GENERATED_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    filename = f"{index:03d}_{safe_name(case['project_name'])}_{safe_name(case['target_name'])}.yaml"
    output_path = GENERATED_CONFIG_DIR / filename

    with output_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)

    return output_path


def append_result(row: dict[str, str | int | float]) -> None:
    BATCH_RESULTS.parent.mkdir(parents=True, exist_ok=True)

    file_exists = BATCH_RESULTS.exists()

    fieldnames = [
        "index",
        "project_name",
        "repo_path",
        "target_name",
        "config_path",
        "returncode",
        "seconds",
        "status",
    ]

    with BATCH_RESULTS.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def main() -> int:
    if not BASE_CONFIG.exists():
        print(f"Base config not found: {BASE_CONFIG}", file=sys.stderr)
        return 1

    if not CASES_FILE.exists():
        print(f"Cases file not found: {CASES_FILE}", file=sys.stderr)
        return 1

    cases = load_cases()

    if not cases:
        print("No cases found.", file=sys.stderr)
        return 1

    base_commits: dict[str, str] = {}

    for case in cases:
        repo_path = Path(case["repo_path"]).resolve()

        if str(repo_path) not in base_commits:
            base_commits[str(repo_path)] = get_head(repo_path)

    for index, case in enumerate(cases, start=1):
        repo_path = Path(case["repo_path"]).resolve()
        base_commit = base_commits[str(repo_path)]

        print("=" * 80)
        print(f"[{index}/{len(cases)}] {case['project_name']} :: {case['target_name']}")
        print(f"Resetting repository to {base_commit}")
        reset_repo(repo_path, base_commit)

        config_path = create_case_config(case, index)

        command = [
            sys.executable,
            "-m",
            "codex_refactoring.main",
            "--config",
            str(config_path),
        ]

        print("Running:")
        print(" ".join(command))

        start = time.time()

        result = subprocess.run(
            command,
            text=True,
            check=False,
        )

        elapsed = round(time.time() - start, 2)

        status = "ok" if result.returncode == 0 else "failed"

        append_result(
            {
                "index": index,
                "project_name": case["project_name"],
                "repo_path": str(repo_path),
                "target_name": case["target_name"],
                "config_path": str(config_path),
                "returncode": result.returncode,
                "seconds": elapsed,
                "status": status,
            }
        )

        print(f"Finished with status={status}, returncode={result.returncode}, seconds={elapsed}")

    print("=" * 80)
    print(f"Batch finished. Results: {BATCH_RESULTS}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())