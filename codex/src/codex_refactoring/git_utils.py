import subprocess
from pathlib import Path

# function to run a generic git command and return the completed process
def run_git_command(repo_path: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=False,
    )

# function to check if a given path is a git repository and return the resolved path
def ensure_git_repository(repo_path: str | Path) -> Path:
    path = Path(repo_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Repository path does not exist: {path}")

    result = run_git_command(path, ["rev-parse", "--is-inside-work-tree"])

    if result.returncode != 0 or result.stdout.strip() != "true":
        raise ValueError(f"Path is not inside a Git repository: {path}")

    return path

# return the current commit
def get_current_commit(repo_path: str | Path) -> str:
    path = Path(repo_path).resolve()

    result = run_git_command(path, ["rev-parse", "HEAD"])

    if result.returncode != 0:
        raise RuntimeError(
            "Could not get current Git commit.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    return result.stdout.strip()

# return the git status in porcelain format
def get_git_status(repo_path: str | Path) -> str:
    path = Path(repo_path).resolve()

    result = run_git_command(path, ["status", "--porcelain"])

    if result.returncode != 0:
        raise RuntimeError(
            "Could not get Git status.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    return result.stdout

# check if the git repository hasn't uncommitted
def is_repo_clean(repo_path: str | Path) -> bool:
    return get_git_status(repo_path).strip() == ""

# raise an error if the git repository has uncommitted changes
def ensure_clean_repo(repo_path: str | Path) -> None:
    status = get_git_status(repo_path)

    if status.strip():
        raise RuntimeError(
            "Target repository is not clean. Commit, stash, or discard changes before running the experiment.\n\n"
            f"Git status:\n{status}"
        )

# save the git diff to a file in a format that can be applied with git apply
def save_git_diff(repo_path: str | Path, output_path: str | Path) -> None:
    path = Path(repo_path).resolve()

    result = run_git_command(path, ["diff", "--binary"])

    if result.returncode != 0:
        raise RuntimeError(
            "Could not save Git diff.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    Path(output_path).write_text(result.stdout, encoding="utf-8")

# get the number of changed files, lines added, and lines deleted in the git diff
def get_diff_stats(repo_path: str | Path) -> dict[str, int]:
    path = Path(repo_path).resolve()

    result = run_git_command(path, ["diff", "--numstat"])

    if result.returncode != 0:
        raise RuntimeError(
            "Could not get Git diff stats.\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    changed_files = 0
    lines_added = 0
    lines_deleted = 0

    for line in result.stdout.splitlines():
        parts = line.split("\t")

        if len(parts) < 3:
            continue

        added, deleted, _file_path = parts[0], parts[1], parts[2]

        changed_files += 1

        if added.isdigit():
            lines_added += int(added)

        if deleted.isdigit():
            lines_deleted += int(deleted)

    return {
        "changed_files_count": changed_files,
        "lines_added": lines_added,
        "lines_deleted": lines_deleted,
    }