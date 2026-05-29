from pathlib import Path

from mvp.source_refactor.lib.subprocess_utils import run_command


def git_current_commit(repo_path: str | Path) -> str:
    code, out = run_command(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_path,
    )

    if code != 0:
        raise RuntimeError(f"Could not get current git commit:\n{out}")

    return out.strip()


def git_status_porcelain(repo_path: str | Path) -> str:
    code, out = run_command(
        ["git", "status", "--porcelain"],
        cwd=repo_path,
    )

    if code != 0:
        raise RuntimeError(f"Could not get git status:\n{out}")

    return out.strip()


def ensure_clean_git_workspace(repo_path: str | Path) -> None:
    status = git_status_porcelain(repo_path)

    if status:
        raise RuntimeError(
            "Repository has uncommitted changes. "
            "Please commit, stash, or reset before running source_refactor.\n\n"
            f"Git status:\n{status}"
        )


def git_commit_all(repo_path: str | Path, message: str) -> str:
    code, out = run_command(
        ["git", "add", "-A"],
        cwd=repo_path,
    )

    if code != 0:
        raise RuntimeError(f"Could not git add files:\n{out}")

    status = git_status_porcelain(repo_path)

    if not status:
        return git_current_commit(repo_path)

    code, out = run_command(
        ["git", "commit", "-m", message],
        cwd=repo_path,
    )

    if code != 0:
        raise RuntimeError(f"Could not create git commit:\n{out}")

    return git_current_commit(repo_path)


def git_clean_workspace(repo_path: str | Path) -> None:
    code, out = run_command(
        ["git", "clean", "-fd"],
        cwd=repo_path,
    )

    if code != 0:
        raise RuntimeError(f"Could not clean repository:\n{out}")


def git_reset_hard(repo_path: str | Path, commit: str) -> str:
    code, out = run_command(
        ["git", "reset", "--hard", commit],
        cwd=repo_path,
    )

    if code != 0:
        raise RuntimeError(f"Could not reset repository to {commit}:\n{out}")

    return out