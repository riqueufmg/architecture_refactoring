from pathlib import Path

from util.subprocess_utils import _run, _tail


def _git_current_commit(repo_path: Path) -> str:
    p = _run(["git", "rev-parse", "HEAD"], cwd=repo_path)

    if p.returncode != 0:
        raise RuntimeError("git rev-parse HEAD failed:\n" + _tail(p.stderr))

    return p.stdout.strip()