from pathlib import Path
import os
import subprocess

from util.subprocess_utils import _run

def _run_build(
    repo_path: Path,
    tmp_dir: Path,
    command: list[str] | None = None,
) -> subprocess.CompletedProcess:
    cmd = command
    env = os.environ.copy()
    env["MAVEN_OPTS"] = f"-Xshare:off -Djava.io.tmpdir={tmp_dir}"

    return _run(cmd, cwd=repo_path, env=env)