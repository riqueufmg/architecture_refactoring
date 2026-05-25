import os
import subprocess
from pathlib import Path

from util.subprocess_utils import _run

def _run_build(repo_path: Path, tmp_dir: Path) -> subprocess.CompletedProcess:
    cmd = [
        "mvn", "-q",
        "-Dmaven.test.skip=true",
        "-Djapicmp.skip=true",
        "-Drat.skip=true",
        "-Dcheckstyle.skip=true",
        "-Dspotbugs.skip=true",
        "-Dpmd.skip=true",
        "-DskipITs",
        "clean", "verify",
    ]

    env = os.environ.copy()
    env["MAVEN_OPTS"] = f"-Xshare:off -Djava.io.tmpdir={tmp_dir}"

    return _run(cmd, cwd=repo_path, env=env)