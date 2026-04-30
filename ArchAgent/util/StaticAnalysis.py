from pathlib import Path
from typing import Tuple

import shutil

class StaticAnalysisNode():
    
    def __init__(self, name: str, input_keys: list[str], output_keys: list[str]):
        super().__init__(name, input_keys, output_keys)
    
    def _run_designite(
        repo_path: Path,
        out_dir: Path,
        jar_path: Path,
    ) -> Tuple[Path, list[str]]:

        if out_dir.exists():
            shutil.rmtree(out_dir)

        out_dir.mkdir(parents=True, exist_ok=True)

        java22 = "/usr/lib/jvm/jdk-22.0.2-oracle-x64/bin/java"

        #cmd = [
        #    "java", "-jar", str(jar_path),
        #    "-i", str(repo_path),
        #    "-o", str(out_dir),
        #]

        cmd = [java22, "-jar", str(jar_path), "-g", "-i", str(repo_path), "-o", str(out_dir)]

        p = _run(cmd, cwd=repo_path)

        log = (p.stdout or "") + ("\n" if p.stdout else "") + (p.stderr or "")
        (out_dir / "designite.log").write_text(log, encoding="utf-8")

        if p.returncode != 0:
            raise RuntimeError(
                f"Designite failed (rc={p.returncode}). See log at {out_dir / 'designite.log'}"
            )

        return out_dir, cmd