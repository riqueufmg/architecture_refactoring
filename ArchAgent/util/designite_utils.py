import csv
import shutil
from pathlib import Path
from typing import Tuple

from util.Dependencies import Dependencies
from util.subprocess_utils import _run


def _run_designite(
    repo_path: Path,
    out_dir: Path,
    jar_path: Path,
) -> Tuple[Path, list[str]]:

    if out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    java22 = "/usr/lib/jvm/jdk-22.0.2-oracle-x64/bin/java"

    cmd = [
        java22,
        "-jar",
        str(jar_path),
        "-g",
        "-i",
        str(repo_path),
        "-o",
        str(out_dir),
    ]

    p = _run(cmd, cwd=repo_path)

    log = (p.stdout or "") + ("\n" if p.stdout else "") + (p.stderr or "")
    (out_dir / "designite.log").write_text(log, encoding="utf-8")

    if p.returncode != 0:
        raise RuntimeError(
            f"Designite failed (rc={p.returncode}). See log at {out_dir / 'designite.log'}"
        )

    return out_dir, cmd


def _designite_smell_present(
    designite_dir: Path,
    target_name: str,
    smell_name: str,
    csv_name: str = "DesignSmells.csv",
    target_type: str = "class",
) -> bool:
    csv_path = designite_dir / csv_name
    if not csv_path.exists():
        return False

    target = (target_name or "").strip()
    smell = (smell_name or "").strip()

    if not target or not smell:
        return False

    with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            row_smell = (row.get("Smell") or "").strip()
            if row_smell != smell:
                continue

            pkg = (row.get("Package") or "").strip()
            cls = (row.get("Class") or "").strip()

            if target_type == "package":
                if pkg == target:
                    return True

                component = (
                    row.get("Component")
                    or row.get("Package Name")
                    or row.get("Element")
                    or ""
                ).strip()

                if component == target:
                    return True

            else:
                if pkg and cls:
                    row_fqn = f"{pkg}.{cls}"
                    if row_fqn == target:
                        return True

    return False


def get_package_dependencies(graphml_path: str, target_name: str):
    deps = Dependencies(target_name)
    return deps._get_package_dependencies(graphml_path)