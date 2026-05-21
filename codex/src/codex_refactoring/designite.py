import csv
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


def run_designite(
    repo_path: str | Path,
    jar_path: str | Path,
    output_dir: str | Path,
    log_path: str | Path,
    java_path: str | Path | None = None,
) -> dict[str, Any]:
    repo = Path(repo_path).resolve()
    jar = Path(jar_path).resolve()
    output = Path(output_dir).resolve()
    log_file = Path(log_path)

    if not repo.exists():
        raise FileNotFoundError(f"Repository path not found: {repo}")

    if not jar.exists():
        raise FileNotFoundError(f"Designite jar not found: {jar}")

    if java_path is None:
        java_executable = "java"
    else:
        java_executable = str(Path(java_path).resolve())

        if not Path(java_executable).exists():
            raise FileNotFoundError(f"Java executable not found: {java_executable}")

    if output.exists():
        shutil.rmtree(output)

    output.mkdir(parents=True, exist_ok=True)

    command = [
        java_executable,
        "-jar",
        str(jar),
        "-g",
        "-i",
        str(repo),
        "-o",
        str(output),
    ]

    start = time.time()

    with log_file.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=repo,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    elapsed = round(time.time() - start, 2)

    return {
        "command": command,
        "returncode": result.returncode,
        "seconds": elapsed,
        "ok": result.returncode == 0,
        "output_dir": str(output),
        "log_path": str(log_file),
    }


def normalize(value: str) -> str:
    return value.strip().lower()


def row_contains_target_smell(
    row: dict[str, str],
    smell_name: str,
    smell_code: str,
    target_name: str,
) -> bool:
    values = [str(value) for value in row.values() if value is not None]

    joined = " | ".join(values).lower()

    smell_name_norm = normalize(smell_name)
    smell_code_norm = normalize(smell_code)
    target_norm = normalize(target_name)

    has_target = target_norm in joined

    has_smell = (
        smell_name_norm in joined
        or smell_code_norm in joined
    )

    return has_target and has_smell


def find_target_smell(
    designite_output_dir: str | Path,
    smell_name: str,
    smell_code: str,
    target_name: str,
) -> dict[str, Any]:
    output = Path(designite_output_dir)

    if not output.exists():
        raise FileNotFoundError(f"Designite output directory not found: {output}")

    matches: list[dict[str, Any]] = []

    csv_files = sorted(output.rglob("*.csv"))

    for csv_file in csv_files:
        with csv_file.open("r", encoding="utf-8", errors="replace", newline="") as file:
            reader = csv.DictReader(file)

            for row_index, row in enumerate(reader, start=2):
                if row_contains_target_smell(
                    row=row,
                    smell_name=smell_name,
                    smell_code=smell_code,
                    target_name=target_name,
                ):
                    matches.append(
                        {
                            "file": str(csv_file),
                            "row_index": row_index,
                            "row": row,
                        }
                    )

    return {
        "present": len(matches) > 0,
        "matches_count": len(matches),
        "matches": matches,
    }