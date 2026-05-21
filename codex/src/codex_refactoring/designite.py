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


def row_matches_target_smell(
    row: dict[str, str],
    smell_name: str,
    smell_code: str,
    target_name: str,
    smell_column: str | None = None,
    package_column: str | None = None,
    class_column: str | None = None,
    target_column: str | None = None,
) -> bool:
    smell_name_norm = normalize(smell_name)
    smell_code_norm = normalize(smell_code)
    target_norm = normalize(target_name)

    normalized_row = {
        normalize(str(key)): normalize(str(value))
        for key, value in row.items()
        if value is not None
    }

    # ------------------------------------------------------------
    # 1. Explicit smell check
    # ------------------------------------------------------------
    if smell_column is not None:
        smell_key = normalize(smell_column)
        smell_value = normalized_row.get(smell_key, "")

        has_smell = (
            smell_value == smell_name_norm
            or smell_value == smell_code_norm
            or smell_name_norm in smell_value
        )
    else:
        values_joined = " | ".join(normalized_row.values())
        has_smell = (
            smell_name_norm in values_joined
            or smell_code_norm in values_joined
        )

    if not has_smell:
        return False

    # ------------------------------------------------------------
    # 2. Class-level smells: Package + Class
    # Example DesignSmells.csv:
    # Project, Package, Class, Smell, Description, File
    # ------------------------------------------------------------
    if package_column is not None and class_column is not None:
        package_key = normalize(package_column)
        class_key = normalize(class_column)

        package_value = normalized_row.get(package_key, "")
        class_value = normalized_row.get(class_key, "")

        full_class_name = f"{package_value}.{class_value}"

        return (
            full_class_name == target_norm
            or class_value == target_norm
        )

    # ------------------------------------------------------------
    # 3. Package-level smells: Package
    # Example ArchitectureSmells.csv:
    # Project, Package, Smell, Description
    # ------------------------------------------------------------
    if package_column is not None:
        package_key = normalize(package_column)
        package_value = normalized_row.get(package_key, "")

        return package_value == target_norm

    # ------------------------------------------------------------
    # 4. Generic explicit target column
    # ------------------------------------------------------------
    if target_column is not None:
        target_key = normalize(target_column)
        target_value = normalized_row.get(target_key, "")

        return target_value == target_norm

    # ------------------------------------------------------------
    # 5. Backward-compatible fallback
    # ------------------------------------------------------------
    return row_contains_target_smell(
        row=row,
        smell_name=smell_name,
        smell_code=smell_code,
        target_name=target_name,
    )


def find_target_smell(
    designite_output_dir: str | Path,
    smell_name: str,
    smell_code: str,
    target_name: str,
    csv_file: str | None = None,
    smell_column: str | None = None,
    package_column: str | None = None,
    class_column: str | None = None,
    target_column: str | None = None,
) -> dict[str, Any]:
    output = Path(designite_output_dir)

    if not output.exists():
        raise FileNotFoundError(f"Designite output directory not found: {output}")

    matches: list[dict[str, Any]] = []

    if csv_file is not None:
        csv_files = [output / csv_file]
    else:
        csv_files = sorted(output.rglob("*.csv"))

    searched_files: list[str] = []

    for csv_path in csv_files:
        searched_files.append(str(csv_path))

        if not csv_path.exists():
            continue

        with csv_path.open("r", encoding="utf-8", errors="replace", newline="") as file:
            reader = csv.DictReader(file)

            for row_index, row in enumerate(reader, start=2):
                if row_matches_target_smell(
                    row=row,
                    smell_name=smell_name,
                    smell_code=smell_code,
                    target_name=target_name,
                    smell_column=smell_column,
                    package_column=package_column,
                    class_column=class_column,
                    target_column=target_column,
                ):
                    matches.append(
                        {
                            "file": str(csv_path),
                            "row_index": row_index,
                            "row": row,
                        }
                    )

    return {
        "present": len(matches) > 0,
        "matches_count": len(matches),
        "matches": matches,
        "searched_files": searched_files,
    }