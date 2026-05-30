from pathlib import Path

def read_text_file(path: str | Path) -> str:
    p = Path(path).resolve()

    with p.open("r", encoding="utf-8") as f:
        return f.read()


def write_text_file(path: str | Path, content: str) -> None:
    p = Path(path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)

    with p.open("w", encoding="utf-8") as f:
        f.write(content)


def delete_file(path: str | Path) -> None:
    p = Path(path).resolve()

    if p.exists() and p.is_file():
        p.unlink()


def ensure_path_inside_repo(repo_path: str | Path, relative_path: str) -> Path:
    repo = Path(repo_path).resolve()
    target = (repo / relative_path).resolve()

    try:
        target.relative_to(repo)
    except ValueError:
        raise ValueError(f"Path escapes repository: {relative_path}")

    return target


def load_files_context(repo_path: str | Path, files: list[str]) -> list[dict[str, str]]:
    context: list[dict[str, str]] = []

    for file_path in files:
        abs_path = ensure_path_inside_repo(repo_path, file_path)

        if not abs_path.exists():
            context.append(
                {
                    "path": file_path,
                    "exists": "false",
                    "content": "",
                }
            )
            continue

        if not abs_path.is_file():
            raise ValueError(f"Expected file but found directory: {file_path}")

        context.append(
            {
                "path": file_path,
                "exists": "true",
                "content": read_text_file(abs_path),
            }
        )

    return context