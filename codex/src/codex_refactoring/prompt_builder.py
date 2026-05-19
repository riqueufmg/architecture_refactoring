from pathlib import Path
from typing import Any

# Load prompt template
def read_text(path: str | Path) -> str:
    file_path = Path(path)

    if not file_path.exists():
        raise FileNotFoundError(f"Prompt template not found: {file_path}")

    return file_path.read_text(encoding="utf-8")

# use configuration to complete the prompt
def render_template(template: str, values: dict[str, str]) -> str:
    rendered = template

    for key, value in values.items():
        rendered = rendered.replace("{{ " + key + " }}", value)

    return rendered

# return smell prompt path
def get_smell_prompt_path(smell: str, prompts_dir: str | Path = "prompts") -> Path:
    smell_normalized = smell.lower()

    mapping = {
        "gc": "codex_refactor_gc.md",
        "im": "codex_refactor_im.md",
        "hm": "codex_refactor_hm.md",
    }

    if smell_normalized not in mapping:
        raise ValueError(f"Unsupported smell for prompt generation: {smell}")

    return Path(prompts_dir) / mapping[smell_normalized]

# format maven command list to string
def format_command(command: list[str]) -> str:
    return " ".join(command)


# replace prompt placeholders with configuration values
def build_prompt(config: dict[str, Any], prompts_dir: str | Path = "prompts") -> str:
    base_template_path = Path(prompts_dir) / "codex_refactor_base.md"
    smell_template_path = get_smell_prompt_path(config["smell"], prompts_dir)

    base_template = read_text(base_template_path)
    smell_guidance = read_text(smell_template_path)

    values = {
        "smell_name": str(config["smell_name"]),
        "smell": str(config["smell"]),
        "target_type": str(config["target_type"]),
        "target_name": str(config["target_name"]),
        "maven_command": format_command(config["maven_command"]),
        "smell_specific_guidance": smell_guidance,
    }

    return render_template(base_template, values)

# save prompt in run directory
def save_prompt(prompt: str, output_path: str | Path) -> None:
    Path(output_path).write_text(prompt, encoding="utf-8")