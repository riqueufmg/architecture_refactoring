import os
import dotenv
from pathlib import Path

class Prompt:

    def __init__(self):
        self.templates_path = os.getenv("TEMPLATES_PATH")

    def load_custom_prompt(self, type: str, input: str) -> str:
        if type == "agent_planner":
            prompt_path = Path(self.templates_path, "agent_planner.prompt")
        
        with open(prompt_path, "r") as file:
            return file.read().replace("{input}", input)