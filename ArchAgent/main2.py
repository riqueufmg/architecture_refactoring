import os
import dotenv
import argparse

from pathlib import Path

if __name__ == "__main__":
    dotenv.load_dotenv()
    prefix = os.getenv("REPO_PATH")

    # 1. parse command line arguments for smell type
    parser = argparse.ArgumentParser()
    parser.add_argument("--smell", required=True, help="Type of smell (GC, HM or IM)")
    args = parser.parse_args()
    smell = args.smell
    if smell not in {"GC", "HM", "IM"}:
        raise ValueError("Invalid smell type. Must be 'GC', 'HM' or 'IM'.")

    # 2. load smell planner prompt
    with open(f"data/prompts/planner_{smell}.prompt", "r", encoding="utf-8") as f:
        PROMPT_TEMPLATE = f.read()

    # 3. load and iterate instances to be refactored
    file = open(f"data/dataset/{smell.lower()}_smells.txt", "r", encoding="utf-8")
    for line in file:

        # 3.1 load target class or path
        PATH = line.strip()
        PROJECT_PATH, TARGET_FILE = PATH.removeprefix(prefix).split("/", 1)
        repo_path = Path(prefix+PROJECT_PATH).resolve()
        print(repo_path, TARGET_FILE)

        target_rel, target_code = _read_target_file(repo_path, TARGET_FILE)

        if smell == "IM":
            planner_input = {
                "smell": "Insufficient Modularization",
                "target_file": target_rel,
                "target_code": target_code,
            }
        elif smell == "HM":
            observed_external_calls = extract_observed_external_calls(target_code)
            planner_input = {
                "smell": "Hub-like Modularization", # smell type
                "target_file": target_rel, # path to the target class
                "target_code": target_code, # raw code of target class
                "observed_external_calls": observed_external_calls, # list of external calls
            }
        elif smell == "GC":
            planner_input = {
                "smell": "God Component", # smell type
                "target_file": target_rel, # path to the target class
                "target_code": target_code, # raw code of target class
            }
        else:
            raise ValueError("Invalid smell type")

        out = app.invoke(
            {
                "repo_path": str(repo_path),
                "target_file": target_rel,
                "planner_prompt": PROMPT_TEMPLATE,
                "planner_input_json": json.dumps(planner_input, indent=2)
            }
        )

        open(f"{repo_path}/state.json", "w", encoding="utf-8").write(json.dumps(out, indent=2))