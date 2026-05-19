import argparse

from codex_refactoring.config import load_config
from codex_refactoring.runner import run_experiment

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Codex for automatic Java smell refactoring."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML config.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    run_dir = run_experiment(config)

    print("Experiment initialized successfully.")
    print(f"Run directory: {run_dir}")

if __name__ == "__main__":
    main()