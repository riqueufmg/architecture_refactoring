import argparse

from codex_refactoring.config import load_config
from codex_refactoring.runner import run_experiment

# function to load system arguments
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

# main function of the system
def main() -> None:

    # load arguments
    args = parse_args()

    # load experiment configuration file
    config = load_config(args.config)

    # run the experiment based on the configuration file
    run_dir = run_experiment(config)

    # feedback message
    print("Experiment initialized successfully.")
    print(f"Run directory: {run_dir}")

if __name__ == "__main__":
    main()