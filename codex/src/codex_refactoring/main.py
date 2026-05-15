import argparse
import json

from codex_refactoring.config import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Codex baseline for automatic Java smell refactoring."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment YAML config."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    print("Config loaded successfully.")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
