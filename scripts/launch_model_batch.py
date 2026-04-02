import argparse
from pathlib import Path
import sys

from loguru import logger

from mhs_llms.model_batch import launch_model_batch
from mhs_llms.paths import REPO_ROOT


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a provider batch job for MHS comments.")
    parser.add_argument(
        "config_path",
        nargs="?",
        default=str(REPO_ROOT / "configs" / "model_batch_openai_example.yaml"),
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    outputs = launch_model_batch(config_path=Path(args.config_path))
    print(f"run_dir={outputs.run_dir}")
    print(f"batch_metadata={outputs.batch_metadata_path}")
    print(f"batch_id={outputs.batch_id}")
    print(f"status={outputs.status}")


if __name__ == "__main__":
    main()
