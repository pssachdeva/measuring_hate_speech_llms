import argparse
from pathlib import Path
import sys

from loguru import logger

from mhs_llms.batch import launch_batch
from mhs_llms.paths import REPO_ROOT


def _print_summary(title: str, rows: list[tuple[str, str]]) -> None:
    """Print a compact, aligned CLI summary block."""

    label_width = max(len(label) for label, _ in rows)
    print()
    print(title)
    for label, value in rows:
        print(f"  {label:<{label_width}} : {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a provider batch job for MHS comments.")
    parser.add_argument(
        "config_path",
        nargs="?",
        default=str(REPO_ROOT / "configs" / "queries" / "reference_oai_gpt54_low.yaml"),
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    outputs = launch_batch(config_path=Path(args.config_path))
    _print_summary(
        "Batch Launched",
        [
            ("Config", str(Path(args.config_path).resolve())),
            ("Run Dir", str(outputs.run_dir)),
            ("Metadata", str(outputs.batch_metadata_path)),
            ("Batch ID", outputs.batch_id),
            ("Status", outputs.status),
        ],
    )


if __name__ == "__main__":
    main()
