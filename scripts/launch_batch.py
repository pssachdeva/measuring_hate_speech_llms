import argparse
from pathlib import Path
import sys

from loguru import logger

from mhs_llms.batch import launch_batches
from mhs_llms.paths import REPO_ROOT


def _print_summary(title: str, rows: list[tuple[str, str]]) -> None:
    """Print a compact, aligned CLI summary block."""

    label_width = max(len(label) for label, _ in rows)
    print()
    print(title)
    for label, value in rows:
        print(f"  {label:<{label_width}} : {value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch one provider batch job per configured model for MHS comments."
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        default=str(REPO_ROOT / "configs" / "queries" / "reference_oai_gpt54_low.yaml"),
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    config_path = Path(args.config_path).resolve()
    outputs = launch_batches(config_path=config_path)
    for output in outputs:
        _print_summary(
            f"Batch Launched: {output.model_id}",
            [
                ("Config", str(config_path)),
                ("Run Dir", str(output.run_dir)),
                ("Metadata", str(output.batch_metadata_path)),
                ("Batch ID", output.batch_id),
                ("Status", output.status),
            ],
        )


if __name__ == "__main__":
    main()
