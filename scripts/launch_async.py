import argparse
from pathlib import Path
import sys

from loguru import logger

from mhs_llms.async_jobs import launch_async
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
        description="Run one sequential async query per comment for each configured model."
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        default=str(REPO_ROOT / "configs" / "queries" / "reference_multi_model_example.yaml"),
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    config_path = Path(args.config_path).resolve()
    outputs = launch_async(config_path=config_path)
    for output in outputs.outputs:
        _print_summary(
            f"Async Launch: {output.model_id}",
            [
                ("Config", str(config_path)),
                ("Run Dir", str(output.run_dir)),
                ("Metadata", str(output.metadata_path)),
                ("Completed", str(output.completed_count)),
                ("Skipped Existing", str(output.skipped_existing_count)),
                ("Errors", str(output.error_count)),
                ("Total", str(output.total_requests)),
            ],
        )
    _print_summary("Async Launch Summary", [("All Complete", str(outputs.all_complete))])


if __name__ == "__main__":
    main()
