import argparse
from pathlib import Path
import sys

from loguru import logger

from mhs_llms.batch import process_batch, write_processed_annotations
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
        description="Refresh a provider batch job and process results when complete."
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        default=str(REPO_ROOT / "configs" / "exp1_openai.yaml"),
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        help="Optional CSV or JSONL file to create or append processed annotations to.",
    )
    parser.add_argument(
        "--all-cols",
        action="store_true",
        help="Include optional columns such as metadata in the processed outputs.",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    outputs = process_batch(config_path=Path(args.config_path), include_all_cols=args.all_cols)
    rows = [
        ("Config", str(Path(args.config_path).resolve())),
        ("Run Dir", str(outputs.run_dir)),
        ("Metadata", str(outputs.batch_metadata_path)),
        ("Status", outputs.status),
    ]
    if outputs.raw_results_path is not None:
        rows.append(("Raw Results", str(outputs.raw_results_path)))
    if outputs.processed_csv_path is not None:
        rows.append(("Processed CSV", str(outputs.processed_csv_path)))
    if args.output_path and outputs.processed_records_path and outputs.processed_csv_path:
        output_path = write_processed_annotations(
            processed_records_path=outputs.processed_records_path,
            processed_csv_path=outputs.processed_csv_path,
            output_path=Path(args.output_path).resolve(),
        )
        rows.append(("Output", str(output_path)))
    _print_summary("Batch Status", rows)


if __name__ == "__main__":
    main()
