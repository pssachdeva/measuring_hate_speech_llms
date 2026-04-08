import argparse
from pathlib import Path
import sys

from loguru import logger

from mhs_llms.batch import process_batches
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
        description="Refresh configured provider batch jobs and process results when complete."
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        default=str(REPO_ROOT / "configs" / "queries" / "reference_oai_gpt54_low.yaml"),
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        help="Optional CSV or JSONL file to rebuild from all processed model annotations.",
    )
    parser.add_argument(
        "--all-cols",
        action="store_true",
        help="Include optional columns such as metadata in the processed outputs.",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    config_path = Path(args.config_path).resolve()
    outputs = process_batches(
        config_path=config_path,
        include_all_cols=args.all_cols,
        output_path=Path(args.output_path).resolve() if args.output_path else None,
    )
    for output in outputs.outputs:
        rows = [
            ("Config", str(config_path)),
            ("Run Dir", str(output.run_dir)),
            ("Metadata", str(output.batch_metadata_path)),
            ("Status", output.status),
        ]
        if output.raw_results_path is not None:
            rows.append(("Raw Results", str(output.raw_results_path)))
        if output.processed_csv_path is not None:
            rows.append(("Processed CSV", str(output.processed_csv_path)))
        _print_summary(f"Batch Status: {output.model_id}", rows)

    summary_rows = [
        ("All Terminal", str(outputs.all_terminal)),
        ("All Successful", str(outputs.all_successful)),
    ]
    if outputs.combined_output_path is not None:
        summary_rows.append(("Combined Output", str(outputs.combined_output_path)))
    _print_summary("Combined Output", summary_rows)


if __name__ == "__main__":
    main()
