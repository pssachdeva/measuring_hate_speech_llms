import argparse
from pathlib import Path
import sys

from loguru import logger

from mhs_llms.async_jobs import process_async
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
        description="Normalize saved async query responses for each configured model."
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        default=str(REPO_ROOT / "configs" / "queries" / "reference_multi_model_example.yaml"),
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        help="Optional CSV or JSONL file to rebuild from all processed async model annotations.",
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
    outputs = process_async(
        config_path=config_path,
        include_all_cols=args.all_cols,
        output_path=Path(args.output_path).resolve() if args.output_path else None,
    )
    for output in outputs.outputs:
        _print_summary(
            f"Async Process: {output.model_id}",
            [
                ("Config", str(config_path)),
                ("Run Dir", str(output.run_dir)),
                ("Metadata", str(output.metadata_path)),
                ("Processed JSONL", str(output.processed_records_path)),
                ("Processed CSV", str(output.processed_csv_path)),
                ("Processing Errors", str(output.processing_errors_path)),
                ("Completed", str(output.completed_count)),
                ("Total", str(output.total_requests)),
                ("Processing Errors Count", str(output.processing_error_count)),
                ("Is Complete", str(output.is_complete)),
            ],
        )

    summary_rows = [("All Complete", str(outputs.all_complete))]
    if outputs.combined_output_path is not None:
        summary_rows.append(("Combined Output", str(outputs.combined_output_path)))
    _print_summary("Async Process Summary", summary_rows)


if __name__ == "__main__":
    main()
