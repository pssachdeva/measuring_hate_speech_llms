"""Small importable wrapper for directly retrying errored batch items."""

import argparse
from pathlib import Path
import sys
from typing import Sequence

from loguru import logger

from mhs_llms.retry_direct import DirectRetryOutputs, retry_errored_requests


def retry_direct_errors(
    config_path: str | Path,
    *,
    model_ids: Sequence[str] | None = None,
    max_tokens: int | None = None,
    budget_tokens: int | None = None,
    effort: str | None = None,
    retry_root: str | Path | None = None,
    include_all_cols: bool = False,
    concurrency: int = 1,
) -> DirectRetryOutputs:
    """Retry failed rows directly against the provider using one config plus optional overrides."""

    return retry_errored_requests(
        Path(config_path).resolve(),
        model_ids=model_ids,
        max_tokens=max_tokens,
        budget_tokens=budget_tokens,
        effort=effort,
        retry_root=Path(retry_root).resolve() if retry_root is not None else None,
        include_all_cols=include_all_cols,
        concurrency=concurrency,
    )


def main() -> None:
    """Run direct retries from the command line."""

    parser = argparse.ArgumentParser(
        description="Directly retry rows currently listed in processing_errors.jsonl."
    )
    parser.add_argument("config_path", help="Batch config whose errors should be retried.")
    parser.add_argument(
        "--model-id",
        action="append",
        dest="model_ids",
        help="Restrict retries to one model id. Can be passed more than once.",
    )
    parser.add_argument("--max-tokens", type=int, help="Override max tokens for retry requests.")
    parser.add_argument("--budget-tokens", type=int, help="Override reasoning budget tokens.")
    parser.add_argument("--effort", help="Override reasoning effort.")
    parser.add_argument("--retry-root", help="Optional directory for retry artifacts.")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of direct retry requests to run at once.",
    )
    parser.add_argument(
        "--all-cols",
        action="store_true",
        help="Include optional metadata columns in processed outputs.",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    outputs = retry_direct_errors(
        config_path=args.config_path,
        model_ids=args.model_ids,
        max_tokens=args.max_tokens,
        budget_tokens=args.budget_tokens,
        effort=args.effort,
        retry_root=args.retry_root,
        include_all_cols=args.all_cols,
        concurrency=args.concurrency,
    )
    for output in outputs.outputs:
        print()
        print(f"Direct Retry: {output.model_id}")
        print(f"  Original Run Dir : {output.original_run_dir}")
        print(f"  Retry Run Dir    : {output.retry_run_dir}")
        print(f"  Retried          : {output.retried_count}")
        print(f"  Successes        : {output.retry_success_count}")
        print(f"  Errors           : {output.retry_error_count}")
        print(f"  Processed CSV    : {output.merged_processed_csv_path}")
    if outputs.combined_output_path is not None:
        print()
        print(f"Combined Output: {outputs.combined_output_path}")


if __name__ == "__main__":
    main()
