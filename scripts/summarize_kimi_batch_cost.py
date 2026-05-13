import argparse
import json
from pathlib import Path

from mhs_llms.config import load_model_batch_config
from mhs_llms.paths import REPO_ROOT


KIMI_K25_STANDARD_INPUT_PER_MTOK = 0.60
KIMI_K25_STANDARD_OUTPUT_PER_MTOK = 3.00
KIMI_BATCH_PRICE_MULTIPLIER = 0.60


def _read_jsonl(path: Path) -> list[dict]:
    """Read non-empty JSONL rows from a file."""

    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _usage_from_row(row: dict) -> dict:
    """Return the OpenAI-compatible usage block from one batch result row."""

    response = row.get("response", {})
    if not isinstance(response, dict):
        return {}
    body = response.get("body", {})
    if not isinstance(body, dict):
        return {}
    usage = body.get("usage", {})
    return usage if isinstance(usage, dict) else {}


def _estimate_batch_cost(input_tokens: int, output_tokens: int) -> float:
    """Estimate Kimi K2.5 Batch API cost from token counts."""

    input_cost = input_tokens / 1_000_000 * KIMI_K25_STANDARD_INPUT_PER_MTOK
    output_cost = output_tokens / 1_000_000 * KIMI_K25_STANDARD_OUTPUT_PER_MTOK
    return (input_cost + output_cost) * KIMI_BATCH_PRICE_MULTIPLIER


def _print_summary(title: str, rows: list[tuple[str, str]]) -> None:
    """Print a compact aligned summary."""

    label_width = max(len(label) for label, _ in rows)
    print()
    print(title)
    for label, value in rows:
        print(f"  {label:<{label_width}} : {value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize Kimi K2.5 batch token usage and estimated cost."
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        default=str(
            REPO_ROOT
            / "configs"
            / "full_set_moonshot"
            / "queries_full_set_moonshot_kimi-k2.5_test50.yaml"
        ),
    )
    parser.add_argument(
        "--scale-to",
        type=int,
        help="Optional request count for a linear full-run cost projection.",
    )
    args = parser.parse_args()

    config_path = Path(args.config_path).resolve()
    config = load_model_batch_config(config_path)
    raw_results_path = config.batches.run_dir / config.batches.raw_results_filename
    if not raw_results_path.exists():
        raise FileNotFoundError(f"Raw results not found: {raw_results_path}")

    rows = _read_jsonl(raw_results_path)
    usage_rows = [_usage_from_row(row) for row in rows]
    input_tokens = sum(int(usage.get("prompt_tokens", 0) or 0) for usage in usage_rows)
    output_tokens = sum(int(usage.get("completion_tokens", 0) or 0) for usage in usage_rows)
    total_tokens = sum(int(usage.get("total_tokens", 0) or 0) for usage in usage_rows)
    successful_rows = sum(
        1
        for row in rows
        if row.get("response", {}).get("status_code") in {0, 200}
        and row.get("error") is None
    )
    estimated_cost = _estimate_batch_cost(input_tokens=input_tokens, output_tokens=output_tokens)

    summary_rows = [
        ("Config", str(config_path)),
        ("Raw Results", str(raw_results_path)),
        ("Rows", str(len(rows))),
        ("Successful Rows", str(successful_rows)),
        ("Input Tokens", f"{input_tokens:,}"),
        ("Output Tokens", f"{output_tokens:,}"),
        ("Total Tokens", f"{total_tokens:,}"),
        ("Estimated Cost", f"${estimated_cost:.4f}"),
    ]

    if rows:
        summary_rows.append(("Avg Cost / Row", f"${estimated_cost / len(rows):.6f}"))
    if args.scale_to and rows:
        scaled_cost = estimated_cost / len(rows) * args.scale_to
        summary_rows.append((f"Projected {args.scale_to:,}", f"${scaled_cost:.2f}"))

    _print_summary("Kimi Batch Cost Summary", summary_rows)


if __name__ == "__main__":
    main()
