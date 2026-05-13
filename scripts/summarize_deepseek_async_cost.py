import argparse
from datetime import datetime
import json
from pathlib import Path

from mhs_llms.config import load_model_batch_config
from mhs_llms.paths import REPO_ROOT


DEEPSEEK_V4_PRO_CACHE_HIT_PER_MTOK = 0.003625
DEEPSEEK_V4_PRO_CACHE_MISS_PER_MTOK = 0.435
DEEPSEEK_V4_PRO_OUTPUT_PER_MTOK = 0.87


def _read_json(path: Path) -> dict:
    """Read one JSON object from disk."""

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _parse_datetime(value: str | None) -> datetime | None:
    """Parse an ISO timestamp emitted by the async runner."""

    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _usage_from_response(payload: dict) -> dict:
    """Return the DeepSeek usage block from one saved async response."""

    provider_response = payload.get("provider_response")
    if not isinstance(provider_response, dict):
        return {}
    usage = provider_response.get("usage")
    return usage if isinstance(usage, dict) else {}


def _response_duration_seconds(payload: dict) -> float | None:
    """Return request wall time for one saved async response when available."""

    started = _parse_datetime(payload.get("request_started_at"))
    completed = _parse_datetime(payload.get("completed_at"))
    if started is None or completed is None:
        return None
    return max(0.0, (completed - started).total_seconds())


def _estimate_cost(
    *,
    cache_hit_tokens: int,
    cache_miss_tokens: int,
    output_tokens: int,
) -> float:
    """Estimate DeepSeek V4 Pro cost using current official promo pricing."""

    return (
        cache_hit_tokens / 1_000_000 * DEEPSEEK_V4_PRO_CACHE_HIT_PER_MTOK
        + cache_miss_tokens / 1_000_000 * DEEPSEEK_V4_PRO_CACHE_MISS_PER_MTOK
        + output_tokens / 1_000_000 * DEEPSEEK_V4_PRO_OUTPUT_PER_MTOK
    )


def _print_summary(title: str, rows: list[tuple[str, str]]) -> None:
    """Print a compact aligned summary."""

    label_width = max(len(label) for label, _ in rows)
    print()
    print(title)
    for label, value in rows:
        print(f"  {label:<{label_width}} : {value}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize DeepSeek async streaming usage, cost, and timing."
    )
    parser.add_argument(
        "config_path",
        nargs="?",
        default=str(
            REPO_ROOT
            / "configs"
            / "full_set_deepseek"
            / "queries_full_set_deepseek-v4-pro_test50.yaml"
        ),
    )
    parser.add_argument("--scale-to", type=int, default=5990)
    args = parser.parse_args()

    config_path = Path(args.config_path).resolve()
    config = load_model_batch_config(config_path)
    responses_dir = config.batches.run_dir / "async_responses"
    response_paths = sorted(responses_dir.glob("*.json"))
    responses = [_read_json(path) for path in response_paths]

    usages = [_usage_from_response(payload) for payload in responses]
    cache_hit_tokens = sum(int(usage.get("prompt_cache_hit_tokens", 0) or 0) for usage in usages)
    cache_miss_tokens = sum(int(usage.get("prompt_cache_miss_tokens", 0) or 0) for usage in usages)
    prompt_tokens = sum(int(usage.get("prompt_tokens", 0) or 0) for usage in usages)
    output_tokens = sum(int(usage.get("completion_tokens", 0) or 0) for usage in usages)
    total_tokens = sum(int(usage.get("total_tokens", 0) or 0) for usage in usages)
    reasoning_tokens = sum(
        int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0)
        for usage in usages
        if isinstance(usage.get("completion_tokens_details"), dict)
    )
    estimated_cost = _estimate_cost(
        cache_hit_tokens=cache_hit_tokens,
        cache_miss_tokens=cache_miss_tokens,
        output_tokens=output_tokens,
    )

    durations = [
        duration
        for duration in (_response_duration_seconds(payload) for payload in responses)
        if duration is not None
    ]
    elapsed = sum(durations)
    average_seconds = elapsed / len(durations) if durations else 0.0
    projected_cost = estimated_cost / len(responses) * args.scale_to if responses else 0.0
    projected_sequential_hours = average_seconds * args.scale_to / 3600 if responses else 0.0

    _print_summary(
        "DeepSeek Async Cost Summary",
        [
            ("Config", str(config_path)),
            ("Responses", str(len(responses))),
            ("Prompt Tokens", f"{prompt_tokens:,}"),
            ("Cache Hit Tokens", f"{cache_hit_tokens:,}"),
            ("Cache Miss Tokens", f"{cache_miss_tokens:,}"),
            ("Output Tokens", f"{output_tokens:,}"),
            ("Reasoning Tokens", f"{reasoning_tokens:,}"),
            ("Total Tokens", f"{total_tokens:,}"),
            ("Estimated Cost", f"${estimated_cost:.4f}"),
            ("Avg Cost / Row", f"${estimated_cost / len(responses):.6f}" if responses else "$0"),
            (f"Projected {args.scale_to:,}", f"${projected_cost:.2f}"),
            ("Avg Seconds / Row", f"{average_seconds:.1f}"),
            ("Sequential Hours", f"{projected_sequential_hours:.1f}"),
        ],
    )


if __name__ == "__main__":
    main()
