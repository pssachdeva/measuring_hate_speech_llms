"""Small importable wrapper for directly retrying errored batch items."""

from pathlib import Path
from typing import Sequence

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
    )
