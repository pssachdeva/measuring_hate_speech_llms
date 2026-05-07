"""Direct retry helpers for reissuing failed batch requests one item at a time."""

from dataclasses import dataclass, replace
import json
from pathlib import Path
import subprocess
from typing import Any, Sequence

from anthropic import Anthropic
from google import genai
from loguru import logger
from openai import OpenAI
import pandas as pd

from mhs_llms.batch import (
    TOGETHER_API_BASE_URL,
    _apply_anthropic_request_reasoning,
    _build_processing_error_record,
    _coerce_anthropic_content_to_text,
    _coerce_google_response_to_text,
    _coerce_openai_content_to_text,
    _judge_id,
    _parse_response_json,
    _provider_api_key,
    _read_jsonl,
    _to_jsonable,
    _write_json,
    _write_jsonl,
    _xai_api_request,
    write_combined_processed_annotations,
)
from mhs_llms.config import ModelBatchConfig, load_model_batch_configs
from mhs_llms.schema import annotation_record_to_row, normalize_model_annotation


@dataclass(frozen=True)
class DirectRetryModelOutputs:
    """Artifacts produced when retrying one model's failed batch items directly."""

    model_id: str
    original_run_dir: Path
    retry_run_dir: Path | None
    retried_count: int
    retry_success_count: int
    retry_error_count: int
    merged_processed_records_path: Path
    merged_processed_csv_path: Path


@dataclass(frozen=True)
class DirectRetryOutputs:
    """Aggregated direct-retry outputs across every selected model config."""

    outputs: tuple[DirectRetryModelOutputs, ...]
    combined_output_path: Path | None


def retry_errored_requests(
    config_path: Path,
    *,
    model_ids: Sequence[str] | None = None,
    max_tokens: int | None = None,
    budget_tokens: int | None = None,
    effort: str | None = None,
    retry_root: Path | None = None,
    include_all_cols: bool = False,
) -> DirectRetryOutputs:
    """Retry failed batch items directly against the provider and rebuild merged outputs."""

    resolved_config_path = config_path.resolve()
    configs = load_model_batch_configs(resolved_config_path)
    selected_model_ids = set(model_ids) if model_ids is not None else None

    outputs: list[DirectRetryModelOutputs] = []
    merged_paths: list[tuple[Path, Path]] = []

    for config in configs:
        current_model_id = config.model.id or _judge_id(config)
        if selected_model_ids is not None and current_model_id not in selected_model_ids:
            merged_paths.append(_original_processed_paths(config))
            outputs.append(
                DirectRetryModelOutputs(
                    model_id=current_model_id,
                    original_run_dir=config.batches.run_dir,
                    retry_run_dir=None,
                    retried_count=0,
                    retry_success_count=0,
                    retry_error_count=0,
                    merged_processed_records_path=_original_processed_paths(config)[0],
                    merged_processed_csv_path=_original_processed_paths(config)[1],
                )
            )
            continue

        output = _retry_errored_for_config(
            config=config,
            max_tokens=max_tokens,
            budget_tokens=budget_tokens,
            effort=effort,
            retry_root=retry_root,
            include_all_cols=include_all_cols,
            config_path=resolved_config_path,
        )
        outputs.append(output)
        merged_paths.append((output.merged_processed_records_path, output.merged_processed_csv_path))

    combined_output_path: Path | None = None
    combined_base_path = configs[0].batches.combined_output_path if configs else None
    if combined_base_path is not None:
        combined_output_path = combined_base_path
        write_combined_processed_annotations(processed_paths=merged_paths, output_path=combined_output_path)

    return DirectRetryOutputs(outputs=tuple(outputs), combined_output_path=combined_output_path)


def _retry_errored_for_config(
    *,
    config: ModelBatchConfig,
    max_tokens: int | None,
    budget_tokens: int | None,
    effort: str | None,
    retry_root: Path | None,
    include_all_cols: bool,
    config_path: Path,
) -> DirectRetryModelOutputs:
    """Retry one model's errored rows and write merged per-model outputs."""

    original_run_dir = config.batches.run_dir
    manifest_path = original_run_dir / config.batches.request_manifest_filename
    original_raw_results_path = original_run_dir / config.batches.raw_results_filename
    errors_path = original_run_dir / config.batches.errors_filename
    original_metadata_path = original_run_dir / config.batches.batch_metadata_filename
    original_processed_records_path, original_processed_csv_path = _original_processed_paths(config)

    if not errors_path.exists():
        return DirectRetryModelOutputs(
            model_id=config.model.id or _judge_id(config),
            original_run_dir=original_run_dir,
            retry_run_dir=None,
            retried_count=0,
            retry_success_count=0,
            retry_error_count=0,
            merged_processed_records_path=original_processed_records_path,
            merged_processed_csv_path=original_processed_csv_path,
        )

    manifest_rows = _read_jsonl(manifest_path)
    error_rows = _read_jsonl(errors_path)
    retry_manifest_rows = _select_retry_manifest_rows(
        manifest_rows=manifest_rows,
        error_rows=error_rows,
    )
    if not retry_manifest_rows:
        return DirectRetryModelOutputs(
            model_id=config.model.id or _judge_id(config),
            original_run_dir=original_run_dir,
            retry_run_dir=None,
            retried_count=0,
            retry_success_count=0,
            retry_error_count=0,
            merged_processed_records_path=original_processed_records_path,
            merged_processed_csv_path=original_processed_csv_path,
        )

    retry_run_dir = _retry_run_dir(config=config, retry_root=retry_root)
    retry_run_dir.mkdir(parents=True, exist_ok=True)

    retry_config = _retry_config(
        config=config,
        retry_run_dir=retry_run_dir,
        max_tokens=max_tokens,
        budget_tokens=budget_tokens,
        effort=effort,
    )

    system_prompt = retry_config.prompt.system_prompt_path.read_text()
    raw_retry_results: list[dict[str, Any]] = []
    processed_rows: list[dict[str, Any]] = []
    processing_errors: list[dict[str, Any]] = []

    total_retries = len(retry_manifest_rows)
    for retry_index, manifest_row in enumerate(retry_manifest_rows, start=1):
        custom_id = str(manifest_row["custom_id"])
        comment_id = int(manifest_row["comment_id"])
        comment_text = str(manifest_row["text"])
        user_prompt = _user_prompt(config=retry_config, comment_text=comment_text)
        request_payload = _build_direct_provider_request(
            config=retry_config,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        logger.info(
            "Direct retry {}/{} for {} ({})",
            retry_index,
            total_retries,
            custom_id,
            retry_config.model.provider,
        )

        try:
            provider_response, response_text = _execute_direct_request(
                config=retry_config,
                request_payload=request_payload,
            )
            raw_retry_results.append(
                {
                    "comment_id": comment_id,
                    "custom_id": custom_id,
                    "provider_response": provider_response,
                    "request_payload": request_payload,
                    "response_text": response_text,
                }
            )
            payload = _parse_response_json(response_text)
            record = normalize_model_annotation(
                comment_id=comment_id,
                judge_id=_judge_id(retry_config),
                text=comment_text,
                payload=payload,
                metadata={
                    "provider": retry_config.model.provider,
                    "model": retry_config.model.name,
                    "custom_id": custom_id,
                    "retry": True,
                    "retry_overrides": {
                        "max_tokens": max_tokens,
                        "budget_tokens": budget_tokens,
                        "effort": effort,
                    },
                    "provider_response": provider_response,
                },
            )
            processed_rows.append(
                annotation_record_to_row(record, include_metadata=include_all_cols)
            )
        except Exception as exc:  # noqa: BLE001
            processing_error = _direct_retry_error_record(
                config=retry_config,
                custom_id=custom_id,
                comment_id=comment_id,
                request_payload=request_payload,
                exception=exc,
                raw_retry_results=raw_retry_results,
            )
            processing_errors.append(processing_error)

    retry_manifest_path = retry_run_dir / retry_config.batches.request_manifest_filename
    retry_raw_results_path = retry_run_dir / retry_config.batches.raw_results_filename
    retry_processed_records_path = retry_run_dir / retry_config.batches.processed_records_filename
    retry_processed_csv_path = retry_run_dir / retry_config.batches.processed_csv_filename
    retry_errors_path = retry_run_dir / retry_config.batches.errors_filename
    retry_metadata_path = retry_run_dir / retry_config.batches.batch_metadata_filename
    merged_processed_records_path = retry_run_dir / "merged_processed_records.jsonl"
    merged_processed_csv_path = retry_run_dir / "merged_processed_records.csv"

    _write_jsonl(retry_manifest_path, retry_manifest_rows)
    _write_jsonl(retry_raw_results_path, raw_retry_results)
    _write_jsonl(retry_processed_records_path, processed_rows)
    pd.DataFrame(processed_rows).to_csv(retry_processed_csv_path, index=False)
    _write_jsonl(retry_errors_path, processing_errors)
    _write_json(
        retry_metadata_path,
        {
            "config_path": str(config_path),
            "model_id": retry_config.model.id or _judge_id(retry_config),
            "original_run_dir": str(original_run_dir),
            "retry_run_dir": str(retry_run_dir),
            "retried_count": len(retry_manifest_rows),
            "retry_success_count": len(processed_rows),
            "retry_error_count": len(processing_errors),
            "overrides": {
                "max_tokens": max_tokens,
                "budget_tokens": budget_tokens,
                "effort": effort,
            },
        },
    )

    merged_raw_results, merged_rows, remaining_errors = _apply_retry_results_to_original_run(
        manifest_rows=manifest_rows,
        original_raw_results_path=original_raw_results_path,
        original_processed_records_path=original_processed_records_path,
        original_processed_csv_path=original_processed_csv_path,
        original_errors_path=errors_path,
        original_metadata_path=original_metadata_path,
        raw_retry_results=raw_retry_results,
        retry_rows=processed_rows,
        retry_errors=processing_errors,
        retry_metadata={
            "retry_run_dir": str(retry_run_dir),
            "retried_count": len(retry_manifest_rows),
            "retry_success_count": len(processed_rows),
            "retry_error_count": len(processing_errors),
            "overrides": {
                "max_tokens": max_tokens,
                "budget_tokens": budget_tokens,
                "effort": effort,
            },
        },
    )
    _write_jsonl(retry_run_dir / "merged_raw_results.jsonl", merged_raw_results)
    _write_jsonl(retry_run_dir / "remaining_errors.jsonl", remaining_errors)
    _write_jsonl(merged_processed_records_path, merged_rows)
    pd.DataFrame(merged_rows).to_csv(merged_processed_csv_path, index=False)

    return DirectRetryModelOutputs(
        model_id=retry_config.model.id or _judge_id(retry_config),
        original_run_dir=original_run_dir,
        retry_run_dir=retry_run_dir,
        retried_count=len(retry_manifest_rows),
        retry_success_count=len(processed_rows),
        retry_error_count=len(processing_errors),
        merged_processed_records_path=original_processed_records_path,
        merged_processed_csv_path=original_processed_csv_path,
    )


def _apply_retry_results_to_original_run(
    *,
    manifest_rows: Sequence[dict[str, Any]],
    original_raw_results_path: Path,
    original_processed_records_path: Path,
    original_processed_csv_path: Path,
    original_errors_path: Path,
    original_metadata_path: Path,
    raw_retry_results: Sequence[dict[str, Any]],
    retry_rows: Sequence[dict[str, Any]],
    retry_errors: Sequence[dict[str, Any]],
    retry_metadata: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Rewrite the original run files so retries behave like in-place repairs."""

    original_raw_results = (
        _read_jsonl(original_raw_results_path) if original_raw_results_path.exists() else []
    )
    merged_raw_results = _merge_raw_results(
        original_rows=original_raw_results,
        retry_rows=raw_retry_results,
        manifest_rows=manifest_rows,
    )

    original_rows = _read_jsonl(original_processed_records_path)
    merged_rows = _merge_processed_rows(
        original_rows=original_rows,
        retry_rows=retry_rows,
        manifest_rows=manifest_rows,
    )

    original_error_rows = _read_jsonl(original_errors_path)
    remaining_errors = _merge_processing_errors(
        original_error_rows=original_error_rows,
        retry_rows=retry_rows,
        retry_errors=retry_errors,
    )

    _write_jsonl(original_raw_results_path, merged_raw_results)
    _write_jsonl(original_processed_records_path, merged_rows)
    pd.DataFrame(merged_rows).to_csv(original_processed_csv_path, index=False)
    _write_jsonl(original_errors_path, remaining_errors)

    original_metadata = _read_existing_json(original_metadata_path)
    original_metadata["direct_retry"] = retry_metadata
    _write_json(original_metadata_path, original_metadata)

    return merged_raw_results, merged_rows, remaining_errors


def _retry_run_dir(config: ModelBatchConfig, retry_root: Path | None) -> Path:
    """Return the directory used to store direct retry artifacts for one model."""

    if retry_root is not None:
        return retry_root.resolve() / (config.model.id or _judge_id(config))
    return config.batches.run_dir / "direct_retry"


def _retry_config(
    *,
    config: ModelBatchConfig,
    retry_run_dir: Path,
    max_tokens: int | None,
    budget_tokens: int | None,
    effort: str | None,
) -> ModelBatchConfig:
    """Clone one batch config for direct retry with optional reasoning overrides."""

    retry_reasoning = replace(
        config.model.reasoning,
        effort=effort if effort is not None else config.model.reasoning.effort,
        budget_tokens=(
            budget_tokens if budget_tokens is not None else config.model.reasoning.budget_tokens
        ),
    )
    retry_model = replace(
        config.model,
        max_tokens=max_tokens if max_tokens is not None else config.model.max_tokens,
        reasoning=retry_reasoning,
    )
    retry_batches = replace(config.batches, run_dir=retry_run_dir, combined_output_path=None)
    return replace(config, model=retry_model, batches=retry_batches)


def _user_prompt(config: ModelBatchConfig, comment_text: str) -> str:
    """Render the configured user prompt for one comment text."""

    if config.prompt.user_prompt_template.strip():
        return config.prompt.user_prompt_template.format(comment_text=comment_text)
    return comment_text


def _build_direct_provider_request(
    config: ModelBatchConfig,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    """Build one direct provider request using the same config fields as batch mode."""

    provider = config.model.provider
    if provider == "openai":
        payload: dict[str, Any] = {
            "model": config.model.name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if config.model.max_tokens is not None:
            payload["max_completion_tokens"] = config.model.max_tokens
        if config.model.reasoning.effort is not None:
            payload["reasoning_effort"] = config.model.reasoning.effort
        payload.update(config.model.params)
        return payload

    if provider == "anthropic":
        payload = {
            "model": config.model.name,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if config.model.max_tokens is not None:
            payload["max_tokens"] = config.model.max_tokens
        if config.model.params:
            payload.update(config.model.params)
        return _apply_anthropic_request_reasoning(
            request_params=payload,
            model=config.model.name,
            reasoning_effort=config.model.reasoning.effort,
            reasoning_budget_tokens=config.model.reasoning.budget_tokens,
        )

    if provider == "google":
        generation_config = dict(config.model.params)
        if config.model.max_tokens is not None:
            generation_config["max_output_tokens"] = config.model.max_tokens
        if config.model.reasoning.effort is not None and config.model.reasoning.budget_tokens is not None:
            raise ValueError(
                "Google direct retries cannot set both reasoning.effort and reasoning.budget_tokens"
            )
        thinking_config: dict[str, Any] = {}
        if config.model.reasoning.effort is not None:
            thinking_config["thinking_level"] = config.model.reasoning.effort
        if config.model.reasoning.budget_tokens is not None:
            thinking_config["thinking_budget"] = config.model.reasoning.budget_tokens
        if thinking_config:
            generation_config["thinking_config"] = thinking_config
        return {
            "model": config.model.name,
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "config": {
                **generation_config,
                "system_instruction": system_prompt,
            },
        }

    if provider == "xai":
        if config.model.reasoning.budget_tokens is not None:
            raise ValueError("xAI direct retries do not support reasoning.budget_tokens")
        payload = {
            "model": config.model.name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        if config.model.max_tokens is not None:
            payload["max_tokens"] = config.model.max_tokens
        if config.model.reasoning.effort is not None:
            payload["reasoning_effort"] = config.model.reasoning.effort
        payload.update(config.model.params)
        return payload

    if provider == "together":
        if config.model.reasoning.effort is not None or config.model.reasoning.budget_tokens is not None:
            raise ValueError(
                "Together direct retries do not support the shared reasoning config fields; "
                "pass provider-specific controls through model params if needed"
            )
        payload = {
            "model": config.model.name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if config.model.max_tokens is not None:
            payload["max_tokens"] = config.model.max_tokens
        payload.update(config.model.params)
        return payload

    raise ValueError(f"Unsupported provider for direct retry: {provider}")


def _execute_direct_request(
    config: ModelBatchConfig,
    request_payload: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Execute one direct provider request and return the raw response plus extracted text."""

    provider = config.model.provider
    if provider == "openai":
        client = OpenAI(api_key=_provider_api_key(config))
        response = client.chat.completions.create(**request_payload)
        payload = _to_jsonable(response)
        return payload, _coerce_openai_content_to_text(payload["choices"][0]["message"]["content"])

    if provider == "anthropic":
        client = Anthropic(api_key=_provider_api_key(config))
        response = client.messages.create(**request_payload)
        payload = _to_jsonable(response)
        return payload, _coerce_anthropic_content_to_text(payload.get("content", []))

    if provider == "google":
        client = genai.Client(api_key=_provider_api_key(config))
        response = client.models.generate_content(
            model=request_payload["model"],
            contents=request_payload["contents"],
            config=request_payload["config"],
        )
        payload = _to_jsonable(response)
        return payload, _coerce_google_response_to_text(payload)

    if provider == "xai":
        payload = _xai_api_request(
            config=config,
            method="POST",
            path="/v1/chat/completions",
            payload=request_payload,
        )
        return payload, _coerce_openai_content_to_text(payload["choices"][0]["message"]["content"])

    if provider == "together":
        return _execute_together_streaming_request(config=config, request_payload=request_payload)

    raise ValueError(f"Unsupported provider for direct retry: {provider}")


def _execute_together_streaming_request(
    config: ModelBatchConfig,
    request_payload: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Execute one Together chat completion with streaming enabled and collect the text."""

    payload = dict(request_payload)
    payload["stream"] = True
    process = subprocess.Popen(
        [
            "curl",
            "--no-buffer",
            "--silent",
            "--show-error",
            "--fail-with-body",
            "-X",
            "POST",
            f"{TOGETHER_API_BASE_URL}/chat/completions",
            "-H",
            f"Authorization: Bearer {_provider_api_key(config)}",
            "-H",
            "Accept: text/event-stream",
            "-H",
            "Content-Type: application/json",
            "--data-binary",
            json.dumps(payload),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdout is None or process.stderr is None:
        raise ValueError("Could not open curl pipes for Together streaming request")

    provider_response, response_text = _parse_together_streaming_lines(process.stdout)
    stderr_text = process.stderr.read().decode("utf-8", errors="replace")
    returncode = process.wait()
    if returncode != 0:
        raise ValueError(
            "Together streaming request failed "
            f"(exit={returncode}): {stderr_text}{response_text}"
        )
    return provider_response, response_text


def _parse_together_streaming_lines(lines: Any) -> tuple[dict[str, Any], str]:
    """Parse OpenAI-compatible Together SSE chunks into provider metadata and text."""

    events: list[dict[str, Any]] = []
    text_chunks: list[str] = []
    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            break
        event = json.loads(data)
        events.append(event)
        choices = event.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        content = delta.get("content")
        if content:
            text_chunks.append(str(content))

    response_text = "".join(text_chunks)
    return {
        "stream": True,
        "events": events,
        "choices": [{"message": {"content": response_text}}],
    }, response_text


def _select_retry_manifest_rows(
    *,
    manifest_rows: Sequence[dict[str, Any]],
    error_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Select the original manifest rows whose custom ids appeared in processing errors."""

    errored_custom_ids = {
        str(row["custom_id"])
        for row in error_rows
        if row.get("custom_id")
    }
    return [row for row in manifest_rows if str(row["custom_id"]) in errored_custom_ids]


def _merge_processed_rows(
    *,
    original_rows: Sequence[dict[str, Any]],
    retry_rows: Sequence[dict[str, Any]],
    manifest_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge retry successes into original processed rows while preserving manifest order."""

    merged_by_comment_id = {
        int(row["comment_id"]): dict(row)
        for row in original_rows
    }
    for row in retry_rows:
        merged_by_comment_id[int(row["comment_id"])] = dict(row)

    manifest_order = {
        int(row["comment_id"]): index
        for index, row in enumerate(manifest_rows)
    }
    return sorted(
        merged_by_comment_id.values(),
        key=lambda row: manifest_order.get(int(row["comment_id"]), int(row["comment_id"])),
    )

def _merge_raw_results(
    *,
    original_rows: Sequence[dict[str, Any]],
    retry_rows: Sequence[dict[str, Any]],
    manifest_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge retry raw responses into the original raw results in manifest order."""

    merged_by_custom_id = {
        str(row["custom_id"]): dict(row)
        for row in original_rows
        if row.get("custom_id") is not None
    }
    for row in retry_rows:
        merged_by_custom_id[str(row["custom_id"])] = dict(row)

    manifest_order = [str(row["custom_id"]) for row in manifest_rows if row.get("custom_id") is not None]
    merged_rows = [
        merged_by_custom_id[custom_id]
        for custom_id in manifest_order
        if custom_id in merged_by_custom_id
    ]

    remaining_custom_ids = sorted(set(merged_by_custom_id) - set(manifest_order))
    merged_rows.extend(merged_by_custom_id[custom_id] for custom_id in remaining_custom_ids)
    return merged_rows


def _merge_processing_errors(
    *,
    original_error_rows: Sequence[dict[str, Any]],
    retry_rows: Sequence[dict[str, Any]],
    retry_errors: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Replace original errors for retried items with only the still-unresolved errors."""

    resolved_identities = set().union(*(_row_identities(row) for row in retry_rows))
    retried_identities = resolved_identities | set().union(
        *(_row_identities(row) for row in retry_errors)
    )

    remaining_rows = [
        dict(row)
        for row in original_error_rows
        if
        _row_identities(row).isdisjoint(retried_identities)
    ]
    remaining_rows.extend(dict(row) for row in retry_errors)
    return remaining_rows


def _original_processed_paths(config: ModelBatchConfig) -> tuple[Path, Path]:
    """Return the original per-model processed JSONL and CSV paths."""

    run_dir = config.batches.run_dir
    return (
        run_dir / config.batches.processed_records_filename,
        run_dir / config.batches.processed_csv_filename,
    )


def _read_existing_json(path: Path) -> dict[str, Any]:
    """Read one JSON object from disk for in-place metadata updates."""

    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _comment_id_or_none(row: dict[str, Any]) -> int | None:
    """Extract one integer comment id when present in a row-like dictionary."""

    value = row.get("comment_id")
    if value is None:
        return None
    return int(value)


def _row_identities(row: dict[str, Any]) -> set[str]:
    """Build the stable identities used to match original and retried rows."""

    identities: set[str] = set()
    comment_id = _comment_id_or_none(row)
    if comment_id is not None:
        identities.add(f"comment_id:{comment_id}")
        identities.add(f"custom_id:comment-{comment_id}")

    custom_id = row.get("custom_id")
    if custom_id is not None:
        identities.add(f"custom_id:{custom_id}")

    return identities


def _direct_retry_error_record(
    *,
    config: ModelBatchConfig,
    custom_id: str,
    comment_id: int,
    request_payload: dict[str, Any],
    exception: Exception,
    raw_retry_results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Build a structured retry error record for execution or parsing failures."""

    if raw_retry_results and raw_retry_results[-1].get("custom_id") == custom_id:
        latest_result = raw_retry_results[-1]
        return _build_processing_error_record(
            provider_name=config.model.provider,
            custom_id=custom_id,
            comment_id=comment_id,
            response_text=str(latest_result.get("response_text", "")),
            response_metadata=latest_result.get("provider_response"),
            exception=exception,
        ) | {
            "request_payload": request_payload,
            "raw_result": latest_result,
        }

    return {
        "comment_id": comment_id,
        "custom_id": custom_id,
        "error": f"Direct request failed before processing: {exception}",
        "error_type": "direct_request_error",
        "request_payload": request_payload,
    }
