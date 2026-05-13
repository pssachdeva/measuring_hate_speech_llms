"""Sequential per-query launch and processing helpers for provider-hosted jobs."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from anthropic import Anthropic
from google import genai
from loguru import logger
from openai import OpenAI
import pandas as pd
from tqdm.auto import tqdm

from mhs_llms.batch import (
    _apply_anthropic_request_reasoning,
    _build_processing_error_record,
    _coerce_anthropic_content_to_text,
    _coerce_google_response_to_text,
    _coerce_openai_content_to_text,
    _judge_id,
    _load_batch_comments,
    _model_id,
    _parse_response_json,
    _provider_api_key,
    _read_jsonl,
    _looks_like_model_refusal,
    _to_jsonable,
    _utcnow,
    _write_json,
    _write_jsonl,
    write_combined_processed_annotations,
)
from mhs_llms.config import ModelBatchConfig, load_model_batch_configs
from mhs_llms.schema import annotation_record_to_row, normalize_model_annotation


ASYNC_MANIFEST_FILENAME = "async_request_manifest.jsonl"
ASYNC_METADATA_FILENAME = "async_job.json"
ASYNC_REQUEST_ERRORS_FILENAME = "async_request_errors.jsonl"
ASYNC_RESPONSES_DIRNAME = "async_responses"
ASYNC_PROCESSED_RECORDS_FILENAME = "async_processed_records.jsonl"
ASYNC_PROCESSED_CSV_FILENAME = "async_processed_records.csv"
ASYNC_PROCESSING_ERRORS_FILENAME = "async_processing_errors.jsonl"

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_TITLE = "measuring-hate-speech-llms"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MOONSHOT_API_BASE_URL = "https://api.moonshot.ai/v1"


@dataclass(frozen=True)
class AsyncStoragePaths:
    """Resolved storage locations for one model's async run."""

    manifest_path: Path
    metadata_path: Path
    request_errors_path: Path
    responses_dir: Path
    processed_records_path: Path
    processed_csv_path: Path
    processing_errors_path: Path


@dataclass(frozen=True)
class LaunchAsyncOutputs:
    """Summary of one sequential async launch pass for a single model."""

    model_id: str
    run_dir: Path
    metadata_path: Path
    completed_count: int
    skipped_existing_count: int
    error_count: int
    total_requests: int


@dataclass(frozen=True)
class LaunchAsyncAllOutputs:
    """Summary of one sequential async launch pass across all models."""

    outputs: tuple[LaunchAsyncOutputs, ...]
    all_complete: bool


@dataclass(frozen=True)
class ProcessAsyncOutputs:
    """Summary of one async processing pass for a single model."""

    model_id: str
    run_dir: Path
    metadata_path: Path
    processed_records_path: Path
    processed_csv_path: Path
    processing_errors_path: Path
    completed_count: int
    total_requests: int
    processing_error_count: int
    is_complete: bool


@dataclass(frozen=True)
class ProcessAsyncAllOutputs:
    """Summary of one async processing pass across all configured models."""

    outputs: tuple[ProcessAsyncOutputs, ...]
    combined_output_path: Path | None
    all_complete: bool


def launch_async(config_path: Path) -> LaunchAsyncAllOutputs:
    """Issue one sequential provider request per comment for every configured model."""

    config_path = config_path.resolve()
    configs = load_model_batch_configs(config_path)
    outputs = tuple(
        launch_async_for_config(
            config=config,
            config_path=config_path,
            show_progress=True,
        )
        for config in configs
    )
    return LaunchAsyncAllOutputs(
        outputs=outputs,
        all_complete=all(output.completed_count == output.total_requests for output in outputs),
    )


def launch_async_for_config(
    config: ModelBatchConfig,
    config_path: Path | None = None,
    show_progress: bool = False,
) -> LaunchAsyncOutputs:
    """Issue async provider requests for one model and save one file per successful query."""

    paths = _async_storage_paths(config)
    paths.responses_dir.mkdir(parents=True, exist_ok=True)

    request_rows = _build_async_request_rows(config)
    _write_jsonl(paths.manifest_path, [row["manifest"] for row in request_rows])

    request_errors: list[dict[str, Any]] = []
    runnable_rows: list[dict[str, Any]] = []
    skipped_existing_count = _partition_async_request_rows(
        config=config,
        request_rows=request_rows,
        responses_dir=paths.responses_dir,
        runnable_rows=runnable_rows,
    )
    progress_bar = (
        tqdm(
            total=len(request_rows),
            desc=_model_id(config),
            unit="req",
            leave=True,
        )
        if show_progress
        else None
    )

    try:
        if progress_bar is not None and skipped_existing_count:
            progress_bar.update(skipped_existing_count)
        if config.async_retries.concurrency > 1:
            request_errors.extend(
                _launch_async_concurrent(
                    config=config,
                    config_path=config_path,
                    paths=paths,
                    request_rows=runnable_rows,
                    progress_bar=progress_bar,
                )
            )
        else:
            request_errors.extend(
                _launch_async_sequential(
                    config=config,
                    config_path=config_path,
                    paths=paths,
                    request_rows=runnable_rows,
                    progress_bar=progress_bar,
                )
            )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    _write_jsonl(paths.request_errors_path, request_errors)

    completed_count = _count_valid_async_annotation_responses(config=config, request_rows=request_rows, responses_dir=paths.responses_dir)
    _write_json(
        paths.metadata_path,
        {
            "config_path": str(config_path.resolve()) if config_path is not None else None,
            "name": config.name,
            "model_id": _model_id(config),
            "provider": config.model.provider,
            "model": config.model.name,
            "judge_id": _judge_id(config),
            "request_count": len(request_rows),
            "completed_count": completed_count,
            "skipped_existing_count": skipped_existing_count,
            "request_error_count": len(request_errors),
            "max_attempts": config.async_retries.max_attempts,
            "retry_delay_seconds": config.async_retries.retry_delay_seconds,
            "concurrency": config.async_retries.concurrency,
            "responses_dir": str(paths.responses_dir),
            "last_launched_at": _utcnow(),
        },
    )

    logger.info(
        "Async launch for {} saved {} / {} responses",
        _model_id(config),
        completed_count,
        len(request_rows),
    )

    return LaunchAsyncOutputs(
        model_id=_model_id(config),
        run_dir=config.batches.run_dir,
        metadata_path=paths.metadata_path,
        completed_count=completed_count,
        skipped_existing_count=skipped_existing_count,
        error_count=len(request_errors),
        total_requests=len(request_rows),
    )


def _partition_async_request_rows(
    *,
    config: ModelBatchConfig,
    request_rows: list[dict[str, Any]],
    responses_dir: Path,
    runnable_rows: list[dict[str, Any]],
) -> int:
    """Split request rows into already-valid saved rows and rows that need execution."""

    skipped_existing_count = 0
    for request_row in request_rows:
        response_path = responses_dir / f"{request_row['custom_id']}.json"
        if _has_valid_async_annotation_response(
            config=config,
            request_row=request_row,
            response_path=response_path,
        ):
            skipped_existing_count += 1
        else:
            runnable_rows.append(request_row)
    return skipped_existing_count


def _launch_async_sequential(
    *,
    config: ModelBatchConfig,
    config_path: Path | None,
    paths: AsyncStoragePaths,
    request_rows: list[dict[str, Any]],
    progress_bar: Any,
) -> list[dict[str, Any]]:
    """Run async requests with the default sequential pipeline."""

    request_errors: list[dict[str, Any]] = []
    for request_row in request_rows:
        request_errors.extend(
            _run_one_async_request_with_retries(
                config=config,
                config_path=config_path,
                paths=paths,
                request_row=request_row,
            )
        )
        if progress_bar is not None:
            progress_bar.update(1)
    return request_errors


def _launch_async_concurrent(
    *,
    config: ModelBatchConfig,
    config_path: Path | None,
    paths: AsyncStoragePaths,
    request_rows: list[dict[str, Any]],
    progress_bar: Any,
) -> list[dict[str, Any]]:
    """Run async requests through the opt-in bounded-concurrency pipeline."""

    request_errors: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=config.async_retries.concurrency) as executor:
        futures = [
            executor.submit(
                _run_one_async_request_with_retries,
                config=config,
                config_path=config_path,
                paths=paths,
                request_row=request_row,
            )
            for request_row in request_rows
        ]
        for future in as_completed(futures):
            request_errors.extend(future.result())
            if progress_bar is not None:
                progress_bar.update(1)
    return request_errors


def _run_one_async_request_with_retries(
    *,
    config: ModelBatchConfig,
    config_path: Path | None,
    paths: AsyncStoragePaths,
    request_row: dict[str, Any],
) -> list[dict[str, Any]]:
    """Run one async request with configured retries and save the response on success."""

    response_path = paths.responses_dir / f"{request_row['custom_id']}.json"
    request_errors: list[dict[str, Any]] = []
    for attempt_number in range(1, config.async_retries.max_attempts + 1):
        provider_response: dict[str, Any] | None = None
        response_text = ""
        try:
            request_started_at = _utcnow()
            provider_response, response_text = _execute_async_request(
                config=config,
                request_payload=request_row["provider_request"],
            )
            _validate_async_response(
                config=config,
                request_row=request_row,
                response_path=response_path,
                response_text=response_text,
                provider_response=provider_response,
            )
            _write_json(
                response_path,
                {
                    "config_path": str(config_path.resolve()) if config_path is not None else None,
                    "name": config.name,
                    "model_id": _model_id(config),
                    "provider": config.model.provider,
                    "model": config.model.name,
                    "judge_id": _judge_id(config),
                    "custom_id": request_row["custom_id"],
                    "comment_id": request_row["comment_id"],
                    "text": request_row["text"],
                    "system_prompt": request_row["system_prompt"],
                    "user_prompt": request_row["user_prompt"],
                    "provider_request": request_row["provider_request"],
                    "provider_response": provider_response,
                    "response_text": response_text,
                    "attempt_number": attempt_number,
                    "request_started_at": request_started_at,
                    "completed_at": _utcnow(),
                },
            )
            break
        except Exception as exc:  # noqa: BLE001
            request_errors.append(
                _build_async_request_error_record(
                    config=config,
                    request_row=request_row,
                    response_text=response_text,
                    provider_response=provider_response,
                    exception=exc,
                    attempt_number=attempt_number,
                )
            )
            if attempt_number == config.async_retries.max_attempts:
                break
            logger.info(
                "Retrying async request {} for {} after failed attempt {} / {}",
                request_row["custom_id"],
                _model_id(config),
                attempt_number,
                config.async_retries.max_attempts,
            )
            if config.async_retries.retry_delay_seconds > 0:
                time.sleep(config.async_retries.retry_delay_seconds)
    return request_errors


def process_async(
    config_path: Path,
    include_all_cols: bool = False,
    output_path: Path | None = None,
) -> ProcessAsyncAllOutputs:
    """Normalize saved async query responses for every model and optionally rebuild one combined file."""

    config_path = config_path.resolve()
    outputs = tuple(
        process_async_for_config(
            config=config,
            include_all_cols=include_all_cols,
            config_path=config_path,
        )
        for config in load_model_batch_configs(config_path)
    )

    all_complete = all(output.is_complete for output in outputs)
    resolved_output_path = output_path.resolve() if output_path is not None else None
    configs = load_model_batch_configs(config_path)
    if resolved_output_path is None and configs:
        resolved_output_path = configs[0].batches.combined_output_path

    combined_output_path: Path | None = None
    if resolved_output_path is not None and outputs and all_complete:
        combined_output_path = write_combined_processed_annotations(
            processed_paths=[
                (output.processed_records_path, output.processed_csv_path)
                for output in outputs
            ],
            output_path=resolved_output_path,
        )

    return ProcessAsyncAllOutputs(
        outputs=outputs,
        combined_output_path=combined_output_path,
        all_complete=all_complete,
    )


def process_async_for_config(
    config: ModelBatchConfig,
    include_all_cols: bool = False,
    config_path: Path | None = None,
) -> ProcessAsyncOutputs:
    """Normalize saved async query responses for one model into JSONL and CSV outputs."""

    paths = _async_storage_paths(config)
    request_rows = _build_async_request_rows(config)
    if not paths.manifest_path.exists():
        _write_jsonl(paths.manifest_path, [row["manifest"] for row in request_rows])
    manifest_rows = _read_jsonl(paths.manifest_path)

    processed_rows: list[dict[str, Any]] = []
    processing_errors: list[dict[str, Any]] = []
    completed_count = 0

    for manifest_row in manifest_rows:
        response_path = paths.responses_dir / f"{manifest_row['custom_id']}.json"
        if not _is_valid_async_response_file(response_path):
            continue

        completed_count += 1
        response_payload = json.loads(response_path.read_text())
        response_text = str(response_payload.get("response_text", ""))
        try:
            payload = _parse_response_json(response_text)
            record = normalize_model_annotation(
                comment_id=int(manifest_row["comment_id"]),
                judge_id=_judge_id(config),
                text=str(manifest_row["text"]),
                payload=payload,
                metadata={
                    "provider": config.model.provider,
                    "model": config.model.name,
                    "custom_id": manifest_row["custom_id"],
                    "response_path": str(response_path),
                    "provider_response": response_payload.get("provider_response"),
                },
            )
            processed_rows.append(annotation_record_to_row(record, include_metadata=include_all_cols))
        except Exception as exc:  # noqa: BLE001
            processing_error = _build_processing_error_record(
                provider_name=config.model.provider,
                custom_id=manifest_row["custom_id"],
                comment_id=manifest_row["comment_id"],
                response_text=response_text,
                response_metadata=response_payload.get("provider_response"),
                exception=exc,
            )
            processing_error["response_path"] = str(response_path)
            processing_errors.append(processing_error)

    _write_jsonl(paths.processed_records_path, processed_rows)
    pd.DataFrame(processed_rows).to_csv(paths.processed_csv_path, index=False)
    _write_jsonl(paths.processing_errors_path, processing_errors)

    _write_json(
        paths.metadata_path,
        {
            "config_path": str(config_path.resolve()) if config_path is not None else None,
            "name": config.name,
            "model_id": _model_id(config),
            "provider": config.model.provider,
            "model": config.model.name,
            "judge_id": _judge_id(config),
            "request_count": len(manifest_rows),
            "completed_count": completed_count,
            "processed_row_count": len(processed_rows),
            "processing_error_count": len(processing_errors),
            "responses_dir": str(paths.responses_dir),
            "processed_at": _utcnow(),
        },
    )

    unused_request_count = len(manifest_rows) - completed_count
    if unused_request_count > 0:
        logger.info(
            "Async processing for {} is partial: {} / {} saved responses",
            _model_id(config),
            completed_count,
            len(manifest_rows),
        )

    return ProcessAsyncOutputs(
        model_id=_model_id(config),
        run_dir=config.batches.run_dir,
        metadata_path=paths.metadata_path,
        processed_records_path=paths.processed_records_path,
        processed_csv_path=paths.processed_csv_path,
        processing_errors_path=paths.processing_errors_path,
        completed_count=completed_count,
        total_requests=len(manifest_rows),
        processing_error_count=len(processing_errors),
        is_complete=completed_count == len(manifest_rows) and not processing_errors,
    )


def _async_storage_paths(config: ModelBatchConfig) -> AsyncStoragePaths:
    """Resolve filesystem paths used by one model's async execution flow."""

    run_dir = config.batches.run_dir
    return AsyncStoragePaths(
        manifest_path=run_dir / ASYNC_MANIFEST_FILENAME,
        metadata_path=run_dir / ASYNC_METADATA_FILENAME,
        request_errors_path=run_dir / ASYNC_REQUEST_ERRORS_FILENAME,
        responses_dir=run_dir / ASYNC_RESPONSES_DIRNAME,
        processed_records_path=run_dir / ASYNC_PROCESSED_RECORDS_FILENAME,
        processed_csv_path=run_dir / ASYNC_PROCESSED_CSV_FILENAME,
        processing_errors_path=run_dir / ASYNC_PROCESSING_ERRORS_FILENAME,
    )


def _build_async_request_rows(config: ModelBatchConfig) -> list[dict[str, Any]]:
    """Build deterministic per-comment provider requests for sequential async execution."""

    system_prompt = config.prompt.system_prompt_path.read_text()
    comments = _load_batch_comments(config)
    request_rows: list[dict[str, Any]] = []

    for comment in comments:
        comment_id = int(comment["comment_id"])
        comment_text = str(comment["text"])
        custom_id = f"comment-{comment_id}"
        if config.prompt.user_prompt_template.strip():
            user_prompt = config.prompt.user_prompt_template.format(comment_text=comment_text)
        else:
            user_prompt = comment_text

        request_rows.append(
            {
                "custom_id": custom_id,
                "comment_id": comment_id,
                "text": comment_text,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "manifest": {
                    "custom_id": custom_id,
                    "comment_id": comment_id,
                    "text": comment_text,
                },
                "provider_request": _build_async_provider_request(
                    config=config,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                ),
            }
        )
    return request_rows


def _build_async_provider_request(
    config: ModelBatchConfig,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    """Build one provider-specific request payload for sequential execution."""

    provider = config.model.provider
    if provider == "openai":
        payload: dict[str, Any] = {
            "model": config.model.name,
            "messages": _openai_messages(system_prompt, user_prompt),
        }
        if config.model.max_tokens is not None:
            payload["max_completion_tokens"] = config.model.max_tokens
        if config.model.reasoning.effort is not None:
            payload["reasoning_effort"] = config.model.reasoning.effort
        if config.model.params:
            payload.update(config.model.params)
        return payload

    if provider == "openrouter":
        payload = {
            "model": config.model.name,
            "messages": _openai_messages(system_prompt, user_prompt),
        }
        if config.model.max_tokens is not None:
            payload["max_tokens"] = config.model.max_tokens
        if config.model.params:
            payload.update(config.model.params)
        reasoning_payload: dict[str, Any] = {}
        if config.model.reasoning.effort is not None:
            reasoning_payload["effort"] = config.model.reasoning.effort
        if config.model.reasoning.budget_tokens is not None:
            reasoning_payload["max_tokens"] = config.model.reasoning.budget_tokens
        if reasoning_payload:
            payload["reasoning"] = reasoning_payload
        return payload

    if provider == "deepseek":
        payload = {
            "model": config.model.name,
            "messages": _openai_messages(system_prompt, user_prompt),
        }
        if config.model.max_tokens is not None:
            payload["max_tokens"] = config.model.max_tokens
        if config.model.reasoning.effort is not None:
            payload["reasoning_effort"] = config.model.reasoning.effort
        if config.model.reasoning.budget_tokens is not None:
            raise ValueError("DeepSeek async requests do not support reasoning.budget_tokens")
        if config.model.params:
            payload.update(config.model.params)
        return payload

    if provider == "moonshot":
        if config.model.reasoning.effort is not None or config.model.reasoning.budget_tokens is not None:
            raise ValueError(
                "Moonshot async requests support thinking only as a provider-specific "
                "model param, e.g. params.thinking.type=enabled"
            )
        payload = {
            "model": config.model.name,
            "messages": _openai_messages(system_prompt, user_prompt),
        }
        if config.model.max_tokens is not None:
            payload["max_tokens"] = config.model.max_tokens
        if config.model.params:
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
                "Google async requests cannot set both reasoning.effort and reasoning.budget_tokens"
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

    raise ValueError(f"Unsupported async provider: {provider}")


def _execute_async_request(
    config: ModelBatchConfig,
    request_payload: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Execute one provider request and return the raw provider payload plus extracted text."""

    provider = config.model.provider

    if provider == "openai":
        client = OpenAI(api_key=_provider_api_key(config))
        response = client.chat.completions.create(**request_payload)
        payload = _to_jsonable(response)
        return payload, _coerce_async_openai_text(provider_name=provider, payload=payload)

    if provider == "openrouter":
        client = OpenAI(
            api_key=_provider_api_key(config),
            base_url=OPENROUTER_BASE_URL,
            default_headers={"X-Title": OPENROUTER_TITLE},
        )
        openrouter_payload = dict(request_payload)
        reasoning_payload = openrouter_payload.pop("reasoning", None)
        if reasoning_payload is not None:
            openrouter_payload["extra_body"] = {"reasoning": reasoning_payload}
        response = client.chat.completions.create(**openrouter_payload)
        payload = _to_jsonable(response)
        return payload, _coerce_async_openai_text(provider_name=provider, payload=payload)

    if provider == "deepseek":
        return _execute_deepseek_streaming_request(config=config, request_payload=request_payload)

    if provider == "moonshot":
        return _execute_openai_compatible_streaming_request(
            config=config,
            request_payload=request_payload,
            base_url=MOONSHOT_API_BASE_URL,
            provider_label="Moonshot",
            include_usage=False,
        )

    if provider == "anthropic":
        client = Anthropic(api_key=_provider_api_key(config))
        response = client.messages.create(**request_payload)
        payload = _to_jsonable(response)
        try:
            return payload, _coerce_anthropic_content_to_text(payload.get("content", []))
        except ValueError:
            if _looks_like_model_refusal(
                provider_name=provider,
                response_text="",
                response_metadata=payload,
            ):
                return payload, ""
            raise

    if provider == "google":
        client = genai.Client(api_key=_provider_api_key(config))
        response = client.models.generate_content(
            model=request_payload["model"],
            contents=request_payload["contents"],
            config=request_payload["config"],
        )
        payload = _to_jsonable(response)
        return payload, _coerce_google_response_to_text(payload)

    raise ValueError(f"Unsupported async provider: {provider}")


def _execute_openai_compatible_streaming_request(
    config: ModelBatchConfig,
    request_payload: dict[str, Any],
    base_url: str,
    provider_label: str,
    include_usage: bool,
) -> tuple[dict[str, Any], str]:
    """Execute one OpenAI-compatible streaming chat completion and collect text."""

    payload = dict(request_payload)
    payload["stream"] = True
    if include_usage:
        payload.setdefault("stream_options", {"include_usage": True})
    process = subprocess.Popen(
        [
            "curl",
            "--no-buffer",
            "--silent",
            "--show-error",
            "--fail-with-body",
            "-X",
            "POST",
            f"{base_url}/chat/completions",
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
        raise ValueError(f"Could not open curl pipes for {provider_label} streaming request")

    provider_response, response_text = _parse_openai_compatible_streaming_lines(process.stdout)
    stderr_text = process.stderr.read().decode("utf-8", errors="replace")
    returncode = process.wait()
    if returncode != 0:
        raise ValueError(
            f"{provider_label} streaming request failed "
            f"(exit={returncode}): {stderr_text}{response_text}"
        )
    return provider_response, response_text


def _execute_deepseek_streaming_request(
    config: ModelBatchConfig,
    request_payload: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    """Execute one DeepSeek chat completion with streaming enabled and collect text."""

    return _execute_openai_compatible_streaming_request(
        config=config,
        request_payload=request_payload,
        base_url=DEEPSEEK_BASE_URL,
        provider_label="DeepSeek",
        include_usage=True,
    )


def _parse_openai_compatible_streaming_lines(lines: Any) -> tuple[dict[str, Any], str]:
    """Parse OpenAI-compatible SSE chunks into provider metadata and final text."""

    events: list[dict[str, Any]] = []
    text_chunks: list[str] = []
    usage: dict[str, Any] | None = None
    for raw_line in lines:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or line.startswith(":") or not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            break
        event = json.loads(data)
        events.append(event)
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        choices = event.get("choices", [])
        if not choices:
            continue
        delta = choices[0].get("delta", {})
        content = delta.get("content")
        if content:
            text_chunks.append(str(content))

    response_text = "".join(text_chunks)
    provider_response: dict[str, Any] = {
        "stream": True,
        "events": events,
        "choices": [{"message": {"content": response_text}}],
    }
    if usage is not None:
        provider_response["usage"] = usage
    return provider_response, response_text


def _is_valid_async_response_file(path: Path) -> bool:
    """Return whether a saved async response file is present and structurally usable."""

    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    response_text = payload.get("response_text")
    return isinstance(response_text, str) and bool(response_text.strip())


def _count_valid_async_responses(responses_dir: Path) -> int:
    """Count structurally valid saved response files in the async responses directory."""

    if not responses_dir.exists():
        return 0
    return sum(1 for path in responses_dir.glob("*.json") if _is_valid_async_response_file(path))


def _count_valid_async_annotation_responses(
    config: ModelBatchConfig,
    request_rows: list[dict[str, Any]],
    responses_dir: Path,
) -> int:
    """Count saved responses that already validate as usable MHS annotations."""

    request_row_by_custom_id = {row["custom_id"]: row for row in request_rows}
    valid_count = 0
    for response_path in responses_dir.glob("*.json"):
        custom_id = response_path.stem
        request_row = request_row_by_custom_id.get(custom_id)
        if request_row is None:
            continue
        if _has_valid_async_annotation_response(
            config=config,
            request_row=request_row,
            response_path=response_path,
        ):
            valid_count += 1
    return valid_count


def _has_valid_async_annotation_response(
    config: ModelBatchConfig,
    request_row: dict[str, Any],
    response_path: Path,
) -> bool:
    """Return whether one saved async response already parses into a valid annotation."""

    if not _is_valid_async_response_file(response_path):
        return False
    response_payload = json.loads(response_path.read_text())
    try:
        _validate_async_response(
            config=config,
            request_row=request_row,
            response_path=response_path,
            response_text=str(response_payload.get("response_text", "")),
            provider_response=response_payload.get("provider_response"),
        )
    except Exception:  # noqa: BLE001
        return False
    return True


def _validate_async_response(
    *,
    config: ModelBatchConfig,
    request_row: dict[str, Any],
    response_path: Path,
    response_text: str,
    provider_response: dict[str, Any] | None,
) -> None:
    """Validate one async provider response against the shared MHS schema."""

    payload = _parse_response_json(response_text)
    normalize_model_annotation(
        comment_id=int(request_row["comment_id"]),
        judge_id=_judge_id(config),
        text=str(request_row["text"]),
        payload=payload,
        metadata={
            "provider": config.model.provider,
            "model": config.model.name,
            "custom_id": request_row["custom_id"],
            "response_path": str(response_path),
            "provider_response": provider_response,
        },
    )


def _build_async_request_error_record(
    *,
    config: ModelBatchConfig,
    request_row: dict[str, Any],
    response_text: str,
    provider_response: dict[str, Any] | None,
    exception: Exception,
    attempt_number: int,
) -> dict[str, Any]:
    """Record one failed async attempt with parsing-aware error classification."""

    if provider_response is not None:
        error_record = _build_processing_error_record(
            provider_name=config.model.provider,
            custom_id=request_row["custom_id"],
            comment_id=request_row["comment_id"],
            response_text=response_text,
            response_metadata=provider_response,
            exception=exception,
        )
    else:
        error_record = {
            "custom_id": request_row["custom_id"],
            "comment_id": request_row["comment_id"],
            "error": str(exception),
            "error_type": "request_error",
            "response_text": response_text,
        }
    error_record["attempt_number"] = attempt_number
    error_record["provider_request"] = request_row["provider_request"]
    error_record["recorded_at"] = _utcnow()
    if provider_response is not None:
        error_record["provider_response"] = provider_response
    return error_record


def _coerce_async_openai_text(provider_name: str, payload: dict[str, Any]) -> str:
    """Extract OpenAI-compatible text while preserving explicit refusals for retry handling."""

    try:
        return _coerce_openai_content_to_text(payload["choices"][0]["message"]["content"])
    except ValueError:
        if _looks_like_model_refusal(
            provider_name=provider_name,
            response_text="",
            response_metadata=payload,
        ):
            return ""
        raise


def _openai_messages(system_prompt: str, user_prompt: str) -> list[dict[str, Any]]:
    """Build OpenAI-compatible message arrays used by OpenAI and OpenRouter."""

    messages: list[dict[str, Any]] = []
    if system_prompt.strip():
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages
