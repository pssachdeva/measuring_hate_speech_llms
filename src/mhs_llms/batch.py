"""Batch launch and processing helpers for provider-hosted model jobs."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request

from anthropic import Anthropic
from google import genai
from loguru import logger
from openai import OpenAI
import pandas as pd

from mhs_llms.config import ModelBatchConfig, load_model_batch_config, load_model_batch_configs
from mhs_llms.constants import REFERENCE_SET_PLATFORM
from mhs_llms.dataset import build_comment_frame, load_mhs_dataframe
from mhs_llms.schema import annotation_record_to_row, normalize_model_annotation


TERMINAL_BATCH_STATUSES = {
    "openai": {"completed", "failed", "expired", "cancelled"},
    "anthropic": {"ended"},
    "google": {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_PARTIALLY_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
        "BATCH_STATE_SUCCEEDED",
        "BATCH_STATE_FAILED",
        "BATCH_STATE_CANCELLED",
        "BATCH_STATE_EXPIRED",
    },
    "xai": {"completed", "completed_with_errors"},
}

SUCCESSFUL_RESULT_STATUSES = {
    "openai": {"completed"},
    "anthropic": {"ended"},
    "google": {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_PARTIALLY_SUCCEEDED",
        "BATCH_STATE_SUCCEEDED",
    },
    "xai": {"completed", "completed_with_errors"},
}

PROVIDER_API_ENV_VARS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
    "xai": "XAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}

XAI_API_BASE_URL = "https://api.x.ai"


@dataclass(frozen=True)
class LaunchBatchOutputs:
    """Paths and ids created when a provider batch job is launched."""

    model_id: str
    run_dir: Path
    batch_metadata_path: Path
    batch_id: str
    status: str


@dataclass(frozen=True)
class ProcessBatchOutputs:
    """Paths and status emitted when processing a provider batch job."""

    model_id: str
    run_dir: Path
    batch_metadata_path: Path
    status: str
    raw_results_path: Path | None
    processed_records_path: Path | None
    processed_csv_path: Path | None


@dataclass(frozen=True)
class ProcessBatchesOutputs:
    """Summarize one processing pass across every configured model batch."""

    outputs: tuple[ProcessBatchOutputs, ...]
    combined_output_path: Path | None
    all_terminal: bool
    all_successful: bool


def launch_batch(config_path: Path) -> LaunchBatchOutputs:
    """Create a provider batch job from the configured MHS comment slice."""

    config = load_model_batch_config(config_path)
    return launch_batch_for_config(config=config, config_path=config_path)


def launch_batch_for_config(config: ModelBatchConfig, config_path: Path | None = None) -> LaunchBatchOutputs:
    """Create a provider batch job for one already-resolved model config."""

    run_dir = config.batches.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    request_manifest_path = run_dir / config.batches.request_manifest_filename
    provider_requests_path = run_dir / config.batches.provider_requests_filename
    batch_metadata_path = run_dir / config.batches.batch_metadata_filename

    comments = _load_batch_comments(config)
    logger.info("Preparing {} comments for {} batch launch", len(comments), config.model.provider)

    request_manifest, provider_requests = _build_requests(config=config, comments=comments)
    _write_jsonl(request_manifest_path, request_manifest)
    _write_jsonl(provider_requests_path, provider_requests)
    logger.info("Wrote request manifest to {}", request_manifest_path)

    batch_object = _create_provider_batch(
        config=config,
        provider_requests_path=provider_requests_path,
        provider_requests=provider_requests,
    )
    batch_id = _batch_identifier(config.model.provider, batch_object)
    status = _batch_status(config.model.provider, batch_object)

    metadata = {
        "config_path": str(config_path.resolve()) if config_path is not None else None,
        "name": config.name,
        "model_id": _model_id(config),
        "provider": config.model.provider,
        "model": config.model.name,
        "judge_id": _judge_id(config),
        "reasoning": {
            "effort": config.model.reasoning.effort,
            "budget_tokens": config.model.reasoning.budget_tokens,
        },
        "model_params": config.model.params,
        "batch_id": batch_id,
        "status": status,
        "subset": config.subset,
        "request_count": len(request_manifest),
        "submitted_at": _utcnow(),
        "provider_batch": _to_jsonable(batch_object),
    }
    _write_json(batch_metadata_path, metadata)
    logger.info("Launched {} batch {} with status {}", config.model.provider, batch_id, status)

    return LaunchBatchOutputs(
        model_id=_model_id(config),
        run_dir=run_dir,
        batch_metadata_path=batch_metadata_path,
        batch_id=batch_id,
        status=status,
    )


def process_batch(config_path: Path, include_all_cols: bool = False) -> ProcessBatchOutputs:
    """Refresh batch status and download/process results once complete."""

    config = load_model_batch_config(config_path)
    return process_batch_for_config(
        config=config,
        include_all_cols=include_all_cols,
        config_path=config_path,
    )


def process_batch_for_config(
    config: ModelBatchConfig,
    include_all_cols: bool = False,
    config_path: Path | None = None,
) -> ProcessBatchOutputs:
    """Refresh one resolved batch status and process results once complete."""

    run_dir = config.batches.run_dir
    batch_metadata_path = run_dir / config.batches.batch_metadata_filename
    request_manifest_path = run_dir / config.batches.request_manifest_filename
    raw_results_path = run_dir / config.batches.raw_results_filename
    processed_records_path = run_dir / config.batches.processed_records_filename
    processed_csv_path = run_dir / config.batches.processed_csv_filename
    errors_path = run_dir / config.batches.errors_filename

    metadata = json.loads(batch_metadata_path.read_text())
    batch_id = str(metadata["batch_id"])
    logger.info("Checking {} batch {}", config.model.provider, batch_id)

    batch_object = _retrieve_provider_batch(config=config, batch_id=batch_id)
    status = _batch_status(config.model.provider, batch_object)
    metadata["status"] = status
    metadata["last_checked_at"] = _utcnow()
    metadata["provider_batch"] = _to_jsonable(batch_object)
    _write_json(batch_metadata_path, metadata)
    logger.info("Current {} batch status: {}", config.model.provider, status)

    if status not in TERMINAL_BATCH_STATUSES[config.model.provider]:
        logger.info("Batch {} is still running; no result download yet", batch_id)
        return ProcessBatchOutputs(
            model_id=_model_id(config),
            run_dir=run_dir,
            batch_metadata_path=batch_metadata_path,
            status=status,
            raw_results_path=None,
            processed_records_path=None,
            processed_csv_path=None,
        )

    if status not in SUCCESSFUL_RESULT_STATUSES[config.model.provider]:
        logger.warning("Batch {} finished in terminal state {} with no downloadable results", batch_id, status)
        return ProcessBatchOutputs(
            model_id=_model_id(config),
            run_dir=run_dir,
            batch_metadata_path=batch_metadata_path,
            status=status,
            raw_results_path=None,
            processed_records_path=None,
            processed_csv_path=None,
        )

    raw_entries = _download_provider_results(config=config, batch_object=batch_object)
    _write_jsonl(raw_results_path, raw_entries)
    logger.info("Stored {} raw batch results at {}", len(raw_entries), raw_results_path)

    manifest_rows = _read_jsonl(request_manifest_path)
    manifest_by_custom_id = {row["custom_id"]: row for row in manifest_rows}

    extractor = _result_extractor(config.model.provider)
    processed_rows: list[dict[str, Any]] = []
    processing_errors: list[dict[str, Any]] = []

    for entry in raw_entries:
        try:
            custom_id, response_text, response_metadata, response_error = extractor(entry)
        except Exception as exc:  # noqa: BLE001
            processing_errors.append(
                {
                    "custom_id": _result_entry_custom_id(config.model.provider, entry),
                    "error": f"Provider result extraction failed: {exc}",
                    "error_type": "provider_response_format_error",
                    "raw_result": entry,
                }
            )
            continue
        manifest_row = manifest_by_custom_id.get(custom_id)
        if manifest_row is None:
            processing_errors.append(
                {
                    "custom_id": custom_id,
                    "error": "Result returned an unknown custom_id",
                    "raw_result": entry,
                }
            )
            continue

        if response_error is not None:
            processing_errors.append(
                {
                    "custom_id": custom_id,
                    "comment_id": manifest_row["comment_id"],
                    "error": response_error,
                    "raw_result": entry,
                }
            )
            continue

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
                    "batch_id": batch_id,
                    "custom_id": custom_id,
                    "provider_response": response_metadata,
                },
            )
            processed_rows.append(
                annotation_record_to_row(record, include_metadata=include_all_cols)
            )
        except Exception as exc:  # noqa: BLE001
            processing_error = _build_processing_error_record(
                provider_name=config.model.provider,
                custom_id=custom_id,
                comment_id=manifest_row["comment_id"],
                response_text=response_text,
                response_metadata=response_metadata,
                exception=exc,
            )
            processing_error["raw_result"] = entry
            processing_errors.append(processing_error)

    _write_jsonl(processed_records_path, processed_rows)
    pd.DataFrame(processed_rows).to_csv(processed_csv_path, index=False)
    _write_jsonl(errors_path, processing_errors)
    logger.info(
        "Processed {} successful rows and {} errors",
        len(processed_rows),
        len(processing_errors),
    )

    metadata["processed_at"] = _utcnow()
    metadata["processed_row_count"] = len(processed_rows)
    metadata["processing_error_count"] = len(processing_errors)
    if config_path is not None:
        metadata["config_path"] = str(config_path.resolve())
    _write_json(batch_metadata_path, metadata)

    return ProcessBatchOutputs(
        model_id=_model_id(config),
        run_dir=run_dir,
        batch_metadata_path=batch_metadata_path,
        status=status,
        raw_results_path=raw_results_path,
        processed_records_path=processed_records_path,
        processed_csv_path=processed_csv_path,
    )


def launch_batches(config_path: Path) -> tuple[LaunchBatchOutputs, ...]:
    """Launch one provider batch for each model declared in the shared config."""

    config_path = config_path.resolve()
    return tuple(
        launch_batch_for_config(config=config, config_path=config_path)
        for config in load_model_batch_configs(config_path)
    )


def process_batches(
    config_path: Path,
    include_all_cols: bool = False,
    output_path: Path | None = None,
) -> ProcessBatchesOutputs:
    """Process every configured batch and rebuild the combined file only when all succeed."""

    config_path = config_path.resolve()
    configs = load_model_batch_configs(config_path)
    outputs = tuple(
        process_batch_for_config(
            config=config,
            include_all_cols=include_all_cols,
            config_path=config_path,
        )
        for config in configs
    )

    all_terminal = all(
        output.status in TERMINAL_BATCH_STATUSES[config.model.provider]
        for config, output in zip(configs, outputs, strict=True)
    )
    all_successful = all(
        output.status in SUCCESSFUL_RESULT_STATUSES[config.model.provider]
        for config, output in zip(configs, outputs, strict=True)
    )

    resolved_output_path = output_path.resolve() if output_path is not None else None
    if resolved_output_path is None and configs:
        resolved_output_path = configs[0].batches.combined_output_path

    combined_output_path: Path | None = None
    if resolved_output_path is not None and all_terminal and all_successful:
        processed_paths = [
            (output.processed_records_path, output.processed_csv_path)
            for output in outputs
            if output.processed_records_path is not None and output.processed_csv_path is not None
        ]
        if len(processed_paths) != len(outputs):
            raise ValueError("Not every successful batch produced processed annotation files")
        combined_output_path = write_combined_processed_annotations(
            processed_paths=processed_paths,
            output_path=resolved_output_path,
        )

    return ProcessBatchesOutputs(
        outputs=outputs,
        combined_output_path=combined_output_path,
        all_terminal=all_terminal,
        all_successful=all_successful,
    )


def write_processed_annotations(
    processed_records_path: Path,
    processed_csv_path: Path,
    output_path: Path,
) -> Path:
    """Rewrite one CSV or JSONL file from a single processed batch output."""

    return write_combined_processed_annotations(
        processed_paths=[(processed_records_path, processed_csv_path)],
        output_path=output_path,
    )


def write_combined_processed_annotations(
    processed_paths: Iterable[tuple[Path, Path]],
    output_path: Path,
) -> Path:
    """Rebuild one combined CSV or JSONL file from one or more processed batch outputs."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    normalized_paths = list(processed_paths)

    if output_path.suffix == ".csv":
        dataframes = [_read_processed_csv(csv_path) for _, csv_path in normalized_paths]
        dataframe = pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()
        dataframe.to_csv(output_path, index=False)
        return output_path

    if output_path.suffix == ".jsonl":
        with output_path.open("w") as handle:
            for processed_records_path, _ in normalized_paths:
                lines = processed_records_path.read_text()
                if not lines:
                    continue
                handle.write(lines)
                if not lines.endswith("\n"):
                    handle.write("\n")
        return output_path

    raise ValueError("Output path must end in .csv or .jsonl")


def _load_batch_comments(config: ModelBatchConfig) -> list[dict[str, Any]]:
    """Load one text row per comment for the configured evaluation slice."""

    dataset = load_mhs_dataframe(
        dataset_name="ucberkeley-dlab/measuring-hate-speech",
        split="train",
        config_name=None,
    )
    comment_ids = _select_comment_ids(dataframe=dataset, subset=config.subset)

    comments = build_comment_frame(dataset, comment_ids=comment_ids, limit=config.limit)
    return comments.to_dict(orient="records")


def _select_comment_ids(dataframe: pd.DataFrame, subset: str | dict[str, Any]) -> list[int] | None:
    """Return the in-code comment selection for a named batch subset."""

    if subset == "reference_set":
        return (
            dataframe.loc[dataframe["platform"] == REFERENCE_SET_PLATFORM, "comment_id"]
            .drop_duplicates()
            .astype(int)
            .sort_values(kind="stable")
            .tolist()
        )
    if subset == "all_comments":
        return (
            dataframe["comment_id"]
            .drop_duplicates()
            .astype(int)
            .sort_values(kind="stable")
            .tolist()
        )
    if isinstance(subset, dict):
        subset_type = str(subset.get("type", ""))
        if subset_type == "annotator_count_threshold":
            return _select_annotator_count_threshold_comment_ids(dataframe=dataframe, subset=subset)
        if subset_type == "comment_ids_file":
            return _select_comment_ids_from_file(subset=subset)
        raise ValueError(f"Unsupported subset type: {subset_type}")
    raise ValueError(f"Unsupported subset: {subset}")


def _select_comment_ids_from_file(subset: dict[str, Any]) -> list[int]:
    """Return comment ids from a CSV file column configured by a structured subset."""

    path_value = subset.get("path")
    if path_value is None:
        raise ValueError("comment_ids_file subset requires 'path'")
    column = str(subset.get("column", "comment_id"))
    path = Path(path_value)
    if not path.is_absolute():
        path = Path.cwd() / path
    frame = pd.read_csv(path)
    if column not in frame.columns:
        raise ValueError(f"comment_ids_file column '{column}' not found in {path}")
    return frame[column].drop_duplicates().astype(int).tolist()


def _select_annotator_count_threshold_comment_ids(
    dataframe: pd.DataFrame,
    subset: dict[str, Any],
) -> list[int]:
    """Return comment ids whose annotation count falls within configured bounds."""

    min_count = subset.get("min")
    max_count = subset.get("max")
    if min_count is None and max_count is None:
        raise ValueError("annotator_count_threshold subset requires 'min' or 'max'")

    counts = dataframe.groupby("comment_id").size()
    selected = counts
    if min_count is not None:
        selected = selected[selected >= int(min_count)]
    if max_count is not None:
        selected = selected[selected <= int(max_count)]
    return selected.index.astype(int).sort_values().tolist()


def _build_requests(
    config: ModelBatchConfig,
    comments: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build a manifest plus the provider-specific requests for a batch launch."""

    system_prompt = config.prompt.system_prompt_path.read_text()
    request_manifest: list[dict[str, Any]] = []
    provider_requests: list[dict[str, Any]] = []

    for comment in comments:
        comment_id = int(comment["comment_id"])
        comment_text = str(comment["text"])
        custom_id = f"comment-{comment_id}"
        if config.prompt.user_prompt_template.strip():
            user_prompt = config.prompt.user_prompt_template.format(comment_text=comment_text)
        else:
            user_prompt = comment_text
        request_manifest.append(
            {
                "custom_id": custom_id,
                "comment_id": comment_id,
                "text": comment_text,
            }
        )
        provider_requests.append(
            _provider_request(
                provider_name=config.model.provider,
                custom_id=custom_id,
                model=config.model.name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=config.model.max_tokens,
                model_params=config.model.params,
                reasoning_effort=config.model.reasoning.effort,
                reasoning_budget_tokens=config.model.reasoning.budget_tokens,
                comment_id=comment_id,
            )
        )
    return request_manifest, provider_requests


def _provider_request(
    provider_name: str,
    custom_id: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int | None,
    model_params: dict[str, Any],
    reasoning_effort: str | None,
    reasoning_budget_tokens: int | None,
    comment_id: int,
) -> dict[str, Any]:
    """Build one provider-specific request payload."""

    if provider_name == "openai":
        body = {
            **model_params,
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if max_tokens is not None:
            body["max_completion_tokens"] = max_tokens
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }
    if provider_name == "anthropic":
        params = {
            **model_params,
            "model": model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
        }
        if max_tokens is not None:
            params["max_tokens"] = max_tokens
        params = _apply_anthropic_request_reasoning(
            request_params=params,
            model=model,
            reasoning_effort=reasoning_effort,
            reasoning_budget_tokens=reasoning_budget_tokens,
        )
        return {
            "custom_id": custom_id,
            "params": params,
        }
    if provider_name == "google":
        config = {
            **model_params,
        }
        if max_tokens is not None:
            config["max_output_tokens"] = max_tokens
        config = _apply_google_batch_reasoning(
            generation_config=config,
            reasoning_effort=reasoning_effort,
            reasoning_budget_tokens=reasoning_budget_tokens,
        )
        request_payload = {
            "metadata": {"key": custom_id},
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": user_prompt}],
                    }
                ],
                "generationConfig": _google_batch_alias_dict(config),
            },
        }
        if system_prompt.strip():
            request_payload["request"]["systemInstruction"] = {
                "parts": [{"text": system_prompt}]
            }
        return request_payload
    if provider_name == "xai":
        if reasoning_budget_tokens is not None:
            raise ValueError("xAI batch requests do not support reasoning.budget_tokens")
        body = {
            **model_params,
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        return {
            "batch_request_id": custom_id,
            "batch_request": {
                "chat_get_completion": body,
            },
        }
    raise ValueError(f"Unsupported provider: {provider_name}")


def _create_provider_batch(
    config: ModelBatchConfig,
    provider_requests_path: Path,
    provider_requests: list[dict[str, Any]],
) -> Any:
    """Submit the prepared requests to the configured provider."""

    if config.model.provider == "openai":
        client = OpenAI(api_key=_provider_api_key(config))
        with provider_requests_path.open("rb") as handle:
            input_file = client.files.create(file=handle, purpose="batch")
        return client.batches.create(
            completion_window="24h",
            endpoint="/v1/chat/completions",
            input_file_id=input_file.id,
            metadata={"job_name": config.name},
        )
    if config.model.provider == "anthropic":
        client = Anthropic(api_key=_provider_api_key(config))
        return client.messages.batches.create(requests=provider_requests)
    if config.model.provider == "google":
        client = genai.Client(api_key=_provider_api_key(config))
        inline_requests = [
            _google_inline_request_from_batch_entry(request)
            for request in provider_requests
        ]
        return client.batches.create(
            model=config.model.name,
            src=inline_requests,
            config={"display_name": config.name},
        )
    if config.model.provider == "xai":
        batch = _xai_api_request(
            config=config,
            method="POST",
            path="/v1/batches",
            payload={"name": config.name},
        )
        for request_chunk in _chunks(provider_requests, size=1000):
            _xai_api_request(
                config=config,
                method="POST",
                path=f"/v1/batches/{batch['batch_id']}/requests",
                payload={"batch_requests": request_chunk},
            )
        return _xai_api_request(
            config=config,
            method="GET",
            path=f"/v1/batches/{batch['batch_id']}",
        )
    raise ValueError(f"Unsupported provider: {config.model.provider}")


def _chunks(items: list[dict[str, Any]], size: int) -> list[list[dict[str, Any]]]:
    """Split provider request payloads into fixed-size chunks."""

    return [items[index : index + size] for index in range(0, len(items), size)]


def _retrieve_provider_batch(config: ModelBatchConfig, batch_id: str) -> Any:
    """Fetch the latest provider-side status for an existing batch."""

    if config.model.provider == "openai":
        client = OpenAI(api_key=_provider_api_key(config))
        return client.batches.retrieve(batch_id)
    if config.model.provider == "anthropic":
        client = Anthropic(api_key=_provider_api_key(config))
        return client.messages.batches.retrieve(batch_id)
    if config.model.provider == "google":
        client = genai.Client(api_key=_provider_api_key(config))
        return client.batches.get(name=batch_id)
    if config.model.provider == "xai":
        return _xai_api_request(
            config=config,
            method="GET",
            path=f"/v1/batches/{batch_id}",
        )
    raise ValueError(f"Unsupported provider: {config.model.provider}")


def _download_provider_results(config: ModelBatchConfig, batch_object: Any) -> list[dict[str, Any]]:
    """Download provider results and convert them into JSON-serializable entries."""

    if config.model.provider == "openai":
        client = OpenAI(api_key=_provider_api_key(config))
        output_file_id = getattr(batch_object, "output_file_id", None)
        if output_file_id is None:
            raise ValueError("OpenAI batch completed without an output_file_id")
        raw_text = client.files.content(output_file_id).text
        return [json.loads(line) for line in raw_text.splitlines() if line.strip()]

    if config.model.provider == "anthropic":
        client = Anthropic(api_key=_provider_api_key(config))
        return [_to_jsonable(entry) for entry in client.messages.batches.results(batch_object.id)]

    if config.model.provider == "google":
        destination = getattr(batch_object, "dest", None)
        inlined_responses = getattr(destination, "inlined_responses", None)
        if inlined_responses is None:
            destination_payload = _to_jsonable(destination) if destination is not None else {}
            if isinstance(destination_payload, dict):
                inlined_responses = destination_payload.get("inlined_responses") or destination_payload.get(
                    "inlinedResponses"
                )
        if inlined_responses is not None:
            request_manifest = _load_request_manifest_entries(config)
            response_rows = [_to_jsonable(item) for item in inlined_responses]
            if len(response_rows) != len(request_manifest):
                raise ValueError(
                    "Google inline batch response count did not match the request manifest "
                    f"({len(response_rows)} != {len(request_manifest)})"
                )
            results: list[dict[str, Any]] = []
            for response_row in response_rows:
                entry = dict(response_row)
                metadata = entry.get("metadata")
                metadata_key = metadata.get("key") if isinstance(metadata, dict) else None
                if metadata_key is None:
                    raise ValueError(
                        "Google inline batch response did not include metadata.key; "
                        "refusing to assign comment ids positionally"
                    )
                entry.setdefault("custom_id", str(metadata_key))
                results.append(entry)
            return results

        file_name = getattr(getattr(batch_object, "dest", None), "file_name", None)
        if file_name is None:
            raise ValueError("Google batch completed without a destination file")
        raw_bytes = _download_google_batch_output(file_name=file_name, api_key=_provider_api_key(config))
        return [json.loads(line) for line in raw_bytes.decode("utf-8").splitlines() if line.strip()]

    if config.model.provider == "xai":
        batch_id = str(batch_object["batch_id"])
        results: list[dict[str, Any]] = []
        pagination_token: str | None = None
        while True:
            payload = _xai_api_request(
                config=config,
                method="GET",
                path=f"/v1/batches/{batch_id}/results",
                query={
                    "page_size": 1000,
                    "pagination_token": pagination_token,
                },
            )
            results.extend(payload.get("results", []))
            pagination_token = payload.get("pagination_token")
            if not pagination_token:
                break
        return results

    raise ValueError(f"Unsupported provider: {config.model.provider}")


def _result_extractor(
    provider_name: str,
) -> Callable[[dict[str, Any]], tuple[str, str, dict[str, Any], str | None]]:
    """Return the provider-specific result extraction function."""

    if provider_name == "openai":
        return _extract_openai_result
    if provider_name == "anthropic":
        return _extract_anthropic_result
    if provider_name == "google":
        return _extract_google_result
    if provider_name == "xai":
        return _extract_xai_result
    raise ValueError(f"Unsupported provider: {provider_name}")


def _extract_openai_result(
    entry: dict[str, Any],
) -> tuple[str, str, dict[str, Any], str | None]:
    """Extract text content and status metadata from one OpenAI batch result."""

    custom_id = str(entry["custom_id"])
    error = entry.get("error")
    if error is not None:
        return custom_id, "", {"error": error}, json.dumps(error, sort_keys=True)

    response = entry.get("response", {})
    status_code = int(response.get("status_code", 0))
    body = response.get("body", {})
    if status_code != 200:
        return custom_id, "", body, f"OpenAI request failed with status_code={status_code}"

    choices = body.get("choices", [])
    if not choices:
        return custom_id, "", body, "OpenAI response did not include any choices"

    message = choices[0].get("message", {})
    return custom_id, _coerce_openai_content_to_text(message.get("content")), body, None


def _extract_anthropic_result(
    entry: dict[str, Any],
) -> tuple[str, str, dict[str, Any], str | None]:
    """Extract text content and status metadata from one Anthropic batch result."""

    custom_id = str(entry["custom_id"])
    result = entry.get("result", {})
    result_type = result.get("type")
    if result_type != "succeeded":
        return custom_id, "", result, f"Anthropic request ended with result type '{result_type}'"

    message = result.get("message", {})
    content = message.get("content", [])
    if not any(isinstance(block, dict) and block.get("type") == "text" for block in content):
        stop_reason = message.get("stop_reason")
        if stop_reason == "refusal":
            return custom_id, "", message, "Anthropic model refusal with no text content"
        return custom_id, "", message, "Anthropic response did not include a text block"
    return custom_id, _coerce_anthropic_content_to_text(content), message, None


def _extract_google_result(
    entry: dict[str, Any],
) -> tuple[str, str, dict[str, Any], str | None]:
    """Extract text content and status metadata from one Google batch result."""

    metadata = entry.get("metadata", {})
    custom_id = str(
        entry.get("key")
        or entry.get("custom_id")
        or (metadata.get("key") if isinstance(metadata, dict) else "")
        or ""
    )
    response = entry.get("response")
    if response is not None:
        if "error" in response:
            return custom_id, "", response, json.dumps(response["error"], sort_keys=True)
        return custom_id, _coerce_google_response_to_text(response), response, None

    if "error" in entry:
        return custom_id, "", entry, json.dumps(entry["error"], sort_keys=True)
    if "code" in entry and "message" in entry:
        return custom_id, "", entry, json.dumps(
            {"code": entry["code"], "message": entry["message"]},
            sort_keys=True,
        )

    return custom_id, _coerce_google_response_to_text(entry), entry, None


def _extract_xai_result(
    entry: dict[str, Any],
) -> tuple[str, str, dict[str, Any], str | None]:
    """Extract text content and status metadata from one xAI batch result."""

    custom_id = str(entry.get("batch_request_id", ""))
    error_message = entry.get("error_message")
    if error_message:
        return custom_id, "", entry, str(error_message)

    batch_result = entry.get("batch_result", {})
    response = batch_result.get("response", {})
    completion = response.get("chat_get_completion")
    if completion is None:
        return custom_id, "", entry, "xAI batch result did not include chat_get_completion"

    choices = completion.get("choices", [])
    if not choices:
        return custom_id, "", completion, "xAI chat_get_completion did not include any choices"

    message = choices[0].get("message", {})
    return custom_id, _coerce_openai_content_to_text(message.get("content")), completion, None


def _coerce_openai_content_to_text(content: Any) -> str:
    """Convert OpenAI content payloads into a single text string."""

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_chunks = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"output_text", "text"}:
                text_chunks.append(str(item.get("text", "")))
        return "".join(text_chunks)
    raise ValueError("OpenAI response content was not text")


def _coerce_anthropic_content_to_text(content: list[dict[str, Any]]) -> str:
    """Convert Anthropic content blocks into a single text string."""

    text_chunks = [str(block.get("text", "")) for block in content if block.get("type") == "text"]
    if not text_chunks:
        raise ValueError("Anthropic response did not include a text block")
    return "".join(text_chunks)


def _coerce_google_response_to_text(response: dict[str, Any]) -> str:
    """Convert a Gemini batch response into a single text string."""

    candidates = response.get("candidates", [])
    if not candidates:
        raise ValueError("Google response did not include any candidates")

    parts = candidates[0].get("content", {}).get("parts", [])
    text_chunks = [str(part.get("text", "")) for part in parts if "text" in part]
    if not text_chunks:
        raise ValueError("Google response did not include text parts")
    return "".join(text_chunks)


def _result_entry_custom_id(provider_name: str, entry: dict[str, Any]) -> str:
    """Extract the provider-specific request identifier from one raw result entry."""

    if provider_name == "openai":
        return str(entry.get("custom_id", ""))
    if provider_name == "anthropic":
        return str(entry.get("custom_id", ""))
    if provider_name == "google":
        metadata = entry.get("metadata", {})
        return str(
            entry.get("key")
            or entry.get("custom_id")
            or (metadata.get("key") if isinstance(metadata, dict) else "")
            or ""
        )
    if provider_name == "xai":
        return str(entry.get("batch_request_id", ""))
    return str(entry.get("custom_id", ""))


def _google_batch_alias_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert Google batch request keys to the REST alias casing expected by Batch API."""

    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            converted: dict[str, Any] = {}
            for key, nested_value in value.items():
                parts = key.split("_")
                alias = parts[0] + "".join(part.capitalize() for part in parts[1:])
                converted[alias] = convert(nested_value)
            return converted
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    return convert(payload)


def _google_inline_request_from_batch_entry(entry: dict[str, Any]) -> dict[str, Any]:
    """Convert one stored Google batch row into the SDK inline-request shape."""

    request_payload = dict(entry.get("request", entry))
    inline_request = {
        "contents": request_payload["contents"],
    }
    metadata = entry.get("metadata")
    if isinstance(metadata, dict):
        inline_request["metadata"] = {str(key): str(value) for key, value in metadata.items()}

    config: dict[str, Any] = {}
    generation_config = request_payload.get("generationConfig") or request_payload.get("generation_config")
    if generation_config is not None:
        config.update(_google_batch_sdk_dict(generation_config))

    system_instruction = request_payload.get("systemInstruction") or request_payload.get("system_instruction")
    if system_instruction is not None:
        config["system_instruction"] = _google_batch_sdk_dict(system_instruction)

    if config:
        inline_request["config"] = config

    return inline_request


def _google_batch_sdk_dict(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert Google REST alias keys back to the SDK snake_case request shape."""

    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            converted: dict[str, Any] = {}
            for key, nested_value in value.items():
                snake_key = []
                for index, char in enumerate(key):
                    if char.isupper() and index > 0:
                        snake_key.append("_")
                    snake_key.append(char.lower())
                converted["".join(snake_key)] = convert(nested_value)
            return converted
        if isinstance(value, list):
            return [convert(item) for item in value]
        return value

    return convert(payload)


def _load_request_manifest_entries(config: ModelBatchConfig) -> list[dict[str, Any]]:
    """Read the request manifest rows for one launched batch run."""

    manifest_path = config.batches.run_dir / config.batches.request_manifest_filename
    return [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]


def _parse_response_json(response_text: str) -> dict[str, Any]:
    """Parse the provider text output into the expected JSON response object."""

    cleaned_text = response_text.strip()
    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.strip("`")
        if cleaned_text.startswith("json"):
            cleaned_text = cleaned_text[4:]
        cleaned_text = cleaned_text.strip()

    try:
        payload = json.loads(cleaned_text)
    except json.JSONDecodeError:
        # Some providers occasionally prepend reasoning or wrap the JSON in
        # extra text. Recover only when we can isolate one valid JSON object.
        payload = json.loads(_extract_embedded_json_object(cleaned_text))
    if not isinstance(payload, dict):
        raise ValueError("Model output must decode to a JSON object")
    return payload


def _extract_embedded_json_object(response_text: str) -> str:
    """Extract one embedded JSON object from a text response with extra prose."""

    fenced_json = _extract_fenced_json_block(response_text)
    if fenced_json is not None:
        return fenced_json

    decoder = json.JSONDecoder()
    for index, character in enumerate(response_text):
        if character != "{":
            continue
        try:
            payload, end_index = decoder.raw_decode(response_text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            trailing_text = response_text[index + end_index :].strip()
            if not trailing_text or trailing_text.startswith("```"):
                return response_text[index : index + end_index]

    raise json.JSONDecodeError("Expecting value", response_text, 0)


def _extract_fenced_json_block(response_text: str) -> str | None:
    """Return the contents of the first fenced JSON block when one is present."""

    fence_start = response_text.find("```json")
    if fence_start == -1:
        fence_start = response_text.find("```")
        if fence_start == -1:
            return None

    block_start = response_text.find("\n", fence_start)
    if block_start == -1:
        return None

    fence_end = response_text.find("```", block_start + 1)
    if fence_end == -1:
        return None

    return response_text[block_start + 1 : fence_end].strip()


def _build_processing_error_record(
    *,
    provider_name: str,
    custom_id: str,
    comment_id: int | str,
    response_text: str,
    response_metadata: dict[str, Any] | None,
    exception: Exception,
) -> dict[str, Any]:
    """Convert one processing exception into a structured error record."""

    error_record = {
        "custom_id": custom_id,
        "comment_id": comment_id,
        "response_text": response_text,
    }
    error_record.update(
        _classify_processing_failure(
            provider_name=provider_name,
            response_text=response_text,
            response_metadata=response_metadata,
            exception=exception,
        )
    )
    return error_record


def _classify_processing_failure(
    *,
    provider_name: str,
    response_text: str,
    response_metadata: dict[str, Any] | None,
    exception: Exception,
) -> dict[str, Any]:
    """Classify one processing failure so refusal cases are easy to inspect downstream."""

    details: dict[str, Any] = {
        "error": str(exception),
        "error_type": "processing_error",
    }
    stop_reason = _response_stop_reason(response_metadata)
    if stop_reason:
        details["provider_stop_reason"] = stop_reason

    if isinstance(exception, json.JSONDecodeError):
        if _looks_like_model_refusal(
            provider_name=provider_name,
            response_text=response_text,
            response_metadata=response_metadata,
        ):
            details["error_type"] = "model_refusal"
            details["error"] = "Model returned an unstructured refusal instead of the expected JSON object"
        else:
            details["error_type"] = "invalid_json_response"
            details["error"] = "Model returned non-JSON text instead of the expected JSON object"
        details["parse_error"] = str(exception)

    return details


def _response_stop_reason(response_metadata: dict[str, Any] | None) -> str | None:
    """Extract the provider stop reason from normalized response metadata when present."""

    if not isinstance(response_metadata, dict):
        return None
    stop_reason = response_metadata.get("stop_reason")
    if stop_reason is None:
        return None
    return str(stop_reason)


def _looks_like_model_refusal(
    *,
    provider_name: str,
    response_text: str,
    response_metadata: dict[str, Any] | None,
) -> bool:
    """Heuristically identify when a provider returned a refusal rather than malformed JSON."""

    stop_reason = _response_stop_reason(response_metadata)
    if stop_reason == "refusal":
        return True

    if provider_name == "openai" and isinstance(response_metadata, dict):
        choices = response_metadata.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message", {})
            refusal = message.get("refusal")
            if refusal:
                return True

    lowered_text = response_text.strip().lower()
    refusal_markers = (
        "i can't",
        "i cannot",
        "i wont",
        "i won't",
        "i'm not able",
        "i am not able",
        "unable to",
        "not able to engage",
        "not able to help",
        "can't analyze this content",
        "cannot analyze this content",
        "can't help with",
        "cannot help with",
    )
    return any(marker in lowered_text for marker in refusal_markers)


def _apply_google_batch_reasoning(
    generation_config: dict[str, Any],
    reasoning_effort: str | None,
    reasoning_budget_tokens: int | None,
) -> dict[str, Any]:
    """Apply Gemini batch-compatible thinking controls to generation config."""

    payload = dict(generation_config)
    if reasoning_effort is not None and reasoning_budget_tokens is not None:
        raise ValueError(
            "Google Gemini batch requests cannot set both reasoning.effort and reasoning.budget_tokens"
        )

    if reasoning_budget_tokens is not None:
        payload["thinking_config"] = {"thinking_budget": reasoning_budget_tokens}
        return payload

    if reasoning_effort is None:
        return payload

    payload["thinking_config"] = {"thinking_level": reasoning_effort.strip().lower()}
    return payload


def _apply_anthropic_request_reasoning(
    *,
    request_params: dict[str, Any],
    model: str,
    reasoning_effort: str | None,
    reasoning_budget_tokens: int | None,
) -> dict[str, Any]:
    """Apply Anthropic reasoning controls using the current model-specific guidance."""

    payload = dict(request_params)
    max_tokens = payload.get("max_tokens")
    if reasoning_budget_tokens is not None and max_tokens is not None and reasoning_budget_tokens >= int(max_tokens):
        raise ValueError("Anthropic reasoning.budget_tokens must be less than max_tokens")

    if reasoning_effort is not None:
        output_config = dict(payload.get("output_config", {}))
        output_config["effort"] = reasoning_effort
        payload["output_config"] = output_config

    # Claude 4.6 models should use adaptive thinking with effort rather than manual budgets.
    if _anthropic_prefers_adaptive_thinking(model):
        if reasoning_budget_tokens is not None:
            raise ValueError(
                "Anthropic Claude 4.6 models should use reasoning.effort without "
                "reasoning.budget_tokens; budget_tokens is deprecated for adaptive thinking"
            )
        if reasoning_effort is not None:
            payload["thinking"] = {"type": "adaptive"}
        return payload

    if reasoning_budget_tokens is not None:
        payload["thinking"] = {
            "type": "enabled",
            "budget_tokens": reasoning_budget_tokens,
        }

    return payload


def _anthropic_prefers_adaptive_thinking(model: str) -> bool:
    """Return whether Anthropic recommends adaptive thinking for this model family."""

    normalized_model = model.strip().lower()
    return normalized_model.startswith("claude-opus-4-6") or normalized_model.startswith(
        "claude-sonnet-4-6"
    )


def _download_google_batch_output(file_name: str, api_key: str) -> bytes:
    """Download the Gemini batch output file bytes from the Generative Language API."""

    url = (
        "https://generativelanguage.googleapis.com/v1beta/"
        f"{file_name}:download?alt=media&key={api_key}"
    )
    with urllib_request.urlopen(url) as response:  # noqa: S310
        return response.read()


def _provider_api_key(config: ModelBatchConfig) -> str:
    """Read the provider API key from the configured environment variable."""

    api_key_env = PROVIDER_API_ENV_VARS.get(config.model.provider)
    if api_key_env is None:
        raise ValueError(f"Unsupported provider: {config.model.provider}")
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise ValueError(f"Missing API key in environment variable {api_key_env}")
    return api_key


def _xai_api_request(
    config: ModelBatchConfig,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    query: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Issue one JSON request to the xAI REST API and decode the response body."""

    url = f"{XAI_API_BASE_URL}{path}"
    normalized_query = {
        key: value
        for key, value in (query or {}).items()
        if value is not None
    }
    if normalized_query:
        url = f"{url}?{urllib_parse.urlencode(normalized_query)}"

    request_body = None
    headers = {
        "Authorization": f"Bearer {_provider_api_key(config)}",
        "Accept": "application/json",
    }
    if payload is not None:
        request_body = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib_request.Request(url=url, data=request_body, method=method, headers=headers)
    try:
        with urllib_request.urlopen(request) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))
    except urllib_error.HTTPError as exc:
        response_text = exc.read().decode("utf-8", errors="replace")
        raise ValueError(f"xAI API request failed ({exc.code} {method} {path}): {response_text}") from exc


def _judge_id(config: ModelBatchConfig) -> str:
    """Return the identifier used for processed model rows."""

    if config.model.id:
        return config.model.id
    return f"{config.model.provider}:{config.model.name}"


def _model_id(config: ModelBatchConfig) -> str:
    """Return a stable model identifier for summaries and metadata."""

    return config.model.id or _judge_id(config)


def _batch_identifier(provider_name: str, batch_object: Any) -> str:
    """Return the provider-specific batch identifier."""

    if provider_name == "google":
        return str(batch_object.name)
    if provider_name == "xai":
        return str(batch_object["batch_id"])
    return str(batch_object.id)


def _batch_status(provider_name: str, batch_object: Any) -> str:
    """Return a normalized status string for the provider batch object."""

    if provider_name == "openai":
        return str(batch_object.status)
    if provider_name == "anthropic":
        return str(batch_object.processing_status)
    if provider_name == "google":
        state = getattr(batch_object, "state", None)
        return getattr(state, "name", getattr(state, "value", str(state)))
    if provider_name == "xai":
        state = batch_object.get("state", {})
        num_pending = int(state.get("num_pending", 0) or 0)
        num_error = int(state.get("num_error", 0) or 0)
        num_cancelled = int(state.get("num_cancelled", 0) or 0)
        if num_pending > 0:
            return "in_progress"
        if num_error > 0 or num_cancelled > 0:
            return "completed_with_errors"
        return "completed"
    raise ValueError(f"Unsupported provider: {provider_name}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a single JSON payload to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Write an iterable of dict rows as JSONL."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=_json_default))
            handle.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dict rows."""

    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _read_processed_csv(path: Path) -> pd.DataFrame:
    """Read one processed CSV, allowing empty files from all-error batches."""

    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def _to_jsonable(value: Any) -> Any:
    """Convert SDK objects into plain JSON-serializable structures."""

    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_jsonable(item) for item in value]
    return value


def _json_default(value: Any) -> Any:
    """Fallback serializer used for metadata and JSONL writes."""

    if isinstance(value, datetime):
        return value.isoformat()
    return _to_jsonable(value)


def _utcnow() -> str:
    """Return the current UTC timestamp in ISO 8601 form."""

    return datetime.now(timezone.utc).isoformat()
