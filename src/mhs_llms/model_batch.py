"""Batch launch and processing helpers for provider-hosted model jobs."""

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable

from anthropic import Anthropic
from google import genai
from loguru import logger
from openai import OpenAI
import pandas as pd

from mhs_llms.config import ModelBatchConfig, load_model_batch_config
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
    },
}

SUCCESSFUL_RESULT_STATUSES = {
    "openai": {"completed"},
    "anthropic": {"ended"},
    "google": {"JOB_STATE_SUCCEEDED", "JOB_STATE_PARTIALLY_SUCCEEDED"},
}


@dataclass(frozen=True)
class LaunchBatchOutputs:
    """Paths and ids created when a provider batch job is launched."""

    run_dir: Path
    batch_metadata_path: Path
    batch_id: str
    status: str


@dataclass(frozen=True)
class ProcessBatchOutputs:
    """Paths and status emitted when processing a provider batch job."""

    run_dir: Path
    batch_metadata_path: Path
    status: str
    raw_results_path: Path | None
    processed_csv_path: Path | None


def launch_model_batch(config_path: Path) -> LaunchBatchOutputs:
    """Create a provider batch job from the configured MHS comment slice."""

    config = load_model_batch_config(config_path)
    run_dir = config.output.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    request_manifest_path = run_dir / config.output.request_manifest_filename
    provider_requests_path = run_dir / config.output.provider_requests_filename
    batch_metadata_path = run_dir / config.output.batch_metadata_filename

    comments = _load_batch_comments(config)
    logger.info("Preparing {} comments for {} batch launch", len(comments), config.provider.name)

    request_manifest, provider_requests = _build_requests(config=config, comments=comments)
    _write_jsonl(request_manifest_path, request_manifest)
    _write_jsonl(provider_requests_path, provider_requests)
    logger.info("Wrote request manifest to {}", request_manifest_path)

    batch_object = _create_provider_batch(
        config=config,
        provider_requests_path=provider_requests_path,
        provider_requests=provider_requests,
    )
    batch_id = _batch_identifier(config.provider.name, batch_object)
    status = _batch_status(config.provider.name, batch_object)

    metadata = {
        "config_path": str(config_path.resolve()),
        "name": config.name,
        "provider": config.provider.name,
        "model": config.provider.model,
        "judge_id": _judge_id(config),
        "batch_id": batch_id,
        "status": status,
        "request_count": len(request_manifest),
        "submitted_at": _utcnow(),
        "provider_batch": _to_jsonable(batch_object),
    }
    _write_json(batch_metadata_path, metadata)
    logger.info("Launched {} batch {} with status {}", config.provider.name, batch_id, status)

    return LaunchBatchOutputs(
        run_dir=run_dir,
        batch_metadata_path=batch_metadata_path,
        batch_id=batch_id,
        status=status,
    )


def process_model_batch(config_path: Path) -> ProcessBatchOutputs:
    """Refresh batch status and download/process results once complete."""

    config = load_model_batch_config(config_path)
    run_dir = config.output.run_dir
    batch_metadata_path = run_dir / config.output.batch_metadata_filename
    request_manifest_path = run_dir / config.output.request_manifest_filename
    raw_results_path = run_dir / config.output.raw_results_filename
    processed_records_path = run_dir / config.output.processed_records_filename
    processed_csv_path = run_dir / config.output.processed_csv_filename
    errors_path = run_dir / config.output.errors_filename

    metadata = json.loads(batch_metadata_path.read_text())
    batch_id = str(metadata["batch_id"])
    logger.info("Checking {} batch {}", config.provider.name, batch_id)

    batch_object = _retrieve_provider_batch(config=config, batch_id=batch_id)
    status = _batch_status(config.provider.name, batch_object)
    metadata["status"] = status
    metadata["last_checked_at"] = _utcnow()
    metadata["provider_batch"] = _to_jsonable(batch_object)
    _write_json(batch_metadata_path, metadata)
    logger.info("Current {} batch status: {}", config.provider.name, status)

    if status not in TERMINAL_BATCH_STATUSES[config.provider.name]:
        logger.info("Batch {} is still running; no result download yet", batch_id)
        return ProcessBatchOutputs(
            run_dir=run_dir,
            batch_metadata_path=batch_metadata_path,
            status=status,
            raw_results_path=None,
            processed_csv_path=None,
        )

    if status not in SUCCESSFUL_RESULT_STATUSES[config.provider.name]:
        logger.warning("Batch {} finished in terminal state {} with no downloadable results", batch_id, status)
        return ProcessBatchOutputs(
            run_dir=run_dir,
            batch_metadata_path=batch_metadata_path,
            status=status,
            raw_results_path=None,
            processed_csv_path=None,
        )

    raw_entries = _download_provider_results(config=config, batch_object=batch_object)
    _write_jsonl(raw_results_path, raw_entries)
    logger.info("Stored {} raw batch results at {}", len(raw_entries), raw_results_path)

    manifest_rows = _read_jsonl(request_manifest_path)
    manifest_by_custom_id = {row["custom_id"]: row for row in manifest_rows}

    extractor = _result_extractor(config.provider.name)
    processed_rows: list[dict[str, Any]] = []
    processing_errors: list[dict[str, Any]] = []

    for entry in raw_entries:
        custom_id, response_text, response_metadata, response_error = extractor(entry)
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
                    "provider": config.provider.name,
                    "model": config.provider.model,
                    "batch_id": batch_id,
                    "custom_id": custom_id,
                    "provider_response": response_metadata,
                },
            )
            processed_rows.append(annotation_record_to_row(record))
        except Exception as exc:  # noqa: BLE001
            processing_errors.append(
                {
                    "custom_id": custom_id,
                    "comment_id": manifest_row["comment_id"],
                    "error": str(exc),
                    "response_text": response_text,
                    "raw_result": entry,
                }
            )

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
    _write_json(batch_metadata_path, metadata)

    return ProcessBatchOutputs(
        run_dir=run_dir,
        batch_metadata_path=batch_metadata_path,
        status=status,
        raw_results_path=raw_results_path,
        processed_csv_path=processed_csv_path,
    )


def _load_batch_comments(config: ModelBatchConfig) -> list[dict[str, Any]]:
    """Load one text row per comment for the configured evaluation slice."""

    dataset = load_mhs_dataframe(
        dataset_name=config.dataset.name,
        split=config.dataset.split,
        config_name=config.dataset.config_name,
    )
    comment_ids = None
    if config.selection.comment_ids_path is not None:
        comment_ids_frame = pd.read_csv(config.selection.comment_ids_path)
        comment_ids = comment_ids_frame["comment_id"].astype(int).tolist()

    comments = build_comment_frame(dataset, comment_ids=comment_ids, limit=config.selection.limit)
    return comments.to_dict(orient="records")


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
        user_prompt = config.prompt.user_prompt_template.format(comment_text=comment_text)
        request_manifest.append(
            {
                "custom_id": custom_id,
                "comment_id": comment_id,
                "text": comment_text,
            }
        )
        provider_requests.append(
            _provider_request(
                provider_name=config.provider.name,
                custom_id=custom_id,
                model=config.provider.model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model_params=config.provider.model_params,
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
    model_params: dict[str, Any],
    comment_id: int,
) -> dict[str, Any]:
    """Build one provider-specific request payload."""

    if provider_name == "openai":
        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                **model_params,
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            },
        }
    if provider_name == "anthropic":
        return {
            "custom_id": custom_id,
            "params": {
                **model_params,
                "model": model,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}],
            },
        }
    if provider_name == "google":
        return {
            "model": model,
            "contents": user_prompt,
            "metadata": {
                "custom_id": custom_id,
                "comment_id": str(comment_id),
            },
            "config": {
                **model_params,
                "system_instruction": system_prompt,
            },
        }
    raise ValueError(f"Unsupported provider: {provider_name}")


def _create_provider_batch(
    config: ModelBatchConfig,
    provider_requests_path: Path,
    provider_requests: list[dict[str, Any]],
) -> Any:
    """Submit the prepared requests to the configured provider."""

    if config.provider.name == "openai":
        client = OpenAI(api_key=_provider_api_key(config))
        with provider_requests_path.open("rb") as handle:
            input_file = client.files.create(file=handle, purpose="batch")
        completion_window = str(config.provider.batch_params.get("completion_window", "24h"))
        return client.batches.create(
            completion_window=completion_window,
            endpoint="/v1/chat/completions",
            input_file_id=input_file.id,
            metadata={"job_name": config.name},
        )
    if config.provider.name == "anthropic":
        client = Anthropic(api_key=_provider_api_key(config))
        return client.messages.batches.create(requests=provider_requests)
    if config.provider.name == "google":
        client = genai.Client(api_key=_provider_api_key(config))
        batch_config = config.provider.batch_params or {"display_name": config.name}
        return client.batches.create(
            model=config.provider.model,
            src=provider_requests,
            config=batch_config,
        )
    raise ValueError(f"Unsupported provider: {config.provider.name}")


def _retrieve_provider_batch(config: ModelBatchConfig, batch_id: str) -> Any:
    """Fetch the latest provider-side status for an existing batch."""

    if config.provider.name == "openai":
        client = OpenAI(api_key=_provider_api_key(config))
        return client.batches.retrieve(batch_id)
    if config.provider.name == "anthropic":
        client = Anthropic(api_key=_provider_api_key(config))
        return client.messages.batches.retrieve(batch_id)
    if config.provider.name == "google":
        client = genai.Client(api_key=_provider_api_key(config))
        return client.batches.get(name=batch_id)
    raise ValueError(f"Unsupported provider: {config.provider.name}")


def _download_provider_results(config: ModelBatchConfig, batch_object: Any) -> list[dict[str, Any]]:
    """Download provider results and convert them into JSON-serializable entries."""

    if config.provider.name == "openai":
        client = OpenAI(api_key=_provider_api_key(config))
        output_file_id = getattr(batch_object, "output_file_id", None)
        if output_file_id is None:
            raise ValueError("OpenAI batch completed without an output_file_id")
        raw_text = client.files.content(output_file_id).text
        return [json.loads(line) for line in raw_text.splitlines() if line.strip()]

    if config.provider.name == "anthropic":
        client = Anthropic(api_key=_provider_api_key(config))
        return [_to_jsonable(entry) for entry in client.messages.batches.results(batch_object.id)]

    if config.provider.name == "google":
        responses = getattr(getattr(batch_object, "dest", None), "inlined_responses", None)
        if responses is None:
            raise ValueError("Google batch completed without inline responses")
        return [_to_jsonable(entry) for entry in responses]

    raise ValueError(f"Unsupported provider: {config.provider.name}")


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
    return custom_id, _coerce_anthropic_content_to_text(message.get("content", [])), message, None


def _extract_google_result(
    entry: dict[str, Any],
) -> tuple[str, str, dict[str, Any], str | None]:
    """Extract text content and status metadata from one Google batch result."""

    metadata = entry.get("metadata", {})
    custom_id = str(metadata.get("custom_id"))
    error = entry.get("error")
    if error is not None:
        return custom_id, "", metadata, json.dumps(error, sort_keys=True)

    response = entry.get("response", {})
    return custom_id, _coerce_google_response_to_text(response), response, None


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


def _parse_response_json(response_text: str) -> dict[str, Any]:
    """Parse the provider text output into the expected JSON response object."""

    cleaned_text = response_text.strip()
    if cleaned_text.startswith("```"):
        cleaned_text = cleaned_text.strip("`")
        if cleaned_text.startswith("json"):
            cleaned_text = cleaned_text[4:]
        cleaned_text = cleaned_text.strip()

    payload = json.loads(cleaned_text)
    if not isinstance(payload, dict):
        raise ValueError("Model output must decode to a JSON object")
    return payload


def _provider_api_key(config: ModelBatchConfig) -> str:
    """Read the provider API key from the configured environment variable."""

    api_key = os.environ.get(config.provider.api_key_env)
    if not api_key:
        raise ValueError(
            f"Missing API key in environment variable {config.provider.api_key_env}"
        )
    return api_key


def _judge_id(config: ModelBatchConfig) -> str:
    """Return the identifier used for processed model rows."""

    if config.provider.judge_id:
        return config.provider.judge_id
    return f"{config.provider.name}:{config.provider.model}"


def _batch_identifier(provider_name: str, batch_object: Any) -> str:
    """Return the provider-specific batch identifier."""

    if provider_name == "google":
        return str(batch_object.name)
    return str(batch_object.id)


def _batch_status(provider_name: str, batch_object: Any) -> str:
    """Return a normalized status string for the provider batch object."""

    if provider_name == "openai":
        return str(batch_object.status)
    if provider_name == "anthropic":
        return str(batch_object.processing_status)
    if provider_name == "google":
        state = getattr(batch_object, "state", None)
        return getattr(state, "value", str(state))
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
