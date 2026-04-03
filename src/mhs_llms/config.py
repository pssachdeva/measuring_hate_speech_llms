"""Configuration loading for reproducible experiment stages."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml


@dataclass(frozen=True)
class OutputConfig:
    run_dir: Path
    facets_run_dir: Path
    comment_ids_filename: str
    cleaned_annotations_filename: str
    facets_data_filename: str
    facets_spec_filename: str
    facets_score_filename: str
    facets_output_filename: str


@dataclass(frozen=True)
class FacetsConfig:
    title: str
    model: str = "?, ?, #, R"
    noncenter: int = 1
    positive: int = 1
    arrange: str = "N"
    subset_detection: str = "No"
    delements: tuple[str, ...] = ("1N", "2N", "3N")
    csv: str = "Tab"


@dataclass(frozen=True)
class HumanBaselineConfig:
    output: OutputConfig
    facets: FacetsConfig


@dataclass(frozen=True)
class BatchPromptConfig:
    system_prompt_path: Path
    user_prompt_template: str = "SOCIAL MEDIA COMMENT:\n{comment_text}"


@dataclass(frozen=True)
class BatchReasoningConfig:
    effort: Optional[str] = None
    budget_tokens: Optional[int] = None


@dataclass(frozen=True)
class BatchModelConfig:
    provider: str
    name: str
    id: Optional[str] = None
    max_tokens: Optional[int] = None
    params: dict[str, Any] = field(default_factory=dict)
    reasoning: BatchReasoningConfig = field(default_factory=BatchReasoningConfig)


@dataclass(frozen=True)
class BatchStorageConfig:
    run_dir: Path
    request_manifest_filename: str
    provider_requests_filename: str
    batch_metadata_filename: str
    raw_results_filename: str
    processed_records_filename: str
    processed_csv_filename: str
    errors_filename: str


@dataclass(frozen=True)
class ModelBatchConfig:
    name: str
    prompt: BatchPromptConfig
    model: BatchModelConfig
    batches: BatchStorageConfig
    subset: str = "reference_set"
    limit: Optional[int] = None


def _resolve_path(path_value: Union[str, Path], base_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _normalize_yes_no(value: object, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def load_human_baseline_config(config_path: Path) -> HumanBaselineConfig:
    config_path = config_path.resolve()
    data = yaml.safe_load(config_path.read_text())
    base_dir = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent

    output = OutputConfig(
        run_dir=_resolve_path(data["output"]["run_dir"], base_dir),
        facets_run_dir=_resolve_path(data["output"]["facets_run_dir"], base_dir),
        comment_ids_filename=data["output"]["comment_ids_filename"],
        cleaned_annotations_filename=data["output"]["cleaned_annotations_filename"],
        facets_data_filename=data["output"]["facets_data_filename"],
        facets_spec_filename=data["output"]["facets_spec_filename"],
        facets_score_filename=data["output"]["facets_score_filename"],
        facets_output_filename=data["output"]["facets_output_filename"],
    )
    facets = FacetsConfig(
        title=data["facets"]["title"],
        model=data["facets"].get("model", "?, ?, #, R"),
        noncenter=int(data["facets"].get("noncenter", 1)),
        positive=int(data["facets"].get("positive", 1)),
        arrange=str(data["facets"].get("arrange", "N")),
        subset_detection=_normalize_yes_no(data["facets"].get("subset_detection"), "No"),
        delements=tuple(data["facets"].get("delements", ["1N", "2N", "3N"])),
        csv=str(data["facets"].get("csv", "Tab")),
    )
    return HumanBaselineConfig(
        output=output,
        facets=facets,
    )


def load_model_batch_config(config_path: Path) -> ModelBatchConfig:
    """Load the config for launching and processing a provider batch job."""

    config_path = config_path.resolve()
    data = yaml.safe_load(config_path.read_text())
    base_dir = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent

    prompt = BatchPromptConfig(
        system_prompt_path=_resolve_path(data["prompt"]["system_prompt_path"], base_dir),
        user_prompt_template=str(data["prompt"].get("user_prompt_template", "")),
    )
    model = BatchModelConfig(
        provider=str(data["model"]["provider"]).lower(),
        name=str(data["model"]["name"]),
        id=str(data["model"].get("id")) if data["model"].get("id") else None,
        max_tokens=(
            int(data["model"]["max_tokens"]) if data["model"].get("max_tokens") is not None else None
        ),
        params=dict(data["model"].get("params", {})),
        reasoning=BatchReasoningConfig(
            effort=(
                str(data["model"].get("reasoning", {}).get("effort"))
                if data["model"].get("reasoning", {}).get("effort")
                else None
            ),
            budget_tokens=(
                int(data["model"].get("reasoning", {}).get("budget_tokens"))
                if data["model"].get("reasoning", {}).get("budget_tokens") is not None
                else None
            ),
        ),
    )
    batches = BatchStorageConfig(
        run_dir=_resolve_path(data["batches"]["run_dir"], base_dir),
        request_manifest_filename=str(
            data["batches"].get("request_manifest_filename", "request_manifest.jsonl")
        ),
        provider_requests_filename=str(
            data["batches"].get("provider_requests_filename", "provider_requests.jsonl")
        ),
        batch_metadata_filename=str(
            data["batches"].get("batch_metadata_filename", "batch_job.json")
        ),
        raw_results_filename=str(data["batches"].get("raw_results_filename", "raw_results.jsonl")),
        processed_records_filename=str(
            data["batches"].get("processed_records_filename", "processed_records.jsonl")
        ),
        processed_csv_filename=str(
            data["batches"].get("processed_csv_filename", "processed_records.csv")
        ),
        errors_filename=str(data["batches"].get("errors_filename", "processing_errors.jsonl")),
    )
    return ModelBatchConfig(
        name=str(data["name"]),
        subset=str(data.get("subset", "reference_set")),
        limit=int(data["limit"]) if data.get("limit") is not None else None,
        prompt=prompt,
        model=model,
        batches=batches,
    )
