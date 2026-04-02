"""Configuration loading for reproducible experiment stages."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    split: str = "train"
    config_name: Optional[str] = None


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
    dataset: DatasetConfig
    output: OutputConfig
    facets: FacetsConfig


@dataclass(frozen=True)
class BatchPromptConfig:
    system_prompt_path: Path
    user_prompt_template: str = "SOCIAL MEDIA COMMENT:\n{comment_text}"


@dataclass(frozen=True)
class BatchSelectionConfig:
    comment_ids_path: Optional[Path] = None
    limit: Optional[int] = None


@dataclass(frozen=True)
class BatchProviderConfig:
    name: str
    api_key_env: str
    model: str
    judge_id: Optional[str] = None
    model_params: dict[str, Any] = field(default_factory=dict)
    batch_params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchOutputConfig:
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
    dataset: DatasetConfig
    prompt: BatchPromptConfig
    selection: BatchSelectionConfig
    provider: BatchProviderConfig
    output: BatchOutputConfig


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

    dataset = DatasetConfig(**data["dataset"])
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
        dataset=dataset,
        output=output,
        facets=facets,
    )


def load_model_batch_config(config_path: Path) -> ModelBatchConfig:
    """Load the config for launching and processing a provider batch job."""

    config_path = config_path.resolve()
    data = yaml.safe_load(config_path.read_text())
    base_dir = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent

    dataset_data = data["dataset"].copy()
    comment_ids_path = dataset_data.pop("comment_ids_path", None)
    limit = dataset_data.pop("limit", None)
    dataset = DatasetConfig(**dataset_data)

    prompt = BatchPromptConfig(
        system_prompt_path=_resolve_path(data["prompt"]["system_prompt_path"], base_dir),
        user_prompt_template=str(
            data["prompt"].get("user_prompt_template", "SOCIAL MEDIA COMMENT:\n{comment_text}")
        ),
    )
    selection = BatchSelectionConfig(
        comment_ids_path=_resolve_path(comment_ids_path, base_dir) if comment_ids_path else None,
        limit=int(limit) if limit is not None else None,
    )
    provider = BatchProviderConfig(
        name=str(data["provider"]["name"]).lower(),
        api_key_env=str(data["provider"]["api_key_env"]),
        model=str(data["provider"]["model"]),
        judge_id=str(data["provider"].get("judge_id")) if data["provider"].get("judge_id") else None,
        model_params=dict(data["provider"].get("model_params", {})),
        batch_params=dict(data["provider"].get("batch_params", {})),
    )
    output = BatchOutputConfig(
        run_dir=_resolve_path(data["output"]["run_dir"], base_dir),
        request_manifest_filename=str(
            data["output"].get("request_manifest_filename", "request_manifest.jsonl")
        ),
        provider_requests_filename=str(
            data["output"].get("provider_requests_filename", "provider_requests.jsonl")
        ),
        batch_metadata_filename=str(
            data["output"].get("batch_metadata_filename", "batch_job.json")
        ),
        raw_results_filename=str(data["output"].get("raw_results_filename", "raw_results.jsonl")),
        processed_records_filename=str(
            data["output"].get("processed_records_filename", "processed_records.jsonl")
        ),
        processed_csv_filename=str(
            data["output"].get("processed_csv_filename", "processed_records.csv")
        ),
        errors_filename=str(data["output"].get("errors_filename", "processing_errors.jsonl")),
    )
    return ModelBatchConfig(
        name=str(data["name"]),
        dataset=dataset,
        prompt=prompt,
        selection=selection,
        provider=provider,
        output=output,
    )
