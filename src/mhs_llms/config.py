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
    bias: Optional[str] = None
    zscore: Optional[str] = None


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
    combined_output_path: Optional[Path] = None


@dataclass(frozen=True)
class AsyncRetryConfig:
    max_attempts: int = 3
    retry_delay_seconds: float = 0.0


@dataclass(frozen=True)
class ModelBatchConfig:
    name: str
    prompt: BatchPromptConfig
    model: BatchModelConfig
    batches: BatchStorageConfig
    subset: str | dict[str, Any] = "reference_set"
    limit: Optional[int] = None
    async_retries: AsyncRetryConfig = field(default_factory=AsyncRetryConfig)


@dataclass(frozen=True)
class LLMFacetsConfig:
    """Config for an LLM-only FACETS run anchored to the human baseline."""

    annotation_paths: tuple[Path, ...]
    comment_scores_path: Path
    item_scores_path: Path
    facets_run_dir: Path
    facets_data_filename: str
    facets_spec_filename: str
    facets_score_filename: str
    facets_output_filename: str
    facets: FacetsConfig


@dataclass(frozen=True)
class SeverityDecompositionConfig:
    """Config for a FACETS run estimating LLM judge-by-item interactions."""

    annotation_paths: tuple[Path, ...]
    comment_scores_path: Path
    item_scores_path: Path
    facets_run_dir: Path
    facets_data_filename: str
    facets_spec_filename: str
    facets_score_filename: str
    facets_output_filename: str
    facets: FacetsConfig


@dataclass(frozen=True)
class OrderEffectConfig:
    """Config for a pooled FACETS run estimating an order-condition effect."""

    original_annotation_paths: tuple[Path, ...]
    reverse_annotation_paths: tuple[Path, ...]
    comment_scores_path: Path
    item_scores_path: Path
    facets_run_dir: Path
    facets_data_filename: str
    facets_spec_filename: str
    facets_score_filename: str
    facets_output_filename: str
    original_order_label: str
    reverse_order_label: str
    facets: FacetsConfig


@dataclass(frozen=True)
class TargetDRFConfig:
    """Config for a FACETS run estimating LLM judge-by-target interactions."""

    annotation_paths: tuple[Path, ...]
    dataset_name: str
    split: str
    min_annotators: int
    agreement_threshold: float
    min_comments_per_target: int
    anchor_targets: bool
    collapse_targets: dict[str, str]
    exclude_targets: tuple[str, ...]
    comment_scores_path: Path
    item_scores_path: Path
    judge_scores_path: Path
    facets_run_dir: Path
    target_labels_path: Path
    facets_data_filename: str
    facets_spec_filename: str
    facets_score_filename: str
    facets_output_filename: str
    facets: FacetsConfig


def _resolve_path(path_value: Union[str, Path]) -> Path:
    """Resolve config paths from the current working directory."""

    path = Path(path_value)
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _normalize_yes_no(value: object, default: str) -> str:
    if value is None:
        return default
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def _load_yaml_config(config_path: Path) -> dict[str, Any]:
    """Read one YAML config file into a plain dictionary."""

    return yaml.safe_load(config_path.resolve().read_text())


def _parse_batch_prompt_config(prompt_data: dict[str, Any]) -> BatchPromptConfig:
    """Parse the shared prompt settings for batch launches."""

    return BatchPromptConfig(
        system_prompt_path=_resolve_path(prompt_data["system_prompt_path"]),
        user_prompt_template=str(prompt_data.get("user_prompt_template", "")),
    )


def _parse_batch_reasoning_config(reasoning_data: dict[str, Any] | None) -> BatchReasoningConfig:
    """Parse provider-specific reasoning settings for one model."""

    reasoning_data = reasoning_data or {}
    return BatchReasoningConfig(
        effort=str(reasoning_data.get("effort")) if reasoning_data.get("effort") else None,
        budget_tokens=(
            int(reasoning_data["budget_tokens"])
            if reasoning_data.get("budget_tokens") is not None
            else None
        ),
    )


def _parse_batch_model_config(model_data: dict[str, Any]) -> BatchModelConfig:
    """Parse one batch model entry from the config file."""

    return BatchModelConfig(
        provider=str(model_data["provider"]).lower(),
        name=str(model_data["name"]),
        id=str(model_data.get("id")) if model_data.get("id") else None,
        max_tokens=(
            int(model_data["max_tokens"]) if model_data.get("max_tokens") is not None else None
        ),
        params=dict(model_data.get("params", {})),
        reasoning=_parse_batch_reasoning_config(model_data.get("reasoning")),
    )


def _parse_async_retry_config(async_data: dict[str, Any] | None) -> AsyncRetryConfig:
    """Parse optional async retry settings shared by one or more models."""

    async_data = async_data or {}
    max_attempts = int(async_data.get("max_attempts", 3))
    retry_delay_seconds = float(async_data.get("retry_delay_seconds", 0.0))
    if max_attempts < 1:
        raise ValueError("async.max_attempts must be at least 1")
    if retry_delay_seconds < 0:
        raise ValueError("async.retry_delay_seconds must be non-negative")
    return AsyncRetryConfig(
        max_attempts=max_attempts,
        retry_delay_seconds=retry_delay_seconds,
    )


def _parse_batch_storage_config(
    batch_data: dict[str, Any], run_dir: Path | None = None
) -> BatchStorageConfig:
    """Parse batch storage paths and filenames from the config file."""

    resolved_run_dir = run_dir if run_dir is not None else _resolve_path(batch_data["run_dir"])
    combined_output_path = batch_data.get("combined_output_path")
    return BatchStorageConfig(
        run_dir=resolved_run_dir,
        request_manifest_filename=str(
            batch_data.get("request_manifest_filename", "request_manifest.jsonl")
        ),
        provider_requests_filename=str(
            batch_data.get("provider_requests_filename", "provider_requests.jsonl")
        ),
        batch_metadata_filename=str(batch_data.get("batch_metadata_filename", "batch_job.json")),
        raw_results_filename=str(batch_data.get("raw_results_filename", "raw_results.jsonl")),
        processed_records_filename=str(
            batch_data.get("processed_records_filename", "processed_records.jsonl")
        ),
        processed_csv_filename=str(
            batch_data.get("processed_csv_filename", "processed_records.csv")
        ),
        errors_filename=str(batch_data.get("errors_filename", "processing_errors.jsonl")),
        combined_output_path=(
            _resolve_path(combined_output_path) if combined_output_path is not None else None
        ),
    )


def _parse_facets_config(
    facets_data: dict[str, Any],
    default_model: str = "?, ?, #, R",
) -> FacetsConfig:
    """Parse common FACETS control settings from config data."""

    return FacetsConfig(
        title=facets_data["title"],
        model=facets_data.get("model", default_model),
        noncenter=int(facets_data.get("noncenter", 1)),
        positive=int(facets_data.get("positive", 1)),
        arrange=str(facets_data.get("arrange", "N")),
        subset_detection=_normalize_yes_no(facets_data.get("subset_detection"), "No"),
        delements=tuple(facets_data.get("delements", ["1N", "2N", "3N"])),
        csv=str(facets_data.get("csv", "Tab")),
        bias=str(facets_data["bias"]) if facets_data.get("bias") is not None else None,
        zscore=str(facets_data["zscore"]) if facets_data.get("zscore") is not None else None,
    )


def _build_model_batch_config(
    *,
    name: str,
    prompt: BatchPromptConfig,
    model: BatchModelConfig,
    batches: BatchStorageConfig,
    subset: str | dict[str, Any],
    limit: Optional[int],
    async_retries: AsyncRetryConfig,
) -> ModelBatchConfig:
    """Build one fully resolved per-model batch config."""

    return ModelBatchConfig(
        name=name,
        subset=subset,
        limit=limit,
        prompt=prompt,
        model=model,
        batches=batches,
        async_retries=async_retries,
    )


def _clone_batch_storage_config(
    batch_config: BatchStorageConfig, run_dir: Path
) -> BatchStorageConfig:
    """Copy the shared storage settings while swapping in a model-specific run dir."""

    return BatchStorageConfig(
        run_dir=run_dir,
        request_manifest_filename=batch_config.request_manifest_filename,
        provider_requests_filename=batch_config.provider_requests_filename,
        batch_metadata_filename=batch_config.batch_metadata_filename,
        raw_results_filename=batch_config.raw_results_filename,
        processed_records_filename=batch_config.processed_records_filename,
        processed_csv_filename=batch_config.processed_csv_filename,
        errors_filename=batch_config.errors_filename,
        combined_output_path=batch_config.combined_output_path,
    )


def load_human_baseline_config(config_path: Path) -> HumanBaselineConfig:
    config_path = config_path.resolve()
    data = _load_yaml_config(config_path)

    output = OutputConfig(
        run_dir=_resolve_path(data["output"]["run_dir"]),
        facets_run_dir=_resolve_path(data["output"]["facets_run_dir"]),
        comment_ids_filename=data["output"]["comment_ids_filename"],
        cleaned_annotations_filename=data["output"]["cleaned_annotations_filename"],
        facets_data_filename=data["output"]["facets_data_filename"],
        facets_spec_filename=data["output"]["facets_spec_filename"],
        facets_score_filename=data["output"]["facets_score_filename"],
        facets_output_filename=data["output"]["facets_output_filename"],
    )
    facets = _parse_facets_config(data["facets"])
    return HumanBaselineConfig(
        output=output,
        facets=facets,
    )


def load_model_batch_config(config_path: Path) -> ModelBatchConfig:
    """Load the config for launching and processing a provider batch job."""

    model_configs = load_model_batch_configs(config_path)
    if len(model_configs) != 1:
        raise ValueError(
            "Config defines more than one model; use load_model_batch_configs() for multi-model runs"
        )
    return model_configs[0]


def load_model_batch_configs(config_path: Path) -> tuple[ModelBatchConfig, ...]:
    """Load one or more provider batch configs from a shared experiment YAML."""

    config_path = config_path.resolve()
    data = _load_yaml_config(config_path)
    prompt = _parse_batch_prompt_config(data["prompt"])
    subset = data.get("subset", "reference_set")
    limit = int(data["limit"]) if data.get("limit") is not None else None
    async_retries = _parse_async_retry_config(data.get("async"))

    # Legacy configs keep their exact run_dir. Multi-model configs treat run_dir as
    # a base directory and resolve one subdirectory per model id.
    if data.get("model") is not None:
        model = _parse_batch_model_config(data["model"])
        batches = _parse_batch_storage_config(data["batches"])
        return (
            _build_model_batch_config(
                name=str(data["name"]),
                prompt=prompt,
                model=model,
                batches=batches,
                subset=subset,
                limit=limit,
                async_retries=async_retries,
            ),
        )

    if data.get("models") is None:
        raise ValueError("Batch config must define either 'model' or 'models'")

    shared_batches = _parse_batch_storage_config(data["batches"])
    model_configs: list[ModelBatchConfig] = []
    for model_entry in data["models"]:
        model = _parse_batch_model_config(model_entry)
        if not model.id:
            raise ValueError("Each model entry must define an id when using the 'models' list")
        model_batches = _clone_batch_storage_config(
            shared_batches,
            run_dir=(shared_batches.run_dir / model.id).resolve(),
        )
        model_configs.append(
            _build_model_batch_config(
                name=model.id,
                prompt=prompt,
                model=model,
                batches=model_batches,
                subset=subset,
                limit=limit,
                async_retries=async_retries,
            )
        )
    return tuple(model_configs)


def load_llm_facets_config(config_path: Path) -> LLMFacetsConfig:
    """Load the config for an LLM FACETS run linked to human anchors."""

    config_path = config_path.resolve()
    data = yaml.safe_load(config_path.read_text())

    facets = _parse_facets_config(data["facets"])

    annotation_values = data["annotations"].get("paths")
    if annotation_values is None:
        annotation_values = [data["annotations"]["path"]]

    return LLMFacetsConfig(
        annotation_paths=tuple(_resolve_path(path_value) for path_value in annotation_values),
        comment_scores_path=_resolve_path(data["anchors"]["comment_scores_path"]),
        item_scores_path=_resolve_path(data["anchors"]["item_scores_path"]),
        facets_run_dir=_resolve_path(data["output"]["facets_run_dir"]),
        facets_data_filename=str(data["output"].get("facets_data_filename", "llm_facets_data.tsv")),
        facets_spec_filename=str(data["output"].get("facets_spec_filename", "llm_facets_spec.txt")),
        facets_score_filename=str(
            data["output"].get("facets_score_filename", "llm_facets_scores.txt")
        ),
        facets_output_filename=str(
            data["output"].get("facets_output_filename", "llm_facets_output.txt")
        ),
        facets=facets,
    )


def load_severity_decomposition_config(config_path: Path) -> SeverityDecompositionConfig:
    """Load config for the LLM item-dependent severity FACETS prep run."""

    config_path = config_path.resolve()
    data = yaml.safe_load(config_path.read_text())
    annotation_values = data["annotations"].get("paths")
    if annotation_values is None:
        annotation_values = [data["annotations"]["path"]]

    return SeverityDecompositionConfig(
        annotation_paths=tuple(_resolve_path(path_value) for path_value in annotation_values),
        comment_scores_path=_resolve_path(data["anchors"]["comment_scores_path"]),
        item_scores_path=_resolve_path(data["anchors"]["item_scores_path"]),
        facets_run_dir=_resolve_path(data["output"]["facets_run_dir"]),
        facets_data_filename=str(
            data["output"].get("facets_data_filename", "severity_decomposition_data.tsv")
        ),
        facets_spec_filename=str(
            data["output"].get("facets_spec_filename", "severity_decomposition_spec.txt")
        ),
        facets_score_filename=str(
            data["output"].get("facets_score_filename", "severity_decomposition_scores.txt")
        ),
        facets_output_filename=str(
            data["output"].get("facets_output_filename", "severity_decomposition_output.txt")
        ),
        facets=_parse_facets_config(data["facets"], default_model="?, ?B, #B, R"),
    )


def load_order_effect_config(config_path: Path) -> OrderEffectConfig:
    """Load config for the pooled original/reverse order FACETS run."""

    config_path = config_path.resolve()
    data = yaml.safe_load(config_path.read_text())
    annotations = data["annotations"]
    original_values = annotations.get("original_paths")
    if original_values is None:
        original_values = [annotations["original_path"]]
    reverse_values = annotations.get("reverse_paths")
    if reverse_values is None:
        reverse_values = [annotations["reverse_path"]]

    order_data = data.get("order", {})
    return OrderEffectConfig(
        original_annotation_paths=tuple(_resolve_path(path_value) for path_value in original_values),
        reverse_annotation_paths=tuple(_resolve_path(path_value) for path_value in reverse_values),
        comment_scores_path=_resolve_path(data["anchors"]["comment_scores_path"]),
        item_scores_path=_resolve_path(data["anchors"]["item_scores_path"]),
        facets_run_dir=_resolve_path(data["output"]["facets_run_dir"]),
        facets_data_filename=str(data["output"].get("facets_data_filename", "order_effect_data.tsv")),
        facets_spec_filename=str(data["output"].get("facets_spec_filename", "order_effect_spec.txt")),
        facets_score_filename=str(data["output"].get("facets_score_filename", "order_effect_scores.txt")),
        facets_output_filename=str(data["output"].get("facets_output_filename", "order_effect_output.txt")),
        original_order_label=str(order_data.get("original_label", "original_order")),
        reverse_order_label=str(order_data.get("reverse_label", "reverse_order")),
        facets=_parse_facets_config(data["facets"], default_model="?, ?, #, ?, R"),
    )


def load_target_drf_config(config_path: Path) -> TargetDRFConfig:
    """Load config for the LLM target-identity DRF FACETS prep run."""

    config_path = config_path.resolve()
    data = yaml.safe_load(config_path.read_text())
    annotation_values = data["annotations"].get("paths")
    if annotation_values is None:
        annotation_values = [data["annotations"]["path"]]

    targets = data["targets"]
    return TargetDRFConfig(
        annotation_paths=tuple(_resolve_path(path_value) for path_value in annotation_values),
        dataset_name=str(targets.get("dataset_name", "ucberkeley-dlab/measuring-hate-speech")),
        split=str(targets.get("split", "train")),
        min_annotators=int(targets.get("min_annotators", 4)),
        agreement_threshold=float(targets.get("agreement_threshold", 0.75)),
        min_comments_per_target=int(targets.get("min_comments_per_target", 1)),
        anchor_targets=bool(targets.get("anchor_targets", False)),
        collapse_targets=dict(targets.get("collapse", {})),
        exclude_targets=tuple(str(value) for value in targets.get("exclude", [])),
        comment_scores_path=_resolve_path(data["anchors"]["comment_scores_path"]),
        item_scores_path=_resolve_path(data["anchors"]["item_scores_path"]),
        judge_scores_path=_resolve_path(data["anchors"]["judge_scores_path"]),
        facets_run_dir=_resolve_path(data["output"]["facets_run_dir"]),
        target_labels_path=_resolve_path(data["output"]["target_labels_path"]),
        facets_data_filename=str(data["output"].get("facets_data_filename", "target_drf_data.tsv")),
        facets_spec_filename=str(data["output"].get("facets_spec_filename", "target_drf_spec.txt")),
        facets_score_filename=str(
            data["output"].get("facets_score_filename", "target_drf_scores.txt")
        ),
        facets_output_filename=str(
            data["output"].get("facets_output_filename", "target_drf_output.txt")
        ),
        facets=_parse_facets_config(data["facets"], default_model="?, ?B, #, ?B, R"),
    )
