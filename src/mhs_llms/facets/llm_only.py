"""Helpers for unanchored FACETS runs estimated from LLM annotations only."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mhs_llms.config import load_llm_only_facets_config
from mhs_llms.constants import HUMAN_FACETS_RECODE_MAP
from mhs_llms.facets.anchored import _build_facets_judge_map
from mhs_llms.facets.facets import build_facets_frame, build_facets_spec, write_facets_data, write_facets_spec
from mhs_llms.schema import ITEM_NAMES, prompt_letter_to_hf_value
from mhs_llms.utils import recode_responses

LLM_ONLY_REQUIRED_COLUMNS = (
    "comment_id",
    "judge_id",
    *ITEM_NAMES,
)


@dataclass(frozen=True)
class LLMOnlyFacetsOutputs:
    """Paths produced when preparing one unanchored LLM-only FACETS run."""

    facets_data_path: Path
    facets_spec_path: Path


def _prepare_llm_only_annotations(
    annotations: pd.DataFrame,
    recode_like_humans: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert prompt-letter model annotations into FACETS-ready numeric rows."""

    selected = annotations.copy()

    # Keep the original model response categories; only translate letters to numeric scores.
    for item_name in ITEM_NAMES:
        selected[item_name] = selected[item_name].map(
            lambda value: prompt_letter_to_hf_value(item_name, value)
        )
    if recode_like_humans:
        selected = recode_responses(selected, **HUMAN_FACETS_RECODE_MAP)

    selected["judge_label"] = selected["judge_id"]
    selected["judge_id"] = selected["judge_id"].map(_build_facets_judge_map(selected["judge_id"]))
    selected["judge_id"] = selected["judge_id"].astype(int).astype(str)

    judge_mapping = (
        selected[["judge_id", "judge_label"]]
        .drop_duplicates()
        .sort_values("judge_id", key=lambda series: series.astype(int))
        .rename(columns={"judge_label": "external_judge_id"})
    )
    return selected, judge_mapping


def run_llm_only_facets(config_path: Path) -> LLMOnlyFacetsOutputs:
    """Prepare an unanchored FACETS run using model annotations as the full scale source."""

    config = load_llm_only_facets_config(config_path)
    annotation_frames = [
        pd.read_csv(annotation_path, usecols=LLM_ONLY_REQUIRED_COLUMNS)
        for annotation_path in config.annotation_paths
    ]
    annotations = pd.concat(annotation_frames, ignore_index=True)
    prepared_annotations, judge_mapping = _prepare_llm_only_annotations(
        annotations=annotations,
        recode_like_humans=config.recode_like_humans,
    )

    facets_frame = build_facets_frame(prepared_annotations)
    judge_labels = dict(
        zip(
            judge_mapping["judge_id"].tolist(),
            judge_mapping["external_judge_id"].tolist(),
            strict=True,
        )
    )

    facets_run_dir = config.facets_run_dir
    facets_run_dir.mkdir(parents=True, exist_ok=True)

    facets_data_path = facets_run_dir / config.facets_data_filename
    write_facets_data(facets_frame, facets_data_path)

    spec_text = build_facets_spec(
        facets_frame=facets_frame,
        facets_config=config.facets,
        data_filename=config.facets_data_filename,
        score_filename=config.facets_score_filename,
        output_filename=config.facets_output_filename,
        judge_labels=judge_labels,
    )
    facets_spec_path = facets_run_dir / config.facets_spec_filename
    write_facets_spec(spec_text, facets_spec_path)

    return LLMOnlyFacetsOutputs(
        facets_data_path=facets_data_path,
        facets_spec_path=facets_spec_path,
    )
