"""Helpers for linked FACETS runs anchored to the human baseline."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from mhs_llms.config import load_llm_facets_config
from mhs_llms.constants import FACETS_LLM_JUDGE_IDS, FACETS_LLM_JUDGE_ID_START, HUMAN_FACETS_RECODE_MAP
from mhs_llms.facets.facets import build_facets_frame, build_facets_spec, write_facets_data, write_facets_spec
from mhs_llms.facets.postprocess import load_measure_anchors
from mhs_llms.schema import ITEM_NAMES, prompt_letter_to_hf_value
from mhs_llms.utils import recode_responses


@dataclass(frozen=True)
class AnchoredLLMFacetsOutputs:
    """Paths produced when preparing one linked LLM FACETS run."""

    facets_data_path: Path
    facets_spec_path: Path


def _build_facets_judge_map(judge_ids: pd.Series) -> dict[str, int]:
    """Extend the reserved FACETS judge map with deterministic ids for new judges."""

    judge_map = dict(FACETS_LLM_JUDGE_IDS)
    next_judge_id = max([FACETS_LLM_JUDGE_ID_START - 1, *judge_map.values()]) + 1

    for judge_id in sorted(judge_ids.dropna().astype(str).unique().tolist()):
        if judge_id in judge_map:
            continue
        # Assign unseen judges in sorted order so reruns over the same inputs stay stable.
        judge_map[judge_id] = next_judge_id
        next_judge_id += 1

    return judge_map


def _prepare_llm_annotations(
    annotations: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter, score, and map model annotations into FACETS-ready numeric rows."""

    selected = annotations.copy()

    # Convert prompt-letter responses back into the human numeric coding space.
    for item_name in ITEM_NAMES:
        selected[item_name] = selected[item_name].map(
            lambda value: prompt_letter_to_hf_value(item_name, value)
        )

    # Apply the same collapsing used in the human baseline so the linked run shares categories.
    selected = recode_responses(selected, **HUMAN_FACETS_RECODE_MAP)

    # Replace external string judge ids with reserved FACETS-safe numeric ids.
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


def run_anchored_llm_facets(config_path: Path) -> AnchoredLLMFacetsOutputs:
    """Prepare an LLM-only FACETS run linked to the human baseline scale."""

    config = load_llm_facets_config(config_path)
    annotation_frames = [pd.read_csv(annotation_path) for annotation_path in config.annotation_paths]
    annotations = pd.concat(annotation_frames, ignore_index=True)
    prepared_annotations, judge_mapping = _prepare_llm_annotations(annotations=annotations)

    facets_frame = build_facets_frame(prepared_annotations)

    comment_anchors = load_measure_anchors(
        score_path=config.comment_scores_path,
        key_column="facet_id",
        measure_column="measure",
    )
    item_anchors = load_measure_anchors(
        score_path=config.item_scores_path,
        key_column="facet_label",
        measure_column="measure",
    )

    comment_labels = {str(comment_id): str(comment_id) for comment_id in facets_frame["comment_id"].astype(int)}
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

    relevant_comment_anchors = {
        str(comment_id): comment_anchors[str(comment_id)]
        for comment_id in facets_frame["comment_id"].astype(int).astype(str).unique().tolist()
    }
    relevant_item_anchors = {item_name: item_anchors[item_name] for item_name in ITEM_NAMES}

    spec_text = build_facets_spec(
        facets_frame=facets_frame,
        facets_config=config.facets,
        data_filename=config.facets_data_filename,
        score_filename=config.facets_score_filename,
        output_filename=config.facets_output_filename,
        comment_labels=comment_labels,
        judge_labels=judge_labels,
        comment_anchors=relevant_comment_anchors,
        item_anchors=relevant_item_anchors,
    )
    facets_spec_path = facets_run_dir / config.facets_spec_filename
    write_facets_spec(spec_text, facets_spec_path)

    return AnchoredLLMFacetsOutputs(
        facets_data_path=facets_data_path,
        facets_spec_path=facets_spec_path,
    )
