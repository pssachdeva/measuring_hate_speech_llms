"""Prepare FACETS inputs for pooled question-order effect estimation."""

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import pandas as pd

from mhs_llms.config import OrderEffectConfig, load_order_effect_config
from mhs_llms.facets.anchored import _build_facets_judge_map, _prepare_llm_annotations
from mhs_llms.facets.facets import _format_measure, write_facets_data, write_facets_spec
from mhs_llms.facets.postprocess import load_measure_anchors, parse_facets_score_file
from mhs_llms.schema import ITEM_NAMES


ORIGINAL_ORDER_ID = "1"
REVERSE_ORDER_ID = "2"


@dataclass(frozen=True)
class OrderEffectOutputs:
    """Paths produced when preparing one pooled order-effect FACETS run."""

    facets_data_path: Path
    facets_spec_path: Path


@dataclass(frozen=True)
class OrderEffectPostprocessOutputs:
    """Paths produced when processing one pooled order-effect FACETS run."""

    order_conditions_path: Path
    order_contrast_path: Path


def run_order_effect_facets(config_path: Path) -> OrderEffectOutputs:
    """Prepare a four-facet FACETS run with a shared model severity and order effect."""

    config = load_order_effect_config(config_path)
    original = _load_condition_annotations(
        annotation_paths=config.original_annotation_paths,
        order_id=ORIGINAL_ORDER_ID,
    )
    reverse = _load_condition_annotations(
        annotation_paths=config.reverse_annotation_paths,
        order_id=REVERSE_ORDER_ID,
    )
    annotations = pd.concat([original, reverse], ignore_index=True)
    _validate_matched_condition_rows(original=original, reverse=reverse)

    prepared_annotations, judge_mapping = _prepare_order_effect_annotations(annotations)
    facets_frame = build_order_effect_facets_frame(prepared_annotations)

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

    facets_run_dir = config.facets_run_dir
    facets_run_dir.mkdir(parents=True, exist_ok=True)

    facets_data_path = facets_run_dir / config.facets_data_filename
    write_facets_data(facets_frame, facets_data_path)

    spec_text = build_order_effect_facets_spec(
        facets_frame=facets_frame,
        config=config,
        judge_mapping=judge_mapping,
        comment_anchors=comment_anchors,
        item_anchors=item_anchors,
    )
    facets_spec_path = facets_run_dir / config.facets_spec_filename
    write_facets_spec(spec_text, facets_spec_path)

    return OrderEffectOutputs(
        facets_data_path=facets_data_path,
        facets_spec_path=facets_spec_path,
    )


def process_order_effect_run(
    config_path: Path,
    order_conditions_path: Path | None = None,
    order_contrast_path: Path | None = None,
) -> OrderEffectPostprocessOutputs:
    """Parse pooled order-condition FACETS scores and write condition/contrast CSVs."""

    config = load_order_effect_config(config_path)
    order_conditions = parse_order_condition_scores(config)
    order_contrast = build_order_condition_contrast(
        order_conditions=order_conditions,
        original_label=config.original_order_label,
        reverse_label=config.reverse_order_label,
    )

    if order_conditions_path is None:
        order_conditions_path = Path.cwd() / "data" / f"{config.facets_run_dir.name}_order_conditions.csv"
    if order_contrast_path is None:
        order_contrast_path = Path.cwd() / "data" / f"{config.facets_run_dir.name}_order_contrast.csv"

    order_conditions_path.parent.mkdir(parents=True, exist_ok=True)
    order_contrast_path.parent.mkdir(parents=True, exist_ok=True)
    order_conditions.to_csv(order_conditions_path, index=False)
    order_contrast.to_csv(order_contrast_path, index=False)

    return OrderEffectPostprocessOutputs(
        order_conditions_path=order_conditions_path,
        order_contrast_path=order_contrast_path,
    )


def build_order_effect_facets_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert order-labeled annotations into a four-facet FACETS input table."""

    facets = dataframe[["comment_id", "judge_id", "order_id", *ITEM_NAMES]].copy()
    for item_name in ITEM_NAMES:
        facets[item_name] = facets[item_name].astype(int)
    facets["item_id"] = f"1-{len(ITEM_NAMES)}"
    return facets[["comment_id", "judge_id", "item_id", "order_id", *ITEM_NAMES]]


def build_order_effect_facets_spec(
    facets_frame: pd.DataFrame,
    config: OrderEffectConfig,
    judge_mapping: pd.DataFrame,
    comment_anchors: Mapping[str, float],
    item_anchors: Mapping[str, float],
) -> str:
    """Build a four-facet FACETS spec for the pooled order-effect model."""

    comment_ids = facets_frame["comment_id"].drop_duplicates().astype(int).astype(str).tolist()
    judge_ids = facets_frame["judge_id"].drop_duplicates().astype(str).tolist()
    item_ids = [str(index) for index, _ in enumerate(ITEM_NAMES, start=1)]
    order_ids = [ORIGINAL_ORDER_ID, REVERSE_ORDER_ID]
    comment_label_map = {comment_id: comment_id for comment_id in comment_ids}
    judge_label_map = dict(
        zip(
            judge_mapping["judge_id"].astype(str).tolist(),
            judge_mapping["external_judge_id"].astype(str).tolist(),
            strict=True,
        )
    )
    order_label_map = {
        ORIGINAL_ORDER_ID: config.original_order_label,
        REVERSE_ORDER_ID: config.reverse_order_label,
    }

    relevant_comment_anchors = _select_anchor_subset(comment_anchors, comment_ids, "comment")
    relevant_item_anchors = _select_anchor_subset(item_anchors, ITEM_NAMES, "item")
    delements = ", ".join(config.facets.delements)
    optional_lines = []
    if config.facets.bias is not None:
        optional_lines.append(f"Bias = {config.facets.bias}")
    if config.facets.zscore is not None:
        optional_lines.append(f"Zscore = {config.facets.zscore}")
    optional_text = "\n".join(optional_lines)
    if optional_text:
        optional_text = f"{optional_text}\n"

    return (
        f"Title = {config.facets.title}\n"
        "Facets = 4\n"
        f"Model = {config.facets.model}\n"
        f"Noncenter = {config.facets.noncenter}\n"
        f"Positive = {config.facets.positive}\n"
        f"Arrange = {config.facets.arrange}\n"
        f"Subset detection = {config.facets.subset_detection}\n"
        f"Delements = {delements}\n"
        f"{optional_text}"
        f"Scorefile = {config.facets_score_filename}\n"
        f"Output file = {config.facets_output_filename}\n"
        f"CSV = {config.facets.csv}\n\n"
        "Labels =\n"
        f"{_build_anchor_label_block(1, 'Comments', comment_ids, comment_label_map, relevant_comment_anchors)}\n"
        f"{_build_plain_label_block(2, 'Judges', judge_ids, judge_label_map)}\n"
        f"{_build_anchor_label_block(3, 'Items', item_ids, dict(enumerate(ITEM_NAMES, start=1)), relevant_item_anchors, anchor_key_by_label=True)}\n"
        f"{_build_plain_label_block(4, 'Order', order_ids, order_label_map)}\n\n"
        f"Data = {config.facets_data_filename}\n"
    )


def parse_order_condition_scores(config: OrderEffectConfig) -> pd.DataFrame:
    """Read FACETS order-condition scores from the fourth score file."""

    score_path = config.facets_run_dir / config.facets_score_filename.replace(".txt", ".4.txt")
    order_scores = parse_facets_score_file(score_path).rename(
        columns={
            "facet_id": "order_id",
            "facet_label": "order_condition",
            "measure": "order_measure",
            "s_e": "order_se",
            "infit_ms": "order_infit_ms",
            "outfit_ms": "order_outfit_ms",
            "t_count": "response_count",
            "t_score": "observed_score",
        }
    )
    columns = [
        "order_id",
        "order_condition",
        "order_measure",
        "order_se",
        "order_infit_ms",
        "order_outfit_ms",
        "response_count",
        "observed_score",
    ]
    return order_scores[columns].sort_values("order_id", key=lambda series: series.astype(int)).reset_index(drop=True)


def build_order_condition_contrast(
    order_conditions: pd.DataFrame,
    original_label: str = "original_order",
    reverse_label: str = "reverse_order",
) -> pd.DataFrame:
    """Calculate the reverse-minus-original order measure contrast."""

    indexed = order_conditions.set_index("order_condition")
    missing = [label for label in (original_label, reverse_label) if label not in indexed.index]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Order condition scores are missing required labels: {missing_text}")

    original = indexed.loc[original_label]
    reverse = indexed.loc[reverse_label]
    contrast = float(reverse["order_measure"]) - float(original["order_measure"])
    contrast_se = (float(reverse["order_se"]) ** 2 + float(original["order_se"]) ** 2) ** 0.5
    return pd.DataFrame(
        [
            {
                "contrast": "reverse_order_minus_original_order",
                "original_order_measure": float(original["order_measure"]),
                "reverse_order_measure": float(reverse["order_measure"]),
                "order_measure_delta": contrast,
                "order_delta_se_independent": contrast_se,
            }
        ]
    )


def _load_condition_annotations(annotation_paths: tuple[Path, ...], order_id: str) -> pd.DataFrame:
    """Load one order condition and attach the FACETS order id."""

    if not annotation_paths:
        raise ValueError("At least one annotation path is required for each order condition")
    frames = [pd.read_csv(annotation_path) for annotation_path in annotation_paths]
    annotations = pd.concat(frames, ignore_index=True)
    annotations["judge_id"] = annotations["judge_id"].map(_normalize_condition_judge_id)
    annotations["order_id"] = order_id
    return annotations


def _prepare_order_effect_annotations(annotations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score annotations while preserving the pooled order-condition column."""

    prepared, _ = _prepare_llm_annotations(annotations)
    judge_mapping = _build_order_effect_judge_mapping(prepared["judge_label"])
    prepared["judge_id"] = prepared["judge_label"].map(_build_facets_judge_map(prepared["judge_label"]))
    prepared["judge_id"] = prepared["judge_id"].astype(int).astype(str)
    return prepared, judge_mapping


def _build_order_effect_judge_mapping(judge_labels: pd.Series) -> pd.DataFrame:
    """Build the numeric-to-external judge map after condition labels are normalized."""

    judge_map = _build_facets_judge_map(judge_labels)
    rows = [
        {"judge_id": numeric_id, "external_judge_id": judge_label}
        for judge_label, numeric_id in judge_map.items()
        if judge_label in set(judge_labels.astype(str).tolist())
    ]
    return pd.DataFrame(rows).sort_values("judge_id", kind="stable").reset_index(drop=True)


def _validate_matched_condition_rows(original: pd.DataFrame, reverse: pd.DataFrame) -> None:
    """Require original and reverse conditions to cover identical judge/comment pairs."""

    original_pairs = set(zip(original["judge_id"].astype(str), original["comment_id"].astype(int), strict=False))
    reverse_pairs = set(zip(reverse["judge_id"].astype(str), reverse["comment_id"].astype(int), strict=False))
    if original_pairs == reverse_pairs:
        return

    missing_reverse = sorted(original_pairs.difference(reverse_pairs))[:5]
    missing_original = sorted(reverse_pairs.difference(original_pairs))[:5]
    raise ValueError(
        "Original and reverse order annotations must contain the same judge/comment pairs; "
        f"missing reverse examples={missing_reverse}, missing original examples={missing_original}"
    )


def _normalize_condition_judge_id(value: object) -> str:
    """Remove known order-condition suffixes so pooled rows share one judge label."""

    normalized = str(value)
    suffixes = (
        "__original_order",
        "__reverse_order",
        "_original_order",
        "_reverse_order",
        "-original-order",
        "-reverse-order",
    )
    for suffix in suffixes:
        if normalized.endswith(suffix):
            return normalized[: -len(suffix)]
    return normalized


def _select_anchor_subset(
    anchors: Mapping[str, float],
    keys: list[str] | tuple[str, ...],
    anchor_name: str,
) -> dict[str, float]:
    """Return anchors for the requested keys or report missing values."""

    missing = [str(key) for key in keys if str(key) not in anchors]
    if missing:
        missing_text = ", ".join(missing[:10])
        raise ValueError(f"Missing {anchor_name} anchors for values: {missing_text}")
    return {str(key): anchors[str(key)] for key in keys}


def _build_anchor_label_block(
    facet_number: int,
    facet_name: str,
    element_ids: list[str],
    labels: Mapping[object, str],
    anchors: Mapping[str, float],
    anchor_key_by_label: bool = False,
) -> str:
    """Build one FACETS label block with anchored measures."""

    label_lines = [f"{facet_number},{facet_name},A"]
    for element_id in element_ids:
        label = str(labels.get(element_id, labels.get(int(element_id), element_id)))
        anchor_key = label if anchor_key_by_label else element_id
        label_lines.append(f"{element_id}={label},{_format_measure(anchors[str(anchor_key)])}")
    label_lines.append("*")
    return "\n".join(label_lines)


def _build_plain_label_block(
    facet_number: int,
    facet_name: str,
    element_ids: list[str],
    labels: Mapping[str, str],
) -> str:
    """Build one unanchored FACETS label block."""

    label_lines = [f"{facet_number},{facet_name}"]
    for element_id in element_ids:
        label_lines.append(f"{element_id}={labels[element_id]}")
    label_lines.append("*")
    return "\n".join(label_lines)
