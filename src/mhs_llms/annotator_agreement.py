"""Krippendorff alpha utilities for human and LLM MHS annotations."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mhs_llms.dataset import load_mhs_dataframe
from mhs_llms.plotting import format_plot_text, save_figure
from mhs_llms.schema import ITEM_NAMES
from mhs_llms.score_distribution import align_item_responses, read_annotation_table


ITEM_DISPLAY_LABELS = {
    "sentiment": "Sentiment",
    "respect": "Respect",
    "insult": "Insult",
    "humiliate": "Humiliate",
    "status": "Status",
    "dehumanize": "Dehumanize",
    "violence": "Violence",
    "genocide": "Genocide",
    "attack_defend": "Attack/defend",
    "hate_speech": "Hate speech",
}

AGREEMENT_GROUP_ORDER = ("Humans", "LLMs", "Humans + LLMs")


def load_annotation_files(paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate one or more annotation files."""

    if not paths:
        raise ValueError("At least one annotation file is required")

    frames = []
    for path in paths:
        frame = read_annotation_table(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def load_human_annotations(
    human_path: Path | None,
    dataset_name: str,
    split: str,
    config_name: str | None,
) -> pd.DataFrame:
    """Load human annotations from a local file or the configured MHS dataset."""

    if human_path is not None:
        return read_annotation_table(human_path)
    return load_mhs_dataframe(dataset_name=dataset_name, split=split, config_name=config_name)


def reference_comment_ids_from_annotations(
    llm_annotations: pd.DataFrame,
    reference_set_path: Path | None = None,
) -> list[int]:
    """Return reference comment ids from a file or from the loaded LLM annotations."""

    if reference_set_path is not None:
        reference_frame = read_annotation_table(reference_set_path)
        if "comment_id" not in reference_frame.columns:
            raise ValueError(f"Reference set is missing comment_id: {reference_set_path}")
        return (
            reference_frame["comment_id"]
            .dropna()
            .astype(int)
            .drop_duplicates()
            .sort_values(kind="stable")
            .tolist()
        )

    if "comment_id" not in llm_annotations.columns:
        raise ValueError("LLM annotations must include comment_id")
    return sorted(llm_annotations["comment_id"].dropna().astype(int).unique().tolist())


def build_item_agreement_summary(
    llm_annotations: pd.DataFrame,
    human_annotations: pd.DataFrame,
    reference_comment_ids: list[int],
    item_names: tuple[str, ...] = ITEM_NAMES,
    distance_metric: str = "ordinal",
) -> pd.DataFrame:
    """Compute item-level alpha for human-only, LLM-only, and combined annotator pools."""

    llm_frame = prepare_agreement_annotations(
        annotation_frame=llm_annotations,
        reference_comment_ids=reference_comment_ids,
        annotator_prefix="llm",
    )
    human_frame = prepare_agreement_annotations(
        annotation_frame=human_annotations,
        reference_comment_ids=reference_comment_ids,
        annotator_prefix="human",
    )
    combined_frame = pd.concat([human_frame, llm_frame], ignore_index=True)

    rows = []
    for item_name in item_names:
        if item_name not in ITEM_NAMES:
            raise ValueError(f"Unsupported MHS item: {item_name}")
        rows.append(
            _agreement_summary_row(
                annotation_frame=human_frame,
                item_name=item_name,
                agreement_group="Humans",
                distance_metric=distance_metric,
            )
        )
        rows.append(
            _agreement_summary_row(
                annotation_frame=llm_frame,
                item_name=item_name,
                agreement_group="LLMs",
                distance_metric=distance_metric,
            )
        )
        rows.append(
            _agreement_summary_row(
                annotation_frame=combined_frame,
                item_name=item_name,
                agreement_group="Humans + LLMs",
                distance_metric=distance_metric,
            )
        )

    summary = pd.DataFrame(rows)
    summary["item_label"] = summary["item_name"].map(ITEM_DISPLAY_LABELS)
    summary["item_order"] = summary["item_name"].map({item: index for index, item in enumerate(item_names)})
    summary["agreement_group"] = pd.Categorical(
        summary["agreement_group"],
        categories=list(AGREEMENT_GROUP_ORDER),
        ordered=True,
    )
    return summary.sort_values(["item_order", "agreement_group"], kind="stable").reset_index(drop=True)


def prepare_agreement_annotations(
    annotation_frame: pd.DataFrame,
    reference_comment_ids: list[int],
    annotator_prefix: str,
) -> pd.DataFrame:
    """Normalize annotation ids and item responses for agreement calculations."""

    if "comment_id" not in annotation_frame.columns:
        raise ValueError("Annotation frame must include comment_id")

    normalized = annotation_frame.copy()
    if "violence_phys" in normalized.columns and "violence" not in normalized.columns:
        normalized = normalized.rename(columns={"violence_phys": "violence"})
    if "judge_id" not in normalized.columns:
        if "annotator_id" not in normalized.columns:
            raise ValueError("Annotation frame must include judge_id or annotator_id")
        normalized["judge_id"] = normalized["annotator_id"]

    selected = normalized.loc[normalized["comment_id"].astype(int).isin(reference_comment_ids)].copy()
    if selected.empty:
        raise ValueError(f"No {annotator_prefix} annotations matched the reference comments")

    aligned = align_item_responses(selected)
    aligned["comment_id"] = aligned["comment_id"].astype(int)
    aligned["judge_id"] = annotator_prefix + ":" + aligned["judge_id"].astype(str)

    # Ensure each annotator contributes at most one response per comment and item.
    grouped = (
        aligned.groupby(["comment_id", "judge_id"], as_index=False)
        .agg({item_name: "mean" for item_name in ITEM_NAMES})
        .sort_values(["comment_id", "judge_id"], kind="stable")
    )
    return grouped


def krippendorff_alpha(
    annotations: pd.DataFrame,
    value_column: str,
    unit_column: str = "comment_id",
    annotator_column: str = "judge_id",
    distance_metric: str = "ordinal",
) -> float:
    """Return Krippendorff's alpha for one item, allowing missing annotator cells."""

    required_columns = {unit_column, annotator_column, value_column}
    missing_columns = required_columns.difference(annotations.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Annotations are missing required columns: {missing_text}")

    selected = annotations[[unit_column, annotator_column, value_column]].dropna().copy()
    if selected.empty:
        return float("nan")

    selected = selected.groupby([unit_column, annotator_column], as_index=False).agg(
        {value_column: "mean"}
    )
    pivot = selected.pivot_table(
        index=unit_column,
        columns=annotator_column,
        values=value_column,
        aggfunc="mean",
    )
    values = sorted(selected[value_column].dropna().astype(float).unique().tolist())
    if len(values) <= 1:
        return float("nan")

    coincidence = _build_coincidence_matrix(pivot, values)
    total_coincidences = float(coincidence.sum())
    if total_coincidences <= 0.0:
        return float("nan")

    marginals = coincidence.sum(axis=0)
    distance = _build_distance_matrix(values, marginals, distance_metric)
    observed_disagreement = float((coincidence * distance).sum())
    expected = _build_expected_coincidence_matrix(marginals)
    expected_disagreement = float((expected * distance).sum())
    if expected_disagreement == 0.0:
        return 1.0 if observed_disagreement == 0.0 else float("nan")
    return 1.0 - observed_disagreement / expected_disagreement


def plot_item_agreement_summary(
    summary_frame: pd.DataFrame,
    output_path: Path,
    figsize: tuple[float, float],
    dpi: int,
    marker_size: float,
    x_offset: float,
    y_limits: tuple[float, float] | None,
    tick_label_size: float,
    axis_label_size: float,
    legend_font_size: float,
) -> Path:
    """Plot item-level alpha for human-only, LLM-only, and combined annotator pools."""

    if summary_frame.empty:
        raise ValueError("Agreement summary is empty")

    figure, axis = plt.subplots(figsize=figsize, dpi=dpi)
    colors = {
        "Humans": "#009E73",
        "LLMs": "#0072B2",
        "Humans + LLMs": "#D55E00",
    }
    offsets = {
        "Humans": -x_offset,
        "LLMs": 0.0,
        "Humans + LLMs": x_offset,
    }

    item_labels = (
        summary_frame[["item_order", "item_label"]]
        .drop_duplicates()
        .sort_values("item_order", kind="stable")["item_label"]
        .tolist()
    )
    x_positions = list(range(len(item_labels)))

    for agreement_group in AGREEMENT_GROUP_ORDER:
        group_frame = summary_frame.loc[summary_frame["agreement_group"].astype(str) == agreement_group]
        if group_frame.empty:
            continue
        axis.scatter(
            group_frame["item_order"].astype(float) + offsets[agreement_group],
            group_frame["alpha"],
            s=marker_size,
            color=colors[agreement_group],
            edgecolor="white",
            linewidth=0.8,
            label=format_plot_text(agreement_group),
            zorder=3,
        )

    axis.axhline(0.0, color="#9A9A9A", linewidth=0.9, linestyle="--", zorder=1)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(
        [format_plot_text(label) for label in item_labels],
        rotation=35,
        ha="right",
        fontsize=tick_label_size,
    )
    axis.set_ylabel(format_plot_text("Krippendorff's alpha"), fontsize=axis_label_size)
    axis.set_xlabel(format_plot_text("Survey item"), fontsize=axis_label_size)
    axis.tick_params(axis="y", labelsize=tick_label_size)
    axis.grid(axis="y", alpha=0.22)
    axis.set_xlim(-0.6, len(item_labels) - 0.4)
    if y_limits is not None:
        axis.set_ylim(*y_limits)
    axis.legend(frameon=False, fontsize=legend_font_size)

    figure.tight_layout()
    plotted_path = save_figure(figure, output_path, dpi=dpi)
    plt.close(figure)
    return plotted_path


def _agreement_summary_row(
    annotation_frame: pd.DataFrame,
    item_name: str,
    agreement_group: str,
    distance_metric: str,
) -> dict[str, object]:
    """Build the alpha and coverage metadata for one agreement series."""

    selected = annotation_frame[["comment_id", "judge_id", item_name]].dropna()
    alpha = krippendorff_alpha(
        annotations=annotation_frame,
        value_column=item_name,
        distance_metric=distance_metric,
    )
    return {
        "item_name": item_name,
        "agreement_group": agreement_group,
        "alpha": alpha,
        "num_units": int(selected["comment_id"].nunique()),
        "num_annotators": int(selected["judge_id"].nunique()),
        "num_ratings": int(len(selected)),
    }


def _build_coincidence_matrix(pivot: pd.DataFrame, values: list[float]) -> np.ndarray:
    """Build Krippendorff's coincidence matrix from an annotator-by-unit table."""

    value_to_index = {value: index for index, value in enumerate(values)}
    coincidence = np.zeros((len(values), len(values)), dtype=float)

    for _, row in pivot.iterrows():
        unit_values = row.dropna().astype(float).tolist()
        unit_count = len(unit_values)
        if unit_count <= 1:
            continue

        counts = np.zeros(len(values), dtype=float)
        for value in unit_values:
            counts[value_to_index[value]] += 1.0

        unit_coincidence = np.outer(counts, counts)
        unit_coincidence[np.diag_indices_from(unit_coincidence)] -= counts
        coincidence += unit_coincidence / (unit_count - 1.0)

    return coincidence


def _build_expected_coincidence_matrix(marginals: np.ndarray) -> np.ndarray:
    """Build the expected coincidence matrix from observed value marginals."""

    total = float(marginals.sum())
    if total <= 1.0:
        return np.zeros((len(marginals), len(marginals)), dtype=float)

    expected = np.outer(marginals, marginals) / (total - 1.0)
    expected[np.diag_indices_from(expected)] = marginals * (marginals - 1.0) / (total - 1.0)
    return expected


def _build_distance_matrix(
    values: list[float],
    marginals: np.ndarray,
    distance_metric: str,
) -> np.ndarray:
    """Build the alpha distance matrix for nominal, ordinal, or interval data."""

    metric = distance_metric.lower().strip()
    if metric not in {"nominal", "ordinal", "interval"}:
        raise ValueError("distance_metric must be one of: nominal, ordinal, interval")

    distance = np.zeros((len(values), len(values)), dtype=float)
    for left_index, left_value in enumerate(values):
        for right_index, right_value in enumerate(values):
            if left_index == right_index:
                continue
            if metric == "nominal":
                distance[left_index, right_index] = 1.0
            elif metric == "interval":
                distance[left_index, right_index] = (left_value - right_value) ** 2
            else:
                low_index = min(left_index, right_index)
                high_index = max(left_index, right_index)
                cumulative = marginals[low_index : high_index + 1].sum()
                endpoint_adjustment = (marginals[low_index] + marginals[high_index]) / 2.0
                distance[left_index, right_index] = (cumulative - endpoint_adjustment) ** 2
    return distance
