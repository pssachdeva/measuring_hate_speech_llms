"""Utilities for comparing human and LLM hate-score distributions."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mhs_llms.paths import ARTIFACTS_DIR
from mhs_llms.plotting import build_gaussian_kde_curve
from mhs_llms.schema import ITEM_NAMES, prompt_letter_to_hf_value

RAW_ITEM_COLUMN_ALIASES = {
    "hate_speech": ("hatespeech",),
}


def read_annotation_table(path: Path) -> pd.DataFrame:
    """Load a CSV, TSV, or JSONL annotation table from disk."""

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported annotation file type: {path}")


def load_reference_comment_ids(reference_set_path: Path) -> list[int]:
    """Load the reference-set comment ids used to filter both input tables."""

    reference_frame = read_annotation_table(reference_set_path)
    if "comment_id" not in reference_frame.columns:
        raise ValueError(f"Reference set file is missing comment_id: {reference_set_path}")

    return (
        reference_frame["comment_id"]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values(kind="stable")
        .tolist()
    )


def align_item_responses(annotation_frame: pd.DataFrame) -> pd.DataFrame:
    """Convert item responses into a shared numeric scale where larger is more hateful."""

    aligned = annotation_frame.copy()

    # Normalize each survey item independently so human numeric values and LLM letters
    # end up on the same higher-means-more-hateful scale.
    for item_name in ITEM_NAMES:
        aligned[item_name] = _resolve_aligned_item_series(annotation_frame, item_name)

    return aligned


def build_comment_score_frame(
    annotation_frame: pd.DataFrame,
    reference_comment_ids: list[int],
    source_label: str,
) -> pd.DataFrame:
    """Aggregate one annotation table into one summed score per reference-set comment."""

    if "comment_id" not in annotation_frame.columns:
        raise ValueError("Annotation file must include a comment_id column")

    aligned = align_item_responses(annotation_frame)
    filtered = aligned.loc[aligned["comment_id"].isin(reference_comment_ids)].copy()
    if filtered.empty:
        raise ValueError(f"No rows matched the reference set for {source_label}")

    # Sum the ten aligned item scores per annotation, then average within comment so
    # multi-rater human files still contribute one score per reference-set sample.
    filtered["hate_speech_score"] = filtered[list(ITEM_NAMES)].sum(axis=1)

    comment_scores = (
        filtered.groupby("comment_id", as_index=False)
        .agg(
            hate_speech_score=("hate_speech_score", "mean"),
            num_annotations=("hate_speech_score", "size"),
        )
        .sort_values("comment_id", kind="stable")
    )
    comment_scores["source"] = source_label
    return comment_scores


def infer_llm_label(annotation_frame: pd.DataFrame, fallback: str) -> str:
    """Infer a concise LLM label from provider/model metadata when available."""

    if "model" in annotation_frame.columns:
        models = annotation_frame["model"].dropna().astype(str).unique().tolist()
        if len(models) == 1 and models[0]:
            return f"LLM ({models[0]})"
    if "judge_id" in annotation_frame.columns:
        judge_ids = annotation_frame["judge_id"].dropna().astype(str).unique().tolist()
        if len(judge_ids) == 1 and judge_ids[0]:
            return f"LLM ({judge_ids[0]})"
    return fallback


def build_default_output_path(llm_path: Path) -> Path:
    """Return the default artifacts path for the score-distribution plot."""

    return ARTIFACTS_DIR / f"{llm_path.stem}_vs_humans_hate_score_kde.png"


def plot_score_distributions(
    score_frame: pd.DataFrame,
    output_path: Path,
    title: str,
) -> Path:
    """Plot KDE curves plus mean markers for each source and save the figure."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    source_to_scores = {
        source: group["hate_speech_score"].astype(float).tolist()
        for source, group in score_frame.groupby("source", sort=False)
    }
    if not source_to_scores:
        raise ValueError("No score data was provided for plotting")

    all_scores = [score for scores in source_to_scores.values() for score in scores]
    x_min = min(all_scores)
    x_max = max(all_scores)
    padding = max(1.0, (x_max - x_min) * 0.1)
    plot_min = max(0.0, x_min - padding)
    plot_max = x_max + padding

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {
        "Humans": "#1f77b4",
    }

    for index, (source, scores) in enumerate(source_to_scores.items()):
        color = colors.get(source, f"C{index}")
        x_values, y_values = build_gaussian_kde_curve(scores, plot_min, plot_max)
        ax.plot(x_values, y_values, linewidth=2, label=source, color=color)
        ax.fill_between(x_values, y_values, alpha=0.18, color=color)

        mean_score = sum(scores) / len(scores)
        ax.axvline(
            mean_score,
            color=color,
            linestyle="--",
            linewidth=1.5,
            label=f"{source} mean",
        )

    ax.set_title(title)
    ax.set_xlabel("Summed hate speech score across 10 items")
    ax.set_ylabel("Density")
    ax.set_xlim(plot_min, plot_max)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def _resolve_aligned_item_series(annotation_frame: pd.DataFrame, item_name: str) -> pd.Series:
    """Read one item column and align it onto the common hate-score scale."""

    if item_name in annotation_frame.columns:
        source_series = annotation_frame[item_name]
    else:
        raw_aliases = RAW_ITEM_COLUMN_ALIASES.get(item_name, ())
        alias_match = next((alias for alias in raw_aliases if alias in annotation_frame.columns), None)
        if alias_match is not None:
            source_series = annotation_frame[alias_match]
            return source_series.map(lambda value: _align_item_value(item_name, value))

        letter_column = f"{item_name}_letter"
        if letter_column not in annotation_frame.columns:
            raise ValueError(f"Annotation file is missing {item_name} or {letter_column}")
        source_series = annotation_frame[letter_column]

    return source_series.map(lambda value: _align_item_value(item_name, value))


def _align_item_value(item_name: str, value: object) -> int:
    """Align one raw value or prompt letter onto the more-hateful-is-larger scale."""

    if pd.isna(value):
        raise ValueError(f"{item_name} contains a missing response")

    normalized_value = str(value).strip().upper()
    if normalized_value.isalpha():
        return prompt_letter_to_hf_value(item_name, normalized_value)
    return int(float(normalized_value))
