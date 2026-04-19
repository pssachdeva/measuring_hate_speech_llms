"""Build average hate-score summaries for model and human annotations."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mhs_llms.dataset import load_mhs_dataframe
from mhs_llms.labels import infer_provider, model_id_to_label, provider_display_name
from mhs_llms.plotting import (
    build_gaussian_kde_curve,
    format_plot_text,
    get_provider_color,
    save_figure,
)
from mhs_llms.schema import ITEM_NAMES
from mhs_llms.score_distribution import align_item_responses, read_annotation_table


SCORE_COLUMN = "hate_speech_score"
RELEASE_DATE_PATTERN = re.compile(r"(20\d{6})")

PROVIDER_ORDER = [
    "openai",
    "anthropic",
    "google",
    "xai",
    "deepseek",
    "minimax",
    "moonshotai",
    "qwen",
    "xiaomi",
    "zai",
    "openrouter",
    "unknown",
]

REASONING_ORDER = {
    "none": 0,
    "minimal": 1,
    "low": 2,
    "medium": 3,
    "high": 4,
    "xhigh": 5,
}

RELEASE_DATE_OVERRIDES = {
    "openai_gpt-4o": "2024-05-13",
    "openai_gpt-4.1": "2025-04-14",
    "openai_gpt-5.2": "2026-02-01",
    "openai_gpt-5.4": "2026-04-01",
    "openai_gpt-5.4-mini": "2026-04-01",
    "openai_gpt-5.4-nano": "2026-04-01",
    "google_gemini-2.5-pro": "2025-06-17",
    "google_gemini-2.5-flash": "2025-06-17",
    "google_gemini-3-flash-preview": "2026-03-01",
    "google_gemini-3.1-flash-lite-preview": "2026-04-09",
    "google_gemini-3.1-pro-preview": "2026-04-09",
    "xai_grok-3": "2025-02-17",
    "xai_grok-4-fast-non-reasoning": "2025-07-09",
    "xai_grok-4-fast-reasoning": "2025-07-09",
    "xai_grok-4-1-fast-non-reasoning": "2025-11-17",
    "xai_grok-4-1-fast-reasoning": "2025-11-17",
}


def load_model_annotation_files(paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate model annotation files."""

    if not paths:
        raise ValueError("At least one model annotation file is required")

    frames = []
    for path in paths:
        frame = read_annotation_table(path)
        frame["source_file"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def reference_comment_ids(model_annotations: pd.DataFrame) -> list[int]:
    """Return the sorted comment ids covered by the model annotation set."""

    if "comment_id" not in model_annotations.columns:
        raise ValueError("Model annotations must include comment_id")
    return sorted(model_annotations["comment_id"].dropna().astype(int).unique().tolist())


def build_model_comment_scores(model_annotations: pd.DataFrame) -> pd.DataFrame:
    """Compute one aligned hate score per model run and comment."""

    required_columns = {"comment_id", "judge_id", "model"}
    missing_columns = required_columns.difference(model_annotations.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Model annotations are missing required columns: {missing_text}")

    aligned = align_item_responses(model_annotations)
    aligned[SCORE_COLUMN] = aligned[list(ITEM_NAMES)].sum(axis=1)
    if "provider" in aligned.columns:
        aligned["provider_slug"] = aligned["provider"].fillna("").astype(str).str.strip()
        aligned.loc[aligned["provider_slug"] == "", "provider_slug"] = "unknown"
    else:
        aligned["provider_slug"] = aligned["judge_id"].astype(str).map(infer_provider)
    aligned["color_slug"] = aligned["judge_id"].astype(str).map(infer_provider)

    # Collapse accidental duplicates without hiding missing comments from failed runs.
    score_frame = (
        aligned.groupby(["provider_slug", "color_slug", "judge_id", "model", "comment_id"], as_index=False)
        .agg(**{SCORE_COLUMN: (SCORE_COLUMN, "mean")})
        .sort_values(["provider_slug", "judge_id", "comment_id"], kind="stable")
    )
    return score_frame


def summarize_model_scores(
    model_comment_scores: pd.DataFrame,
    n_bootstrap: int,
    random_seed: int,
) -> pd.DataFrame:
    """Summarize each model run with mean score and bootstrap confidence interval."""

    summaries = []
    rng = np.random.default_rng(random_seed)
    for (provider_slug, color_slug, judge_id, model_name), group in model_comment_scores.groupby(
        ["provider_slug", "color_slug", "judge_id", "model"],
        sort=False,
    ):
        values = group[SCORE_COLUMN].astype(float).to_numpy()
        mean_score, ci_low, ci_high = _bootstrap_mean_ci(values, n_bootstrap, rng)
        summaries.append(
            {
                "provider_slug": provider_slug,
                "provider_label": provider_display_name(provider_slug),
                "color_slug": color_slug,
                "judge_id": judge_id,
                "model": model_name,
                "display_label": model_id_to_label(judge_id),
                "mean_score": mean_score,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "num_comments": int(len(values)),
                "release_date": _release_date_for_model(judge_id, model_name),
                "reasoning_order": _reasoning_order(judge_id),
            }
        )

    summary_frame = pd.DataFrame(summaries)
    if summary_frame.empty:
        raise ValueError("No model scores were available to summarize")
    return _sort_model_summary(summary_frame)


def build_human_average_scores(
    comment_ids: list[int],
    dataset_name: str,
    split: str,
    config_name: str | None,
) -> pd.DataFrame:
    """Compute each human annotator's average aligned hate score for selected comments."""

    human_frame = load_mhs_dataframe(dataset_name=dataset_name, split=split, config_name=config_name)
    selected = human_frame.loc[human_frame["comment_id"].astype(int).isin(comment_ids)].copy()
    if selected.empty:
        raise ValueError("No human annotations matched the selected model comment ids")

    aligned = align_item_responses(selected)
    aligned[SCORE_COLUMN] = aligned[list(ITEM_NAMES)].sum(axis=1)
    aligned["judge_id"] = aligned["annotator_id"].astype(int).astype(str)

    # Average annotations per annotator, preserving how many reference comments each saw.
    human_scores = (
        aligned.groupby("judge_id", as_index=False)
        .agg(
            average_score=(SCORE_COLUMN, "mean"),
            num_comments=(SCORE_COLUMN, "size"),
        )
        .sort_values("average_score", kind="stable")
    )
    return human_scores


def assign_grouped_x_positions(
    model_summary: pd.DataFrame,
    provider_gap: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign x positions with visible gaps between provider groups."""

    positioned_rows = []
    group_rows = []
    current_position = 0.0
    for provider_slug, provider_frame in model_summary.groupby("provider_slug", sort=False):
        provider_positions = []
        for row in provider_frame.itertuples(index=False):
            row_dict = row._asdict()
            row_dict["x_position"] = current_position
            positioned_rows.append(row_dict)
            provider_positions.append(current_position)
            current_position += 1.0

        if provider_positions:
            group_rows.append(
                {
                    "provider_slug": provider_slug,
                    "provider_label": provider_display_name(provider_slug),
                    "x_min": min(provider_positions),
                    "x_max": max(provider_positions),
                    "x_mid": (min(provider_positions) + max(provider_positions)) / 2.0,
                }
            )
            current_position += provider_gap

    return pd.DataFrame(positioned_rows), pd.DataFrame(group_rows)


def plot_average_hate_scores(
    model_summary: pd.DataFrame,
    human_average_scores: pd.DataFrame,
    output_path: Path,
    provider_gap: float,
    figsize: tuple[float, float],
    dpi: int,
    score_min: float,
    score_max: float,
    kde_points: int,
    marker_size: float,
    errorbar_linewidth: float,
    capsize: float,
    tick_label_size: float,
    axis_label_size: float,
    provider_label_size: float,
) -> Path:
    """Plot model mean scores with CIs next to human average-score KDE."""

    positioned, provider_groups = assign_grouped_x_positions(model_summary, provider_gap)
    human_scores = human_average_scores["average_score"].astype(float).tolist()
    if not human_scores:
        raise ValueError("Human average score frame is empty")

    y_min, y_max = _build_y_limits(positioned, human_scores, score_min, score_max)
    kde_y, kde_density = build_gaussian_kde_curve(human_scores, y_min, y_max, point_count=kde_points)

    figure, (model_axis, human_axis) = plt.subplots(
        1,
        2,
        figsize=figsize,
        dpi=dpi,
        sharey=True,
        gridspec_kw={"width_ratios": [5.8, 1.25], "wspace": 0.04},
    )

    for row in positioned.itertuples(index=False):
        color = get_provider_color(row.color_slug)
        yerr_low = row.mean_score - row.ci_low
        yerr_high = row.ci_high - row.mean_score
        model_axis.errorbar(
            row.x_position,
            row.mean_score,
            yerr=[[yerr_low], [yerr_high]],
            fmt="o",
            markersize=marker_size,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.7,
            ecolor="#202020",
            elinewidth=errorbar_linewidth,
            capsize=capsize,
            zorder=3,
        )

    for group_row in provider_groups.itertuples(index=False):
        if group_row.x_min > positioned["x_position"].min():
            model_axis.axvline(group_row.x_min - provider_gap / 2.0, color="#BFBFBF", linewidth=0.8, alpha=0.7)
        model_axis.text(
            group_row.x_mid,
            1.015,
            format_plot_text(group_row.provider_label),
            ha="center",
            va="bottom",
            fontsize=provider_label_size,
            transform=model_axis.get_xaxis_transform(),
        )

    model_axis.set_xticks(positioned["x_position"].tolist())
    model_axis.set_xticklabels(
        [format_plot_text(label) for label in positioned["display_label"].tolist()],
        rotation=90,
        ha="center",
        va="top",
        fontsize=tick_label_size,
    )
    model_axis.set_xlim(positioned["x_position"].min() - 0.8, positioned["x_position"].max() + 0.8)
    model_axis.set_ylim(y_min, y_max)
    model_axis.set_ylabel(format_plot_text("Average hate speech score"), fontsize=axis_label_size)
    model_axis.set_xlabel(format_plot_text("Model run"), fontsize=axis_label_size)
    model_axis.tick_params(axis="y", labelsize=tick_label_size)
    model_axis.grid(axis="y", alpha=0.22)

    human_color = "#8CA2CF"
    human_axis.plot(kde_density, kde_y, color=human_color, linewidth=2.2)
    human_axis.fill_betweenx(kde_y, 0.0, kde_density, color=human_color, alpha=0.35)
    human_axis.axhline(
        sum(human_scores) / len(human_scores),
        color="#4A5E89",
        linestyle="--",
        linewidth=1.1,
        alpha=0.8,
    )
    human_axis.set_xlabel(format_plot_text("Human\nDensity"), fontsize=axis_label_size)
    human_axis.tick_params(axis="x", labelsize=tick_label_size)
    human_axis.tick_params(axis="y", labelleft=False)
    human_axis.set_xlim(left=0.0)
    human_axis.grid(axis="y", alpha=0.16)
    human_axis.spines["left"].set_visible(False)

    figure.align_ylabels([model_axis])
    plotted_path = save_figure(figure, output_path, dpi=dpi)
    plt.close(figure)
    return plotted_path


def _bootstrap_mean_ci(
    values: np.ndarray,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Return the sample mean and percentile bootstrap interval for one score vector."""

    if len(values) == 0:
        raise ValueError("Cannot bootstrap an empty score vector")
    mean_score = float(values.mean())
    if len(values) == 1 or n_bootstrap <= 0:
        return mean_score, mean_score, mean_score

    sample_indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    bootstrap_means = values[sample_indices].mean(axis=1)
    ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    return mean_score, float(ci_low), float(ci_high)


def _sort_model_summary(summary_frame: pd.DataFrame) -> pd.DataFrame:
    """Sort model runs by provider and release date within provider."""

    provider_rank = {provider_slug: index for index, provider_slug in enumerate(PROVIDER_ORDER)}
    sorted_frame = summary_frame.copy()
    sorted_frame["provider_rank"] = sorted_frame["provider_slug"].map(
        lambda provider_slug: provider_rank.get(provider_slug, len(provider_rank))
    )
    sorted_frame = sorted_frame.sort_values(
        ["provider_rank", "release_date", "model", "reasoning_order", "display_label"],
        kind="stable",
    ).reset_index(drop=True)
    return sorted_frame.drop(columns=["provider_rank"])


def _release_date_for_model(judge_id: str, model_name: str) -> str:
    """Infer a release-order date from model metadata or an explicit local mapping."""

    normalized_judge_id = str(judge_id)
    normalized_model_name = str(model_name)
    date_match = RELEASE_DATE_PATTERN.search(normalized_model_name)
    if date_match is not None:
        raw_date = date_match.group(1)
        return f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"

    base_model_id, _ = _split_reasoning_suffix(normalized_judge_id)
    return RELEASE_DATE_OVERRIDES.get(base_model_id, "9999-12-31")


def _reasoning_order(judge_id: str) -> int:
    """Return a stable ordering for reasoning-effort variants."""

    _, suffix = _split_reasoning_suffix(str(judge_id))
    if suffix is None:
        return -1
    return REASONING_ORDER.get(suffix, len(REASONING_ORDER))


def _split_reasoning_suffix(model_id: str) -> tuple[str, str | None]:
    """Split the trailing reasoning suffix from one model id when present."""

    parts = model_id.split("_")
    if len(parts) < 2:
        return model_id, None
    suffix = parts[-1]
    if suffix not in REASONING_ORDER:
        return model_id, None
    return "_".join(parts[:-1]), suffix


def _build_y_limits(
    model_summary: pd.DataFrame,
    human_scores: list[float],
    score_min: float,
    score_max: float,
) -> tuple[float, float]:
    """Build score-axis bounds that include human averages and model intervals."""

    observed_min = min([min(human_scores), float(model_summary["ci_low"].min())])
    observed_max = max([max(human_scores), float(model_summary["ci_high"].max())])
    padding = max((observed_max - observed_min) * 0.08, 0.8)
    y_min = max(score_min, observed_min - padding)
    y_max = min(score_max, observed_max + padding)
    if y_max <= y_min:
        return score_min, score_max
    return y_min, y_max
