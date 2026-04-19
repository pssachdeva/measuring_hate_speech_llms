"""Figure helpers for comparing human comment severities with model severities."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mhs_llms.facets.postprocess import parse_facets_score_file
from mhs_llms.labels import infer_provider, model_id_to_plot_label, provider_display_name
from mhs_llms.plotting import (
    apply_plot_style,
    build_gaussian_kde_curve,
    format_plot_text,
    get_provider_color,
    save_figure,
)


def load_human_judge_severities(judges_path: Path) -> pd.DataFrame:
    """Load FACETS human judge severity measures for the top-panel distribution."""

    score_frame = _read_facets_score_table(judges_path)
    _validate_required_columns(score_frame, judges_path, {"measure"})
    return score_frame.loc[:, ["measure"]].copy()


def load_model_judge_severities(judges_paths: list[Path]) -> pd.DataFrame:
    """Load, label, and sort FACETS judge severities across one or more runs."""

    if not judges_paths:
        raise ValueError("At least one judges score file is required")

    severity_frames: list[pd.DataFrame] = []
    for judges_path in judges_paths:
        score_frame = _read_facets_score_table(judges_path)
        _validate_required_columns(score_frame, judges_path, {"facet_label", "measure", "s_e"})
        selected = score_frame.loc[:, ["facet_label", "measure", "s_e"]].copy()
        severity_frames.append(selected)

    combined = pd.concat(severity_frames, ignore_index=True)
    if combined["facet_label"].duplicated().any():
        duplicate_labels = combined.loc[combined["facet_label"].duplicated(), "facet_label"].tolist()
        duplicates_text = ", ".join(sorted(set(duplicate_labels)))
        raise ValueError(f"Duplicate model ids found across judge score files: {duplicates_text}")

    combined["provider"] = combined["facet_label"].map(infer_provider)
    combined["provider_label"] = combined["provider"].map(provider_display_name)
    combined["display_label"] = combined["facet_label"].map(model_id_to_plot_label)

    # Sort from the highest estimated severity to the lowest for top-to-bottom plotting.
    combined = combined.sort_values(["measure", "display_label"], ascending=[False, True], kind="stable")
    combined = combined.reset_index(drop=True)
    return combined


def plot_model_severity_figure(
    human_severity_frame: pd.DataFrame,
    model_severity_frame: pd.DataFrame,
    output_path: Path,
    title: str = "",
    figure_width: float = 8.5,
    figure_height: float | None = None,
    legend_font_size: float | None = None,
    bottom_label_font_size: float = 9.0,
    value_label_font_size: float = 8.5,
    x_min: float | None = None,
    x_max: float | None = None,
) -> Path:
    """Plot the requested two-panel severity figure and save it."""

    apply_plot_style()

    human_measures = human_severity_frame["measure"].astype(float).tolist()
    if not human_measures:
        raise ValueError("Human severity frame is empty")
    if model_severity_frame.empty:
        raise ValueError("Model severity frame is empty")

    model_min = float((model_severity_frame["measure"] - model_severity_frame["s_e"]).min())
    model_max = float((model_severity_frame["measure"] + model_severity_frame["s_e"]).max())
    human_min = min(human_measures)
    human_max = max(human_measures)
    plot_min, plot_max = _build_plot_bounds([human_min, model_min], [human_max, model_max])
    if x_min is not None:
        plot_min = x_min
    if x_max is not None:
        plot_max = x_max

    x_values, y_values = build_gaussian_kde_curve(human_measures, plot_min, plot_max)
    y_positions = list(range(len(model_severity_frame)))
    colors = [get_provider_color(provider) for provider in model_severity_frame["provider"].tolist()]

    if figure_height is None:
        figure_height = len(model_severity_frame) * 0.25 + 3.0
    figure, (top_axis, bottom_axis) = plt.subplots(
        2,
        1,
        figsize=(figure_width, figure_height),
        sharex=True,
        gridspec_kw={"height_ratios": [0.9, 1.9], "hspace": 0.02},
    )

    human_distribution_color = "#8CA2CF"
    axis_background_color = "#F7F7F7"
    top_axis.set_facecolor(axis_background_color)
    bottom_axis.set_facecolor(axis_background_color)
    top_axis.plot(x_values, y_values, color=human_distribution_color, linewidth=2.6, zorder=2)
    top_axis.fill_between(x_values, y_values, color=human_distribution_color, alpha=0.32, zorder=1)
    for row in model_severity_frame.itertuples(index=False):
        top_axis.axvline(
            row.measure,
            color=get_provider_color(row.provider),
            linewidth=1.6,
            alpha=0.85,
            linestyle="--",
            zorder=3,
        )

    top_axis.set_ylabel(format_plot_text("Human Annotator\nSeverity Density"), fontsize=8)
    if title:
        top_axis.set_title(format_plot_text(title))
    top_axis.axvline(0.0, color="#202020", linewidth=1.1, zorder=4)
    top_axis.set_ylim(bottom=0.0)
    top_axis.grid(alpha=0.18)
    top_axis.tick_params(axis="y", labelsize=8)
    top_axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    bar_container = bottom_axis.barh(
        y_positions,
        model_severity_frame["measure"],
        xerr=model_severity_frame["s_e"],
        color=colors,
        edgecolor="white",
        linewidth=1.1,
        alpha=0.95,
        error_kw={"elinewidth": 1.3, "capsize": 3.2, "ecolor": "#202020"},
    )
    _ = bar_container

    bottom_axis.set_yticks(y_positions)
    bottom_axis.set_yticklabels([])
    bottom_axis.tick_params(
        axis="y",
        which="both",
        right=False,
        left=False,
    )
    bottom_axis.tick_params(axis="x", labelsize=8)
    bottom_axis.invert_yaxis()
    bottom_axis.set_ylim(len(model_severity_frame) - 0.1, -0.90)
    bottom_axis.axvline(0.0, color="#444444", linewidth=1.1)
    bottom_axis.set_xlabel(format_plot_text("Severity"))
    bottom_axis.set_ylabel(format_plot_text("Models"), fontsize=10)
    bottom_axis.yaxis.labelpad = -8
    bottom_axis.grid(axis="x", alpha=0.18)

    label_pad = (plot_max - plot_min) * 0.012
    value_pad = (plot_max - plot_min) * 0.020
    for y_position, row in zip(y_positions, model_severity_frame.itertuples(index=False)):
        value = float(row.measure)
        standard_error = float(row.s_e)
        model_label_x = label_pad if value <= 0.0 else -label_pad
        model_label_alignment = "left" if value <= 0.0 else "right"
        value_x = (
            value - standard_error - value_pad
            if value <= 0.0
            else value + standard_error + value_pad
        )
        value_alignment = "right" if value <= 0.0 else "left"
        bottom_axis.text(
            model_label_x,
            y_position,
            format_plot_text(row.display_label),
            va="center",
            ha=model_label_alignment,
            fontsize=bottom_label_font_size,
        )
        bottom_axis.text(
            value_x,
            y_position,
            format_plot_text(f"{value:.3f}"),
            va="center",
            ha=value_alignment,
            fontsize=value_label_font_size,
        )

    legend_handles = [
        plt.matplotlib.patches.Patch(
            facecolor=get_provider_color(provider_slug),
            edgecolor="none",
            label=format_plot_text(provider_display_name(provider_slug)),
        )
        for provider_slug in _provider_order(model_severity_frame["provider"].tolist())
    ]
    bottom_axis.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        fontsize=legend_font_size,
    )

    x_padding = 0.0 if x_min is not None or x_max is not None else max((plot_max - plot_min) * 0.06, 0.10)
    bottom_axis.set_xlim(plot_min - x_padding, plot_max + x_padding)
    top_axis.set_xlim(plot_min - x_padding, plot_max + x_padding)
    direction_label_box = {
        "boxstyle": "round,pad=0.42,rounding_size=0.28",
        "facecolor": "#F4F4F4",
        "edgecolor": "#D0D0D0",
        "linewidth": 0.8,
        "alpha": 0.72,
    }
    direction_label_inset = (plot_max - plot_min) * 0.09
    top_axis.text(
        plot_min - x_padding + direction_label_inset,
        0.14,
        format_plot_text("More Likely to\nLabel as Hateful"),
        transform=top_axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
        bbox=direction_label_box,
    )
    top_axis.text(
        plot_max + x_padding - direction_label_inset,
        0.14,
        format_plot_text("Less Likely to\nLabel as Hateful"),
        transform=top_axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=8,
        color="#555555",
        bbox=direction_label_box,
    )

    figure.align_ylabels([top_axis, bottom_axis])
    plotted_path = save_figure(figure, output_path)
    plt.close(figure)
    return plotted_path


def _read_facets_score_table(score_path: Path) -> pd.DataFrame:
    """Read a FACETS score table from either raw text or processed CSV."""

    if score_path.suffix.lower() == ".csv":
        return pd.read_csv(score_path)
    return parse_facets_score_file(score_path)


def _validate_required_columns(
    score_frame: pd.DataFrame,
    score_path: Path,
    required_columns: set[str],
) -> None:
    """Validate that a FACETS score table contains the requested columns."""

    missing_columns = required_columns.difference(score_frame.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"FACETS score file is missing required columns in {score_path}: {missing_list}")


def _build_plot_bounds(min_values: list[float], max_values: list[float]) -> tuple[float, float]:
    """Build shared x-axis limits that comfortably include all values."""

    x_min = min(min_values)
    x_max = max(max_values)
    padding = max((x_max - x_min) * 0.08, 0.12)
    return x_min - padding, x_max + padding


def _provider_order(provider_slugs: list[str]) -> list[str]:
    """Return a stable provider order for legends."""

    preferred_order = [
        "anthropic",
        "openai",
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
    seen = set(provider_slugs)
    return [provider_slug for provider_slug in preferred_order if provider_slug in seen]
