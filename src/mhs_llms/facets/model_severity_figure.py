"""Figure helpers for comparing human comment severities with model severities."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mhs_llms.facets.postprocess import parse_facets_score_file
from mhs_llms.labels import infer_provider, model_id_to_label, provider_display_name
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
    combined["display_label"] = combined["facet_label"].map(model_id_to_label)

    # Sort from the highest estimated severity to the lowest for top-to-bottom plotting.
    combined = combined.sort_values(["measure", "display_label"], ascending=[False, True], kind="stable")
    combined = combined.reset_index(drop=True)
    return combined


def plot_model_severity_figure(
    human_severity_frame: pd.DataFrame,
    model_severity_frame: pd.DataFrame,
    output_path: Path,
    title: str = "Human Severity Distribution and Model Judge Severities",
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

    x_values, y_values = build_gaussian_kde_curve(human_measures, plot_min, plot_max)
    y_positions = list(range(len(model_severity_frame)))
    colors = [get_provider_color(provider) for provider in model_severity_frame["provider"].tolist()]

    figure_height = max(9.5, len(model_severity_frame) * 0.34 + 3.8)
    figure, (top_axis, bottom_axis) = plt.subplots(
        2,
        1,
        figsize=(11.5, figure_height),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.9], "hspace": 0.02},
    )

    top_axis.plot(x_values, y_values, color="#8CA2CF", linewidth=2.6, zorder=2)
    top_axis.fill_between(x_values, y_values, color="#8CA2CF", alpha=0.32, zorder=1)
    for row in model_severity_frame.itertuples(index=False):
        top_axis.axvline(
            row.measure,
            color=get_provider_color(row.provider),
            linewidth=1.6,
            alpha=0.85,
            linestyle="--",
            zorder=3,
        )

    top_axis.set_ylabel(format_plot_text("Density"))
    top_axis.set_title(format_plot_text(title))
    top_axis.grid(alpha=0.18)

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
    bottom_axis.set_yticklabels(format_plot_text(model_severity_frame["display_label"].tolist()))
    bottom_axis.yaxis.tick_right()
    bottom_axis.tick_params(
        axis="y",
        which="both",
        labelright=True,
        labelleft=False,
        right=False,
        left=False,
        pad=10,
        labelsize=9,
    )
    bottom_axis.invert_yaxis()
    bottom_axis.axvline(0.0, color="#444444", linewidth=1.1)
    bottom_axis.set_xlabel(format_plot_text("Severity"))
    bottom_axis.grid(axis="x", alpha=0.18)

    value_pad = (plot_max - plot_min) * 0.012
    for y_position, row in zip(y_positions, model_severity_frame.itertuples(index=False)):
        value = float(row.measure)
        value_x = value - value_pad if value <= 0.0 else value + value_pad
        value_alignment = "right" if value <= 0.0 else "left"
        bottom_axis.text(
            value_x,
            y_position,
            format_plot_text(f"{value:.3f}"),
            va="center",
            ha=value_alignment,
            fontsize=8.5,
        )

    legend_handles = [
        plt.matplotlib.patches.Patch(
            facecolor=get_provider_color(provider_slug),
            edgecolor="none",
            label=format_plot_text(provider_display_name(provider_slug)),
        )
        for provider_slug in _provider_order(model_severity_frame["provider"].tolist())
    ]
    top_axis.legend(handles=legend_handles, loc="upper right", frameon=True)

    x_padding = max((plot_max - plot_min) * 0.06, 0.10)
    bottom_axis.set_xlim(plot_min - x_padding, plot_max + x_padding)
    top_axis.set_xlim(plot_min - x_padding, plot_max + x_padding)

    top_axis.text(
        -0.07,
        0.5,
        format_plot_text("Human severities"),
        rotation=90,
        va="center",
        ha="center",
        transform=top_axis.transAxes,
        fontsize=11,
    )
    bottom_axis.text(
        -0.07,
        0.5,
        format_plot_text("Model severities"),
        rotation=90,
        va="center",
        ha="center",
        transform=bottom_axis.transAxes,
        fontsize=11,
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

    preferred_order = ["anthropic", "openai", "google", "deepseek", "xai", "unknown"]
    seen = set(provider_slugs)
    return [provider_slug for provider_slug in preferred_order if provider_slug in seen]
