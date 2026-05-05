"""Plot helpers for question-order severity shift analyses."""

from dataclasses import dataclass
from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd

from mhs_llms.facets.model_severity_figure import _read_facets_score_table
from mhs_llms.labels import infer_provider, model_id_to_plot_label
from mhs_llms.plotting import apply_plot_style, format_plot_text, get_provider_color, save_figure


OPEN_MODEL_PROVIDERS = {"deepseek", "minimax", "moonshotai", "qwen", "xiaomi", "zai"}
MODEL_GROUP_SORT_COLUMNS = ["model_group", "display_sort_label"]
MODEL_GROUP_SORT_ASCENDING = [True, True]
MODEL_GROUP_SORT_KIND = "stable"


@dataclass(frozen=True)
class OrderShiftPlotStyle:
    """Figure-specific style settings for the order-shift comparison plot."""

    subplot_row_count: int
    subplot_column_count: int
    figure_width: float
    figure_min_height: float
    figure_row_height: float
    figure_height_padding: float
    subplot_width_ratios: list[float]
    subplot_wspace: float
    marker_format: str
    severity_condition_y_offset: float
    severity_marker_size: float
    severity_error_color: str
    severity_error_capsize: float
    severity_marker_edge_width: float
    severity_original_marker_face: str | None
    severity_reverse_marker_face: str
    severity_original_zorder: int
    severity_reverse_zorder: int
    severity_null_line_value: float
    severity_null_line_color: str
    severity_null_line_width: float
    severity_null_line_style: str
    severity_xlabel: str
    odds_ratio_marker_size: float
    odds_ratio_error_color: str
    odds_ratio_error_capsize: float
    odds_ratio_marker_edge_color: str
    odds_ratio_marker_edge_width: float
    odds_ratio_zorder: int
    odds_ratio_null_value: float
    odds_ratio_null_line_color: str
    odds_ratio_null_line_width: float
    odds_ratio_xlabel: str
    odds_ratio_scale: str
    pooled_band_color: str
    pooled_band_alpha: float
    pooled_band_zorder: int
    pooled_line_color: str
    pooled_line_style: str
    pooled_line_width: float
    pooled_line_zorder: int
    left_band_color: str
    left_band_half_height: float
    left_band_zorder: int
    left_band_row_interval: int
    left_band_row_remainder: int
    right_face_color: str
    right_spines_visible: bool
    right_y_ticks_visible: bool
    grid_axis: str
    grid_alpha: float
    x_tick_label_size: float
    y_tick_label_size: float
    empty_y_tick_label: str
    panel_a_label: str
    panel_b_label: str
    panel_a_label_x: float
    panel_b_label_x: float
    panel_label_y: float
    subplot_label_size: float
    subplot_label_horizontal_alignment: str
    subplot_label_vertical_alignment: str
    direction_left_label: str
    direction_right_label: str
    direction_left_x: float
    direction_right_x: float
    direction_label_y: float
    direction_label_size: float
    direction_left_horizontal_alignment: str
    direction_right_horizontal_alignment: str
    direction_vertical_alignment: str
    legend_original_label: str
    legend_reverse_label: str
    legend_marker_color: str
    legend_line_coordinates: list[float]
    legend_line_color: str
    legend_marker_scale: float
    legend_location: str
    legend_font_size: float
    legend_frame_on: bool
    right_tick_step: float
    right_tick_epsilon: float
    balanced_log_limit_floor: float
    balanced_log_limit_padding: float
    significance_marker_x: float
    significance_marker_x_offset_points: float
    significance_marker_y_offset_points: float
    significance_marker_text_coords: str
    significance_marker_font_size: float
    significance_marker_color: str
    significance_marker_weight: str
    significance_marker_horizontal_alignment: str
    significance_marker_vertical_alignment: str
    significance_marker_clip_on: bool
    significance_marker_zorder: int
    significance_p001_marker: str
    significance_p05_marker: str
    significance_p10_marker: str
    significance_p001_threshold: float
    significance_p05_threshold: float
    significance_p10_threshold: float


def load_order_shift_comparison(
    original_judges_path: Path,
    reverse_judges_path: Path,
) -> pd.DataFrame:
    """Load and pair original/reverse separated judge severity estimates."""

    original = _load_judge_scores(original_judges_path, "original")
    reverse = _load_judge_scores(reverse_judges_path, "reverse")
    original_ids = set(original["facet_label"].tolist())
    reverse_ids = set(reverse["facet_label"].tolist())
    if original_ids != reverse_ids:
        missing_reverse = ", ".join(sorted(original_ids.difference(reverse_ids)))
        missing_original = ", ".join(sorted(reverse_ids.difference(original_ids)))
        raise ValueError(
            "Original and reverse judge score files must contain the same model ids; "
            f"missing reverse={missing_reverse}; missing original={missing_original}"
        )

    comparison = original.merge(reverse, on="facet_label", how="inner")
    comparison["severity_delta"] = comparison["reverse_measure"] - comparison["original_measure"]
    comparison["delta_se_independent"] = (
        comparison["reverse_s_e"].astype(float) ** 2
        + comparison["original_s_e"].astype(float) ** 2
    ) ** 0.5
    comparison["provider"] = comparison["facet_label"].map(infer_provider)
    comparison["display_label"] = comparison["facet_label"].map(model_id_to_plot_label)
    comparison["model_group"] = comparison["provider"].map(_model_group_order)
    comparison["display_sort_label"] = comparison["display_label"].str.casefold()
    return comparison.sort_values(
        MODEL_GROUP_SORT_COLUMNS,
        ascending=MODEL_GROUP_SORT_ASCENDING,
        kind=MODEL_GROUP_SORT_KIND,
    ).reset_index(drop=True)


def load_pooled_order_delta(order_contrast_path: Path) -> tuple[float, float | None]:
    """Load the pooled reverse-minus-original order contrast and optional SE."""

    contrast = pd.read_csv(order_contrast_path)
    required_columns = {"order_measure_delta"}
    missing = required_columns.difference(contrast.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Order contrast file is missing required columns: {missing_text}")
    delta = float(contrast.loc[0, "order_measure_delta"])
    delta_se = None
    if "order_delta_se_independent" in contrast.columns:
        delta_se = float(contrast.loc[0, "order_delta_se_independent"])
    return delta, delta_se


def plot_order_shift_comparison(
    comparison: pd.DataFrame,
    output_path: Path,
    style: OrderShiftPlotStyle,
    pooled_delta: float | None = None,
    pooled_delta_se: float | None = None,
) -> Path:
    """Plot separated original/reverse severities and model-level deltas."""

    if comparison.empty:
        raise ValueError("Order-shift comparison frame is empty")
    apply_plot_style()

    row_count = len(comparison)
    figure, (severity_axis, delta_axis) = plt.subplots(
        style.subplot_row_count,
        style.subplot_column_count,
        figsize=(
            style.figure_width,
            max(style.figure_min_height, row_count * style.figure_row_height + style.figure_height_padding),
        ),
        gridspec_kw={"width_ratios": style.subplot_width_ratios, "wspace": style.subplot_wspace},
    )
    y_positions = list(range(row_count))
    colors = [get_provider_color(provider) for provider in comparison["provider"].tolist()]

    for y_position in y_positions:
        if y_position % style.left_band_row_interval == style.left_band_row_remainder:
            severity_axis.axhspan(
                y_position - style.left_band_half_height,
                y_position + style.left_band_half_height,
                color=style.left_band_color,
                zorder=style.left_band_zorder,
            )

    for y_position, row, color in zip(y_positions, comparison.itertuples(index=False), colors, strict=True):
        original_y = y_position - style.severity_condition_y_offset
        reverse_y = y_position + style.severity_condition_y_offset
        severity_axis.errorbar(
            row.original_measure,
            original_y,
            xerr=row.original_s_e,
            fmt=style.marker_format,
            color=color,
            ecolor=style.severity_error_color,
            capsize=style.severity_error_capsize,
            markerfacecolor=color if style.severity_original_marker_face is None else style.severity_original_marker_face,
            markeredgecolor=color,
            markersize=style.severity_marker_size,
            zorder=style.severity_original_zorder,
        )
        severity_axis.errorbar(
            row.reverse_measure,
            reverse_y,
            xerr=row.reverse_s_e,
            fmt=style.marker_format,
            color=color,
            ecolor=style.severity_error_color,
            capsize=style.severity_error_capsize,
            markerfacecolor=style.severity_reverse_marker_face,
            markeredgecolor=color,
            markeredgewidth=style.severity_marker_edge_width,
            markersize=style.severity_marker_size,
            zorder=style.severity_reverse_zorder,
        )

    odds_ratio = comparison["severity_delta"].map(math.exp)
    lower_ratio = (comparison["severity_delta"] - comparison["delta_se_independent"]).map(math.exp)
    upper_ratio = (comparison["severity_delta"] + comparison["delta_se_independent"]).map(math.exp)
    balanced_log_limit = _balanced_log_limit(
        lower_log_values=comparison["severity_delta"] - comparison["delta_se_independent"],
        upper_log_values=comparison["severity_delta"] + comparison["delta_se_independent"],
        style=style,
    )
    xerr = [odds_ratio - lower_ratio, upper_ratio - odds_ratio]
    for y_position, ratio, error, color in zip(
        y_positions,
        odds_ratio,
        zip(xerr[0], xerr[1], strict=True),
        colors,
        strict=True,
    ):
        delta_axis.errorbar(
            ratio,
            y_position,
            xerr=[[error[0]], [error[1]]],
            fmt=style.marker_format,
            color=color,
            ecolor=style.odds_ratio_error_color,
            capsize=style.odds_ratio_error_capsize,
            markersize=style.odds_ratio_marker_size,
            markeredgecolor=style.odds_ratio_marker_edge_color,
            markeredgewidth=style.odds_ratio_marker_edge_width,
            zorder=style.odds_ratio_zorder,
        )
    if pooled_delta is not None:
        pooled_ratio = math.exp(pooled_delta)
        if pooled_delta_se is not None:
            delta_axis.axvspan(
                math.exp(pooled_delta - pooled_delta_se),
                math.exp(pooled_delta + pooled_delta_se),
                color=style.pooled_band_color,
                alpha=style.pooled_band_alpha,
                zorder=style.pooled_band_zorder,
            )
        delta_axis.axvline(
            pooled_ratio,
            color=style.pooled_line_color,
            linestyle=style.pooled_line_style,
            linewidth=style.pooled_line_width,
            zorder=style.pooled_line_zorder,
        )
    odds_xlim = _odds_ratio_xlim(balanced_log_limit, style)
    delta_axis.set_xlim(*odds_xlim)
    _set_odds_ratio_ticks(delta_axis, odds_xlim, style)
    _add_significance_markers(delta_axis, comparison, style)

    labels = format_plot_text(comparison["display_label"].tolist())
    severity_axis.set_yticks(y_positions)
    severity_axis.set_yticklabels(labels, fontsize=style.y_tick_label_size)
    delta_axis.set_yticks(y_positions)
    delta_axis.set_yticklabels([style.empty_y_tick_label for _ in y_positions])
    delta_axis.tick_params(axis="y", left=style.right_y_ticks_visible)
    shared_y_limits = (row_count - style.left_band_half_height, -style.left_band_half_height)
    severity_axis.set_ylim(*shared_y_limits)
    delta_axis.set_ylim(*shared_y_limits)

    severity_axis.axvline(
        style.severity_null_line_value,
        color=style.severity_null_line_color,
        linewidth=style.severity_null_line_width,
        linestyle=style.severity_null_line_style,
    )
    delta_axis.axvline(
        style.odds_ratio_null_value,
        color=style.odds_ratio_null_line_color,
        linewidth=style.odds_ratio_null_line_width,
    )
    delta_axis.set_facecolor(style.right_face_color)
    for spine in delta_axis.spines.values():
        spine.set_visible(style.right_spines_visible)
    severity_axis.set_xlabel(format_plot_text(style.severity_xlabel))
    delta_axis.set_xlabel(format_plot_text(style.odds_ratio_xlabel))
    severity_axis.grid(axis=style.grid_axis, alpha=style.grid_alpha)
    delta_axis.grid(axis=style.grid_axis, alpha=style.grid_alpha)
    severity_axis.tick_params(axis=style.grid_axis, labelsize=style.x_tick_label_size)
    delta_axis.tick_params(axis=style.grid_axis, labelsize=style.x_tick_label_size)
    severity_axis.text(
        style.panel_a_label_x,
        style.panel_label_y,
        format_plot_text(style.panel_a_label),
        transform=severity_axis.transAxes,
        fontsize=style.subplot_label_size,
        ha=style.subplot_label_horizontal_alignment,
        va=style.subplot_label_vertical_alignment,
    )
    delta_axis.text(
        style.panel_b_label_x,
        style.panel_label_y,
        format_plot_text(style.panel_b_label),
        transform=delta_axis.transAxes,
        fontsize=style.subplot_label_size,
        ha=style.subplot_label_horizontal_alignment,
        va=style.subplot_label_vertical_alignment,
    )
    delta_axis.text(
        style.direction_left_x,
        style.direction_label_y,
        format_plot_text(style.direction_left_label),
        transform=delta_axis.transAxes,
        fontsize=style.direction_label_size,
        ha=style.direction_left_horizontal_alignment,
        va=style.direction_vertical_alignment,
    )
    delta_axis.text(
        style.direction_right_x,
        style.direction_label_y,
        format_plot_text(style.direction_right_label),
        transform=delta_axis.transAxes,
        fontsize=style.direction_label_size,
        ha=style.direction_right_horizontal_alignment,
        va=style.direction_vertical_alignment,
    )

    legend_handles = [
        Line2D(
            style.legend_line_coordinates,
            style.legend_line_coordinates,
            marker=style.marker_format,
            color=style.legend_line_color,
            markerfacecolor=style.legend_marker_color,
            markeredgecolor=style.legend_marker_color,
            label=style.legend_original_label,
            markersize=style.severity_marker_size,
        ),
        Line2D(
            style.legend_line_coordinates,
            style.legend_line_coordinates,
            marker=style.marker_format,
            color=style.legend_line_color,
            markerfacecolor=style.severity_reverse_marker_face,
            markeredgecolor=style.legend_marker_color,
            markeredgewidth=style.severity_marker_edge_width,
            label=style.legend_reverse_label,
            markersize=style.severity_marker_size,
        ),
    ]
    severity_axis.legend(
        handles=legend_handles,
        loc=style.legend_location,
        fontsize=style.legend_font_size,
        markerscale=style.legend_marker_scale,
        frameon=style.legend_frame_on,
    )

    plotted_path = save_figure(figure, output_path)
    plt.close(figure)
    return plotted_path


def _model_group_order(provider: str) -> int:
    """Return sorting order with closed models before open-weight models."""

    return int(provider in OPEN_MODEL_PROVIDERS)


def _balanced_log_limit(
    lower_log_values: pd.Series,
    upper_log_values: pd.Series,
    style: OrderShiftPlotStyle,
) -> float:
    """Return a symmetric log-scale axis limit around an odds ratio of one."""

    min_log_value = float(lower_log_values.min())
    max_log_value = float(upper_log_values.max())
    return max(abs(min_log_value), abs(max_log_value), style.balanced_log_limit_floor) * style.balanced_log_limit_padding


def _odds_ratio_xlim(log_limit: float, style: OrderShiftPlotStyle) -> tuple[float, float]:
    """Return odds-ratio x-limits, optionally linear-balanced around one."""

    log_limits = (math.exp(-log_limit), math.exp(log_limit))
    if style.odds_ratio_scale == "log":
        return log_limits
    linear_half_width = max(style.odds_ratio_null_value - log_limits[0], log_limits[1] - style.odds_ratio_null_value)
    return (
        style.odds_ratio_null_value - linear_half_width,
        style.odds_ratio_null_value + linear_half_width,
    )


def _set_odds_ratio_ticks(
    axis: plt.Axes,
    odds_xlim: tuple[float, float],
    style: OrderShiftPlotStyle,
) -> None:
    """Place simple odds-ratio ticks while preserving a log-balanced axis."""

    lower, upper = odds_xlim
    first_tick = math.ceil(lower / style.right_tick_step) * style.right_tick_step
    tick_values: list[float] = []
    tick_value = first_tick
    while tick_value <= upper + style.right_tick_epsilon:
        tick_values.append(round(tick_value, 1))
        tick_value += style.right_tick_step
    if style.odds_ratio_null_value not in tick_values:
        tick_values.append(style.odds_ratio_null_value)
    tick_values = sorted(value for value in tick_values if lower <= value <= upper)
    axis.set_xticks(tick_values)
    axis.set_xticklabels([f"{value:g}" for value in tick_values])


def _add_significance_markers(
    axis: plt.Axes,
    comparison: pd.DataFrame,
    style: OrderShiftPlotStyle,
) -> None:
    """Annotate significant order-shift deltas on the right side of the odds-ratio panel."""

    for y_position, row in enumerate(comparison.itertuples(index=False)):
        marker = _significance_marker(
            delta=float(row.severity_delta),
            standard_error=float(row.delta_se_independent),
            style=style,
        )
        if not marker:
            continue
        axis.annotate(
            marker,
            xy=(style.significance_marker_x, y_position),
            xycoords=axis.get_yaxis_transform(),
            xytext=(style.significance_marker_x_offset_points, style.significance_marker_y_offset_points),
            textcoords=style.significance_marker_text_coords,
            ha=style.significance_marker_horizontal_alignment,
            va=style.significance_marker_vertical_alignment,
            fontsize=style.significance_marker_font_size,
            color=style.significance_marker_color,
            fontweight=style.significance_marker_weight,
            clip_on=style.significance_marker_clip_on,
            zorder=style.significance_marker_zorder,
        )


def _significance_marker(delta: float, standard_error: float, style: OrderShiftPlotStyle) -> str:
    """Return a star marker from a two-sided normal approximation."""

    if standard_error <= 0:
        return ""
    p_value = math.erfc(abs(delta / standard_error) / math.sqrt(2.0))
    if p_value < style.significance_p001_threshold:
        return style.significance_p001_marker
    if p_value < style.significance_p05_threshold:
        return style.significance_p05_marker
    if p_value < style.significance_p10_threshold:
        return style.significance_p10_marker
    return ""


def _load_judge_scores(judges_path: Path, prefix: str) -> pd.DataFrame:
    """Load one separated judge score file with condition-prefixed columns."""

    score_frame = _read_facets_score_table(judges_path)
    required_columns = {"facet_label", "measure", "s_e"}
    missing = required_columns.difference(score_frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Judge score file is missing required columns in {judges_path}: {missing_text}")
    selected = score_frame.loc[:, ["facet_label", "measure", "s_e"]].copy()
    return selected.rename(
        columns={
            "measure": f"{prefix}_measure",
            "s_e": f"{prefix}_s_e",
        }
    )
