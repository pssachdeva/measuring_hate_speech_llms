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
    pooled_delta: float | None = None,
    pooled_delta_se: float | None = None,
) -> Path:
    """Plot separated original/reverse severities and model-level deltas."""

    if comparison.empty:
        raise ValueError("Order-shift comparison frame is empty")
    apply_plot_style()

    row_count = len(comparison)
    figure, (severity_axis, delta_axis) = plt.subplots(
        SUBPLOT_ROW_COUNT,
        SUBPLOT_COLUMN_COUNT,
        figsize=(
            FIGURE_WIDTH,
            max(FIGURE_MIN_HEIGHT, row_count * FIGURE_ROW_HEIGHT + FIGURE_HEIGHT_PADDING),
        ),
        gridspec_kw={"width_ratios": SUBPLOT_WIDTH_RATIOS, "wspace": SUBPLOT_WSPACE},
    )
    y_positions = list(range(row_count))
    colors = [get_provider_color(provider) for provider in comparison["provider"].tolist()]

    for y_position in y_positions:
        if y_position % LEFT_BAND_ROW_INTERVAL == LEFT_BAND_ROW_REMAINDER:
            severity_axis.axhspan(
                y_position - LEFT_BAND_HALF_HEIGHT,
                y_position + LEFT_BAND_HALF_HEIGHT,
                color=LEFT_BAND_COLOR,
                zorder=LEFT_BAND_ZORDER,
            )

    for y_position, row, color in zip(y_positions, comparison.itertuples(index=False), colors, strict=True):
        original_y = y_position - SEVERITY_CONDITION_Y_OFFSET
        reverse_y = y_position + SEVERITY_CONDITION_Y_OFFSET
        severity_axis.errorbar(
            row.original_measure,
            original_y,
            xerr=row.original_s_e,
            fmt=MARKER_FORMAT,
            color=color,
            ecolor=SEVERITY_ERROR_COLOR,
            capsize=SEVERITY_ERROR_CAPSIZE,
            markerfacecolor=color if SEVERITY_ORIGINAL_MARKER_FACE is None else SEVERITY_ORIGINAL_MARKER_FACE,
            markeredgecolor=color,
            markersize=SEVERITY_MARKER_SIZE,
            zorder=SEVERITY_ORIGINAL_ZORDER,
        )
        severity_axis.errorbar(
            row.reverse_measure,
            reverse_y,
            xerr=row.reverse_s_e,
            fmt=MARKER_FORMAT,
            color=color,
            ecolor=SEVERITY_ERROR_COLOR,
            capsize=SEVERITY_ERROR_CAPSIZE,
            markerfacecolor=SEVERITY_REVERSE_MARKER_FACE,
            markeredgecolor=color,
            markeredgewidth=SEVERITY_MARKER_EDGE_WIDTH,
            markersize=SEVERITY_MARKER_SIZE,
            zorder=SEVERITY_REVERSE_ZORDER,
        )

    odds_ratio = comparison["severity_delta"].map(math.exp)
    lower_ratio = (comparison["severity_delta"] - comparison["delta_se_independent"]).map(math.exp)
    upper_ratio = (comparison["severity_delta"] + comparison["delta_se_independent"]).map(math.exp)
    balanced_log_limit = _balanced_log_limit(
        lower_log_values=comparison["severity_delta"] - comparison["delta_se_independent"],
        upper_log_values=comparison["severity_delta"] + comparison["delta_se_independent"],
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
            fmt=MARKER_FORMAT,
            color=color,
            ecolor=ODDS_RATIO_ERROR_COLOR,
            capsize=ODDS_RATIO_ERROR_CAPSIZE,
            markersize=ODDS_RATIO_MARKER_SIZE,
            markeredgecolor=ODDS_RATIO_MARKER_EDGE_COLOR,
            markeredgewidth=ODDS_RATIO_MARKER_EDGE_WIDTH,
            zorder=ODDS_RATIO_ZORDER,
        )
    if pooled_delta is not None:
        pooled_ratio = math.exp(pooled_delta)
        if pooled_delta_se is not None:
            delta_axis.axvspan(
                math.exp(pooled_delta - pooled_delta_se),
                math.exp(pooled_delta + pooled_delta_se),
                color=POOLED_BAND_COLOR,
                alpha=POOLED_BAND_ALPHA,
                zorder=POOLED_BAND_ZORDER,
            )
        delta_axis.axvline(
            pooled_ratio,
            color=POOLED_LINE_COLOR,
            linestyle=POOLED_LINE_STYLE,
            linewidth=POOLED_LINE_WIDTH,
            zorder=POOLED_LINE_ZORDER,
        )
    delta_axis.set_xscale(ODDS_RATIO_SCALE)
    odds_xlim = (math.exp(-balanced_log_limit), math.exp(balanced_log_limit))
    delta_axis.set_xlim(*odds_xlim)
    _set_odds_ratio_ticks(delta_axis, odds_xlim)
    _add_significance_markers(delta_axis, comparison)

    labels = format_plot_text(comparison["display_label"].tolist())
    severity_axis.set_yticks(y_positions)
    severity_axis.set_yticklabels(labels, fontsize=Y_TICK_LABEL_SIZE)
    delta_axis.set_yticks(y_positions)
    delta_axis.set_yticklabels([EMPTY_Y_TICK_LABEL for _ in y_positions])
    shared_y_limits = (row_count - LEFT_BAND_HALF_HEIGHT, -LEFT_BAND_HALF_HEIGHT)
    severity_axis.set_ylim(*shared_y_limits)
    delta_axis.set_ylim(*shared_y_limits)

    severity_axis.axvline(
        SEVERITY_NULL_LINE_VALUE,
        color=SEVERITY_NULL_LINE_COLOR,
        linewidth=SEVERITY_NULL_LINE_WIDTH,
        linestyle=SEVERITY_NULL_LINE_STYLE,
    )
    delta_axis.axvline(
        ODDS_RATIO_NULL_VALUE,
        color=ODDS_RATIO_NULL_LINE_COLOR,
        linewidth=ODDS_RATIO_NULL_LINE_WIDTH,
    )
    delta_axis.set_facecolor(RIGHT_FACE_COLOR)
    severity_axis.set_xlabel(format_plot_text(SEVERITY_XLABEL))
    delta_axis.set_xlabel(format_plot_text(ODDS_RATIO_XLABEL))
    severity_axis.grid(axis=GRID_AXIS, alpha=GRID_ALPHA)
    delta_axis.grid(axis=GRID_AXIS, alpha=GRID_ALPHA)
    severity_axis.tick_params(axis=GRID_AXIS, labelsize=X_TICK_LABEL_SIZE)
    delta_axis.tick_params(axis=GRID_AXIS, labelsize=X_TICK_LABEL_SIZE)
    severity_axis.text(
        PANEL_A_LABEL_X,
        PANEL_LABEL_Y,
        format_plot_text(PANEL_A_LABEL),
        transform=severity_axis.transAxes,
        fontsize=SUBPLOT_LABEL_SIZE,
        ha=SUBPLOT_LABEL_HORIZONTAL_ALIGNMENT,
        va=SUBPLOT_LABEL_VERTICAL_ALIGNMENT,
    )
    delta_axis.text(
        PANEL_B_LABEL_X,
        PANEL_LABEL_Y,
        format_plot_text(PANEL_B_LABEL),
        transform=delta_axis.transAxes,
        fontsize=SUBPLOT_LABEL_SIZE,
        ha=SUBPLOT_LABEL_HORIZONTAL_ALIGNMENT,
        va=SUBPLOT_LABEL_VERTICAL_ALIGNMENT,
    )
    delta_axis.text(
        DIRECTION_LEFT_X,
        DIRECTION_LABEL_Y,
        format_plot_text(DIRECTION_LEFT_LABEL),
        transform=delta_axis.transAxes,
        fontsize=DIRECTION_LABEL_SIZE,
        ha=DIRECTION_LEFT_HORIZONTAL_ALIGNMENT,
        va=DIRECTION_VERTICAL_ALIGNMENT,
    )
    delta_axis.text(
        DIRECTION_RIGHT_X,
        DIRECTION_LABEL_Y,
        format_plot_text(DIRECTION_RIGHT_LABEL),
        transform=delta_axis.transAxes,
        fontsize=DIRECTION_LABEL_SIZE,
        ha=DIRECTION_RIGHT_HORIZONTAL_ALIGNMENT,
        va=DIRECTION_VERTICAL_ALIGNMENT,
    )

    legend_handles = [
        Line2D(
            LEGEND_LINE_COORDINATES,
            LEGEND_LINE_COORDINATES,
            marker=MARKER_FORMAT,
            color=LEGEND_LINE_COLOR,
            markerfacecolor=LEGEND_MARKER_COLOR,
            markeredgecolor=LEGEND_MARKER_COLOR,
            label=LEGEND_ORIGINAL_LABEL,
            markersize=SEVERITY_MARKER_SIZE,
        ),
        Line2D(
            LEGEND_LINE_COORDINATES,
            LEGEND_LINE_COORDINATES,
            marker=MARKER_FORMAT,
            color=LEGEND_LINE_COLOR,
            markerfacecolor=SEVERITY_REVERSE_MARKER_FACE,
            markeredgecolor=LEGEND_MARKER_COLOR,
            markeredgewidth=SEVERITY_MARKER_EDGE_WIDTH,
            label=LEGEND_REVERSE_LABEL,
            markersize=SEVERITY_MARKER_SIZE,
        ),
    ]
    severity_axis.legend(
        handles=legend_handles,
        loc=LEGEND_LOCATION,
        fontsize=LEGEND_FONT_SIZE,
        markerscale=LEGEND_MARKER_SCALE,
        frameon=LEGEND_FRAME_ON,
    )

    plotted_path = save_figure(figure, output_path)
    plt.close(figure)
    return plotted_path


def _model_group_order(provider: str) -> int:
    """Return sorting order with closed models before open-weight models."""

    return int(provider in OPEN_MODEL_PROVIDERS)


def _balanced_log_limit(lower_log_values: pd.Series, upper_log_values: pd.Series) -> float:
    """Return a symmetric log-scale axis limit around an odds ratio of one."""

    min_log_value = float(lower_log_values.min())
    max_log_value = float(upper_log_values.max())
    return max(abs(min_log_value), abs(max_log_value), BALANCED_LOG_LIMIT_FLOOR) * BALANCED_LOG_LIMIT_PADDING


def _set_odds_ratio_ticks(axis: plt.Axes, odds_xlim: tuple[float, float]) -> None:
    """Place simple odds-ratio ticks while preserving a log-balanced axis."""

    lower, upper = odds_xlim
    first_tick = math.ceil(lower / RIGHT_TICK_STEP) * RIGHT_TICK_STEP
    tick_values: list[float] = []
    tick_value = first_tick
    while tick_value <= upper + RIGHT_TICK_EPSILON:
        tick_values.append(round(tick_value, 1))
        tick_value += RIGHT_TICK_STEP
    if ODDS_RATIO_NULL_VALUE not in tick_values:
        tick_values.append(ODDS_RATIO_NULL_VALUE)
    tick_values = sorted(value for value in tick_values if lower <= value <= upper)
    axis.set_xticks(tick_values)
    axis.set_xticklabels([f"{value:g}" for value in tick_values])


def _add_significance_markers(
    axis: plt.Axes,
    comparison: pd.DataFrame,
) -> None:
    """Annotate significant order-shift deltas on the right side of the odds-ratio panel."""

    for y_position, row in enumerate(comparison.itertuples(index=False)):
        marker = _significance_marker(
            delta=float(row.severity_delta),
            standard_error=float(row.delta_se_independent),
        )
        if not marker:
            continue
        axis.annotate(
            marker,
            xy=(SIGNIFICANCE_MARKER_X, y_position),
            xycoords=axis.get_yaxis_transform(),
            xytext=(SIGNIFICANCE_MARKER_X_OFFSET_POINTS, SIGNIFICANCE_MARKER_Y_OFFSET_POINTS),
            textcoords=SIGNIFICANCE_MARKER_TEXT_COORDS,
            ha=SIGNIFICANCE_MARKER_HORIZONTAL_ALIGNMENT,
            va=SIGNIFICANCE_MARKER_VERTICAL_ALIGNMENT,
            fontsize=SIGNIFICANCE_MARKER_FONT_SIZE,
            color=SIGNIFICANCE_MARKER_COLOR,
            fontweight=SIGNIFICANCE_MARKER_WEIGHT,
            clip_on=SIGNIFICANCE_MARKER_CLIP_ON,
            zorder=SIGNIFICANCE_MARKER_ZORDER,
        )


def _significance_marker(delta: float, standard_error: float) -> str:
    """Return a star marker from a two-sided normal approximation."""

    if standard_error <= 0:
        return ""
    p_value = math.erfc(abs(delta / standard_error) / math.sqrt(2.0))
    if p_value < SIGNIFICANCE_P001_THRESHOLD:
        return SIGNIFICANCE_P001_MARKER
    if p_value < SIGNIFICANCE_P05_THRESHOLD:
        return SIGNIFICANCE_P05_MARKER
    if p_value < SIGNIFICANCE_P10_THRESHOLD:
        return SIGNIFICANCE_P10_MARKER
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
