"""Plot original versus reverse question-order severity shifts."""

from mhs_llms.facets.order_effect_plot import (
    OrderShiftPlotStyle,
    load_order_shift_comparison,
    load_pooled_order_delta,
    plot_order_shift_comparison,
)
from mhs_llms.paths import ARTIFACTS_DIR, DATA_DIR, FACETS_DIR


ORIGINAL_JUDGES_PATH = FACETS_DIR / "question_order_original_matched" / "judges_scores.csv"
REVERSE_JUDGES_PATH = FACETS_DIR / "question_order_reverse_matched" / "judges_scores.csv"
POOLED_ORDER_CONTRAST_PATH = DATA_DIR / "question_order_pooled_effect_order_contrast.csv"
OUTPUT_PATH = ARTIFACTS_DIR / "question_order_severity_shift.png"

SUBPLOT_ROW_COUNT = 1
SUBPLOT_COLUMN_COUNT = 2
FIGURE_WIDTH = 9.5
FIGURE_MIN_HEIGHT = 3.2
FIGURE_ROW_HEIGHT = 0.38
FIGURE_HEIGHT_PADDING = 1.0
SUBPLOT_WIDTH_RATIOS = [1.6, 0.9]
SUBPLOT_WSPACE = 0.08

MARKER_FORMAT = "o"
SEVERITY_CONDITION_Y_OFFSET = 0.11
SEVERITY_MARKER_SIZE = 6.4
SEVERITY_ERROR_COLOR = "#303030"
SEVERITY_ERROR_CAPSIZE = 2.5
SEVERITY_MARKER_EDGE_WIDTH = 1.3
SEVERITY_ORIGINAL_MARKER_FACE = None
SEVERITY_REVERSE_MARKER_FACE = "white"
SEVERITY_ORIGINAL_ZORDER = 3
SEVERITY_REVERSE_ZORDER = 4
SEVERITY_NULL_LINE_VALUE = 0.0
SEVERITY_NULL_LINE_COLOR = "#888888"
SEVERITY_NULL_LINE_WIDTH = 1.0
SEVERITY_NULL_LINE_STYLE = "--"
SEVERITY_XLABEL = r"Severity ($\alpha_j$)"

ODDS_RATIO_MARKER_SIZE = 5.8
ODDS_RATIO_ERROR_COLOR = "#303030"
ODDS_RATIO_ERROR_CAPSIZE = 2.8
ODDS_RATIO_MARKER_EDGE_COLOR = "white"
ODDS_RATIO_MARKER_EDGE_WIDTH = 0.8
ODDS_RATIO_ZORDER = 3
ODDS_RATIO_NULL_VALUE = 1.0
ODDS_RATIO_NULL_LINE_COLOR = "#444444"
ODDS_RATIO_NULL_LINE_WIDTH = 1.0
ODDS_RATIO_XLABEL = "Change in Odds Ratio"
ODDS_RATIO_SCALE = "linear"

POOLED_BAND_COLOR = "#666666"
POOLED_BAND_ALPHA = 0.12
POOLED_BAND_ZORDER = 0
POOLED_LINE_COLOR = "#202020"
POOLED_LINE_STYLE = "--"
POOLED_LINE_WIDTH = 1.1
POOLED_LINE_ZORDER = 3

LEFT_BAND_COLOR = "#F3F3F3"
LEFT_BAND_HALF_HEIGHT = 0.5
LEFT_BAND_ZORDER = 0
LEFT_BAND_ROW_INTERVAL = 2
LEFT_BAND_ROW_REMAINDER = 0
RIGHT_FACE_COLOR = "#FAFAFA"
RIGHT_SPINES_VISIBLE = True
RIGHT_Y_TICKS_VISIBLE = False

GRID_AXIS = "x"
GRID_ALPHA = 0.18
X_TICK_LABEL_SIZE = 8
Y_TICK_LABEL_SIZE = 8
EMPTY_Y_TICK_LABEL = ""

PANEL_A_LABEL = "a"
PANEL_B_LABEL = "b"
PANEL_A_LABEL_X = -0.05
PANEL_B_LABEL_X = -0.08
PANEL_LABEL_Y = 1.02
SUBPLOT_LABEL_SIZE = 11
SUBPLOT_LABEL_HORIZONTAL_ALIGNMENT = "left"
SUBPLOT_LABEL_VERTICAL_ALIGNMENT = "bottom"
DIRECTION_LEFT_LABEL = "Reversal Lowers\nHate Threshold"
DIRECTION_RIGHT_LABEL = "Reversal Raises\nHate Threshold"
DIRECTION_LEFT_X = 0.01
DIRECTION_RIGHT_X = 0.99
DIRECTION_LABEL_Y = 1.08
DIRECTION_LABEL_SIZE = 7.5
DIRECTION_LEFT_HORIZONTAL_ALIGNMENT = "left"
DIRECTION_RIGHT_HORIZONTAL_ALIGNMENT = "right"
DIRECTION_VERTICAL_ALIGNMENT = "top"

LEGEND_ORIGINAL_LABEL = "Original"
LEGEND_REVERSE_LABEL = "Reverse"
LEGEND_MARKER_COLOR = "#555555"
LEGEND_LINE_COORDINATES = [0]
LEGEND_LINE_COLOR = "none"
LEGEND_MARKER_SCALE = 1.12
LEGEND_LOCATION = "lower left"
LEGEND_FONT_SIZE = 9
LEGEND_FRAME_ON = True

RIGHT_TICK_STEP = 0.2
RIGHT_TICK_EPSILON = 1e-9
BALANCED_LOG_LIMIT_FLOOR = 0.1
BALANCED_LOG_LIMIT_PADDING = 1.1

SIGNIFICANCE_MARKER_X = 1.18
SIGNIFICANCE_MARKER_X_OFFSET_POINTS = -20
SIGNIFICANCE_MARKER_Y_OFFSET_POINTS = 0
SIGNIFICANCE_MARKER_TEXT_COORDS = "offset points"
SIGNIFICANCE_MARKER_FONT_SIZE = 10
SIGNIFICANCE_MARKER_COLOR = "#202020"
SIGNIFICANCE_MARKER_WEIGHT = "bold"
SIGNIFICANCE_MARKER_HORIZONTAL_ALIGNMENT = "right"
SIGNIFICANCE_MARKER_VERTICAL_ALIGNMENT = "center"
SIGNIFICANCE_MARKER_CLIP_ON = False
SIGNIFICANCE_MARKER_ZORDER = 4
SIGNIFICANCE_P001_MARKER = "***"
SIGNIFICANCE_P05_MARKER = "**"
SIGNIFICANCE_P10_MARKER = "*"
SIGNIFICANCE_P001_THRESHOLD = 0.001
SIGNIFICANCE_P05_THRESHOLD = 0.05
SIGNIFICANCE_P10_THRESHOLD = 0.1

ORDER_SHIFT_STYLE = OrderShiftPlotStyle(
    subplot_row_count=SUBPLOT_ROW_COUNT,
    subplot_column_count=SUBPLOT_COLUMN_COUNT,
    figure_width=FIGURE_WIDTH,
    figure_min_height=FIGURE_MIN_HEIGHT,
    figure_row_height=FIGURE_ROW_HEIGHT,
    figure_height_padding=FIGURE_HEIGHT_PADDING,
    subplot_width_ratios=SUBPLOT_WIDTH_RATIOS,
    subplot_wspace=SUBPLOT_WSPACE,
    marker_format=MARKER_FORMAT,
    severity_condition_y_offset=SEVERITY_CONDITION_Y_OFFSET,
    severity_marker_size=SEVERITY_MARKER_SIZE,
    severity_error_color=SEVERITY_ERROR_COLOR,
    severity_error_capsize=SEVERITY_ERROR_CAPSIZE,
    severity_marker_edge_width=SEVERITY_MARKER_EDGE_WIDTH,
    severity_original_marker_face=SEVERITY_ORIGINAL_MARKER_FACE,
    severity_reverse_marker_face=SEVERITY_REVERSE_MARKER_FACE,
    severity_original_zorder=SEVERITY_ORIGINAL_ZORDER,
    severity_reverse_zorder=SEVERITY_REVERSE_ZORDER,
    severity_null_line_value=SEVERITY_NULL_LINE_VALUE,
    severity_null_line_color=SEVERITY_NULL_LINE_COLOR,
    severity_null_line_width=SEVERITY_NULL_LINE_WIDTH,
    severity_null_line_style=SEVERITY_NULL_LINE_STYLE,
    severity_xlabel=SEVERITY_XLABEL,
    odds_ratio_marker_size=ODDS_RATIO_MARKER_SIZE,
    odds_ratio_error_color=ODDS_RATIO_ERROR_COLOR,
    odds_ratio_error_capsize=ODDS_RATIO_ERROR_CAPSIZE,
    odds_ratio_marker_edge_color=ODDS_RATIO_MARKER_EDGE_COLOR,
    odds_ratio_marker_edge_width=ODDS_RATIO_MARKER_EDGE_WIDTH,
    odds_ratio_zorder=ODDS_RATIO_ZORDER,
    odds_ratio_null_value=ODDS_RATIO_NULL_VALUE,
    odds_ratio_null_line_color=ODDS_RATIO_NULL_LINE_COLOR,
    odds_ratio_null_line_width=ODDS_RATIO_NULL_LINE_WIDTH,
    odds_ratio_xlabel=ODDS_RATIO_XLABEL,
    odds_ratio_scale=ODDS_RATIO_SCALE,
    pooled_band_color=POOLED_BAND_COLOR,
    pooled_band_alpha=POOLED_BAND_ALPHA,
    pooled_band_zorder=POOLED_BAND_ZORDER,
    pooled_line_color=POOLED_LINE_COLOR,
    pooled_line_style=POOLED_LINE_STYLE,
    pooled_line_width=POOLED_LINE_WIDTH,
    pooled_line_zorder=POOLED_LINE_ZORDER,
    left_band_color=LEFT_BAND_COLOR,
    left_band_half_height=LEFT_BAND_HALF_HEIGHT,
    left_band_zorder=LEFT_BAND_ZORDER,
    left_band_row_interval=LEFT_BAND_ROW_INTERVAL,
    left_band_row_remainder=LEFT_BAND_ROW_REMAINDER,
    right_face_color=RIGHT_FACE_COLOR,
    right_spines_visible=RIGHT_SPINES_VISIBLE,
    right_y_ticks_visible=RIGHT_Y_TICKS_VISIBLE,
    grid_axis=GRID_AXIS,
    grid_alpha=GRID_ALPHA,
    x_tick_label_size=X_TICK_LABEL_SIZE,
    y_tick_label_size=Y_TICK_LABEL_SIZE,
    empty_y_tick_label=EMPTY_Y_TICK_LABEL,
    panel_a_label=PANEL_A_LABEL,
    panel_b_label=PANEL_B_LABEL,
    panel_a_label_x=PANEL_A_LABEL_X,
    panel_b_label_x=PANEL_B_LABEL_X,
    panel_label_y=PANEL_LABEL_Y,
    subplot_label_size=SUBPLOT_LABEL_SIZE,
    subplot_label_horizontal_alignment=SUBPLOT_LABEL_HORIZONTAL_ALIGNMENT,
    subplot_label_vertical_alignment=SUBPLOT_LABEL_VERTICAL_ALIGNMENT,
    direction_left_label=DIRECTION_LEFT_LABEL,
    direction_right_label=DIRECTION_RIGHT_LABEL,
    direction_left_x=DIRECTION_LEFT_X,
    direction_right_x=DIRECTION_RIGHT_X,
    direction_label_y=DIRECTION_LABEL_Y,
    direction_label_size=DIRECTION_LABEL_SIZE,
    direction_left_horizontal_alignment=DIRECTION_LEFT_HORIZONTAL_ALIGNMENT,
    direction_right_horizontal_alignment=DIRECTION_RIGHT_HORIZONTAL_ALIGNMENT,
    direction_vertical_alignment=DIRECTION_VERTICAL_ALIGNMENT,
    legend_original_label=LEGEND_ORIGINAL_LABEL,
    legend_reverse_label=LEGEND_REVERSE_LABEL,
    legend_marker_color=LEGEND_MARKER_COLOR,
    legend_line_coordinates=LEGEND_LINE_COORDINATES,
    legend_line_color=LEGEND_LINE_COLOR,
    legend_marker_scale=LEGEND_MARKER_SCALE,
    legend_location=LEGEND_LOCATION,
    legend_font_size=LEGEND_FONT_SIZE,
    legend_frame_on=LEGEND_FRAME_ON,
    right_tick_step=RIGHT_TICK_STEP,
    right_tick_epsilon=RIGHT_TICK_EPSILON,
    balanced_log_limit_floor=BALANCED_LOG_LIMIT_FLOOR,
    balanced_log_limit_padding=BALANCED_LOG_LIMIT_PADDING,
    significance_marker_x=SIGNIFICANCE_MARKER_X,
    significance_marker_x_offset_points=SIGNIFICANCE_MARKER_X_OFFSET_POINTS,
    significance_marker_y_offset_points=SIGNIFICANCE_MARKER_Y_OFFSET_POINTS,
    significance_marker_text_coords=SIGNIFICANCE_MARKER_TEXT_COORDS,
    significance_marker_font_size=SIGNIFICANCE_MARKER_FONT_SIZE,
    significance_marker_color=SIGNIFICANCE_MARKER_COLOR,
    significance_marker_weight=SIGNIFICANCE_MARKER_WEIGHT,
    significance_marker_horizontal_alignment=SIGNIFICANCE_MARKER_HORIZONTAL_ALIGNMENT,
    significance_marker_vertical_alignment=SIGNIFICANCE_MARKER_VERTICAL_ALIGNMENT,
    significance_marker_clip_on=SIGNIFICANCE_MARKER_CLIP_ON,
    significance_marker_zorder=SIGNIFICANCE_MARKER_ZORDER,
    significance_p001_marker=SIGNIFICANCE_P001_MARKER,
    significance_p05_marker=SIGNIFICANCE_P05_MARKER,
    significance_p10_marker=SIGNIFICANCE_P10_MARKER,
    significance_p001_threshold=SIGNIFICANCE_P001_THRESHOLD,
    significance_p05_threshold=SIGNIFICANCE_P05_THRESHOLD,
    significance_p10_threshold=SIGNIFICANCE_P10_THRESHOLD,
)


def main() -> None:
    """Build and save the question-order severity shift figure."""

    comparison = load_order_shift_comparison(
        original_judges_path=ORIGINAL_JUDGES_PATH,
        reverse_judges_path=REVERSE_JUDGES_PATH,
    )
    pooled_delta = None
    pooled_delta_se = None
    if POOLED_ORDER_CONTRAST_PATH.exists():
        pooled_delta, pooled_delta_se = load_pooled_order_delta(POOLED_ORDER_CONTRAST_PATH)

    plotted_path = plot_order_shift_comparison(
        comparison=comparison,
        output_path=OUTPUT_PATH,
        style=ORDER_SHIFT_STYLE,
        pooled_delta=pooled_delta,
        pooled_delta_se=pooled_delta_se,
    )
    print(f"output={plotted_path.resolve()}")
    print(f"direction_label_y={DIRECTION_LABEL_Y}")
    print(f"significance_x_offset_points={SIGNIFICANCE_MARKER_X_OFFSET_POINTS}")


if __name__ == "__main__":
    main()
