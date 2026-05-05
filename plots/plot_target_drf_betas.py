"""Plot target-identity DRF beta terms across selected LLMs."""

from pathlib import Path

from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_lego.labels import bold_text, fix_labels_for_tex_style
from mpl_lego.style import use_latex_style
import pandas as pd

from mhs_llms.paths import ARTIFACTS_DIR, DATA_DIR


MODEL_TERM_PATHS = {
    "openai_gpt-5.4_medium": DATA_DIR / "full_set_openai_target_drf_target_terms.csv",
    "anthropic_claude-opus-4-6_medium": DATA_DIR / "full_set_anthropic_target_drf_target_terms.csv",
    "google_gemini-3.1-pro-preview_medium": DATA_DIR / "target_drf_google_target_terms.csv",
    "xai_grok-4-1-fast-reasoning": DATA_DIR / "full_set_xai_target_drf_target_terms.csv",
}
MODEL_ORDER = [
    "openai_gpt-5.4_medium",
    "anthropic_claude-opus-4-6_medium",
    "google_gemini-3.1-pro-preview_medium",
    "xai_grok-4-1-fast-reasoning",
]
MODEL_LABELS = {
    "openai_gpt-5.4_medium": "GPT-5.4",
    "anthropic_claude-opus-4-6_medium": "Claude Opus 4.6",
    "google_gemini-3.1-pro-preview_medium": "Gemini 3.1 Pro",
    "xai_grok-4-1-fast-reasoning": "Grok 4.1 Fast",
}
TARGET_ORDER = [
    "target_gender_men",
    "target_gender_women",
    "target_gender_transgender",
    "target_origin",
    "target_race_asian",
    "target_race_black",
    "target_race_latinx",
    "target_race_middle_eastern",
    "target_race_white",
    "target_religion_christian",
    "target_religion_jewish",
    "target_religion_muslim",
    "target_sexuality_bisexual",
    "target_sexuality_gay",
    "target_sexuality_lesbian",
]
TARGET_LABELS = {
    "target_gender_men": "Men",
    "target_gender_women": "Women",
    "target_gender_transgender": "Transgender",
    "target_origin": "Origin",
    "target_race_asian": "Asian",
    "target_race_black": "Black",
    "target_race_latinx": "Latinx",
    "target_race_middle_eastern": "Middle Eastern",
    "target_race_white": "White",
    "target_religion_christian": "Christian",
    "target_religion_jewish": "Jewish",
    "target_religion_muslim": "Muslim",
    "target_sexuality_bisexual": "Bisexual",
    "target_sexuality_gay": "Gay",
    "target_sexuality_lesbian": "Lesbian",
}
TARGET_COLOR_CYCLE = [
    "#332288",
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#999933",
    "#DDCC77",
    "#CC6677",
    "#882255",
    "#AA4499",
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
]

OUTPUT_PATH = ARTIFACTS_DIR / "target_drf_betas.png"
FIGSIZE = (9.2, 3.45)
DPI = 300
FIGURE_TOP_MARGIN = 0.72
X_LIMIT_PADDING = 0.12
ROW_BAND_COLOR = "#F4F4F4"
ROW_BAND_ALPHA = 0.78
MARKER_SIZE = 42
MARKER_EDGE_COLOR = "#333333"
MARKER_EDGE_WIDTH = 0.35
SIGNIFICANT_MARKER = "o"
NONSIGNIFICANT_MARKER = "x"
SIGNIFICANCE_ALPHA = 0.05
SIGNIFICANT_ALPHA = 0.92
NONSIGNIFICANT_ALPHA = 0.60
ZERO_LINE_COLOR = "#8A8A8A"
ZERO_LINE_STYLE = (0, (3, 3))
ZERO_LINE_WIDTH = 0.9
GRID_LINE_WIDTH = 0.6
GRID_ALPHA = 0.20
TICK_LABEL_SIZE = 7.5
MODEL_LABEL_SIZE = 8.5
AXIS_LABEL_SIZE = 9
DIRECTION_LABEL_SIZE = 8
LEGEND_FONT_SIZE = 6.6
TARGET_LEGEND_FONT_SIZE = 7.0
TARGET_LEGEND_ANCHOR = (0.5, 1.0)
SIGNIFICANCE_LEGEND_ANCHOR = (0.82, 0.82)
TARGET_LEGEND_NCOL = 5
SIGNIFICANCE_LEGEND_NCOL = 2
TARGET_LEGEND_COLUMN_SPACING = 1.0
SIGNIFICANCE_LEGEND_COLUMN_SPACING = 1.25
LEGEND_HANDLE_TEXT_PAD = 0.45
SAVE_PAD_INCHES = 0.08
Y_LIMIT_BOTTOM = -0.88
Y_LIMIT_TOP_PADDING = 0.30
DIRECTION_LABEL_Y_POSITION = -0.68
LEFT_DIRECTION_LABEL = r"Negative $\beta_{jm}$"
RIGHT_DIRECTION_LABEL = r"Positive $\beta_{jm}$"
X_LABEL = r"Target-identity DRF $\beta_{jm}$ (logits)"


def main() -> None:
    """Build and save the target-identity DRF beta plot."""

    use_latex_style()
    plt.rcParams["axes.prop_cycle"] = cycler(color=TARGET_COLOR_CYCLE)

    beta_terms = load_target_drf_terms(MODEL_TERM_PATHS, MODEL_ORDER)
    target_colors = dict(zip(TARGET_ORDER, TARGET_COLOR_CYCLE, strict=True))
    y_positions = {model_id: index for index, model_id in enumerate(reversed(MODEL_ORDER))}
    x_limit = symmetric_x_limit(beta_terms["beta_jm"].astype(float).tolist())

    figure, axis = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    figure.subplots_adjust(top=FIGURE_TOP_MARGIN)
    add_row_bands(axis, y_positions)
    plot_beta_terms(axis, beta_terms, y_positions, target_colors)
    style_axis(axis, y_positions, x_limit)
    add_direction_labels(axis, x_limit)
    add_legends(figure, target_colors)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
    plt.close(figure)
    print(f"output={OUTPUT_PATH.resolve()}")


def load_target_drf_terms(data_paths: dict[str, Path], model_order: list[str]) -> pd.DataFrame:
    """Load and combine target-DRF beta terms for selected models."""

    frames = []
    for model_id in model_order:
        data_path = data_paths[model_id]
        if not data_path.exists():
            raise FileNotFoundError(f"Missing target-DRF terms file: {data_path}")
        frame = pd.read_csv(data_path)
        frame["model_id"] = model_id
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    missing_targets = sorted(set(combined["target_identity"]) - set(TARGET_ORDER))
    if missing_targets:
        missing_list = ", ".join(missing_targets)
        raise ValueError(f"Target order is missing identities: {missing_list}")

    combined["model_order"] = combined["model_id"].map(
        {model_id: index for index, model_id in enumerate(model_order)}
    )
    combined["target_order"] = combined["target_identity"].map(
        {target_id: index for index, target_id in enumerate(TARGET_ORDER)}
    )
    return combined.sort_values(["model_order", "target_order"], kind="stable").reset_index(
        drop=True
    )


def symmetric_x_limit(values: list[float]) -> float:
    """Return a padded symmetric x-axis limit around zero."""

    largest_abs_value = max(abs(value) for value in values)
    return round(largest_abs_value + X_LIMIT_PADDING, 1)


def plot_beta_terms(
    axis: plt.Axes,
    beta_terms: pd.DataFrame,
    y_positions: dict[str, int],
    target_colors: dict[str, str],
) -> None:
    """Draw one marker per model-target beta term."""

    for row in beta_terms.itertuples(index=False):
        significant = float(row.p_value) < SIGNIFICANCE_ALPHA
        marker = SIGNIFICANT_MARKER if significant else NONSIGNIFICANT_MARKER
        alpha = SIGNIFICANT_ALPHA if significant else NONSIGNIFICANT_ALPHA
        marker_style = {
            "marker": marker,
            "s": MARKER_SIZE,
            "color": target_colors[row.target_identity],
            "linewidths": 1.0,
            "alpha": alpha,
            "zorder": 3 if significant else 2,
        }
        if significant:
            marker_style["edgecolors"] = MARKER_EDGE_COLOR
            marker_style["linewidths"] = MARKER_EDGE_WIDTH
        axis.scatter(float(row.beta_jm), y_positions[row.model_id], **marker_style)


def add_row_bands(axis: plt.Axes, y_positions: dict[str, int]) -> None:
    """Draw alternating row bands behind model rows."""

    for row_index, y_position in enumerate(y_positions.values()):
        if row_index % 2:
            continue
        axis.axhspan(
            y_position - 0.5,
            y_position + 0.5,
            color=ROW_BAND_COLOR,
            alpha=ROW_BAND_ALPHA,
            linewidth=0,
            zorder=0,
        )


def style_axis(axis: plt.Axes, y_positions: dict[str, int], x_limit: float) -> None:
    """Apply axis labels, model labels, gridlines, and spine cleanup."""

    axis.axvline(
        0,
        color=ZERO_LINE_COLOR,
        linestyle=ZERO_LINE_STYLE,
        linewidth=ZERO_LINE_WIDTH,
        zorder=1,
    )
    axis.set_xlim(-x_limit, x_limit)
    axis.set_ylim(Y_LIMIT_BOTTOM, len(y_positions) - Y_LIMIT_TOP_PADDING)
    axis.set_xlabel(bold_text(X_LABEL), fontsize=AXIS_LABEL_SIZE)
    axis.set_yticks(list(y_positions.values()))
    model_labels = [
        bold_text(label)
        for label in fix_labels_for_tex_style([MODEL_LABELS[model_id] for model_id in y_positions])
    ]
    axis.set_yticklabels(model_labels, fontsize=MODEL_LABEL_SIZE)
    axis.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
    axis.tick_params(axis="y", length=0)
    axis.grid(axis="x", color="#BDBDBD", linewidth=GRID_LINE_WIDTH, alpha=GRID_ALPHA)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.spines["bottom"].set_visible(True)
    axis.spines["bottom"].set_color("#222222")
    axis.spines["bottom"].set_linewidth(ZERO_LINE_WIDTH)


def add_direction_labels(axis: plt.Axes, x_limit: float) -> None:
    """Add text labels describing the sign of the beta axis."""

    axis.text(
        -x_limit,
        DIRECTION_LABEL_Y_POSITION,
        bold_text(LEFT_DIRECTION_LABEL),
        ha="left",
        va="bottom",
        fontsize=DIRECTION_LABEL_SIZE,
        color="#444444",
    )
    axis.text(
        x_limit,
        DIRECTION_LABEL_Y_POSITION,
        bold_text(RIGHT_DIRECTION_LABEL),
        ha="right",
        va="bottom",
        fontsize=DIRECTION_LABEL_SIZE,
        color="#444444",
    )


def add_legends(figure: plt.Figure, target_colors: dict[str, str]) -> None:
    """Add target-color and significance legends."""

    target_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=target_colors[target_id],
            markeredgecolor=MARKER_EDGE_COLOR,
            markeredgewidth=MARKER_EDGE_WIDTH,
            markersize=5,
            label=bold_text(TARGET_LABELS[target_id]),
        )
        for target_id in TARGET_ORDER
    ]
    significance_handles = [
        Line2D(
            [0],
            [0],
            marker=SIGNIFICANT_MARKER,
            color="#333333",
            linestyle="none",
            markersize=5,
            label=r"$p < 0.05$",
        ),
        Line2D(
            [0],
            [0],
            marker=NONSIGNIFICANT_MARKER,
            color="#333333",
            linestyle="none",
            markersize=5,
            label=r"$p \geq 0.05$",
        ),
    ]

    figure.legend(
        handles=target_handles,
        loc="upper center",
        bbox_to_anchor=TARGET_LEGEND_ANCHOR,
        ncol=TARGET_LEGEND_NCOL,
        columnspacing=TARGET_LEGEND_COLUMN_SPACING,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
        frameon=False,
        fontsize=TARGET_LEGEND_FONT_SIZE,
    )
    figure.legend(
        handles=significance_handles,
        loc="upper center",
        bbox_to_anchor=SIGNIFICANCE_LEGEND_ANCHOR,
        ncol=SIGNIFICANCE_LEGEND_NCOL,
        columnspacing=SIGNIFICANCE_LEGEND_COLUMN_SPACING,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
    )


if __name__ == "__main__":
    main()
