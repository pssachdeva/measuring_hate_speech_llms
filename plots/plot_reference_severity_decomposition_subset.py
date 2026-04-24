"""Plot selected model item-dependent severity adjustments."""

from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.lines import Line2D
from mpl_lego.labels import bold_text, fix_labels_for_tex_style
from mpl_lego.style import use_latex_style
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageChops

from mhs_llms.labels import model_id_to_plot_label
from mhs_llms.paths import ARTIFACTS_DIR, DATA_DIR


SUBSET_MODEL_IDS = [
    "openai_gpt-4o",
    "openai_gpt-5.4_medium",
    "google_gemini-2.5-pro",
    "google_gemini-3.1-pro-preview_medium",
    "anthropic_claude-sonnet-4-6_medium",
    "anthropic_claude-opus-4-6_medium",
    "xai_grok-3",
    "xai_grok-4-fast-reasoning",
    "openrouter_moonshotai_kimi-k2.5",
    "openrouter_deepseek_deepseek-v3.2",
]
MODEL_LABELS = {
    "openai_gpt-4o": "GPT-4o",
    "openai_gpt-5.4_medium": "GPT-5.4",
    "google_gemini-2.5-pro": "Gemini 2.5 Pro",
    "google_gemini-3.1-pro-preview_medium": "Gemini 3.1 Pro",
    "anthropic_claude-sonnet-4-6_medium": "Claude Sonnet 4.6",
    "anthropic_claude-opus-4-6_medium": "Claude Opus 4.6",
    "xai_grok-3": "Grok 3",
    "xai_grok-4-fast-reasoning": "Grok 4",
    "openrouter_moonshotai_kimi-k2.5": "Kimi K2.5",
    "openrouter_deepseek_deepseek-v3.2": "DeepSeek V3.2",
}
MODEL_LOGOS = {
    "openai_gpt-4o": "openai.png",
    "openai_gpt-5.4_medium": "openai.png",
    "google_gemini-2.5-pro": "google.png",
    "google_gemini-3.1-pro-preview_medium": "google.png",
    "anthropic_claude-sonnet-4-6_medium": "anthropic.png",
    "anthropic_claude-opus-4-6_medium": "anthropic.png",
    "xai_grok-3": "xai.png",
    "xai_grok-4-fast-reasoning": "xai.png",
    "openrouter_moonshotai_kimi-k2.5": "kimi.png",
    "openrouter_deepseek_deepseek-v3.2": "deepseek.png",
}

ITEM_ORDER = [
    "sentiment",
    "respect",
    "insult",
    "humiliate",
    "status",
    "dehumanize",
    "violence",
    "genocide",
    "attack_defend",
    "hate_speech",
]
ITEM_LABELS = {
    "sentiment": "Sentiment",
    "respect": "Respect",
    "insult": "Insult",
    "humiliate": "Humiliate",
    "status": "Status",
    "dehumanize": "Dehumanize",
    "violence": "Violence",
    "genocide": "Genocide",
    "attack_defend": "Attack/Defend",
    "hate_speech": "Hate Speech",
}

DATA_PATH = DATA_DIR / "reference_set_all_severity_decomposition_bias_terms.csv"
OUTPUT_PATH = ARTIFACTS_DIR / "reference_set_severity_decomposition_subset.png"
LOGO_DIR = ARTIFACTS_DIR / "logos"

FIGSIZE = (9.1, 4.7)
DPI = 300
FIGURE_TOP_MARGIN = 0.86
X_LIMIT_PADDING = 0.25
ROW_BAND_COLOR = "#F4F4F4"
ROW_BAND_ALPHA = 0.78
MARKER_SIZE = 32
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
AXIS_LABEL_SIZE = 9
DIRECTION_LABEL_SIZE = 8
LEGEND_FONT_SIZE = 6.5
LEGEND_TITLE_SIZE = 7
ITEM_LEGEND_FONT_SIZE = 8
ITEM_LEGEND_ANCHOR = (0.5, 0.98)
SIGNIFICANCE_LEGEND_ANCHOR = (0.82, 0.9)
ITEM_LEGEND_NCOL = 5
SIGNIFICANCE_LEGEND_NCOL = 2
ITEM_LEGEND_COLUMN_SPACING = 1.1
SIGNIFICANCE_LEGEND_COLUMN_SPACING = 1.3
LEGEND_HANDLE_TEXT_PAD = 0.5
SAVE_PAD_INCHES = 0.08
LOGO_X_POSITION = 1.018
MODEL_LABEL_X_POSITION = 1.033
LOGO_ZOOM = 0.032
LOGO_CANVAS_SIZE = 256
LOGO_CONTENT_SIZE = 208
LOGO_BACKGROUND_THRESHOLD = 20
LOGO_CROP_BOXES = {
    "deepseek.png": (0, 0, 76, 256),
}

Y_LIMIT_BOTTOM = -1.08
Y_LIMIT_TOP_PADDING = 0.30
DIRECTION_LABEL_Y_POSITION = -0.88

LEFT_DIRECTION_LABEL = "Less severe than expected"
RIGHT_DIRECTION_LABEL = "More severe than expected"
X_LABEL = "Item-dependent severity adjustment"

ITEM_COLOR_CYCLE = [
    "#332288",
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#999933",
    "#DDCC77",
    "#CC6677",
    "#882255",
    "#AA4499",
    "#DDDDDD",
]


def main() -> None:
    """Build and save the selected severity decomposition plot."""

    use_latex_style()
    plt.rcParams["axes.prop_cycle"] = cycler(color=ITEM_COLOR_CYCLE)

    bias_terms = load_subset_bias_terms(DATA_PATH, SUBSET_MODEL_IDS)
    item_colors = dict(zip(ITEM_ORDER, ITEM_COLOR_CYCLE, strict=True))
    y_positions = {
        model_id: index for index, model_id in enumerate(reversed(SUBSET_MODEL_IDS))
    }
    x_limit = symmetric_x_limit(bias_terms["bias_size"].astype(float).tolist())

    fig, axis = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.subplots_adjust(top=FIGURE_TOP_MARGIN)
    add_row_bands(axis, y_positions)
    plot_bias_terms(axis, bias_terms, y_positions, item_colors)
    style_axis(axis, y_positions, x_limit)
    add_model_labels(axis, y_positions)
    add_direction_labels(axis, x_limit)
    add_legends(fig, item_colors)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight", pad_inches=SAVE_PAD_INCHES)
    plt.close(fig)
    print(f"output={OUTPUT_PATH.resolve()}")


def load_subset_bias_terms(data_path: str, model_ids: list[str]) -> pd.DataFrame:
    """Load the bias-term dataset and select models/items in plotting order."""

    bias_terms = pd.read_csv(data_path)
    missing_models = [
        model_id
        for model_id in model_ids
        if model_id not in set(bias_terms["judge_label"].astype(str).tolist())
    ]
    if missing_models:
        missing_list = ", ".join(missing_models)
        raise ValueError(f"Bias-term dataset is missing requested models: {missing_list}")

    selected = bias_terms.loc[bias_terms["judge_label"].isin(model_ids)].copy()
    selected["model_order"] = selected["judge_label"].map(
        {model_id: index for index, model_id in enumerate(model_ids)}
    )
    selected["item_order"] = selected["item_label"].map(
        {item_name: index for index, item_name in enumerate(ITEM_ORDER)}
    )
    selected["display_label"] = selected["judge_label"].map(
        lambda model_id: MODEL_LABELS.get(model_id, model_id_to_plot_label(model_id))
    )
    return selected.sort_values(["model_order", "item_order"], kind="stable").reset_index(drop=True)


def symmetric_x_limit(values: list[float]) -> float:
    """Return a padded symmetric x-axis limit around zero."""

    largest_abs_value = max(abs(value) for value in values)
    return round(largest_abs_value + X_LIMIT_PADDING, 1)


def plot_bias_terms(
    axis: plt.Axes,
    bias_terms: pd.DataFrame,
    y_positions: dict[str, int],
    item_colors: dict[str, str],
) -> None:
    """Draw one marker per model-item bias term."""

    for row in bias_terms.itertuples(index=False):
        significant = float(row.p_value) < SIGNIFICANCE_ALPHA
        marker = SIGNIFICANT_MARKER if significant else NONSIGNIFICANT_MARKER
        alpha = SIGNIFICANT_ALPHA if significant else NONSIGNIFICANT_ALPHA
        y_position = y_positions[row.judge_label]
        marker_style = {
            "marker": marker,
            "s": MARKER_SIZE,
            "color": item_colors[row.item_label],
            "linewidths": 1.0,
            "alpha": alpha,
            "zorder": 3 if significant else 2,
        }
        if significant:
            marker_style["edgecolors"] = MARKER_EDGE_COLOR
            marker_style["linewidths"] = MARKER_EDGE_WIDTH
        axis.scatter(float(row.bias_size), y_position, **marker_style)


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
    """Apply axis labels, ticks, gridlines, and spine removal."""

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
    axis.set_yticklabels([])
    axis.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
    axis.tick_params(axis="y", length=0)
    axis.grid(axis="x", color="#BDBDBD", linewidth=GRID_LINE_WIDTH, alpha=GRID_ALPHA)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.spines["bottom"].set_visible(True)
    axis.spines["bottom"].set_color("#222222")
    axis.spines["bottom"].set_linewidth(ZERO_LINE_WIDTH)


def add_model_labels(axis: plt.Axes, y_positions: dict[str, int]) -> None:
    """Place provider logo images before custom model labels."""

    for model_id, y_position in y_positions.items():
        logo_filename = MODEL_LOGOS.get(model_id)
        if logo_filename is None:
            continue
        logo_path = LOGO_DIR / logo_filename
        if not logo_path.exists():
            continue

        image = load_logo_image(logo_path)
        image_box = OffsetImage(image, zoom=LOGO_ZOOM)
        annotation = AnnotationBbox(
            image_box,
            (LOGO_X_POSITION, y_position),
            xycoords=("axes fraction", "data"),
            box_alignment=(0.5, 0.5),
            frameon=False,
            pad=0,
            clip_on=False,
            zorder=4,
        )
        axis.add_artist(annotation)
        label = MODEL_LABELS.get(model_id, model_id_to_plot_label(model_id))
        label = fix_labels_for_tex_style([label])[0]
        axis.text(
            MODEL_LABEL_X_POSITION,
            y_position,
            bold_text(label),
            transform=axis.get_yaxis_transform(),
            ha="left",
            va="center",
            fontsize=TICK_LABEL_SIZE,
            color="#000000",
            clip_on=False,
        )


def load_logo_image(logo_path: Path) -> np.ndarray:
    """Load a logo, remove white padding, and center it on a transparent canvas."""

    image = Image.open(logo_path).convert("RGBA")
    crop_box = LOGO_CROP_BOXES.get(logo_path.name)
    if crop_box is not None:
        image = image.crop(crop_box)

    mask = visible_pixel_mask(image)
    content_box = mask.getbbox()
    if content_box is not None:
        image = image.crop(content_box)
        mask = mask.crop(content_box)
        image.putalpha(ImageChops.multiply(image.getchannel("A"), mask))

    scale = min(LOGO_CONTENT_SIZE / image.width, LOGO_CONTENT_SIZE / image.height)
    target_size = (round(image.width * scale), round(image.height * scale))
    image = image.resize(target_size, Image.Resampling.LANCZOS)
    canvas = Image.new(
        "RGBA",
        (LOGO_CANVAS_SIZE, LOGO_CANVAS_SIZE),
        (255, 255, 255, 0),
    )
    x_position = (LOGO_CANVAS_SIZE - image.width) // 2
    y_position = (LOGO_CANVAS_SIZE - image.height) // 2
    canvas.alpha_composite(image, (x_position, y_position))
    return np.asarray(canvas)


def visible_pixel_mask(image: Image.Image) -> Image.Image:
    """Return a mask for pixels that differ from a white preview background."""

    white_background = Image.new("RGB", image.size, (255, 255, 255))
    color_difference = ImageChops.difference(image.convert("RGB"), white_background)
    return color_difference.convert("L").point(
        lambda value: 255 if value > LOGO_BACKGROUND_THRESHOLD else 0
    )


def add_direction_labels(axis: plt.Axes, x_limit: float) -> None:
    """Add text labels describing the sign of the bias-size axis."""

    y_position = DIRECTION_LABEL_Y_POSITION
    axis.text(
        -x_limit,
        y_position,
        bold_text(LEFT_DIRECTION_LABEL),
        ha="left",
        va="bottom",
        fontsize=DIRECTION_LABEL_SIZE,
        color="#444444",
    )
    axis.text(
        x_limit,
        y_position,
        bold_text(RIGHT_DIRECTION_LABEL),
        ha="right",
        va="bottom",
        fontsize=DIRECTION_LABEL_SIZE,
        color="#444444",
    )


def add_legends(fig: plt.Figure, item_colors: dict[str, str]) -> None:
    """Add compact legends for item colors and significance markers."""

    item_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=color,
            markeredgecolor=MARKER_EDGE_COLOR,
            markeredgewidth=MARKER_EDGE_WIDTH,
            markersize=5,
            label=bold_text(ITEM_LABELS[item_name]),
        )
        for item_name, color in item_colors.items()
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

    fig.legend(
        handles=item_handles,
        loc="upper center",
        bbox_to_anchor=ITEM_LEGEND_ANCHOR,
        ncol=ITEM_LEGEND_NCOL,
        columnspacing=ITEM_LEGEND_COLUMN_SPACING,
        handletextpad=LEGEND_HANDLE_TEXT_PAD,
        frameon=False,
        fontsize=ITEM_LEGEND_FONT_SIZE,
    )
    fig.legend(
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
