"""Plot selected model severities against the human severity distribution."""

from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_lego.labels import bold_text
from mpl_lego.style import use_latex_style
import pandas as pd

from mhs_llms.facets.model_severity_figure import (
    load_human_judge_severities,
    load_model_judge_severities,
)
from mhs_llms.paths import ARTIFACTS_DIR, FACETS_DIR
from mhs_llms.plotting import build_gaussian_kde_curve, get_provider_color


SUBSET_MODEL_IDS = [
    "openai_gpt-4o",
    "openai_gpt-4.1",
    "openai_gpt-5.2_medium",
    "openai_gpt-5.4_medium",
    "google_gemini-2.5-pro",
    "google_gemini-3-flash-preview_medium",
    "google_gemini-3.1-pro-preview_medium",
    "anthropic_claude-haiku-4-5",
    "anthropic_claude-sonnet-4-5",
    "anthropic_claude-opus-4-5_medium",
    "anthropic_claude-sonnet-4-6_medium",
    "anthropic_claude-opus-4-6_medium",
    "xai_grok-3",
    "xai_grok-4-fast-reasoning",
    "xai_grok-4-1-fast-reasoning",
    "openrouter_moonshotai_kimi-k2.5",
    "openrouter_xiaomi_mimo-v2-pro",
    "openrouter_deepseek_deepseek-v3.2",
    "openrouter_minimax_minimax-m2.5",
]

HUMAN_SCORE_PATH = FACETS_DIR / "human_baseline" / "human_facets_scores.2.txt"
JUDGE_SCORE_PATHS = [
    FACETS_DIR / "reference_set_openai" / "judges_scores.csv",
    FACETS_DIR / "reference_set_anthropic" / "judges_scores.csv",
    FACETS_DIR / "reference_set_google" / "judges_scores.csv",
    FACETS_DIR / "reference_set_open_large" / "judges_scores.csv",
    FACETS_DIR / "reference_set_xai" / "judges_scores.csv",
]
OUTPUT_PATH = ARTIFACTS_DIR / "reference_set_model_severities_subset.png"

FIGSIZE = (8.0, 6.0)
DPI = 300
HEIGHT_RATIOS = [0.9, 1.9]
HSPACE = 0.02
X_LIMITS = (-1.5, 1.5)
BOTTOM_Y_LIMIT_TOP = -0.90
BOTTOM_Y_LIMIT_BOTTOM_OFFSET = -0.1

HUMAN_DISTRIBUTION_COLOR = "#8CA2CF"
AXIS_BACKGROUND_COLOR = "#F7F7F7"
ZERO_LINE_COLOR = "#202020"
BOTTOM_ZERO_LINE_COLOR = "#444444"
GRID_ALPHA = 0.18
BAR_ALPHA = 0.95
DIRECTION_LABEL_COLOR = "#555555"
DIRECTION_BOX_STYLE = {
    "boxstyle": "round,pad=0.42,rounding_size=0.28",
    "facecolor": "#F4F4F4",
    "edgecolor": "#D0D0D0",
    "linewidth": 0.8,
    "alpha": 0.72,
}

COLOR_CYCLE = [
    "#10a37f",
    "#d4a574",
    "#4285f4",
    "#888888",
    "#7c3aed",
    "#0ea5e9",
    "#ff6900",
    "#e11d48",
]

TOP_Y_LABEL = "Human Annotator\nSeverity Density"
BOTTOM_X_LABEL = "Severity"
BOTTOM_Y_LABEL = "Models"
LEFT_DIRECTION_LABEL = "More Likely to\nLabel as Hateful"
RIGHT_DIRECTION_LABEL = "Less Likely to\nLabel as Hateful"

TOP_LABEL_SIZE = 8
BOTTOM_Y_LABEL_SIZE = 10
TICK_LABEL_SIZE = 8
BOTTOM_MODEL_LABEL_SIZE = 6.5
VALUE_LABEL_SIZE = 8
LEGEND_FONT_SIZE = 7
DIRECTION_LABEL_SIZE = 8

HUMAN_LINE_WIDTH = 2.6
MODEL_REFERENCE_LINE_WIDTH = 1.6
ZERO_LINE_WIDTH = 1.1
BAR_EDGE_LINE_WIDTH = 1.1
ERROR_LINE_WIDTH = 1.3
ERROR_CAP_SIZE = 3.2
BOTTOM_Y_LABEL_PAD = -8
MODEL_LABEL_PAD_FRACTION = 0.012
VALUE_LABEL_PAD_FRACTION = 0.020
DIRECTION_LABEL_INSET_FRACTION = 0.09
DIRECTION_LABEL_Y = 0.78

PROVIDER_ORDER = [
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
PROVIDER_LABELS = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "google": "Google",
    "xai": "xAI",
    "deepseek": "DeepSeek",
    "minimax": "MiniMax",
    "moonshotai": "Moonshot AI",
    "qwen": "Qwen",
    "xiaomi": "Xiaomi",
    "zai": "Z.ai",
    "openrouter": "OpenRouter",
    "unknown": "Other",
}


def main() -> None:
    """Build and save the selected model severity figure."""

    use_latex_style()
    plt.rcParams["axes.prop_cycle"] = cycler(color=COLOR_CYCLE)

    human_severity_frame = load_human_judge_severities(HUMAN_SCORE_PATH)
    model_severity_frame = load_model_judge_severities(JUDGE_SCORE_PATHS)
    subset_frame = select_model_subset(model_severity_frame, SUBSET_MODEL_IDS)

    fig, (top_axis, bottom_axis) = plt.subplots(
        2,
        1,
        figsize=FIGSIZE,
        sharex=True,
        gridspec_kw={"height_ratios": HEIGHT_RATIOS, "hspace": HSPACE},
    )
    top_axis.set_facecolor(AXIS_BACKGROUND_COLOR)
    bottom_axis.set_facecolor(AXIS_BACKGROUND_COLOR)

    plot_human_density(top_axis, human_severity_frame, subset_frame)
    plot_model_bars(bottom_axis, subset_frame)
    add_provider_legend(bottom_axis, subset_frame)
    add_direction_labels(top_axis)

    top_axis.set_xlim(*X_LIMITS)
    bottom_axis.set_xlim(*X_LIMITS)
    fig.align_ylabels([top_axis, bottom_axis])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"output={OUTPUT_PATH.resolve()}")


def select_model_subset(model_severity_frame: pd.DataFrame, model_ids: list[str]) -> pd.DataFrame:
    """Filter a loaded severity frame to a required subset and sort by decreasing severity."""

    requested_ids = list(dict.fromkeys(model_ids))
    available_ids = set(model_severity_frame["facet_label"].astype(str).tolist())
    missing_ids = [model_id for model_id in requested_ids if model_id not in available_ids]
    if missing_ids:
        missing_list = ", ".join(missing_ids)
        raise ValueError(f"Model severity frame is missing requested models: {missing_list}")

    subset_frame = model_severity_frame.loc[
        model_severity_frame["facet_label"].isin(requested_ids)
    ].copy()
    return subset_frame.sort_values(
        ["measure", "display_label"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)


def plot_human_density(
    axis: plt.Axes,
    human_severity_frame: pd.DataFrame,
    model_severity_frame: pd.DataFrame,
) -> None:
    """Draw the human severity density and model reference lines on the top axis."""

    human_measures = human_severity_frame["measure"].astype(float).tolist()
    if not human_measures:
        raise ValueError("Human severity frame is empty")

    x_values, y_values = build_gaussian_kde_curve(
        human_measures,
        X_LIMITS[0],
        X_LIMITS[1],
    )
    axis.plot(
        x_values,
        y_values,
        color=HUMAN_DISTRIBUTION_COLOR,
        linewidth=HUMAN_LINE_WIDTH,
        zorder=2,
    )
    axis.fill_between(x_values, y_values, color=HUMAN_DISTRIBUTION_COLOR, alpha=0.32, zorder=1)
    for row in model_severity_frame.itertuples(index=False):
        axis.axvline(
            row.measure,
            color=get_provider_color(row.provider),
            linewidth=MODEL_REFERENCE_LINE_WIDTH,
            alpha=0.85,
            linestyle="--",
            zorder=3,
        )
    axis.axvline(0.0, color=ZERO_LINE_COLOR, linewidth=ZERO_LINE_WIDTH, zorder=4)
    axis.set_ylim(bottom=0.0)
    axis.set_ylabel(bold_text(TOP_Y_LABEL), fontsize=TOP_LABEL_SIZE)
    axis.grid(alpha=GRID_ALPHA)
    axis.tick_params(axis="y", labelsize=TICK_LABEL_SIZE)
    axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)


def plot_model_bars(axis: plt.Axes, model_severity_frame: pd.DataFrame) -> None:
    """Draw model severity bars, error bars, and in-panel model labels."""

    y_positions = list(range(len(model_severity_frame)))
    bar_colors = [get_provider_color(provider) for provider in model_severity_frame["provider"]]
    axis.barh(
        y_positions,
        model_severity_frame["measure"],
        xerr=model_severity_frame["s_e"],
        color=bar_colors,
        edgecolor="white",
        linewidth=BAR_EDGE_LINE_WIDTH,
        alpha=BAR_ALPHA,
        error_kw={
            "elinewidth": ERROR_LINE_WIDTH,
            "capsize": ERROR_CAP_SIZE,
            "ecolor": ZERO_LINE_COLOR,
        },
    )
    axis.set_yticks(y_positions)
    axis.set_yticklabels([])
    axis.tick_params(axis="y", which="both", right=False, left=False)
    axis.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
    axis.invert_yaxis()
    axis.set_ylim(len(model_severity_frame) + BOTTOM_Y_LIMIT_BOTTOM_OFFSET, BOTTOM_Y_LIMIT_TOP)
    axis.axvline(0.0, color=BOTTOM_ZERO_LINE_COLOR, linewidth=ZERO_LINE_WIDTH)
    axis.set_xlabel(bold_text(BOTTOM_X_LABEL))
    axis.set_ylabel(bold_text(BOTTOM_Y_LABEL), fontsize=BOTTOM_Y_LABEL_SIZE)
    axis.yaxis.labelpad = BOTTOM_Y_LABEL_PAD
    axis.grid(axis="x", alpha=GRID_ALPHA)

    label_pad = (X_LIMITS[1] - X_LIMITS[0]) * MODEL_LABEL_PAD_FRACTION
    value_pad = (X_LIMITS[1] - X_LIMITS[0]) * VALUE_LABEL_PAD_FRACTION
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
        axis.text(
            model_label_x,
            y_position,
            bold_text(row.display_label),
            va="center",
            ha=model_label_alignment,
            fontsize=BOTTOM_MODEL_LABEL_SIZE,
        )
        axis.text(
            value_x,
            y_position,
            bold_text(f"{value:.3f}"),
            va="center",
            ha=value_alignment,
            fontsize=VALUE_LABEL_SIZE,
        )


def add_provider_legend(axis: plt.Axes, model_severity_frame: pd.DataFrame) -> None:
    """Add the provider color legend to the bottom panel."""

    providers = _ordered_providers(model_severity_frame["provider"].astype(str).tolist())
    handles = [
        Patch(
            facecolor=get_provider_color(provider),
            edgecolor="none",
            label=bold_text(PROVIDER_LABELS.get(provider, provider)),
        )
        for provider in providers
    ]
    axis.legend(handles=handles, loc="upper right", frameon=True, fontsize=LEGEND_FONT_SIZE)


def add_direction_labels(axis: plt.Axes) -> None:
    """Add interpretation labels inside the top panel near both x-axis endpoints."""

    label_inset = (X_LIMITS[1] - X_LIMITS[0]) * DIRECTION_LABEL_INSET_FRACTION
    axis.text(
        X_LIMITS[0] + label_inset,
        DIRECTION_LABEL_Y,
        bold_text(LEFT_DIRECTION_LABEL),
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=DIRECTION_LABEL_SIZE,
        color=DIRECTION_LABEL_COLOR,
        bbox=DIRECTION_BOX_STYLE,
    )
    axis.text(
        X_LIMITS[1] - label_inset,
        DIRECTION_LABEL_Y,
        bold_text(RIGHT_DIRECTION_LABEL),
        transform=axis.get_xaxis_transform(),
        ha="center",
        va="bottom",
        fontsize=DIRECTION_LABEL_SIZE,
        color=DIRECTION_LABEL_COLOR,
        bbox=DIRECTION_BOX_STYLE,
    )


def _ordered_providers(provider_slugs: list[str]) -> list[str]:
    """Return provider slugs in the figure's preferred legend order."""

    seen = set(provider_slugs)
    return [provider for provider in PROVIDER_ORDER if provider in seen]


if __name__ == "__main__":
    main()
