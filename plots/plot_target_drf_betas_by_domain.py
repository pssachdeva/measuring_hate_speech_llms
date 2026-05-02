"""Plot target-identity DRF beta terms grouped by identity domain."""

from pathlib import Path

from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_lego.labels import bold_text, fix_labels_for_tex_style
from mpl_lego.style import use_latex_style
import pandas as pd

from mhs_llms.labels import infer_provider
from mhs_llms.paths import ARTIFACTS_DIR, DATA_DIR
from mhs_llms.plotting import get_provider_color


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
IDENTITY_GROUPS = {
    "Gender": [
        ("target_gender_men", "Men"),
        ("target_gender_women", "Women"),
        ("target_gender_transgender", "Transgender"),
    ],
    "Race/ethnicity": [
        ("target_race_asian", "Asian"),
        ("target_race_black", "Black"),
        ("target_race_latinx", "Latinx"),
        ("target_race_middle_eastern", "Middle Eastern"),
        ("target_race_white", "White"),
    ],
    "Religion": [
        ("target_religion_christian", "Christian"),
        ("target_religion_jewish", "Jewish"),
        ("target_religion_muslim", "Muslim"),
    ],
    "Sexuality": [
        ("target_sexuality_bisexual", "Bisexual"),
        ("target_sexuality_gay", "Gay"),
        ("target_sexuality_lesbian", "Lesbian"),
    ],
}
TARGET_ORDER = [target_id for targets in IDENTITY_GROUPS.values() for target_id, _ in targets]
TARGET_LABELS = {
    target_id: target_label
    for targets in IDENTITY_GROUPS.values()
    for target_id, target_label in targets
}

OUTPUT_PATH = ARTIFACTS_DIR / "target_drf_betas_by_domain.png"
FIGSIZE = (8.4, 6.3)
DPI = 300
COLOR_CYCLE = [
    get_provider_color("openai"),
    get_provider_color("anthropic"),
    get_provider_color("google"),
    get_provider_color("xai"),
]
FIGURE_TOP_MARGIN = 0.90
FIGURE_BOTTOM_MARGIN = 0.20
FIGURE_LEFT_MARGIN = 0.18
FIGURE_RIGHT_MARGIN = 0.98
X_LIMIT_PADDING = 0.11
MODEL_Y_OFFSETS = [-0.24, -0.08, 0.08, 0.24]
MARKER_SIZE = 5.2
MARKER_EDGE_COLOR = "white"
MARKER_EDGE_WIDTH = 0.45
ERROR_LINE_WIDTH = 1.0
ERROR_CAP_SIZE = 2.2
ERROR_ALPHA = 0.82
SIGNIFICANT_ALPHA = 0.95
NONSIGNIFICANT_ALPHA = 0.42
SIGNIFICANCE_ALPHA = 0.05
GROUP_BAND_COLOR = "#F7F7F7"
GROUP_BAND_ALPHA = 0.74
ROW_BAND_COLOR = "#ECECEC"
ROW_BAND_ALPHA = 0.30
GROUP_DIVIDER_COLOR = "#D0D0D0"
GROUP_DIVIDER_WIDTH = 0.7
ZERO_LINE_COLOR = "#777777"
ZERO_LINE_STYLE = (0, (3, 3))
ZERO_LINE_WIDTH = 0.9
GRID_LINE_COLOR = "#BDBDBD"
GRID_LINE_WIDTH = 0.55
GRID_ALPHA = 0.22
TICK_LABEL_SIZE = 8
AXIS_LABEL_SIZE = 9
GROUP_LABEL_SIZE = 8
LEGEND_FONT_SIZE = 7.5
LEGEND_MARKER_SIZE = 5.4
LEGEND_ANCHOR = (0.56, 0.935)
LEGEND_NCOL = 4
SAVE_PAD_INCHES = 0.08
X_LABEL = r"Target-identity DRF $\beta_{jm}$ (logits)"
GROUP_LABEL_X_POSITION = -0.165
GROUP_GAP = 0.0
LEFT_DIRECTION_LABEL = "Decreased Severity"
RIGHT_DIRECTION_LABEL = "Increased Severity"
DIRECTION_LABEL_Y_POSITION = 0.01
DIRECTION_LABEL_SIZE = 9


def main() -> None:
    """Build and save a grouped target-identity DRF beta plot."""

    use_latex_style()
    plt.rcParams["axes.prop_cycle"] = cycler(color=COLOR_CYCLE)

    beta_terms = load_target_drf_terms(MODEL_TERM_PATHS, MODEL_ORDER)
    row_positions = build_row_positions()
    group_spans = build_group_spans(row_positions)
    x_limit = symmetric_x_limit(
        (beta_terms["beta_jm"].abs() + beta_terms["beta_se"].astype(float)).tolist()
    )

    figure, axis = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    figure.subplots_adjust(
        left=FIGURE_LEFT_MARGIN,
        right=FIGURE_RIGHT_MARGIN,
        top=FIGURE_TOP_MARGIN,
        bottom=FIGURE_BOTTOM_MARGIN,
    )
    add_group_bands(axis, group_spans)
    add_row_bands(axis, row_positions)
    plot_model_betas(axis, beta_terms, row_positions)
    style_axis(axis, row_positions, group_spans, x_limit)
    add_direction_labels(axis)
    add_model_legend(figure)

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
        frame["provider"] = infer_provider(model_id)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    missing_targets = sorted(set(TARGET_ORDER) - set(combined["target_identity"]))
    if missing_targets:
        missing_list = ", ".join(missing_targets)
        raise ValueError(f"Target-DRF terms are missing identities: {missing_list}")
    combined = combined.loc[combined["target_identity"].isin(TARGET_ORDER)].copy()
    combined["model_order"] = combined["model_id"].map(
        {model_id: index for index, model_id in enumerate(model_order)}
    )
    combined["target_order"] = combined["target_identity"].map(
        {target_id: index for index, target_id in enumerate(TARGET_ORDER)}
    )
    return combined.sort_values(["target_order", "model_order"], kind="stable").reset_index(
        drop=True
    )


def build_row_positions() -> dict[str, float]:
    """Return y-axis positions for target identities with gaps between domains."""

    positions = {}
    current_position = 0.0
    for targets in IDENTITY_GROUPS.values():
        for target_id, _ in targets:
            positions[target_id] = current_position
            current_position += 1.0
        current_position += GROUP_GAP
    return positions


def build_group_spans(row_positions: dict[str, float]) -> dict[str, tuple[float, float]]:
    """Return contiguous y-axis lower and upper span bounds for each identity domain."""

    spans = {}
    group_items = list(IDENTITY_GROUPS.items())
    for group_index, (group_name, targets) in enumerate(group_items):
        values = [row_positions[target_id] for target_id, _ in targets]
        lower = min(values) - 0.5
        upper = max(values) + 0.5
        if group_index > 0:
            previous_targets = group_items[group_index - 1][1]
            previous_last = row_positions[previous_targets[-1][0]]
            lower = (previous_last + min(values)) / 2.0
        if group_index < len(group_items) - 1:
            next_first = row_positions[group_items[group_index + 1][1][0][0]]
            upper = (max(values) + next_first) / 2.0
        spans[group_name] = (lower, upper)
    return spans


def symmetric_x_limit(values: list[float]) -> float:
    """Return a padded symmetric x-axis limit around zero."""

    largest_abs_value = max(abs(value) for value in values)
    return round(largest_abs_value + X_LIMIT_PADDING, 1)


def plot_model_betas(
    axis: plt.Axes,
    beta_terms: pd.DataFrame,
    row_positions: dict[str, float],
) -> None:
    """Plot model beta estimates with standard-error bars."""

    model_offsets = dict(zip(MODEL_ORDER, MODEL_Y_OFFSETS, strict=True))
    model_colors = {
        model_id: get_provider_color(infer_provider(model_id)) for model_id in MODEL_ORDER
    }
    for row in beta_terms.itertuples(index=False):
        y_position = row_positions[row.target_identity] + model_offsets[row.model_id]
        significant = float(row.p_value) < SIGNIFICANCE_ALPHA
        alpha = SIGNIFICANT_ALPHA if significant else NONSIGNIFICANT_ALPHA
        axis.errorbar(
            float(row.beta_jm),
            y_position,
            xerr=float(row.beta_se),
            fmt="o",
            markersize=MARKER_SIZE,
            color=model_colors[row.model_id],
            markeredgecolor=MARKER_EDGE_COLOR,
            markeredgewidth=MARKER_EDGE_WIDTH,
            ecolor=model_colors[row.model_id],
            elinewidth=ERROR_LINE_WIDTH,
            capsize=ERROR_CAP_SIZE,
            alpha=alpha,
            zorder=3 if significant else 2,
        )


def add_group_bands(
    axis: plt.Axes,
    group_spans: dict[str, tuple[float, float]],
) -> None:
    """Draw broad horizontal bands aligned to identity domains."""

    for group_index, (group_name, (lower, upper)) in enumerate(group_spans.items()):
        if group_index % 2 == 0:
            axis.axhspan(
                lower,
                upper,
                color=GROUP_BAND_COLOR,
                alpha=GROUP_BAND_ALPHA,
                linewidth=0,
                zorder=0,
            )
        if group_index < len(group_spans) - 1:
            axis.axhline(
                upper,
                color=GROUP_DIVIDER_COLOR,
                linewidth=GROUP_DIVIDER_WIDTH,
                zorder=1,
            )
        axis.text(
            GROUP_LABEL_X_POSITION,
            (lower + upper) / 2.0,
            bold_text(group_name),
            transform=axis.get_yaxis_transform(),
            rotation=90,
            ha="center",
            va="center",
            fontsize=GROUP_LABEL_SIZE,
            color="#333333",
            clip_on=False,
        )


def add_row_bands(axis: plt.Axes, row_positions: dict[str, float]) -> None:
    """Draw lighter sub-bands aligned to individual identity rows."""

    for row_index, y_position in enumerate(row_positions.values()):
        if row_index % 2:
            continue
        axis.axhspan(
            y_position - 0.5,
            y_position + 0.5,
            color=ROW_BAND_COLOR,
            alpha=ROW_BAND_ALPHA,
            linewidth=0,
            zorder=0.5,
        )


def style_axis(
    axis: plt.Axes,
    row_positions: dict[str, float],
    group_spans: dict[str, tuple[float, float]],
    x_limit: float,
) -> None:
    """Apply labels, ticks, gridlines, and spine cleanup."""

    axis.axvline(
        0.0,
        color=ZERO_LINE_COLOR,
        linestyle=ZERO_LINE_STYLE,
        linewidth=ZERO_LINE_WIDTH,
        zorder=1,
    )
    axis.set_xlim(-x_limit, x_limit)
    first_lower = min(lower for lower, _ in group_spans.values())
    last_upper = max(upper for _, upper in group_spans.values())
    axis.set_ylim(last_upper + 0.65, first_lower - 0.15)
    axis.set_xlabel(bold_text(X_LABEL), fontsize=AXIS_LABEL_SIZE)
    axis.set_yticks([row_positions[target_id] for target_id in TARGET_ORDER])
    tick_labels = fix_labels_for_tex_style([TARGET_LABELS[target_id] for target_id in TARGET_ORDER])
    axis.set_yticklabels([bold_text(label) for label in tick_labels], fontsize=TICK_LABEL_SIZE)
    axis.tick_params(axis="x", labelsize=TICK_LABEL_SIZE)
    axis.tick_params(axis="y", length=0)
    axis.grid(axis="x", color=GRID_LINE_COLOR, linewidth=GRID_LINE_WIDTH, alpha=GRID_ALPHA)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.spines["bottom"].set_visible(True)
    axis.spines["bottom"].set_color("#222222")
    axis.spines["bottom"].set_linewidth(ZERO_LINE_WIDTH)


def add_direction_labels(axis: plt.Axes) -> None:
    """Add left and right labels that describe beta sign."""

    axis.text(
        0.0,
        DIRECTION_LABEL_Y_POSITION,
        bold_text(LEFT_DIRECTION_LABEL),
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=DIRECTION_LABEL_SIZE,
        color="#444444",
    )
    axis.text(
        1.0,
        DIRECTION_LABEL_Y_POSITION,
        bold_text(RIGHT_DIRECTION_LABEL),
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=DIRECTION_LABEL_SIZE,
        color="#444444",
    )


def add_model_legend(figure: plt.Figure) -> None:
    """Add a shared model-color legend."""

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=get_provider_color(infer_provider(model_id)),
            markerfacecolor=get_provider_color(infer_provider(model_id)),
            markeredgecolor=MARKER_EDGE_COLOR,
            markeredgewidth=MARKER_EDGE_WIDTH,
            linestyle="none",
            markersize=LEGEND_MARKER_SIZE,
            label=bold_text(MODEL_LABELS[model_id]),
        )
        for model_id in MODEL_ORDER
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=LEGEND_ANCHOR,
        ncol=LEGEND_NCOL,
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        handletextpad=0.45,
        columnspacing=1.35,
    )


if __name__ == "__main__":
    main()
