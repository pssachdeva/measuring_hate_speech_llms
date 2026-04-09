"""Plot FACETS judge severities for a selected set of models."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from mhs_llms.labels import model_id_to_label
from mhs_llms.plotting import apply_plot_style, format_plot_text


REFERENCE_OPENAI_MODEL_ORDER = [
    "openai_gpt-4o",
    "openai_gpt-4.1",
    "openai_gpt-5.2_medium",
    "openai_gpt-5.4_medium",
]

REFERENCE_OPENAI_MODEL_LABELS = {
    "openai_gpt-4o": "gpt-4o",
    "openai_gpt-4.1": "gpt-4.1",
    "openai_gpt-5.2_medium": "5.2 medium",
    "openai_gpt-5.4_medium": "5.4 medium",
}

REFERENCE_ANTHROPIC_MODEL_ORDER = [
    "anthropic_claude-sonnet-4",
    "anthropic_claude-sonnet-4-5",
    "anthropic_claude-sonnet-4-6_medium",
    "anthropic_claude-sonnet-4-6_high",
]

REFERENCE_ANTHROPIC_MODEL_LABELS = {
    "anthropic_claude-sonnet-4": "claude sonnet 4",
    "anthropic_claude-sonnet-4-5": "claude sonnet 4.5",
    "anthropic_claude-sonnet-4-6_medium": "claude sonnet 4.6 (medium)",
    "anthropic_claude-sonnet-4-6_high": "claude sonnet 4.6 (high)",
}

REFERENCE_ANTHROPIC_OPUS_MODEL_ORDER = [
    "anthropic_claude-opus-4",
    "anthropic_claude-opus-4-1",
    "anthropic_claude-opus-4-5_medium",
    "anthropic_claude-opus-4-6_medium",
]

REFERENCE_ANTHROPIC_OPUS_MODEL_LABELS = {
    "anthropic_claude-opus-4": "opus 4",
    "anthropic_claude-opus-4-1": "opus 4.1",
    "anthropic_claude-opus-4-5_medium": "opus 4.5 (medium)",
    "anthropic_claude-opus-4-6_medium": "opus 4.6 (medium)",
}

REFERENCE_OPENAI_REASONING_LEVEL_ORDER = ["none", "low", "medium", "high", "xhigh"]

REFERENCE_OPENAI_REASONING_FAMILY_ORDER = [
    "openai_gpt-5.2",
    "openai_gpt-5.4",
    "openai_gpt-5.4-mini",
    "openai_gpt-5.4-nano",
]

REFERENCE_OPENAI_REASONING_FAMILY_LABELS = {
    "openai_gpt-5.2": "gpt-5.2",
    "openai_gpt-5.4": "gpt-5.4",
    "openai_gpt-5.4-mini": "gpt-5.4 mini",
    "openai_gpt-5.4-nano": "gpt-5.4 nano",
}

REFERENCE_ANTHROPIC_REASONING_LEVEL_ORDER = ["low", "medium", "high"]

REFERENCE_ANTHROPIC_REASONING_FAMILY_ORDER = [
    "anthropic_claude-opus-4-5",
    "anthropic_claude-opus-4-6",
    "anthropic_claude-sonnet-4-6",
]

REFERENCE_ANTHROPIC_REASONING_FAMILY_LABELS = {
    family_name: model_id_to_label(f"{family_name}_medium").replace(" (Medium)", "")
    for family_name in REFERENCE_ANTHROPIC_REASONING_FAMILY_ORDER
}

REFERENCE_REASONING_LEVEL_ORDER = ["none", "low", "medium", "high", "xhigh"]


def load_reference_openai_judge_severities(scores_path: Path) -> pd.DataFrame:
    """Load and order the selected OpenAI judge severities from a FACETS CSV."""

    score_frame = pd.read_csv(scores_path)
    return _load_selected_judge_severities(
        score_frame=score_frame,
        model_order=REFERENCE_OPENAI_MODEL_ORDER,
        model_labels=REFERENCE_OPENAI_MODEL_LABELS,
    )


def load_reference_anthropic_judge_severities(scores_path: Path) -> pd.DataFrame:
    """Load and order the selected Anthropic judge severities from a FACETS CSV."""

    score_frame = pd.read_csv(scores_path)
    return _load_selected_judge_severities(
        score_frame=score_frame,
        model_order=REFERENCE_ANTHROPIC_MODEL_ORDER,
        model_labels=REFERENCE_ANTHROPIC_MODEL_LABELS,
    )


def load_reference_anthropic_opus_judge_severities(scores_path: Path) -> pd.DataFrame:
    """Load and order the selected Anthropic Opus judge severities from a FACETS CSV."""

    score_frame = pd.read_csv(scores_path)
    return _load_selected_judge_severities(
        score_frame=score_frame,
        model_order=REFERENCE_ANTHROPIC_OPUS_MODEL_ORDER,
        model_labels=REFERENCE_ANTHROPIC_OPUS_MODEL_LABELS,
    )


def _load_selected_judge_severities(
    score_frame: pd.DataFrame,
    model_order: list[str],
    model_labels: dict[str, str],
) -> pd.DataFrame:
    """Load a selected set of judge severities and preserve the requested order."""

    required_columns = {"facet_label", "measure", "s_e"}
    missing_columns = required_columns.difference(score_frame.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Judge score file is missing required columns: {missing_list}")

    filtered = score_frame.loc[
        score_frame["facet_label"].isin(model_order),
        ["facet_label", "measure", "s_e"],
    ].copy()
    missing_models = [
        model_name
        for model_name in model_order
        if model_name not in filtered["facet_label"].tolist()
    ]
    if missing_models:
        missing_list = ", ".join(missing_models)
        raise ValueError(f"Judge score file is missing requested models: {missing_list}")

    # Preserve the exact user-requested plotting order rather than the CSV order.
    filtered["plot_order"] = filtered["facet_label"].map(
        {model_name: index for index, model_name in enumerate(model_order)}
    )
    filtered["display_label"] = filtered["facet_label"].map(model_labels)
    filtered = filtered.sort_values("plot_order", kind="stable").reset_index(drop=True)
    return filtered


def plot_reference_openai_judge_severities(
    severity_frame: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot the requested judge severities as a line chart and save the figure."""

    return _plot_selected_judge_severities(
        severity_frame=severity_frame,
        output_path=output_path,
        title="Reference Set OpenAI Judge Severities",
        line_color="#0F4C81",
    )


def plot_reference_anthropic_judge_severities(
    severity_frame: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot the requested Anthropic judge severities as a line chart and save it."""

    return _plot_selected_judge_severities(
        severity_frame=severity_frame,
        output_path=output_path,
        title="Reference Set Anthropic Judge Severities",
        line_color="#4C78A8",
    )


def plot_reference_anthropic_opus_judge_severities(
    severity_frame: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot the requested Anthropic Opus judge severities as a line chart and save it."""

    return _plot_selected_judge_severities(
        severity_frame=severity_frame,
        output_path=output_path,
        title="Reference Set Anthropic Opus Judge Severities",
        line_color="#3E6FB3",
    )


def _plot_selected_judge_severities(
    severity_frame: pd.DataFrame,
    output_path: Path,
    title: str,
    line_color: str,
) -> Path:
    """Plot one selected judge-severity comparison chart and save the figure."""

    apply_plot_style()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_positions = list(range(len(severity_frame)))
    figure, axis = plt.subplots(figsize=(7.5, 4.5))

    axis.plot(
        x_positions,
        severity_frame["measure"],
        color=line_color,
        linewidth=2.5,
        marker="o",
        markersize=8,
    )
    axis.errorbar(
        x_positions,
        severity_frame["measure"],
        yerr=severity_frame["s_e"],
        fmt="none",
        ecolor=line_color,
        elinewidth=1.4,
        capsize=4,
    )

    label_offset = max(float(severity_frame["s_e"].max()) * 1.4, 0.04)
    for x_position, row in zip(x_positions, severity_frame.itertuples(index=False)):
        axis.text(
            x_position,
            row.measure + label_offset,
            format_plot_text(f"{row.measure:.2f}"),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    axis.axhline(0.0, color="#7A7A7A", linestyle="--", linewidth=1.0, alpha=0.8)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(format_plot_text(severity_frame["display_label"].tolist()))
    axis.set_xlabel(format_plot_text("Model"))
    axis.set_ylabel(format_plot_text("FACETS judge measure"))
    axis.set_title(format_plot_text(title))
    axis.grid(axis="y", alpha=0.25)

    y_min = float((severity_frame["measure"] - severity_frame["s_e"]).min())
    y_max = float((severity_frame["measure"] + severity_frame["s_e"]).max())
    padding = max((y_max - y_min) * 0.18, 0.12)
    axis.set_ylim(y_min - padding, y_max + padding)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def load_reference_openai_reasoning_severities(scores_path: Path) -> pd.DataFrame:
    """Load the reasoning-level judge severities for the four OpenAI model families."""

    score_frame = pd.read_csv(scores_path)
    return _load_reasoning_severities_from_frame(
        score_frame=score_frame,
        family_order=REFERENCE_OPENAI_REASONING_FAMILY_ORDER,
        family_labels=REFERENCE_OPENAI_REASONING_FAMILY_LABELS,
        reasoning_level_order=REFERENCE_OPENAI_REASONING_LEVEL_ORDER,
        provider_label="OpenAI",
        family_sort_offset=0,
    )


def load_reference_reasoning_severities(
    openai_scores_path: Path,
    anthropic_scores_path: Path | None = None,
) -> pd.DataFrame:
    """Load the OpenAI reasoning families and optionally append Anthropic families."""

    severity_frames = [load_reference_openai_reasoning_severities(openai_scores_path)]
    if anthropic_scores_path is not None:
        anthropic_score_frame = pd.read_csv(anthropic_scores_path)
        severity_frames.append(
            _load_reasoning_severities_from_frame(
                score_frame=anthropic_score_frame,
                family_order=REFERENCE_ANTHROPIC_REASONING_FAMILY_ORDER,
                family_labels=REFERENCE_ANTHROPIC_REASONING_FAMILY_LABELS,
                reasoning_level_order=REFERENCE_ANTHROPIC_REASONING_LEVEL_ORDER,
                provider_label="Anthropic",
                family_sort_offset=len(REFERENCE_OPENAI_REASONING_FAMILY_ORDER),
            )
        )

    combined = pd.concat(severity_frames, ignore_index=True)
    combined = combined.sort_values(
        ["family_sort_order", "reasoning_order"],
        kind="stable",
    ).reset_index(drop=True)
    return combined


def _load_reasoning_severities_from_frame(
    score_frame: pd.DataFrame,
    family_order: list[str],
    family_labels: dict[str, str],
    reasoning_level_order: list[str],
    provider_label: str,
    family_sort_offset: int,
) -> pd.DataFrame:
    """Load a tidy reasoning-severity frame for one provider family set."""

    required_columns = {"facet_label", "measure", "s_e"}
    missing_columns = required_columns.difference(score_frame.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Judge score file is missing required columns: {missing_list}")

    reasoning_rows: list[dict[str, object]] = []
    for family_index, family_name in enumerate(family_order):
        for reasoning_level in reasoning_level_order:
            facet_label = f"{family_name}_{reasoning_level}"
            matching_rows = score_frame.loc[score_frame["facet_label"] == facet_label]
            if matching_rows.empty:
                raise ValueError(f"Judge score file is missing requested model: {facet_label}")

            row = matching_rows.iloc[0]
            # Store a tidy long-form table so the plotting code can draw one line per family.
            reasoning_rows.append(
                {
                    "facet_label": facet_label,
                    "family_name": family_name,
                    "family_label": family_labels[family_name],
                    "provider_label": provider_label,
                    "reasoning_level": reasoning_level,
                    "reasoning_order": REFERENCE_REASONING_LEVEL_ORDER.index(reasoning_level),
                    "family_sort_order": family_sort_offset + family_index,
                    "measure": row["measure"],
                    "s_e": row["s_e"],
                }
            )

    return pd.DataFrame(reasoning_rows)


def plot_reference_openai_reasoning_severities(
    severity_frame: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot one severity curve per model family across reasoning levels and save it."""

    apply_plot_style()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_positions = list(range(len(REFERENCE_REASONING_LEVEL_ORDER)))
    figure, axis = plt.subplots(figsize=(9.2, 5.6))
    colors = {
        "gpt-5.2": "#0F4C81",
        "gpt-5.4": "#C05A00",
        "gpt-5.4 mini": "#2A7F62",
        "gpt-5.4 nano": "#8B3E95",
        "Claude Opus 4.5": "#3E6FB3",
        "Claude Opus 4.6": "#5A8FD6",
        "Claude Sonnet 4.6": "#7CAFE8",
    }
    line_styles = {
        "OpenAI": "-",
        "Anthropic": "--",
    }
    marker_styles = {
        "OpenAI": "o",
        "Anthropic": "s",
    }

    family_order = (
        severity_frame.loc[:, ["family_name", "family_sort_order"]]
        .drop_duplicates()
        .sort_values("family_sort_order", kind="stable")["family_name"]
        .tolist()
    )
    for family_name in family_order:
        family_frame = severity_frame.loc[severity_frame["family_name"] == family_name].copy()
        family_frame = family_frame.sort_values("reasoning_order", kind="stable")
        family_label = str(family_frame["family_label"].iloc[0])
        provider_label = str(family_frame["provider_label"].iloc[0])
        color = colors.get(family_label)
        family_x_positions = family_frame["reasoning_order"].tolist()

        axis.plot(
            family_x_positions,
            family_frame["measure"],
            color=color,
            linewidth=2.1,
            linestyle=line_styles.get(provider_label, "-"),
            marker=marker_styles.get(provider_label, "o"),
            markersize=6,
            label=format_plot_text(family_label),
        )
        axis.errorbar(
            family_x_positions,
            family_frame["measure"],
            yerr=family_frame["s_e"],
            fmt="none",
            ecolor=color,
            elinewidth=1.1,
            capsize=3,
            alpha=0.9,
        )

    axis.axhline(0.0, color="#7A7A7A", linestyle="--", linewidth=1.0, alpha=0.8)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(format_plot_text(REFERENCE_REASONING_LEVEL_ORDER))
    axis.set_xlabel(format_plot_text("Reasoning effort"))
    axis.set_ylabel(format_plot_text("FACETS judge measure"))
    axis.set_title(format_plot_text("Reasoning Effects on Reference Set Judge Severities"))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=2)

    y_min = float((severity_frame["measure"] - severity_frame["s_e"]).min())
    y_max = float((severity_frame["measure"] + severity_frame["s_e"]).max())
    padding = max((y_max - y_min) * 0.12, 0.10)
    axis.set_ylim(y_min - padding, y_max + padding)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path
