"""Plot FACETS judge severities for a selected set of models."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from mpl_lego.labels import bold_text
from mpl_lego.style import check_latex_style_on, use_latex_style


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


def load_reference_openai_judge_severities(scores_path: Path) -> pd.DataFrame:
    """Load and order the selected OpenAI judge severities from a FACETS CSV."""

    score_frame = pd.read_csv(scores_path)
    required_columns = {"facet_label", "measure", "s_e"}
    missing_columns = required_columns.difference(score_frame.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Judge score file is missing required columns: {missing_list}")

    filtered = score_frame.loc[
        score_frame["facet_label"].isin(REFERENCE_OPENAI_MODEL_ORDER),
        ["facet_label", "measure", "s_e"],
    ].copy()
    missing_models = [
        model_name
        for model_name in REFERENCE_OPENAI_MODEL_ORDER
        if model_name not in filtered["facet_label"].tolist()
    ]
    if missing_models:
        missing_list = ", ".join(missing_models)
        raise ValueError(f"Judge score file is missing requested models: {missing_list}")

    # Preserve the exact user-requested plotting order rather than the CSV order.
    filtered["plot_order"] = filtered["facet_label"].map(
        {model_name: index for index, model_name in enumerate(REFERENCE_OPENAI_MODEL_ORDER)}
    )
    filtered["display_label"] = filtered["facet_label"].map(REFERENCE_OPENAI_MODEL_LABELS)
    filtered = filtered.sort_values("plot_order", kind="stable").reset_index(drop=True)
    return filtered


def plot_reference_openai_judge_severities(
    severity_frame: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Plot the requested judge severities as a line chart and save the figure."""

    use_latex_style()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_positions = list(range(len(severity_frame)))
    figure, axis = plt.subplots(figsize=(7.5, 4.5))

    axis.plot(
        x_positions,
        severity_frame["measure"],
        color="#0F4C81",
        linewidth=2.5,
        marker="o",
        markersize=8,
    )
    axis.errorbar(
        x_positions,
        severity_frame["measure"],
        yerr=severity_frame["s_e"],
        fmt="none",
        ecolor="#0F4C81",
        elinewidth=1.4,
        capsize=4,
    )

    label_offset = max(float(severity_frame["s_e"].max()) * 1.4, 0.04)
    for x_position, row in zip(x_positions, severity_frame.itertuples(index=False)):
        axis.text(
            x_position,
            row.measure + label_offset,
            _format_text(f"{row.measure:.2f}"),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    axis.axhline(0.0, color="#7A7A7A", linestyle="--", linewidth=1.0, alpha=0.8)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(_format_text(severity_frame["display_label"].tolist()))
    axis.set_xlabel(_format_text("Model"))
    axis.set_ylabel(_format_text("FACETS judge measure"))
    axis.set_title(_format_text("Reference Set OpenAI Judge Severities"))
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
    required_columns = {"facet_label", "measure", "s_e"}
    missing_columns = required_columns.difference(score_frame.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Judge score file is missing required columns: {missing_list}")

    reasoning_rows: list[dict[str, object]] = []
    for family_name in REFERENCE_OPENAI_REASONING_FAMILY_ORDER:
        for reasoning_level in REFERENCE_OPENAI_REASONING_LEVEL_ORDER:
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
                    "family_label": REFERENCE_OPENAI_REASONING_FAMILY_LABELS[family_name],
                    "reasoning_level": reasoning_level,
                    "reasoning_order": REFERENCE_OPENAI_REASONING_LEVEL_ORDER.index(reasoning_level),
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

    use_latex_style()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_positions = list(range(len(REFERENCE_OPENAI_REASONING_LEVEL_ORDER)))
    figure, axis = plt.subplots(figsize=(8.2, 5.1))
    colors = {
        "gpt-5.2": "#0F4C81",
        "gpt-5.4": "#C05A00",
        "gpt-5.4 mini": "#2A7F62",
        "gpt-5.4 nano": "#8B3E95",
    }

    for family_name in REFERENCE_OPENAI_REASONING_FAMILY_ORDER:
        family_frame = severity_frame.loc[severity_frame["family_name"] == family_name].copy()
        family_frame = family_frame.sort_values("reasoning_order", kind="stable")
        family_label = str(family_frame["family_label"].iloc[0])
        color = colors.get(family_label)

        axis.plot(
            x_positions,
            family_frame["measure"],
            color=color,
            linewidth=2.1,
            marker="o",
            markersize=6,
            label=_format_text(family_label),
        )
        axis.errorbar(
            x_positions,
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
    axis.set_xticklabels(_format_text(REFERENCE_OPENAI_REASONING_LEVEL_ORDER))
    axis.set_xlabel(_format_text("Reasoning effort"))
    axis.set_ylabel(_format_text("FACETS judge measure"))
    axis.set_title(_format_text("Reasoning Effects on Reference Set Judge Severities"))
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)

    y_min = float((severity_frame["measure"] - severity_frame["s_e"]).min())
    y_max = float((severity_frame["measure"] + severity_frame["s_e"]).max())
    padding = max((y_max - y_min) * 0.12, 0.10)
    axis.set_ylim(y_min - padding, y_max + padding)

    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


def _format_text(text: str | list[str]) -> str | list[str]:
    """Use mpl_lego bold text only when its LaTeX styling is active."""

    if check_latex_style_on():
        return bold_text(text)
    return text
