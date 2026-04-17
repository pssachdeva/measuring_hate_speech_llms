"""Plot a selected subset of model severities against the human distribution."""

import argparse
from pathlib import Path

import pandas as pd

from mhs_llms.facets.model_severity_figure import (
    load_human_judge_severities,
    load_model_judge_severities,
    plot_model_severity_figure,
)
from mhs_llms.paths import ARTIFACTS_DIR, FACETS_DIR


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

DEFAULT_HUMAN_PATH = FACETS_DIR / "human_baseline" / "human_facets_scores.2.txt"
DEFAULT_JUDGES_PATHS = [
    FACETS_DIR / "reference_set_openai" / "judges_scores.csv",
    FACETS_DIR / "reference_set_anthropic" / "judges_scores.csv",
    FACETS_DIR / "reference_set_google" / "judges_scores.csv",
    FACETS_DIR / "reference_set_open_large" / "judges_scores.csv",
    FACETS_DIR / "reference_set_xai" / "judges_scores.csv",
]
DEFAULT_OUTPUT_PATH = ARTIFACTS_DIR / "reference_set_model_severities_subset.png"


def main() -> None:
    """Parse CLI args, filter to the selected model ids, and save the figure."""

    parser = argparse.ArgumentParser(
        description=(
            "Plot a selected subset of FACETS model judge severities against the "
            "human severity distribution."
        )
    )
    parser.add_argument(
        "--human-file",
        dest="human_path",
        default=str(DEFAULT_HUMAN_PATH),
        help="Path to the FACETS human judge score file (.csv or raw FACETS .txt).",
    )
    parser.add_argument(
        "--judges-file",
        dest="judges_paths",
        action="append",
        default=None,
        help=(
            "Path to a FACETS judge score file (.csv or raw FACETS .txt). "
            "Pass multiple times to combine several runs."
        ),
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="PNG output path for the subset figure.",
    )
    parser.add_argument(
        "--title",
        dest="title",
        default="",
        help="Optional figure title.",
    )
    parser.add_argument(
        "--figure-width",
        dest="figure_width",
        type=float,
        default=8,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--figure-height",
        dest="figure_height",
        type=float,
        default=6,
        help="Optional figure height in inches. Defaults to a model-count-based height.",
    )
    parser.add_argument(
        "--legend-font-size",
        dest="legend_font_size",
        type=float,
        default=7,
        help="Optional legend font size.",
    )
    parser.add_argument(
        "--bottom-label-font-size",
        dest="bottom_label_font_size",
        type=float,
        default=6.5,
        help="Font size for model labels in the bottom panel.",
    )
    parser.add_argument(
        "--value-label-font-size",
        dest="value_label_font_size",
        type=float,
        default=8,
        help="Font size for numeric severity labels in the bottom panel.",
    )
    parser.add_argument(
        "--x-min",
        dest="x_min",
        type=float,
        default=-1.5,
        help="Minimum x-axis value.",
    )
    parser.add_argument(
        "--x-max",
        dest="x_max",
        type=float,
        default=1.5,
        help="Maximum x-axis value.",
    )
    args = parser.parse_args()

    human_path = Path(args.human_path).resolve()
    default_judges_paths = [str(path) for path in DEFAULT_JUDGES_PATHS]
    judges_paths = [Path(path).resolve() for path in (args.judges_paths or default_judges_paths)]
    output_path = Path(args.output_path).resolve()

    human_severity_frame = load_human_judge_severities(human_path)
    model_severity_frame = load_model_judge_severities(judges_paths)
    subset_frame = select_model_subset(model_severity_frame, SUBSET_MODEL_IDS)
    plotted_path = plot_model_severity_figure(
        human_severity_frame=human_severity_frame,
        model_severity_frame=subset_frame,
        output_path=output_path,
        title=args.title,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        legend_font_size=args.legend_font_size,
        bottom_label_font_size=args.bottom_label_font_size,
        value_label_font_size=args.value_label_font_size,
        x_min=args.x_min,
        x_max=args.x_max,
    )

    print(f"human_file={human_path}")
    print(f"judges_files={','.join(str(path) for path in judges_paths)}")
    print(f"models={','.join(SUBSET_MODEL_IDS)}")
    print(f"output={plotted_path}")


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
    subset_frame = subset_frame.sort_values(
        ["measure", "display_label"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)
    return subset_frame


if __name__ == "__main__":
    main()
