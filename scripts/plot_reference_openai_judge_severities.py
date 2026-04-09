"""Plot selected FACETS judge severities for the OpenAI reference set."""

import argparse
from pathlib import Path

from mhs_llms.facets.judge_severity_plot import (
    load_reference_reasoning_severities,
    load_reference_openai_judge_severities,
    plot_reference_openai_judge_severities,
    plot_reference_openai_reasoning_severities,
)
from mhs_llms.paths import ARTIFACTS_DIR, FACETS_DIR


DEFAULT_SCORES_PATH = FACETS_DIR / "reference_set_openai" / "judges_scores.csv"
DEFAULT_ANTHROPIC_SCORES_PATH = FACETS_DIR / "reference_set_anthropic" / "judges_scores.csv"
DEFAULT_OUTPUT_PATH = ARTIFACTS_DIR / "reference_set_openai_judge_severities.png"
DEFAULT_REASONING_OUTPUT_PATH = ARTIFACTS_DIR / "reference_set_openai_reasoning_judge_severities.png"


def main() -> None:
    """Parse CLI args, plot both judge-severity figures, and save them."""

    parser = argparse.ArgumentParser(
        description=(
            "Plot FACETS judge severities for gpt-4o, gpt-4.1, 5.2 medium, "
            "and 5.4 medium from the reference_set_openai run."
        )
    )
    parser.add_argument(
        "--scores-file",
        dest="scores_path",
        default=str(DEFAULT_SCORES_PATH),
        help="Path to the FACETS judges_scores.csv file.",
    )
    parser.add_argument(
        "--anthropic-scores-file",
        dest="anthropic_scores_path",
        default=str(DEFAULT_ANTHROPIC_SCORES_PATH),
        help="Path to the Anthropic FACETS judges_scores.csv file used for the reasoning plot.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output path for the four-model comparison PNG.",
    )
    parser.add_argument(
        "--reasoning-output",
        dest="reasoning_output_path",
        default=str(DEFAULT_REASONING_OUTPUT_PATH),
        help="Output path for the reasoning-effects PNG.",
    )
    args = parser.parse_args()

    scores_path = Path(args.scores_path).resolve()
    anthropic_scores_path = Path(args.anthropic_scores_path).resolve()
    output_path = Path(args.output_path).resolve()
    reasoning_output_path = Path(args.reasoning_output_path).resolve()

    severity_frame = load_reference_openai_judge_severities(scores_path)
    plotted_path = plot_reference_openai_judge_severities(severity_frame, output_path)
    reasoning_frame = load_reference_reasoning_severities(scores_path, anthropic_scores_path)
    reasoning_plotted_path = plot_reference_openai_reasoning_severities(
        reasoning_frame,
        reasoning_output_path,
    )

    print(f"scores_file={scores_path}")
    print(f"anthropic_scores_file={anthropic_scores_path}")
    print(f"model_comparison_output={plotted_path}")
    print(f"reasoning_output={reasoning_plotted_path}")


if __name__ == "__main__":
    main()
