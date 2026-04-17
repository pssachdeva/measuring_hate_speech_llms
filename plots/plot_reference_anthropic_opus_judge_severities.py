"""Plot selected FACETS judge severities for Anthropic Opus models."""

import argparse
from pathlib import Path

from mhs_llms.facets.judge_severity_plot import (
    load_reference_anthropic_opus_judge_severities,
    plot_reference_anthropic_opus_judge_severities,
)
from mhs_llms.paths import ARTIFACTS_DIR, FACETS_DIR


DEFAULT_SCORES_PATH = FACETS_DIR / "reference_set_anthropic" / "judges_scores.csv"
DEFAULT_OUTPUT_PATH = ARTIFACTS_DIR / "reference_set_anthropic_opus_judge_severities.png"


def main() -> None:
    """Parse CLI args, plot the selected Anthropic Opus severities, and save the figure."""

    parser = argparse.ArgumentParser(
        description=(
            "Plot FACETS judge severities for opus 4, opus 4.1, "
            "opus 4.5 (medium), and opus 4.6 (medium)."
        )
    )
    parser.add_argument(
        "--scores-file",
        dest="scores_path",
        default=str(DEFAULT_SCORES_PATH),
        help="Path to the FACETS judges_scores.csv file.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Output path for the Anthropic Opus comparison PNG.",
    )
    args = parser.parse_args()

    scores_path = Path(args.scores_path).resolve()
    output_path = Path(args.output_path).resolve()

    severity_frame = load_reference_anthropic_opus_judge_severities(scores_path)
    plotted_path = plot_reference_anthropic_opus_judge_severities(severity_frame, output_path)

    print(f"scores_file={scores_path}")
    print(f"output={plotted_path}")


if __name__ == "__main__":
    main()
