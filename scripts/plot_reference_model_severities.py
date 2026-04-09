"""Plot model severities against the human comment severity distribution."""

import argparse
from pathlib import Path

from mhs_llms.facets.model_severity_figure import (
    load_human_judge_severities,
    load_model_judge_severities,
    plot_model_severity_figure,
)
from mhs_llms.paths import ARTIFACTS_DIR, FACETS_DIR


DEFAULT_HUMAN_PATH = FACETS_DIR / "human_baseline" / "human_facets_scores.2.txt"
DEFAULT_JUDGES_PATHS = [
    FACETS_DIR / "reference_set_openai" / "judges_scores.csv",
    FACETS_DIR / "reference_set_anthropic" / "judges_scores.csv",
]
DEFAULT_OUTPUT_PATH = ARTIFACTS_DIR / "reference_set_model_severities.png"


def main() -> None:
    """Parse CLI args, generate the figure, and print resolved paths."""

    parser = argparse.ArgumentParser(
        description=(
            "Plot a KDE of human FACETS judge severities with vertical lines for "
            "model judge severities, plus a centered horizontal bar chart of model severities."
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
        help="PNG output path for the combined figure.",
    )
    parser.add_argument(
        "--title",
        dest="title",
        default="Human Severity Distribution and Model Judge Severities",
        help="Figure title.",
    )
    args = parser.parse_args()

    human_path = Path(args.human_path).resolve()
    judges_paths = [Path(path).resolve() for path in (args.judges_paths or [str(path) for path in DEFAULT_JUDGES_PATHS])]
    output_path = Path(args.output_path).resolve()

    human_severity_frame = load_human_judge_severities(human_path)
    model_severity_frame = load_model_judge_severities(judges_paths)
    plotted_path = plot_model_severity_figure(
        human_severity_frame=human_severity_frame,
        model_severity_frame=model_severity_frame,
        output_path=output_path,
        title=args.title,
    )

    print(f"human_file={human_path}")
    print(f"judges_files={','.join(str(path) for path in judges_paths)}")
    print(f"output={plotted_path}")


if __name__ == "__main__":
    main()
