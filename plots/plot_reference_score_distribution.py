import argparse
from pathlib import Path

import pandas as pd

from mhs_llms.paths import REPO_ROOT
from mhs_llms.score_distribution import (
    build_comment_score_frame,
    build_default_output_path,
    infer_llm_label,
    load_reference_comment_ids,
    plot_score_distributions,
    read_annotation_table,
)


DEFAULT_REFERENCE_SET_PATH = REPO_ROOT / "data" / "reference_set.csv"


def main() -> None:
    """Parse CLI args, compute comment scores, and write the KDE plot."""

    parser = argparse.ArgumentParser(
        description=(
            "Plot summed hate-score KDE distributions across reference-set comments "
            "for one LLM annotation file versus one human annotation file."
        )
    )
    parser.add_argument(
        "--reference-set",
        dest="reference_set_path",
        default=str(DEFAULT_REFERENCE_SET_PATH),
        help="CSV, TSV, or JSONL file containing the reference-set comment_id values.",
    )
    parser.add_argument(
        "--llm-file",
        dest="llm_path",
        required=True,
        help="CSV, TSV, or JSONL file with LLM annotations.",
    )
    parser.add_argument(
        "--human-file",
        dest="human_path",
        required=True,
        help="CSV, TSV, or JSONL file with human annotations.",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        help="Optional PNG path. Defaults to artifacts/<llm-file-stem>_vs_humans_hate_score_kde.png",
    )
    args = parser.parse_args()

    reference_set_path = Path(args.reference_set_path).resolve()
    llm_path = Path(args.llm_path).resolve()
    human_path = Path(args.human_path).resolve()

    reference_comment_ids = load_reference_comment_ids(reference_set_path)
    llm_frame = read_annotation_table(llm_path)
    human_frame = read_annotation_table(human_path)

    llm_label = infer_llm_label(llm_frame, fallback=f"LLM ({llm_path.stem})")
    human_scores = build_comment_score_frame(
        annotation_frame=human_frame,
        reference_comment_ids=reference_comment_ids,
        source_label="Humans",
    )
    llm_scores = build_comment_score_frame(
        annotation_frame=llm_frame,
        reference_comment_ids=reference_comment_ids,
        source_label=llm_label,
    )
    combined_scores = pd.concat([human_scores, llm_scores], ignore_index=True)

    output_path = (
        Path(args.output_path).resolve() if args.output_path else build_default_output_path(llm_path)
    )
    plotted_path = plot_score_distributions(
        score_frame=combined_scores,
        output_path=output_path,
        title="Reference Set Hate Score Distribution: Humans vs LLM",
    )

    print(f"reference_set={reference_set_path}")
    print(f"human_file={human_path}")
    print(f"llm_file={llm_path}")
    print(f"output={plotted_path}")


if __name__ == "__main__":
    main()
