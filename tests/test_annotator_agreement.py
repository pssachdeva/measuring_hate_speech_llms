from pathlib import Path

import pandas as pd

from mhs_llms.annotator_agreement import (
    build_item_agreement_summary,
    krippendorff_alpha,
    plot_item_agreement_summary,
    prepare_agreement_annotations,
)
from mhs_llms.schema import ITEM_NAMES


def test_krippendorff_alpha_returns_one_for_perfect_agreement_with_missing_cells() -> None:
    annotations = pd.DataFrame(
        [
            {"comment_id": 1, "judge_id": "a", "sentiment": 0},
            {"comment_id": 1, "judge_id": "b", "sentiment": 0},
            {"comment_id": 2, "judge_id": "a", "sentiment": 1},
            {"comment_id": 3, "judge_id": "a", "sentiment": 2},
            {"comment_id": 3, "judge_id": "b", "sentiment": 2},
        ]
    )

    alpha = krippendorff_alpha(
        annotations=annotations,
        value_column="sentiment",
        distance_metric="interval",
    )

    assert alpha == 1.0


def test_prepare_agreement_annotations_accepts_human_alias_columns() -> None:
    annotations = pd.DataFrame(
        [
            {
                "comment_id": 101,
                "annotator_id": 22,
                "sentiment": 4,
                "respect": 4,
                "insult": 4,
                "humiliate": 4,
                "status": 4,
                "dehumanize": 4,
                "violence_phys": 4,
                "genocide": 4,
                "attack_defend": 4,
                "hatespeech": 2,
            }
        ]
    )

    prepared = prepare_agreement_annotations(
        annotation_frame=annotations,
        reference_comment_ids=[101],
        annotator_prefix="human",
    )

    assert prepared.loc[0, "judge_id"] == "human:22"
    assert prepared.loc[0, "violence"] == 4.0
    assert prepared.loc[0, "hate_speech"] == 2.0


def test_build_item_agreement_summary_handles_different_humans_per_comment() -> None:
    llm_annotations = pd.DataFrame(
        [
            _annotation_row(comment_id=1, judge_id="model_a", sentiment="A"),
            _annotation_row(comment_id=1, judge_id="model_b", sentiment="A"),
            _annotation_row(comment_id=2, judge_id="model_a", sentiment="E"),
            _annotation_row(comment_id=2, judge_id="model_b", sentiment="D"),
        ]
    )
    human_annotations = pd.DataFrame(
        [
            _annotation_row(comment_id=1, annotator_id=10, sentiment=4),
            _annotation_row(comment_id=1, annotator_id=11, sentiment=4),
            _annotation_row(comment_id=2, annotator_id=12, sentiment=0),
            _annotation_row(comment_id=2, annotator_id=13, sentiment=1),
        ]
    )

    summary = build_item_agreement_summary(
        llm_annotations=llm_annotations,
        human_annotations=human_annotations,
        reference_comment_ids=[1, 2],
        item_names=("sentiment",),
        distance_metric="interval",
    )

    assert summary["agreement_group"].astype(str).tolist() == ["Humans", "LLMs", "Humans + LLMs"]
    assert summary["num_units"].tolist() == [2, 2, 2]
    assert summary["num_annotators"].tolist() == [4, 2, 6]
    assert summary["num_ratings"].tolist() == [4, 4, 8]
    assert summary["alpha"].notna().all()


def test_plot_item_agreement_summary_writes_png(tmp_path: Path) -> None:
    output_path = tmp_path / "agreement.png"
    summary = pd.DataFrame(
        [
            {
                "item_name": "sentiment",
                "item_label": "Sentiment",
                "item_order": 0,
                "agreement_group": "Humans",
                "alpha": 0.75,
            },
            {
                "item_name": "sentiment",
                "item_label": "Sentiment",
                "item_order": 0,
                "agreement_group": "LLMs",
                "alpha": 0.8,
            },
            {
                "item_name": "sentiment",
                "item_label": "Sentiment",
                "item_order": 0,
                "agreement_group": "Humans + LLMs",
                "alpha": 0.7,
            },
        ]
    )

    plotted_path = plot_item_agreement_summary(
        summary_frame=summary,
        output_path=output_path,
        figsize=(3.5, 2.5),
        dpi=100,
        marker_size=30.0,
        x_offset=0.1,
        y_limits=(0.0, 1.0),
        tick_label_size=8.0,
        axis_label_size=9.0,
        legend_font_size=8.0,
    )

    assert plotted_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def _annotation_row(
    comment_id: int,
    sentiment: object,
    judge_id: str | None = None,
    annotator_id: int | None = None,
) -> dict[str, object]:
    """Build a complete annotation row with one varied item."""

    row: dict[str, object] = {
        "comment_id": comment_id,
        "sentiment": sentiment,
        "respect": "A",
        "insult": "A",
        "humiliate": "A",
        "status": "A",
        "dehumanize": "A",
        "violence": "A",
        "genocide": "A",
        "attack_defend": "A",
        "hate_speech": "A",
    }
    if judge_id is not None:
        row["judge_id"] = judge_id
    if annotator_id is not None:
        row["annotator_id"] = annotator_id
    for item_name in ITEM_NAMES:
        row.setdefault(item_name, "A")
    return row
