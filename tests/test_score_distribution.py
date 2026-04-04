from pathlib import Path

import pandas as pd

from mhs_llms.score_distribution import (
    align_item_responses,
    build_comment_score_frame,
    plot_score_distributions,
)


def test_align_item_responses_maps_letters_onto_hate_aligned_scores() -> None:
    aligned = align_item_responses(
        pd.DataFrame(
            [
                {
                    "comment_id": 1,
                    "sentiment": "A",
                    "respect": "B",
                    "insult": "E",
                    "humiliate": "D",
                    "status": "C",
                    "dehumanize": "B",
                    "violence": "A",
                    "genocide": "C",
                    "attack_defend": "D",
                    "hate_speech": "C",
                }
            ]
        )
    )

    assert aligned.loc[0, "sentiment"] == 4
    assert aligned.loc[0, "respect"] == 3
    assert aligned.loc[0, "insult"] == 4
    assert aligned.loc[0, "humiliate"] == 3
    assert aligned.loc[0, "status"] == 2
    assert aligned.loc[0, "dehumanize"] == 1
    assert aligned.loc[0, "violence"] == 0
    assert aligned.loc[0, "genocide"] == 2
    assert aligned.loc[0, "attack_defend"] == 3
    assert aligned.loc[0, "hate_speech"] == 1


def test_build_comment_score_frame_averages_multiple_human_annotations() -> None:
    human_annotations = pd.DataFrame(
        [
            {
                "comment_id": 101,
                "sentiment": 0,
                "respect": 0,
                "insult": 0,
                "humiliate": 0,
                "status": 0,
                "dehumanize": 0,
                "violence": 0,
                "genocide": 0,
                "attack_defend": 0,
                "hate_speech": 0,
            },
            {
                "comment_id": 101,
                "sentiment": 4,
                "respect": 4,
                "insult": 4,
                "humiliate": 4,
                "status": 4,
                "dehumanize": 4,
                "violence": 4,
                "genocide": 4,
                "attack_defend": 4,
                "hate_speech": 2,
            },
            {
                "comment_id": 202,
                "sentiment": 1,
                "respect": 1,
                "insult": 1,
                "humiliate": 1,
                "status": 1,
                "dehumanize": 1,
                "violence": 1,
                "genocide": 1,
                "attack_defend": 1,
                "hate_speech": 1,
            },
        ]
    )

    scores = build_comment_score_frame(
        annotation_frame=human_annotations,
        reference_comment_ids=[101, 202],
        source_label="Humans",
    )

    assert scores["comment_id"].tolist() == [101, 202]
    assert scores["hate_speech_score"].tolist() == [19.0, 10.0]
    assert scores["num_annotations"].tolist() == [2, 1]
    assert scores["source"].tolist() == ["Humans", "Humans"]


def test_build_comment_score_frame_accepts_raw_mhs_hatespeech_column() -> None:
    human_annotations = pd.DataFrame(
        [
            {
                "comment_id": 101,
                "sentiment": 4,
                "respect": 4,
                "insult": 4,
                "humiliate": 4,
                "status": 4,
                "dehumanize": 4,
                "violence": 4,
                "genocide": 4,
                "attack_defend": 4,
                "hatespeech": 2,
            }
        ]
    )

    scores = build_comment_score_frame(
        annotation_frame=human_annotations,
        reference_comment_ids=[101],
        source_label="Humans",
    )

    assert scores.loc[0, "hate_speech_score"] == 38.0


def test_plot_score_distributions_writes_png(tmp_path: Path) -> None:
    output_path = tmp_path / "distribution.png"
    score_frame = pd.DataFrame(
        [
            {"comment_id": 1, "hate_speech_score": 8.0, "source": "Humans"},
            {"comment_id": 2, "hate_speech_score": 10.0, "source": "Humans"},
            {"comment_id": 1, "hate_speech_score": 12.0, "source": "LLM (gpt-5.4)"},
            {"comment_id": 2, "hate_speech_score": 15.0, "source": "LLM (gpt-5.4)"},
        ]
    )

    plotted_path = plot_score_distributions(
        score_frame=score_frame,
        output_path=output_path,
        title="Test Plot",
    )

    assert plotted_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
