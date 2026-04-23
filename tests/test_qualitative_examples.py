import pandas as pd

from mhs_llms.qualitative_examples import (
    build_comment_response_profile,
    build_comment_summary,
    select_black_woman_reference_comment,
)
from mhs_llms.schema import ITEM_NAMES


def test_select_black_woman_reference_comment_prefers_highest_aligned_score() -> None:
    human_annotations = pd.DataFrame(
        [
            _human_row(comment_id=1, annotator_id=10, value=1, hate_speech=1, black=True),
            _human_row(comment_id=1, annotator_id=11, value=1, hate_speech=1, white=True),
            _human_row(comment_id=2, annotator_id=12, value=4, hate_speech=2, black=True),
            _human_row(comment_id=2, annotator_id=13, value=4, hate_speech=2, white=True),
        ]
    )

    selected_comment = select_black_woman_reference_comment(
        human_annotations=human_annotations,
        reference_comment_ids=[1, 2],
        min_black_annotators=1,
        min_white_annotators=1,
        min_target_share=0.5,
    )

    assert selected_comment == 2


def test_build_comment_response_profile_averages_human_groups_and_llm_providers() -> None:
    human_annotations = pd.DataFrame(
        [
            _human_row(comment_id=7, annotator_id=10, value=4, hate_speech=2, black=True),
            _human_row(comment_id=7, annotator_id=11, value=2, hate_speech=1, white=True),
        ]
    )
    llm_annotations = pd.DataFrame(
        [
            _llm_row(comment_id=7, provider="openai", judge_id="openai_a", sentiment="A"),
            _llm_row(comment_id=7, provider="openai", judge_id="openai_b", sentiment="C"),
            _llm_row(comment_id=7, provider="anthropic", judge_id="anthropic_a", sentiment="E"),
        ]
    )

    profile = build_comment_response_profile(
        human_annotations=human_annotations,
        llm_annotations=llm_annotations,
        comment_id=7,
        provider_order=("openai", "anthropic"),
    )

    sentiment_rows = profile.loc[profile["item_name"] == "sentiment"]
    assert sentiment_rows["group_label"].tolist() == [
        "Black annotators",
        "White annotators",
        "OpenAI",
        "Anthropic",
    ]
    assert sentiment_rows["mean_response"].tolist() == [4.0, 2.0, 3.0, 0.0]
    assert sentiment_rows["num_annotations"].tolist() == [1, 1, 2, 1]


def test_build_comment_summary_reports_target_and_group_counts() -> None:
    human_annotations = pd.DataFrame(
        [
            _human_row(comment_id=9, annotator_id=10, value=4, hate_speech=2, black=True),
            _human_row(comment_id=9, annotator_id=11, value=4, hate_speech=2, white=True),
        ]
    )

    summary = build_comment_summary(human_annotations=human_annotations, comment_id=9)

    assert summary["comment_id"] == 9
    assert summary["num_human_annotations"] == 2
    assert summary["num_black_annotators"] == 1
    assert summary["num_white_annotators"] == 1
    assert summary["target_black_woman_share"] == 1.0


def _human_row(
    comment_id: int,
    annotator_id: int,
    value: int,
    hate_speech: int,
    black: bool = False,
    white: bool = False,
) -> dict[str, object]:
    """Build one synthetic human annotation row."""

    row = {
        "comment_id": comment_id,
        "annotator_id": annotator_id,
        "annotator_race_black": black,
        "annotator_race_white": white,
        "target_race_black": True,
        "target_gender_women": True,
    }
    for item_name in ITEM_NAMES:
        row[item_name] = hate_speech if item_name == "hate_speech" else value
    return row


def _llm_row(
    comment_id: int,
    provider: str,
    judge_id: str,
    sentiment: str,
) -> dict[str, object]:
    """Build one synthetic LLM annotation row."""

    row = {
        "comment_id": comment_id,
        "provider": provider,
        "judge_id": judge_id,
        "sentiment": sentiment,
    }
    for item_name in ITEM_NAMES:
        row.setdefault(item_name, "A")
    return row
