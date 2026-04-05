import json

import pandas as pd
import pytest

from mhs_llms.schema import (
    normalize_human_annotation,
    normalize_human_annotations,
    normalize_model_annotation,
    prompt_letter_to_hf_value,
)


def test_normalize_human_annotation_validates_and_derives_targets() -> None:
    row = pd.Series(
        {
            "comment_id": 10,
            "annotator_id": 77,
            "platform": 1,
            "text": "test comment",
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
            "target_race_black": True,
            "target_gender_women": False,
        }
    )

    record = normalize_human_annotation(row)

    assert record.comment_id == 10
    assert record.judge_id == "77"
    assert record.sentiment == "A"
    assert record.hate_speech == "A"
    assert record.target_groups == ["A"]


def test_normalize_human_annotations_returns_flattened_dataframe() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "comment_id": 10,
                "annotator_id": 77,
                "platform": 1,
                "text": "test comment",
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
                "target_race_black": True,
            }
        ]
    )

    normalized = normalize_human_annotations(dataframe)

    assert normalized.loc[0, "judge_id"] == "77"
    assert normalized.loc[0, "sentiment"] == 4
    assert normalized.loc[0, "sentiment_letter"] == "A"
    targets = json.loads(normalized.loc[0, "target_groups"])
    assert targets == ["A"]


def test_normalize_human_annotation_uses_none_of_the_above_when_no_targets_are_set() -> None:
    row = pd.Series(
        {
            "comment_id": 11,
            "annotator_id": 88,
            "platform": 1,
            "text": "test comment",
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
        }
    )

    record = normalize_human_annotation(row)

    assert record.target_groups == ["I"]


def test_normalize_model_annotation_validates_none_of_the_above_exclusivity() -> None:
    with pytest.raises(ValueError, match="cannot combine 'I' with other codes"):
        normalize_model_annotation(
            comment_id=12,
            judge_id="openai:test",
            text="test comment",
            payload={
                "target_groups": ["I", "A"],
                "sentiment": "C",
                "respect": "C",
                "insult": "A",
                "humiliate": "A",
                "status": "C",
                "dehumanize": "A",
                "violence": "A",
                "genocide": "A",
                "attack_defend": "C",
                "hate_speech": "B",
            },
            metadata={},
        )


def test_prompt_letter_to_hf_value_matches_human_coding() -> None:
    assert prompt_letter_to_hf_value("sentiment", "A") == 4
    assert prompt_letter_to_hf_value("insult", "E") == 4
    assert prompt_letter_to_hf_value("hate_speech", "A") == 2
