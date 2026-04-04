import pandas as pd

from mhs_llms.config import FacetsConfig
from mhs_llms.schema import ITEM_NAMES
from mhs_llms.facets import build_facets_frame, build_facets_spec, load_measure_anchors


def test_build_facets_frame_uses_expected_columns_and_item_order() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "comment_id": 10,
                "judge_id": "101",
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
            }
        ]
    )

    facets = build_facets_frame(dataframe)

    assert facets.columns.tolist() == [
        "comment_id",
        "judge_id",
        "item_id",
        "sentiment",
        "respect",
        "insult",
        "humiliate",
        "status",
        "dehumanize",
        "violence",
        "genocide",
        "attack_defend",
        "hate_speech",
    ]
    assert facets.iloc[0]["item_id"] == "1-10"


def test_build_facets_spec_matches_three_facet_layout() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "comment_id": 10,
                "judge_id": "101",
                "item_id": "1-10",
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
            }
        ]
    )

    spec = build_facets_spec(
        facets_frame=dataframe,
        facets_config=FacetsConfig(title="Test"),
        data_filename="facets_data.tsv",
        score_filename="scores.txt",
        output_filename="output.txt",
    )

    assert "Facets = 3" in spec
    assert "1,Comments" in spec
    assert "2,Judges" in spec
    assert "3,Items,A" in spec
    assert "Data = facets_data.tsv" in spec


def test_build_facets_spec_can_anchor_comments_and_items() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "comment_id": 47777,
                "judge_id": "20001",
                "item_id": "1-10",
                "sentiment": 4,
                "respect": 4,
                "insult": 3,
                "humiliate": 2,
                "status": 1,
                "dehumanize": 1,
                "violence": 1,
                "genocide": 1,
                "attack_defend": 3,
                "hate_speech": 1,
            }
        ]
    )

    spec = build_facets_spec(
        facets_frame=dataframe,
        facets_config=FacetsConfig(title="Anchored"),
        data_filename="facets_data.tsv",
        score_filename="scores.txt",
        output_filename="output.txt",
        comment_labels={"47777": "47777"},
        judge_labels={"20001": "openai:gpt-5.4"},
        comment_anchors={"47777": -3.86},
        item_anchors={item_name: float(index) for index, item_name in enumerate(ITEM_NAMES, start=1)},
    )

    assert "1,Comments,A" in spec
    assert "47777=47777,-3.86" in spec
    assert "20001=openai:gpt-5.4" in spec
    assert "1=sentiment,1" in spec


def test_load_measure_anchors_reads_measure_column() -> None:
    anchors = load_measure_anchors(
        score_path=__import__("pathlib").Path("facets/human_baseline/human_facets_scores.3.txt"),
        key_column="facet_label",
    )

    assert anchors["sentiment"] == -2.63
    assert anchors["hate_speech"] == 0.84
