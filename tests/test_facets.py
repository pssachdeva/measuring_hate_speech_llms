import pandas as pd

from mhs_llms.config import FacetsConfig
from mhs_llms.facets import build_facets_spec, build_human_facets_frame


def test_build_human_facets_frame_uses_expected_columns_and_item_order() -> None:
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

    facets = build_human_facets_frame(dataframe)

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
