import pandas as pd

from mhs_llms.dataset import normalize_mhs_dataframe
from mhs_llms.schema import ITEM_NAMES


def test_normalize_mhs_dataframe_renames_violence_phys_and_preserves_required_columns() -> None:
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
                "violence_phys": 4,
                "genocide": 4,
                "attack_defend": 4,
                "hatespeech": 2,
            }
        ]
    )

    normalized = normalize_mhs_dataframe(dataframe)

    assert "violence" in normalized.columns
    assert "violence_phys" not in normalized.columns
    assert "hate_speech" in normalized.columns
    assert "hatespeech" not in normalized.columns
    for column_name in ("comment_id", "annotator_id", "platform", "text", *ITEM_NAMES):
        assert column_name in normalized.columns
