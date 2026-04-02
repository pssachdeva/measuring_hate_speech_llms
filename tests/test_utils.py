import pandas as pd

from mhs_llms.utils import recode_responses


def test_recode_responses_applies_sparse_integer_mappings() -> None:
    dataframe = pd.DataFrame(
        [
            {
                "insult": 0,
                "humiliate": 1,
                "status": 4,
                "hate_speech": 2,
            }
        ]
    )

    recoded = recode_responses(
        dataframe,
        insult={1: 0, 2: 1, 3: 2, 4: 3},
        humiliate={1: 0, 2: 0, 3: 1, 4: 2},
        status={1: 0, 2: 0, 3: 1, 4: 1},
        hate_speech={1: 0, 2: 1},
    )

    assert recoded.loc[0, "insult"] == 0
    assert recoded.loc[0, "humiliate"] == 0
    assert recoded.loc[0, "status"] == 1
    assert recoded.loc[0, "hate_speech"] == 1


def test_recode_responses_raises_for_missing_columns() -> None:
    dataframe = pd.DataFrame([{"insult": 1}])

    try:
        recode_responses(dataframe, humiliate={1: 0})
    except ValueError as exc:
        assert str(exc) == "Cannot recode missing column: humiliate"
    else:
        raise AssertionError("Expected recode_responses to reject missing columns")
