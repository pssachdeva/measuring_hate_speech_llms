"""Shared dataframe utilities."""

from typing import Mapping

import pandas as pd


def recode_responses(
    dataframe: pd.DataFrame,
    **column_mappings: Mapping[int, int],
) -> pd.DataFrame:
    """Recode response columns with per-column integer mappings.

    Values not present in a column's mapping are left unchanged so sparse
    recodes can collapse only the categories of interest.
    """

    recoded = dataframe.copy()
    for column_name, mapping in column_mappings.items():
        if column_name not in recoded.columns:
            raise ValueError(f"Cannot recode missing column: {column_name}")

        # Apply the mapping only where an explicit replacement is provided.
        recoded[column_name] = recoded[column_name].map(lambda value: mapping.get(int(value), int(value)))
    return recoded
