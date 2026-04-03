"""FACETS export helpers."""

from pathlib import Path

import pandas as pd

from mhs_llms.config import FacetsConfig
from mhs_llms.schema import ITEM_NAMES


def build_human_facets_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert normalized human annotations into the FACETS input table layout."""

    facets = dataframe[["comment_id", "judge_id", *ITEM_NAMES]].copy()
    # FACETS expects numeric item scores rather than string/object dtypes.
    for item in ITEM_NAMES:
        facets[item] = facets[item].astype(int)
    # The current baseline treats the full MHS item set as one ordered item block.
    facets["item_id"] = f"1-{len(ITEM_NAMES)}"
    return facets[["comment_id", "judge_id", "item_id", *ITEM_NAMES]]


def write_facets_data(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Write the FACETS data table as a tab-delimited file without headers."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, sep="\t", header=False, index=False)


def build_facets_spec(
    facets_frame: pd.DataFrame,
    facets_config: FacetsConfig,
    data_filename: str,
    score_filename: str,
    output_filename: str,
) -> str:
    """Build a FACETS spec file for the current three-facet baseline design."""

    # FACETS label blocks enumerate the distinct ids available for each facet.
    comment_ids = "=\n".join(facets_frame["comment_id"].drop_duplicates().astype(str).tolist())
    judge_ids = "=\n".join(facets_frame["judge_id"].drop_duplicates().astype(str).tolist())
    item_ids = "\n".join(f"{index}={item_name}" for index, item_name in enumerate(ITEM_NAMES, start=1))
    delements = ", ".join(facets_config.delements)

    return (
        f"Title = {facets_config.title}\n"
        "Facets = 3\n"
        f"Model = {facets_config.model}\n"
        f"Noncenter = {facets_config.noncenter}\n"
        f"Positive = {facets_config.positive}\n"
        f"Arrange = {facets_config.arrange}\n"
        f"Subset detection = {facets_config.subset_detection}\n"
        f"Delements = {delements}\n"
        f"Scorefile = {score_filename}\n"
        f"Output file = {output_filename}\n"
        f"CSV = {facets_config.csv}\n\n"
        "Labels =\n"
        "1,Comments\n"
        f"{comment_ids}=\n*\n"
        "2,Judges\n"
        f"{judge_ids}=\n*\n"
        "3,Items,A\n"
        f"{item_ids}\n*\n\n"
        f"Data = {data_filename}\n"
    )


def write_facets_spec(spec_text: str, output_path: Path) -> None:
    """Write the FACETS spec text to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(spec_text)
