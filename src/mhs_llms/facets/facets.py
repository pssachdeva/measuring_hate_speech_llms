"""FACETS export helpers."""

from pathlib import Path
from typing import Mapping

import pandas as pd

from mhs_llms.config import FacetsConfig
from mhs_llms.schema import ITEM_NAMES


def build_facets_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert normalized annotations into the FACETS input table layout."""

    facets = dataframe[["comment_id", "judge_id", *ITEM_NAMES]].copy()
    # FACETS expects numeric item scores rather than string/object dtypes.
    for item in ITEM_NAMES:
        facets[item] = facets[item].astype(int)
    # The current baseline treats the full MHS item set as one ordered item block.
    facets["item_id"] = f"1-{len(ITEM_NAMES)}"
    return facets[["comment_id", "judge_id", "item_id", *ITEM_NAMES]]


def build_human_facets_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper for the human baseline FACETS export."""

    return build_facets_frame(dataframe)


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
    comment_labels: Mapping[str, str] | None = None,
    judge_labels: Mapping[str, str] | None = None,
    comment_anchors: Mapping[str, float] | None = None,
    item_anchors: Mapping[str, float] | None = None,
) -> str:
    """Build a FACETS spec file for the current three-facet baseline design."""

    # FACETS label blocks enumerate the distinct ids available for each facet.
    comment_ids = facets_frame["comment_id"].drop_duplicates().astype(int).astype(str).tolist()
    judge_ids = facets_frame["judge_id"].drop_duplicates().astype(str).tolist()
    item_ids = [str(index) for index, _ in enumerate(ITEM_NAMES, start=1)]
    delements = ", ".join(facets_config.delements)
    comment_block = _build_label_block(
        facet_number=1,
        facet_name="Comments",
        element_ids=comment_ids,
        labels=comment_labels,
        anchors=comment_anchors,
    )
    judge_block = _build_label_block(
        facet_number=2,
        facet_name="Judges",
        element_ids=judge_ids,
        labels=judge_labels,
    )
    item_block = _build_label_block(
        facet_number=3,
        facet_name="Items",
        element_ids=item_ids,
        labels={str(index): item_name for index, item_name in enumerate(ITEM_NAMES, start=1)},
        anchors=item_anchors,
        anchor_key_by_label=True,
        anchor_header=True,
    )

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
        f"{comment_block}\n"
        f"{judge_block}\n"
        f"{item_block}\n\n"
        f"Data = {data_filename}\n"
    )


def _build_label_block(
    facet_number: int,
    facet_name: str,
    element_ids: list[str],
    labels: Mapping[str, str] | None = None,
    anchors: Mapping[str, float] | None = None,
    anchor_key_by_label: bool = False,
    anchor_header: bool = False,
) -> str:
    """Build one FACETS label block, optionally anchoring element measures."""

    header = f"{facet_number},{facet_name}"
    if anchor_header or anchors is not None:
        header = f"{header},A"

    label_lines: list[str] = [header]
    for element_id in element_ids:
        label = ""
        if labels is not None and element_id in labels:
            label = labels[element_id]

        anchor_key = label if anchor_key_by_label else element_id
        if anchors is None:
            label_lines.append(f"{element_id}={label}")
            continue

        if not label:
            label = element_id
        anchor_value = anchors[anchor_key]
        label_lines.append(f"{element_id}={label},{_format_measure(anchor_value)}")
    label_lines.append("*")
    return "\n".join(label_lines)


def _format_measure(value: float) -> str:
    """Format a FACETS anchor value compactly while preserving sign and scale."""

    formatted = f"{float(value):.6f}".rstrip("0").rstrip(".")
    if formatted == "-0":
        return "0"
    return formatted


def write_facets_spec(spec_text: str, output_path: Path) -> None:
    """Write the FACETS spec text to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(spec_text)
