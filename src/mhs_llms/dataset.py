"""Dataset access and normalization for MHS."""

from typing import Optional

import pandas as pd
from datasets import load_dataset

from mhs_llms.schema import ITEM_NAMES


def load_mhs_dataframe(
    dataset_name: str,
    split: str = "train",
    config_name: Optional[str] = None,
) -> pd.DataFrame:
    """Load an MHS split from Hugging Face and normalize its column layout."""
    dataset = load_dataset(dataset_name, name=config_name, split=split)
    dataframe = dataset.to_pandas()
    return normalize_mhs_dataframe(dataframe)


def normalize_mhs_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Standardize expected MHS column names and validate required fields."""
    normalized = dataframe.copy()
    # Some dataset versions expose the violence item as `violence_phys`.
    if "violence_phys" in normalized.columns and "violence" not in normalized.columns:
        normalized = normalized.rename(columns={"violence_phys": "violence"})
    # The HF source uses `hatespeech`, but the repo schema uses `hate_speech`.
    if "hatespeech" in normalized.columns and "hate_speech" not in normalized.columns:
        normalized = normalized.rename(columns={"hatespeech": "hate_speech"})

    missing_columns = [column for column in ("comment_id", "annotator_id", "platform", "text", *ITEM_NAMES) if column not in normalized.columns]
    if missing_columns:
        raise ValueError(f"MHS dataset is missing required columns: {missing_columns}")
    return normalized


def build_comment_frame(
    dataframe: pd.DataFrame,
    comment_ids: Optional[list[int]] = None,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """Return one row per comment, optionally filtered to a provided id subset."""

    comments = dataframe[["comment_id", "text"]].drop_duplicates(subset=["comment_id"]).copy()
    comments["comment_id"] = comments["comment_id"].astype(int)

    if comment_ids is not None:
        comment_order = pd.Index(comment_ids, name="comment_id")
        comments = comments.set_index("comment_id").reindex(comment_order).dropna(subset=["text"]).reset_index()
    else:
        comments = comments.sort_values("comment_id", kind="stable").reset_index(drop=True)
    if limit is not None:
        comments = comments.head(limit).copy()
    return comments
