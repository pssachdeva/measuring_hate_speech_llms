"""Helpers for qualitative single-comment response profile figures."""

import pandas as pd

from mhs_llms.annotator_agreement import ITEM_DISPLAY_LABELS
from mhs_llms.labels import provider_display_name
from mhs_llms.schema import ITEM_NAMES
from mhs_llms.score_distribution import align_item_responses


def select_black_woman_reference_comment(
    human_annotations: pd.DataFrame,
    reference_comment_ids: list[int],
    min_black_annotators: int,
    min_white_annotators: int,
    min_target_share: float,
) -> int:
    """Select a reference comment targeting Black women with enough human coverage."""

    candidates = _build_comment_candidate_summary(
        human_annotations=human_annotations,
        reference_comment_ids=reference_comment_ids,
    )
    eligible = candidates.loc[
        (candidates["black_annotators"] >= min_black_annotators)
        & (candidates["white_annotators"] >= min_white_annotators)
        & (candidates["target_black_woman_share"] >= min_target_share)
    ].copy()
    if eligible.empty:
        raise ValueError("No reference comment matched the target and annotator coverage criteria")

    eligible = eligible.sort_values(
        ["average_aligned_score", "target_black_woman_share", "black_annotators"],
        ascending=[False, False, False],
        kind="stable",
    )
    return int(eligible.iloc[0]["comment_id"])


def build_comment_response_profile(
    human_annotations: pd.DataFrame,
    llm_annotations: pd.DataFrame,
    comment_id: int,
    provider_order: tuple[str, ...],
) -> pd.DataFrame:
    """Build item-level average responses for one comment by human group and LLM provider."""

    human_profile = _build_human_group_profile(
        human_annotations=human_annotations,
        comment_id=comment_id,
    )
    llm_profile = _build_provider_profile(
        llm_annotations=llm_annotations,
        comment_id=comment_id,
        provider_order=provider_order,
    )
    profile = pd.concat([human_profile, llm_profile], ignore_index=True)
    if profile.empty:
        raise ValueError(f"No response profile rows were available for comment_id={comment_id}")

    item_order = {item_name: index for index, item_name in enumerate(ITEM_NAMES)}
    profile["item_order"] = profile["item_name"].map(item_order)
    profile["item_label"] = profile["item_name"].map(ITEM_DISPLAY_LABELS)
    return profile.sort_values(["group_order", "item_order"], kind="stable").reset_index(drop=True)


def build_comment_summary(
    human_annotations: pd.DataFrame,
    comment_id: int,
) -> dict[str, object]:
    """Summarize the selected comment without exposing text in generated tables."""

    selected = human_annotations.loc[human_annotations["comment_id"].astype(int) == int(comment_id)]
    if selected.empty:
        raise ValueError(f"No human annotations found for comment_id={comment_id}")

    black_annotators = _human_group_mask(selected, group_name="Black annotators")
    white_annotators = _human_group_mask(selected, group_name="White annotators")
    return {
        "comment_id": int(comment_id),
        "num_human_annotations": int(len(selected)),
        "num_black_annotators": int(black_annotators.sum()),
        "num_white_annotators": int(white_annotators.sum()),
        "target_black_share": float(selected["target_race_black"].astype(bool).mean()),
        "target_women_share": float(selected["target_gender_women"].astype(bool).mean()),
        "target_black_woman_share": float(
            (selected["target_race_black"].astype(bool) & selected["target_gender_women"].astype(bool)).mean()
        ),
    }


def _build_comment_candidate_summary(
    human_annotations: pd.DataFrame,
    reference_comment_ids: list[int],
) -> pd.DataFrame:
    """Build comment-level target and annotator coverage fields for selection."""

    required_columns = {"comment_id", "target_race_black", "target_gender_women"}
    missing_columns = required_columns.difference(human_annotations.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Human annotations are missing required columns: {missing_text}")

    selected = human_annotations.loc[
        human_annotations["comment_id"].astype(int).isin(reference_comment_ids)
    ].copy()
    if selected.empty:
        raise ValueError("No human annotations matched the reference comments")

    aligned = align_item_responses(selected)
    selected["average_aligned_score"] = aligned[list(ITEM_NAMES)].sum(axis=1)
    selected["is_black_annotator"] = _human_group_mask(selected, group_name="Black annotators")
    selected["is_white_annotator"] = _human_group_mask(selected, group_name="White annotators")
    selected["targets_black_woman"] = selected["target_race_black"].astype(bool) & selected[
        "target_gender_women"
    ].astype(bool)

    return (
        selected.groupby("comment_id", as_index=False)
        .agg(
            black_annotators=("is_black_annotator", "sum"),
            white_annotators=("is_white_annotator", "sum"),
            target_black_share=("target_race_black", "mean"),
            target_women_share=("target_gender_women", "mean"),
            target_black_woman_share=("targets_black_woman", "mean"),
            average_aligned_score=("average_aligned_score", "mean"),
        )
        .sort_values("comment_id", kind="stable")
    )


def _build_human_group_profile(
    human_annotations: pd.DataFrame,
    comment_id: int,
) -> pd.DataFrame:
    """Build Black and White human annotator average item responses for one comment."""

    selected = human_annotations.loc[human_annotations["comment_id"].astype(int) == int(comment_id)].copy()
    if selected.empty:
        raise ValueError(f"No human annotations found for comment_id={comment_id}")

    aligned = align_item_responses(selected)
    rows = []
    for group_order, group_label in enumerate(("Black annotators", "White annotators")):
        group_mask = _human_group_mask(selected, group_name=group_label)
        group_values = aligned.loc[group_mask, list(ITEM_NAMES)]
        if group_values.empty:
            continue
        rows.extend(
            _profile_rows_from_wide_frame(
                wide_frame=group_values,
                comment_id=comment_id,
                group_type="Human",
                group_label=group_label,
                group_order=group_order,
            )
        )
    return pd.DataFrame(rows)


def _build_provider_profile(
    llm_annotations: pd.DataFrame,
    comment_id: int,
    provider_order: tuple[str, ...],
) -> pd.DataFrame:
    """Build provider-level average item responses for one LLM-comment profile."""

    required_columns = {"comment_id", "provider"}
    missing_columns = required_columns.difference(llm_annotations.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"LLM annotations are missing required columns: {missing_text}")

    selected = llm_annotations.loc[llm_annotations["comment_id"].astype(int) == int(comment_id)].copy()
    if selected.empty:
        raise ValueError(f"No LLM annotations found for comment_id={comment_id}")

    aligned = align_item_responses(selected)
    aligned["provider_slug"] = selected["provider"].fillna("unknown").astype(str)
    aligned.loc[aligned["provider_slug"] == "", "provider_slug"] = "unknown"

    rows = []
    provider_rank = {provider: index for index, provider in enumerate(provider_order)}
    for provider_slug, group in aligned.groupby("provider_slug", sort=False):
        group_order = 2 + provider_rank.get(provider_slug, len(provider_rank))
        group_label = provider_display_name(provider_slug)
        rows.extend(
            _profile_rows_from_wide_frame(
                wide_frame=group[list(ITEM_NAMES)],
                comment_id=comment_id,
                group_type="LLM",
                group_label=group_label,
                group_order=group_order,
            )
        )
    return pd.DataFrame(rows)


def _profile_rows_from_wide_frame(
    wide_frame: pd.DataFrame,
    comment_id: int,
    group_type: str,
    group_label: str,
    group_order: int,
) -> list[dict[str, object]]:
    """Convert a wide item-response frame into one mean row per item."""

    rows = []
    for item_name in ITEM_NAMES:
        values = wide_frame[item_name].dropna().astype(float)
        rows.append(
            {
                "comment_id": int(comment_id),
                "group_type": group_type,
                "group_label": group_label,
                "group_order": int(group_order),
                "item_name": item_name,
                "mean_response": float(values.mean()),
                "num_annotations": int(len(values)),
            }
        )
    return rows


def _human_group_mask(annotation_frame: pd.DataFrame, group_name: str) -> pd.Series:
    """Return the row mask for the supported human annotator race groups."""

    if group_name == "Black annotators":
        if "annotator_race_black" not in annotation_frame.columns:
            raise ValueError("Human annotations are missing annotator_race_black")
        return annotation_frame["annotator_race_black"].astype(bool)
    if group_name == "White annotators":
        required_columns = {"annotator_race_white", "annotator_race_black"}
        missing_columns = required_columns.difference(annotation_frame.columns)
        if missing_columns:
            missing_text = ", ".join(sorted(missing_columns))
            raise ValueError(f"Human annotations are missing required columns: {missing_text}")
        return annotation_frame["annotator_race_white"].astype(bool) & ~annotation_frame[
            "annotator_race_black"
        ].astype(bool)
    raise ValueError(f"Unsupported human group: {group_name}")
