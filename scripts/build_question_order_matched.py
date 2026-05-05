import argparse
from pathlib import Path

import pandas as pd


TARGET_MODEL_IDS = (
    "openrouter_deepseek_deepseek-v3.2",
    "openai_gpt-5.4_medium",
    "google_gemini-3.1-pro-preview_medium",
    "openrouter_minimax_minimax-m2.5",
    "openrouter_moonshotai_kimi-k2.5",
    "xai_grok-4-1-fast-reasoning",
    "anthropic_claude-opus-4-6_medium",
)

SECONDARY_MODEL_IDS = (
    "openai_gpt-4o",
    "xai_grok-3",
    "google_gemini-2.5-pro",
    "anthropic_claude-sonnet-4-6_medium",
    "anthropic_claude-opus-4-5_medium",
)

MODEL_SETS = {
    "primary": TARGET_MODEL_IDS,
    "secondary": SECONDARY_MODEL_IDS,
    "expanded": (*TARGET_MODEL_IDS, *SECONDARY_MODEL_IDS),
}

ORIGINAL_INPUT_PATHS = (
    Path("data/reference_set_openai_processed.csv"),
    Path("data/reference_set_google_processed.csv"),
    Path("data/reference_set_anthropic_processed.csv"),
    Path("data/reference_set_xai_processed.csv"),
    Path("data/reference_set_open_large_processed.csv"),
)

REVERSE_INPUT_PATHS = (
    Path("data/question_order_reverse_batch_processed.csv"),
    Path("data/question_order_reverse_openrouter_processed.csv"),
)

SECONDARY_REVERSE_INPUT_PATHS = (
    Path("data/question_order_reverse_secondary_batch_processed.csv"),
)

DEFAULT_ORIGINAL_OUTPUT_PATH = Path("data/question_order_original_matched_processed.csv")
DEFAULT_REVERSE_OUTPUT_PATH = Path("data/question_order_reverse_matched_processed.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build matched original/reverse processed CSVs for the question-order FACETS runs."
    )
    parser.add_argument(
        "--model-set",
        choices=sorted(MODEL_SETS),
        default="primary",
        help="Model set to include in the matched output.",
    )
    parser.add_argument(
        "--include-secondary-reverse",
        action="store_true",
        help="Also read reverse-order outputs for the secondary native-batch models.",
    )
    parser.add_argument(
        "--original-output",
        type=Path,
        default=DEFAULT_ORIGINAL_OUTPUT_PATH,
        help="Output path for original-order matched annotations.",
    )
    parser.add_argument(
        "--reverse-output",
        type=Path,
        default=DEFAULT_REVERSE_OUTPUT_PATH,
        help="Output path for reverse-order matched annotations.",
    )
    args = parser.parse_args()

    original = load_processed_annotations(ORIGINAL_INPUT_PATHS)
    reverse_paths = REVERSE_INPUT_PATHS
    if args.include_secondary_reverse:
        reverse_paths = (*reverse_paths, *SECONDARY_REVERSE_INPUT_PATHS)
    reverse = load_processed_annotations(reverse_paths)
    original_matched, reverse_matched = build_matched_question_order_frames(
        original=original,
        reverse=reverse,
        model_ids=MODEL_SETS[args.model_set],
    )

    args.original_output.parent.mkdir(parents=True, exist_ok=True)
    args.reverse_output.parent.mkdir(parents=True, exist_ok=True)
    original_matched.to_csv(args.original_output, index=False)
    reverse_matched.to_csv(args.reverse_output, index=False)
    print(f"original_output={args.original_output.resolve()}")
    print(f"reverse_output={args.reverse_output.resolve()}")
    print(f"rows={len(original_matched)}")
    print(f"models={original_matched['judge_id'].nunique()}")
    print(f"comments={original_matched['comment_id'].nunique()}")


def load_processed_annotations(paths: tuple[Path, ...]) -> pd.DataFrame:
    """Load and combine processed annotation CSV files."""

    missing_paths = [path for path in paths if not path.exists()]
    if missing_paths:
        missing_text = ", ".join(str(path) for path in missing_paths)
        raise FileNotFoundError(f"Missing processed annotation inputs: {missing_text}")
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)


def build_matched_question_order_frames(
    original: pd.DataFrame,
    reverse: pd.DataFrame,
    model_ids: tuple[str, ...] = TARGET_MODEL_IDS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Filter original/reverse annotations to identical model/comment pairs."""

    required_columns = {"comment_id", "judge_id"}
    for label, dataframe in {"original": original, "reverse": reverse}.items():
        missing = required_columns.difference(dataframe.columns)
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"{label} annotations are missing required columns: {missing_text}")

    original_selected = _select_models(original, model_ids)
    reverse_selected = _select_models(reverse, model_ids)
    _assert_all_models_present(original_selected, model_ids, "original")
    _assert_all_models_present(reverse_selected, model_ids, "reverse")

    original_pairs = _pair_frame(original_selected)
    reverse_pairs = _pair_frame(reverse_selected)
    matched_pairs = original_pairs.merge(reverse_pairs, on=["judge_id", "comment_id"], how="inner")
    if matched_pairs.empty:
        raise ValueError("No matched judge/comment pairs found between original and reverse annotations")

    common_comment_ids = _common_complete_comment_ids(matched_pairs, model_ids)
    if not common_comment_ids:
        missing_summary = _summarize_missing_pairs(original_pairs, reverse_pairs)
        raise ValueError(f"No complete shared comments across all models: {missing_summary}")
    matched_pairs = matched_pairs.loc[matched_pairs["comment_id"].isin(common_comment_ids)].copy()

    original_matched = original_selected.merge(matched_pairs, on=["judge_id", "comment_id"], how="inner")
    reverse_matched = reverse_selected.merge(matched_pairs, on=["judge_id", "comment_id"], how="inner")
    return _sort_matched(original_matched), _sort_matched(reverse_matched)


def _select_models(dataframe: pd.DataFrame, model_ids: tuple[str, ...]) -> pd.DataFrame:
    """Keep rows for the requested model ids."""

    selected = dataframe.copy()
    selected["judge_id"] = selected["judge_id"].astype(str).map(_normalize_judge_id)
    return selected.loc[selected["judge_id"].isin(model_ids)].copy()


def _normalize_judge_id(value: str) -> str:
    """Normalize known order-condition suffixes away from judge ids."""

    suffixes = (
        "__original_order",
        "__reverse_order",
        "_original_order",
        "_reverse_order",
        "-original-order",
        "-reverse-order",
    )
    for suffix in suffixes:
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


def _assert_all_models_present(dataframe: pd.DataFrame, model_ids: tuple[str, ...], label: str) -> None:
    """Raise if any requested model is missing from a condition."""

    present = set(dataframe["judge_id"].astype(str).unique())
    missing = [model_id for model_id in model_ids if model_id not in present]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"{label} annotations are missing requested models: {missing_text}")


def _pair_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return one row per judge/comment pair."""

    pairs = dataframe.loc[:, ["judge_id", "comment_id"]].copy()
    pairs["comment_id"] = pairs["comment_id"].astype(int)
    pairs = pairs.drop_duplicates()
    duplicated = pairs.duplicated(["judge_id", "comment_id"])
    if duplicated.any():
        raise ValueError("Duplicate judge/comment pairs found")
    return pairs


def _summarize_missing_pairs(original_pairs: pd.DataFrame, reverse_pairs: pd.DataFrame) -> str:
    """Summarize the first few asymmetric judge/comment pairs."""

    original_set = set(original_pairs.itertuples(index=False, name=None))
    reverse_set = set(reverse_pairs.itertuples(index=False, name=None))
    missing_reverse = sorted(original_set.difference(reverse_set))[:5]
    missing_original = sorted(reverse_set.difference(original_set))[:5]
    return f"missing reverse examples={missing_reverse}; missing original examples={missing_original}"


def _common_complete_comment_ids(matched_pairs: pd.DataFrame, model_ids: tuple[str, ...]) -> set[int]:
    """Return comments present for every requested model in both conditions."""

    counts = matched_pairs.groupby("comment_id")["judge_id"].nunique()
    complete_comment_ids = counts.loc[counts == len(model_ids)].index.astype(int).tolist()
    return set(complete_comment_ids)


def _sort_matched(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Sort matched rows by model and comment for stable downstream FACETS files."""

    return dataframe.sort_values(["judge_id", "comment_id"], kind="stable").reset_index(drop=True)


if __name__ == "__main__":
    main()
