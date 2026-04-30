"""Prepare FACETS inputs for target-identity differential rater functioning."""

from dataclasses import dataclass
import math
from pathlib import Path
import re

import pandas as pd

from mhs_llms.config import TargetDRFConfig, load_target_drf_config
from mhs_llms.dataset import load_mhs_dataframe
from mhs_llms.facets.anchored import _prepare_llm_annotations
from mhs_llms.facets.facets import _format_measure, write_facets_data, write_facets_spec
from mhs_llms.facets.postprocess import load_measure_anchors, parse_facets_score_file
from mhs_llms.schema import ITEM_NAMES, TARGET_GROUP_COLUMNS


@dataclass(frozen=True)
class TargetDRFOutputs:
    """Paths produced when preparing one target-identity DRF FACETS run."""

    facets_data_path: Path
    facets_spec_path: Path
    target_labels_path: Path


@dataclass(frozen=True)
class TargetDRFPostprocessOutputs:
    """Paths produced when processing target-identity DRF FACETS output."""

    target_terms_path: Path
    pairwise_contrasts_path: Path


def run_target_drf_facets(config_path: Path) -> TargetDRFOutputs:
    """Prepare a FACETS run with judge-by-target identity interactions."""

    config = load_target_drf_config(config_path)
    target_labels = build_target_identity_labels(config)
    annotation_frames = [pd.read_csv(annotation_path) for annotation_path in config.annotation_paths]
    annotations = pd.concat(annotation_frames, ignore_index=True)
    target_labels = filter_target_labels_to_annotations(
        target_labels=target_labels,
        annotations=annotations,
        min_comments_per_target=config.min_comments_per_target,
    )

    selected_annotations = annotations.merge(
        target_labels[["comment_id", "target_identity", "target_id"]],
        on="comment_id",
        how="inner",
    )
    if selected_annotations.empty:
        raise ValueError("No LLM annotations matched the target-identity label table")

    prepared_annotations, judge_mapping = _prepare_llm_annotations(annotations=selected_annotations)
    if "target_id" not in prepared_annotations.columns:
        prepared_annotations = prepared_annotations.merge(
            selected_annotations[["comment_id", "target_id"]].drop_duplicates(),
            on="comment_id",
            how="left",
        )
    facets_frame = build_target_drf_facets_frame(prepared_annotations)

    comment_anchors = load_measure_anchors(
        score_path=config.comment_scores_path,
        key_column="facet_id",
        measure_column="measure",
    )
    item_anchors = load_measure_anchors(
        score_path=config.item_scores_path,
        key_column="facet_label",
        measure_column="measure",
    )
    judge_anchors = load_measure_anchors(
        score_path=config.judge_scores_path,
        key_column="facet_label",
        measure_column="measure",
    )

    facets_run_dir = config.facets_run_dir
    facets_run_dir.mkdir(parents=True, exist_ok=True)
    config.target_labels_path.parent.mkdir(parents=True, exist_ok=True)
    target_labels.to_csv(config.target_labels_path, index=False)

    facets_data_path = facets_run_dir / config.facets_data_filename
    write_facets_data(facets_frame, facets_data_path)

    spec_text = build_target_drf_facets_spec(
        facets_frame=facets_frame,
        config=config,
        judge_mapping=judge_mapping,
        target_labels=target_labels,
        comment_anchors=comment_anchors,
        item_anchors=item_anchors,
        judge_anchors=judge_anchors,
    )
    facets_spec_path = facets_run_dir / config.facets_spec_filename
    write_facets_spec(spec_text, facets_spec_path)

    return TargetDRFOutputs(
        facets_data_path=facets_data_path,
        facets_spec_path=facets_spec_path,
        target_labels_path=config.target_labels_path,
    )


def process_target_drf_run(
    config_path: Path,
    target_terms_path: Path | None = None,
    pairwise_contrasts_path: Path | None = None,
) -> TargetDRFPostprocessOutputs:
    """Parse target DRF FACETS outputs into tidy target and contrast tables."""

    config = load_target_drf_config(config_path)
    target_terms = parse_target_term_scores(config)
    pairwise_contrasts = parse_target_pairwise_contrasts(
        config.facets_run_dir / config.facets_output_filename
    )

    if target_terms_path is None:
        target_terms_path = Path.cwd() / "data" / f"{config.facets_run_dir.name}_target_terms.csv"
    if pairwise_contrasts_path is None:
        pairwise_contrasts_path = (
            Path.cwd() / "data" / f"{config.facets_run_dir.name}_pairwise_contrasts.csv"
        )

    target_terms_path.parent.mkdir(parents=True, exist_ok=True)
    pairwise_contrasts_path.parent.mkdir(parents=True, exist_ok=True)
    target_terms.to_csv(target_terms_path, index=False)
    pairwise_contrasts.to_csv(pairwise_contrasts_path, index=False)

    return TargetDRFPostprocessOutputs(
        target_terms_path=target_terms_path,
        pairwise_contrasts_path=pairwise_contrasts_path,
    )


def parse_target_term_scores(config: TargetDRFConfig) -> pd.DataFrame:
    """Read the FACETS target score file and attach target label counts."""

    target_score_path = config.facets_run_dir / config.facets_score_filename.replace(
        ".txt", ".4.txt"
    )
    target_terms = parse_facets_score_file(target_score_path).rename(
        columns={
            "facet_id": "target_id",
            "facet_label": "target_identity",
            "measure": "target_measure",
            "s_e": "target_se",
            "infit_ms": "target_infit_ms",
            "outfit_ms": "target_outfit_ms",
            "t_count": "response_count",
            "t_score": "observed_score",
        }
    )
    columns = [
        "target_id",
        "target_identity",
        "target_measure",
        "target_se",
        "target_infit_ms",
        "target_outfit_ms",
        "response_count",
        "observed_score",
    ]
    target_terms = target_terms[columns].copy()

    if config.target_labels_path.exists():
        labels = pd.read_csv(config.target_labels_path)
        counts = (
            labels.groupby("target_identity")
            .agg(
                comment_count=("comment_id", "nunique"),
                mean_target_share=("target_share", "mean"),
            )
            .reset_index()
        )
        target_terms = target_terms.merge(counts, on="target_identity", how="left")
    return target_terms.sort_values("target_measure", kind="stable").reset_index(drop=True)


def parse_target_pairwise_contrasts(report_path: Path) -> pd.DataFrame:
    """Parse FACETS Table 14 judge-by-target pairwise contrasts."""

    rows = []
    in_table = False
    for line in report_path.read_text(errors="replace").splitlines():
        if line.startswith("Table 14.") and "Bias/Interaction Pairwise Report" in line:
            in_table = True
            continue
        if in_table and line.startswith("+---") and rows:
            break
        if not in_table or not _looks_like_pairwise_row(line):
            continue
        rows.append(_parse_pairwise_row(line))

    if not rows:
        raise ValueError(f"No FACETS target pairwise rows found in {report_path}")
    return pd.DataFrame(rows)


def build_target_identity_labels(config: TargetDRFConfig) -> pd.DataFrame:
    """Create one collapsed target-identity label per high-agreement comment."""

    dataset = load_mhs_dataframe(
        dataset_name=config.dataset_name,
        split=config.split,
        config_name=None,
    )
    detailed_target_columns = [column for columns in TARGET_GROUP_COLUMNS.values() for column in columns]
    selected = dataset[["comment_id", *detailed_target_columns]].copy()
    selected[detailed_target_columns] = selected[detailed_target_columns].fillna(False).astype(bool)

    annotator_counts = selected.groupby("comment_id").size().rename("n_annotators")
    target_shares = selected.groupby("comment_id")[detailed_target_columns].mean().join(annotator_counts)
    target_shares = target_shares.loc[target_shares["n_annotators"] >= config.min_annotators].copy()

    agreement_mask = target_shares[detailed_target_columns] >= config.agreement_threshold
    single_target = target_shares.loc[agreement_mask.sum(axis=1) == 1].copy()
    single_target["raw_target"] = agreement_mask.loc[single_target.index].idxmax(axis=1)
    single_target["target_identity"] = single_target["raw_target"].map(
        lambda value: config.collapse_targets.get(value, value)
    )
    single_target = single_target.loc[~single_target["raw_target"].isin(config.exclude_targets)].copy()

    target_counts = single_target["target_identity"].value_counts()
    kept_targets = target_counts.loc[target_counts >= config.min_comments_per_target].index
    single_target = single_target.loc[single_target["target_identity"].isin(kept_targets)].copy()
    if single_target.empty:
        raise ValueError("No target identities remained after applying target filters")

    target_id_map = {
        target_identity: str(index)
        for index, target_identity in enumerate(sorted(single_target["target_identity"].unique()), start=1)
    }
    single_target["target_id"] = single_target["target_identity"].map(target_id_map)
    single_target["target_share"] = [
        float(single_target.loc[comment_id, raw_target])
        for comment_id, raw_target in single_target["raw_target"].items()
    ]
    return (
        single_target.reset_index()[[
            "comment_id",
            "n_annotators",
            "raw_target",
            "target_identity",
            "target_id",
            "target_share",
        ]]
        .sort_values(["target_identity", "comment_id"], kind="stable")
        .reset_index(drop=True)
    )


def _looks_like_pairwise_row(line: str) -> bool:
    """Return whether one report line has the Table 14 pairwise row shape."""

    parts = line.split("|")
    return len(parts) >= 5 and bool(re.match(r"^\s*\d+\s+", parts[1]))


def _parse_pairwise_row(line: str) -> dict[str, object]:
    """Parse one FACETS Table 14 pairwise target contrast row."""

    parts = line.split("|")
    judge = _parse_pairwise_judge(parts[1])
    target_a = _parse_pairwise_target(parts[2], "a")
    target_b = _parse_pairwise_target(parts[3], "b")
    contrast = _parse_pairwise_contrast(parts[4])
    return {**judge, **target_a, **target_b, **contrast}


def _parse_pairwise_judge(text: str) -> dict[str, object]:
    """Parse the judge identifier block from a pairwise row."""

    match = re.match(r"^\s*(?P<judge_id>\d+)\s+(?P<judge_label>.+?)\s*$", text)
    if match is None:
        raise ValueError(f"Unexpected FACETS pairwise judge format: {text}")
    return {
        "judge_id": int(match.group("judge_id")),
        "judge_label": match.group("judge_label").strip(),
    }


def _parse_pairwise_target(text: str, suffix: str) -> dict[str, object]:
    """Parse one target side from a pairwise contrast row."""

    number_pattern = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    match = re.match(
        rf"^\s*(?P<measure>{number_pattern})\s+"
        rf"(?P<se>{number_pattern})\s+"
        rf"(?P<obs_exp>{number_pattern})\s+"
        rf"(?P<target_id>\d+)\s+"
        rf"(?P<target_identity>\S+)\s*$",
        text,
    )
    if match is None:
        raise ValueError(f"Unexpected FACETS pairwise target format: {text}")
    return {
        f"target_{suffix}_id": int(match.group("target_id")),
        f"target_{suffix}": match.group("target_identity"),
        f"target_{suffix}_measure": _to_float(match.group("measure")),
        f"target_{suffix}_se": _to_float(match.group("se")),
        f"target_{suffix}_obs_exp_average": _to_float(match.group("obs_exp")),
    }


def _parse_pairwise_contrast(text: str) -> dict[str, object]:
    """Parse the contrast statistics from a pairwise row."""

    values = text.split()
    if len(values) != 5:
        raise ValueError(f"Unexpected FACETS pairwise contrast format: {text}")
    contrast = _to_float(values[0])
    return {
        "target_contrast": contrast,
        "contrast_se": _to_float(values[1]),
        "t": _to_float(values[2]),
        "df": int(_to_float(values[3])),
        "p_value": _to_float(values[4]),
        "higher_score_odds_ratio_a_vs_b": math.exp(-contrast),
    }


def _to_float(value: str) -> float:
    """Convert FACETS compact decimal text to float."""

    if value.startswith("."):
        return float(f"0{value}")
    if value.startswith("-."):
        return float(value.replace("-.", "-0.", 1))
    return float(value)


def filter_target_labels_to_annotations(
    target_labels: pd.DataFrame,
    annotations: pd.DataFrame,
    min_comments_per_target: int,
) -> pd.DataFrame:
    """Keep only target labels represented in the provided annotation dataset."""

    annotation_comment_ids = set(annotations["comment_id"].astype(int).tolist())
    selected = target_labels.loc[target_labels["comment_id"].astype(int).isin(annotation_comment_ids)].copy()
    target_counts = selected["target_identity"].value_counts()
    kept_targets = target_counts.loc[target_counts >= min_comments_per_target].index
    selected = selected.loc[selected["target_identity"].isin(kept_targets)].copy()
    if selected.empty:
        raise ValueError("No annotated target identities remained after applying target filters")

    target_id_map = {
        target_identity: str(index)
        for index, target_identity in enumerate(sorted(selected["target_identity"].unique()), start=1)
    }
    selected["target_id"] = selected["target_identity"].map(target_id_map)
    return selected.sort_values(["target_identity", "comment_id"], kind="stable").reset_index(drop=True)


def build_target_drf_facets_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Convert target-labeled annotations into a four-facet FACETS input table."""

    facets = dataframe[["comment_id", "judge_id", "target_id", *ITEM_NAMES]].copy()
    for item_name in ITEM_NAMES:
        facets[item_name] = facets[item_name].astype(int)
    facets["item_id"] = f"1-{len(ITEM_NAMES)}"
    return facets[["comment_id", "judge_id", "item_id", "target_id", *ITEM_NAMES]]


def build_target_drf_facets_spec(
    facets_frame: pd.DataFrame,
    config: TargetDRFConfig,
    judge_mapping: pd.DataFrame,
    target_labels: pd.DataFrame,
    comment_anchors: dict[str, float],
    item_anchors: dict[str, float],
    judge_anchors: dict[str, float],
) -> str:
    """Build a four-facet FACETS spec with judge and target bias terms."""

    comment_ids = facets_frame["comment_id"].drop_duplicates().astype(int).astype(str).tolist()
    judge_ids = facets_frame["judge_id"].drop_duplicates().astype(str).tolist()
    target_ids = sorted(
        facets_frame["target_id"].drop_duplicates().astype(str).tolist(),
        key=lambda value: int(value),
    )
    item_ids = [str(index) for index, _ in enumerate(ITEM_NAMES, start=1)]

    comment_label_map = {comment_id: comment_id for comment_id in comment_ids}
    judge_label_map = dict(
        zip(
            judge_mapping["judge_id"].astype(str).tolist(),
            judge_mapping["external_judge_id"].astype(str).tolist(),
            strict=True,
        )
    )
    target_label_map = (
        target_labels[["target_id", "target_identity"]]
        .drop_duplicates()
        .assign(target_id=lambda frame: frame["target_id"].astype(str))
        .set_index("target_id")["target_identity"]
        .to_dict()
    )

    relevant_comment_anchors = _select_anchor_subset(comment_anchors, comment_ids, "comment")
    relevant_item_anchors = _select_anchor_subset(item_anchors, ITEM_NAMES, "item")
    relevant_judge_anchors = _select_judge_anchors(judge_anchors, judge_label_map)

    delements = ", ".join(config.facets.delements)
    optional_lines = []
    if config.facets.bias is not None:
        optional_lines.append(f"Bias = {config.facets.bias}")
    if config.facets.zscore is not None:
        optional_lines.append(f"Zscore = {config.facets.zscore}")
    optional_text = "\n".join(optional_lines)
    if optional_text:
        optional_text = f"{optional_text}\n"

    return (
        f"Title = {config.facets.title}\n"
        "Facets = 4\n"
        f"Model = {config.facets.model}\n"
        f"Noncenter = {config.facets.noncenter}\n"
        f"Positive = {config.facets.positive}\n"
        f"Arrange = {config.facets.arrange}\n"
        f"Subset detection = {config.facets.subset_detection}\n"
        f"Delements = {delements}\n"
        f"{optional_text}"
        f"Scorefile = {config.facets_score_filename}\n"
        f"Output file = {config.facets_output_filename}\n"
        f"CSV = {config.facets.csv}\n\n"
        "Labels =\n"
        f"{_build_anchor_label_block(1, 'Comments', comment_ids, comment_label_map, relevant_comment_anchors)}\n"
        f"{_build_anchor_label_block(2, 'Judges', judge_ids, judge_label_map, relevant_judge_anchors)}\n"
        f"{_build_anchor_label_block(3, 'Items', item_ids, dict(enumerate(ITEM_NAMES, start=1)), relevant_item_anchors, anchor_key_by_label=True)}\n"
        f"{_build_plain_label_block(4, 'Targets', target_ids, target_label_map)}\n\n"
        f"Data = {config.facets_data_filename}\n"
    )


def _select_anchor_subset(
    anchors: dict[str, float],
    keys: list[str] | tuple[str, ...],
    anchor_name: str,
) -> dict[str, float]:
    """Return anchors for the requested keys or report the missing values."""

    missing = [str(key) for key in keys if str(key) not in anchors]
    if missing:
        missing_text = ", ".join(missing[:10])
        raise ValueError(f"Missing {anchor_name} anchors for values: {missing_text}")
    return {str(key): anchors[str(key)] for key in keys}


def _select_judge_anchors(
    anchors: dict[str, float],
    judge_label_map: dict[str, str],
) -> dict[str, float]:
    """Map current numeric FACETS judge ids to reference-set judge measures."""

    missing = [label for label in judge_label_map.values() if label not in anchors]
    if missing:
        missing_text = ", ".join(missing[:10])
        raise ValueError(f"Missing judge anchors for labels: {missing_text}")
    return {judge_id: anchors[label] for judge_id, label in judge_label_map.items()}


def _build_anchor_label_block(
    facet_number: int,
    facet_name: str,
    element_ids: list[str],
    labels: dict[object, str],
    anchors: dict[str, float],
    anchor_key_by_label: bool = False,
) -> str:
    """Build one FACETS label block with anchored measures."""

    label_lines = [f"{facet_number},{facet_name},A"]
    for element_id in element_ids:
        label = str(labels.get(element_id, labels.get(int(element_id), element_id)))
        anchor_key = label if anchor_key_by_label else element_id
        label_lines.append(f"{element_id}={label},{_format_measure(anchors[str(anchor_key)])}")
    label_lines.append("*")
    return "\n".join(label_lines)


def _build_plain_label_block(
    facet_number: int,
    facet_name: str,
    element_ids: list[str],
    labels: dict[str, str],
) -> str:
    """Build one unanchored FACETS label block."""

    label_lines = [f"{facet_number},{facet_name}"]
    for element_id in element_ids:
        label_lines.append(f"{element_id}={labels[element_id]}")
    label_lines.append("*")
    return "\n".join(label_lines)
