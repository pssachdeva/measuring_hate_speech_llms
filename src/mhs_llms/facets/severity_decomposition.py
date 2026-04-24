"""Prepare FACETS inputs for LLM item-dependent severity decomposition."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Mapping

import pandas as pd

from mhs_llms.config import load_severity_decomposition_config
from mhs_llms.facets.anchored import _prepare_llm_annotations
from mhs_llms.facets.facets import build_facets_frame, build_facets_spec
from mhs_llms.facets.facets import write_facets_data, write_facets_spec
from mhs_llms.facets.postprocess import load_measure_anchors
from mhs_llms.schema import ITEM_NAMES


@dataclass(frozen=True)
class SeverityDecompositionOutputs:
    """Paths produced when preparing one severity decomposition FACETS run."""

    facets_data_path: Path
    facets_spec_path: Path


@dataclass(frozen=True)
class SeverityDecompositionPostprocessOutputs:
    """Paths produced when processing severity decomposition FACETS output."""

    bias_terms_path: Path


def run_severity_decomposition_facets(config_path: Path) -> SeverityDecompositionOutputs:
    """Prepare a FACETS run with judge-by-item bias interactions for LLMs."""

    config = load_severity_decomposition_config(config_path)
    annotation_frames = [pd.read_csv(annotation_path) for annotation_path in config.annotation_paths]
    if not annotation_frames:
        raise ValueError("Severity decomposition config must include at least one annotation path")

    annotations = pd.concat(annotation_frames, ignore_index=True)
    prepared_annotations, judge_mapping = _prepare_llm_annotations(annotations=annotations)
    facets_frame = build_facets_frame(prepared_annotations)

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

    facets_run_dir = config.facets_run_dir
    facets_run_dir.mkdir(parents=True, exist_ok=True)

    facets_data_path = facets_run_dir / config.facets_data_filename
    write_facets_data(facets_frame, facets_data_path)

    relevant_comment_anchors = _select_comment_anchors(
        comment_ids=facets_frame["comment_id"].astype(int).astype(str).unique().tolist(),
        anchors=comment_anchors,
    )
    relevant_item_anchors = _select_item_anchors(item_anchors)

    comment_labels = {str(comment_id): str(comment_id) for comment_id in facets_frame["comment_id"].astype(int)}
    judge_labels = dict(
        zip(
            judge_mapping["judge_id"].tolist(),
            judge_mapping["external_judge_id"].tolist(),
            strict=True,
        )
    )

    spec_text = build_facets_spec(
        facets_frame=facets_frame,
        facets_config=config.facets,
        data_filename=config.facets_data_filename,
        score_filename=config.facets_score_filename,
        output_filename=config.facets_output_filename,
        comment_labels=comment_labels,
        judge_labels=judge_labels,
        comment_anchors=relevant_comment_anchors,
        item_anchors=relevant_item_anchors,
    )
    facets_spec_path = facets_run_dir / config.facets_spec_filename
    write_facets_spec(spec_text, facets_spec_path)

    return SeverityDecompositionOutputs(
        facets_data_path=facets_data_path,
        facets_spec_path=facets_spec_path,
    )


def process_severity_decomposition_run(
    config_path: Path,
    output_path: Path | None = None,
) -> SeverityDecompositionPostprocessOutputs:
    """Parse FACETS judge-by-item bias terms and write a tidy CSV dataset."""

    config = load_severity_decomposition_config(config_path)
    report_path = config.facets_run_dir / config.facets_output_filename
    bias_terms = parse_bias_interaction_report(report_path)

    resolved_output_path = output_path
    if resolved_output_path is None:
        resolved_output_path = Path.cwd() / "data" / f"{config.facets_run_dir.name}_bias_terms.csv"
    resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
    bias_terms.to_csv(resolved_output_path, index=False)

    return SeverityDecompositionPostprocessOutputs(bias_terms_path=resolved_output_path)


def parse_bias_interaction_report(report_path: Path) -> pd.DataFrame:
    """Parse FACETS Table 13 judge-by-item bias terms into a dataframe."""

    rows = []
    in_table = False
    for line in report_path.read_text(errors="replace").splitlines():
        if line.startswith("Table 13.") and "Bias/Interaction Report" in line:
            in_table = True
            continue
        if in_table and line.startswith("Table 14."):
            break
        if not in_table or not _looks_like_bias_row(line):
            continue

        rows.append(_parse_bias_row(line))

    if not rows:
        raise ValueError(f"No FACETS Table 13 bias rows found in {report_path}")
    return pd.DataFrame(rows)


def _select_comment_anchors(
    comment_ids: list[str],
    anchors: Mapping[str, float],
) -> dict[str, float]:
    """Return human comment anchors for the comments present in the LLM run."""

    missing = [comment_id for comment_id in comment_ids if comment_id not in anchors]
    if missing:
        missing_list = ", ".join(missing[:10])
        raise ValueError(f"Missing human comment anchors for comment_id values: {missing_list}")
    return {comment_id: anchors[comment_id] for comment_id in comment_ids}


def _select_item_anchors(anchors: Mapping[str, float]) -> dict[str, float]:
    """Return human item anchors in the canonical MHS item order."""

    missing = [item_name for item_name in ITEM_NAMES if item_name not in anchors]
    if missing:
        missing_list = ", ".join(missing)
        raise ValueError(f"Missing human item anchors for item values: {missing_list}")
    return {item_name: anchors[item_name] for item_name in ITEM_NAMES}


def _looks_like_bias_row(line: str) -> bool:
    """Return whether a report line has the fixed-width Table 13 row shape."""

    parts = line.split("|")
    return (
        len(parts) >= 5
        and bool(re.match(r"^\s*\d", parts[1]))
        and bool(re.match(r"^\s*\d+\s+\d+\s+", parts[4]))
    )


def _parse_bias_row(line: str) -> dict[str, object]:
    """Parse one FACETS Table 13 fixed-width row."""

    parts = line.split("|")
    score_values = parts[1].split()
    bias_values = parts[2].split()
    fit_values = parts[3].split()
    identity_values = _parse_bias_identity(parts[4])

    if len(score_values) != 4 or len(bias_values) != 5 or len(fit_values) != 2:
        raise ValueError(f"Unexpected FACETS bias row format: {line}")

    return {
        "observed_score": _to_float(score_values[0]),
        "expected_score": _to_float(score_values[1]),
        "observed_count": int(_to_float(score_values[2])),
        "obs_exp_average": _to_float(score_values[3]),
        "bias_size": _to_float(bias_values[0]),
        "model_se": _to_float(bias_values[1]),
        "t": _to_float(bias_values[2]),
        "df": int(_to_float(bias_values[3])),
        "p_value": _to_float(bias_values[4]),
        "infit_mnsq": _to_float(fit_values[0]),
        "outfit_mnsq": _to_float(fit_values[1]),
        **identity_values,
    }


def _parse_bias_identity(identity_text: str) -> dict[str, object]:
    """Parse the judge and item identity columns from a FACETS bias row."""

    number_pattern = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
    match = re.match(
        rf"^\s*(?P<sequence>\d+)\s+"
        rf"(?P<judge_id>\d+)\s+"
        rf"(?P<judge_label>.+?)\s+"
        rf"(?P<judge_measure>{number_pattern})\s+"
        rf"(?P<item_id>\d+)\s+"
        rf"(?P<item_label>\S+)\s+"
        rf"(?P<item_measure>{number_pattern})\s*$",
        identity_text,
    )
    if match is None:
        raise ValueError(f"Unexpected FACETS bias identity format: {identity_text}")

    return {
        "bias_sequence": int(match.group("sequence")),
        "judge_id": int(match.group("judge_id")),
        "judge_label": match.group("judge_label").strip(),
        "judge_measure": _to_float(match.group("judge_measure")),
        "item_id": int(match.group("item_id")),
        "item_label": match.group("item_label"),
        "item_measure": _to_float(match.group("item_measure")),
    }


def _to_float(value: str) -> float:
    """Convert FACETS compact decimal text to float."""

    if value.startswith("."):
        return float(f"0{value}")
    if value.startswith("-."):
        return float(value.replace("-.", "-0.", 1))
    return float(value)
