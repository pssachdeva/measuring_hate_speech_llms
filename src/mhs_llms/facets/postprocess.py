"""Post-processing helpers for FACETS score and output files."""

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from loguru import logger
import pandas as pd


@dataclass(frozen=True)
class FacetsPostprocessOutputs:
    """Paths produced when post-processing one FACETS run directory."""

    output_dir: Path
    combined_scores_path: Path
    summary_path: Path


def _normalize_column_name(column_name: str) -> str:
    """Convert FACETS-style column names into pandas-friendly snake case."""

    normalized = column_name.strip()
    normalized = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", normalized)
    normalized = normalized.replace(".", "_")
    normalized = normalized.replace("-", "_")
    normalized = normalized.lower()
    normalized = re.sub(r"[^a-z0-9_]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def parse_facets_score_file(score_path: Path) -> pd.DataFrame:
    """Parse one FACETS score export into a clean dataframe."""

    raw_lines = score_path.read_text().splitlines()
    if len(raw_lines) < 3:
        raise ValueError(f"FACETS score file is too short: {score_path}")

    facet_number_text, facet_name = raw_lines[0].split("\t", maxsplit=1)
    header = [column_name.strip() for column_name in raw_lines[1].split("\t")]

    # FACETS writes a two-line header, then one tab-delimited row per element.
    dataframe = pd.read_csv(score_path, sep="\t", skiprows=2, header=None, names=header)
    dataframe = dataframe.loc[
        :,
        [column_name for column_name in dataframe.columns if str(column_name).strip()],
    ]
    dataframe["facet_number"] = int(facet_number_text)
    dataframe["facet_name"] = facet_name

    # Promote the trailing FACETS identifier columns to consistent names.
    dataframe = dataframe.rename(
        columns={
            str(facet_number_text): "facet_id",
            facet_name: "facet_label",
            "F-Number": "f_number",
            "F-Label": "f_label",
        }
    )
    dataframe = dataframe.rename(columns={column_name: _normalize_column_name(column_name) for column_name in dataframe.columns})

    # Coerce numeric columns so downstream filtering and sorting are straightforward.
    for column_name in dataframe.columns:
        if column_name in {"facet_name", "facet_label", "f_label"}:
            continue
        converted = pd.to_numeric(dataframe[column_name], errors="coerce")
        if converted.notna().all():
            dataframe[column_name] = converted
    return dataframe


def load_measure_anchors(
    score_path: Path,
    key_column: str,
    measure_column: str = "measure",
) -> dict[str, float]:
    """Load a simple anchor mapping from a FACETS score export."""

    dataframe = parse_facets_score_file(score_path)
    return {
        _normalize_anchor_key(row[key_column]): float(row[measure_column])
        for _, row in dataframe[[key_column, measure_column]].iterrows()
    }


def _normalize_anchor_key(value: Any) -> str:
    """Normalize parsed FACETS identifiers so integer ids stay integer-like."""

    if isinstance(value, (int, str)):
        return str(value)
    if pd.notna(value) and float(value).is_integer():
        return str(int(value))
    return str(value)


def extract_facets_run_summary(output_path: Path) -> dict[str, Any]:
    """Extract a small structured summary from the FACETS main output report."""

    text = output_path.read_text()
    summary: dict[str, Any] = {
        "output_path": str(output_path),
        "warnings": re.findall(r"^Warning .*", text, flags=re.MULTILINE),
    }

    patterns = {
        "title": r"^Title = (?P<value>.+)$",
        "data_file": r"^Data file = (?P<value>.+)$",
        "scorefile": r"^Scorefile = (?P<value>.+)$",
        "total_lines_in_data_file": r"^Total lines in data file = (?P<value>\d+)$",
        "total_data_lines": r"^Total data lines = (?P<value>\d+)$",
        "responses_matched": r"^Responses matched to model: .+ = (?P<value>\d+)$",
        "total_non_blank_responses": r"^\s*Total non-blank responses found = (?P<value>\d+)$",
        "valid_responses_used": r"^Valid responses used for estimation = (?P<value>\d+)$",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.MULTILINE)
        if not match:
            continue
        value = match.group("value")
        summary[key] = int(value) if value.isdigit() else value

    iteration_lines = re.findall(r"^\| JMLE.+\|$", text, flags=re.MULTILINE)
    if iteration_lines:
        summary["final_iteration_line"] = iteration_lines[-1]
    return summary


def process_facets_run(facets_dir: Path, output_dir: Path) -> FacetsPostprocessOutputs:
    """Process the score and report files from one FACETS run directory."""

    output_dir.mkdir(parents=True, exist_ok=True)

    score_paths = sorted(facets_dir.glob("*scores.*.txt"))
    if not score_paths:
        raise ValueError(f"No FACETS score files found in {facets_dir}")

    score_frames: list[pd.DataFrame] = []
    for score_path in score_paths:
        logger.info("Parsing FACETS score file {}", score_path)
        score_frame = parse_facets_score_file(score_path)
        score_frames.append(score_frame)

        facet_slug = str(score_frame.loc[0, "facet_name"]).strip().lower()
        facet_output_path = output_dir / f"{facet_slug}_scores.csv"
        score_frame.to_csv(facet_output_path, index=False)

    combined_scores = pd.concat(score_frames, ignore_index=True)
    combined_scores_path = output_dir / "combined_scores.csv"
    combined_scores.to_csv(combined_scores_path, index=False)

    output_paths = list(facets_dir.glob("*_output.txt"))
    if not output_paths:
        raise ValueError(f"No FACETS output report found in {facets_dir}")
    output_path = output_paths[0]

    logger.info("Extracting FACETS run summary from {}", output_path)
    summary = extract_facets_run_summary(output_path)
    summary["facets_dir"] = str(facets_dir)
    summary["score_files"] = [str(path) for path in score_paths]
    summary["facet_counts"] = (
        combined_scores.groupby("facet_name")["facet_id"].count().sort_index().to_dict()
    )

    summary_path = output_dir / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    logger.info(
        "Wrote combined FACETS scores to {} and summary to {}",
        combined_scores_path,
        summary_path,
    )
    return FacetsPostprocessOutputs(
        output_dir=output_dir,
        combined_scores_path=combined_scores_path,
        summary_path=summary_path,
    )
