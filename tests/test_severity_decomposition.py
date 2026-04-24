from pathlib import Path

import pandas as pd
import pytest

from mhs_llms.facets import parse_bias_interaction_report, run_severity_decomposition_facets
from mhs_llms.schema import ITEM_NAMES


def test_run_severity_decomposition_facets_writes_combined_outputs(tmp_path: Path) -> None:
    first_path = tmp_path / "first.csv"
    second_path = tmp_path / "second.csv"
    _annotation_frame(comment_id=20001, judge_id="model_b").to_csv(first_path, index=False)
    _annotation_frame(comment_id=20002, judge_id="model_a").to_csv(second_path, index=False)

    config_path = _write_config(tmp_path, [first_path, second_path])

    outputs = run_severity_decomposition_facets(config_path)

    assert outputs.facets_data_path.exists()
    assert outputs.facets_spec_path.exists()
    assert len(outputs.facets_data_path.read_text().splitlines()) == 2

    spec_text = outputs.facets_spec_path.read_text()
    assert "Model = ?, ?B, #B, R" in spec_text
    assert "Bias = Difficulty" in spec_text
    assert "Zscore = 0, 0" in spec_text
    assert "1,Comments,A" in spec_text
    assert "2,Judges,A" not in spec_text
    assert "2,Judges" in spec_text
    assert "3,Items,A" in spec_text
    assert "model_a" in spec_text
    assert "model_b" in spec_text


def test_run_severity_decomposition_facets_does_not_require_judge_anchor_path(
    tmp_path: Path,
) -> None:
    annotation_path = tmp_path / "annotations.csv"
    _annotation_frame(comment_id=20001, judge_id="model_a").to_csv(annotation_path, index=False)
    config_path = _write_config(tmp_path, [annotation_path])

    outputs = run_severity_decomposition_facets(config_path)

    assert outputs.facets_spec_path.exists()


def test_run_severity_decomposition_facets_reports_missing_comment_anchors(
    tmp_path: Path,
) -> None:
    annotation_path = tmp_path / "annotations.csv"
    _annotation_frame(comment_id=999999999, judge_id="model_a").to_csv(annotation_path, index=False)
    config_path = _write_config(tmp_path, [annotation_path])

    with pytest.raises(ValueError, match="Missing human comment anchors"):
        run_severity_decomposition_facets(config_path)


def test_parse_bias_interaction_report_reads_table_13_rows(tmp_path: Path) -> None:
    report_path = tmp_path / "facets_output.txt"
    report_path.write_text(
        "\n".join(
            [
                "Table 13.1.1  Bias/Interaction Report (arranged by N).",
                "|Observd  Expctd  Observd  Obs-Exp| Bias-  Model                    |Infit Outfit|     Judges                                                    Items                   |",
                "|  Score   Score    Count  Average|  Size   S.E.     t   d.f. Prob. | MnSq  MnSq | Sq  Num   Judges                                       measr- Nu Items         measr- |",
                "|  237     237.64    70       -.01|    .04   .25    .16    69 .8712 |   .6    .2 |   1 20002 anthropic_claude-haiku-4-5                     -.75  1 sentiment      -2.63 |",
                "|   51      38.74    70        .18|  -1.86   .42  -4.45    69 .0000 |   .5    .2 | 541 20002 anthropic_claude-haiku-4-5                     -.75 10 hate_speech      .84 |",
                "Table 14.1.1.2  Bias/Interaction Pairwise Report (arranged by N).",
            ]
        )
    )

    bias_terms = parse_bias_interaction_report(report_path)

    assert bias_terms.shape == (2, 18)
    assert bias_terms.loc[0, "judge_label"] == "anthropic_claude-haiku-4-5"
    assert bias_terms.loc[0, "item_label"] == "sentiment"
    assert bias_terms.loc[0, "bias_size"] == 0.04
    assert bias_terms.loc[1, "item_label"] == "hate_speech"
    assert bias_terms.loc[1, "bias_size"] == -1.86


def _annotation_frame(comment_id: int, judge_id: str) -> pd.DataFrame:
    """Build one minimal processed LLM annotation row using prompt letters."""

    row = {
        "comment_id": comment_id,
        "judge_id": judge_id,
    }
    for item_name in ITEM_NAMES:
        row[item_name] = "B"
    row["hate_speech"] = "A"
    return pd.DataFrame([row])


def _write_config(tmp_path: Path, annotation_paths: list[Path]) -> Path:
    """Write a temporary severity decomposition config for tests."""

    annotation_lines = "\n".join(f"    - {path}" for path in annotation_paths)
    config_path = tmp_path / "severity_decomposition.yaml"
    config_path.write_text(
        f"""
annotations:
  paths:
{annotation_lines}

anchors:
  comment_scores_path: {Path("facets/human_baseline/human_facets_scores.1.txt").resolve()}
  item_scores_path: {Path("facets/human_baseline/human_facets_scores.3.txt").resolve()}

output:
  facets_run_dir: {tmp_path / "facets_run"}
  facets_data_filename: decomposition.tsv
  facets_spec_filename: decomposition.txt
  facets_score_filename: decomposition_scores.txt
  facets_output_filename: decomposition_output.txt

facets:
  title: Severity Decomposition Test
  model: "?, ?B, #B, R"
  noncenter: 2
  positive: 1
  arrange: N
  subset_detection: No
  delements:
    - 1N
    - 2N
    - 3N
  bias: Difficulty
  zscore: "0, 0"
  csv: Tab
""".strip()
    )
    return config_path
