from pathlib import Path

import pandas as pd
import pytest

from mhs_llms.config import FacetsConfig, OrderEffectConfig, load_order_effect_config
from mhs_llms.facets.order_effect import (
    build_order_condition_contrast,
    build_order_effect_facets_frame,
    build_order_effect_facets_spec,
    parse_order_condition_scores,
    run_order_effect_facets,
)
from mhs_llms.facets.order_effect_plot import load_order_shift_comparison
from mhs_llms.schema import ITEM_NAMES


def test_load_order_effect_config_resolves_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "order_effect.yaml"
    config_path.write_text(
        """
annotations:
  original_paths:
    - data/original.csv
  reverse_paths:
    - data/reverse.csv
anchors:
  comment_scores_path: facets/human_baseline/human_facets_scores.1.txt
  item_scores_path: facets/human_baseline/human_facets_scores.3.txt
output:
  facets_run_dir: facets/order_effect
facets:
  title: Order Effect
""".strip()
    )

    config = load_order_effect_config(config_path)

    assert config.original_annotation_paths == ((Path.cwd() / "data" / "original.csv").resolve(),)
    assert config.reverse_annotation_paths == ((Path.cwd() / "data" / "reverse.csv").resolve(),)
    assert config.facets_run_dir == (Path.cwd() / "facets" / "order_effect").resolve()
    assert config.facets.model == "?, ?, #, ?, R"
    assert config.original_order_label == "original_order"
    assert config.reverse_order_label == "reverse_order"


def test_run_order_effect_facets_writes_four_facet_outputs_with_shared_judges(tmp_path: Path) -> None:
    original_path = tmp_path / "original.csv"
    reverse_path = tmp_path / "reverse.csv"
    _annotation_frame(judge_id="model_a").to_csv(original_path, index=False)
    _annotation_frame(judge_id="model_a_reverse_order").to_csv(reverse_path, index=False)
    config_path = _write_order_config(tmp_path, original_path, reverse_path)

    outputs = run_order_effect_facets(config_path)

    assert outputs.facets_data_path.exists()
    assert outputs.facets_spec_path.exists()
    assert outputs.facets_data_path.read_text().splitlines()[0].split("\t")[3] == "1"
    assert outputs.facets_data_path.read_text().splitlines()[1].split("\t")[3] == "2"

    spec_text = outputs.facets_spec_path.read_text()
    assert "Facets = 4" in spec_text
    assert "Model = ?, ?, #, ?, R" in spec_text
    assert "4,Order\n1=original_order\n2=reverse_order\n*" in spec_text
    assert spec_text.count("model_a") == 1


def test_build_order_effect_facets_frame_includes_order_id() -> None:
    row = {"comment_id": 1, "judge_id": "20001", "order_id": "2"}
    for item_name in ITEM_NAMES:
        row[item_name] = 1

    facets = build_order_effect_facets_frame(pd.DataFrame([row]))

    assert facets.columns.tolist()[:4] == ["comment_id", "judge_id", "item_id", "order_id"]
    assert facets.loc[0, "item_id"] == "1-10"
    assert facets.loc[0, "order_id"] == "2"


def test_build_order_effect_facets_spec_labels_order_condition() -> None:
    facets_frame = pd.DataFrame(
        [{"comment_id": 20001, "judge_id": "20002", "item_id": "1-10", "order_id": "1"}]
    )
    judge_mapping = pd.DataFrame([{"judge_id": "20002", "external_judge_id": "model_a"}])
    config = OrderEffectConfig(
        original_annotation_paths=(Path("original.csv"),),
        reverse_annotation_paths=(Path("reverse.csv"),),
        comment_scores_path=Path("comments.txt"),
        item_scores_path=Path("items.txt"),
        facets_run_dir=Path("facets"),
        facets_data_filename="data.tsv",
        facets_spec_filename="spec.txt",
        facets_score_filename="scores.txt",
        facets_output_filename="output.txt",
        original_order_label="original_order",
        reverse_order_label="reverse_order",
        facets=FacetsConfig(title="Order Effect", model="?, ?, #, ?, R", delements=("1N", "2N", "3N", "4N")),
    )

    spec = build_order_effect_facets_spec(
        facets_frame=facets_frame,
        config=config,
        judge_mapping=judge_mapping,
        comment_anchors={"20001": 0.0},
        item_anchors={item_name: 0.0 for item_name in ITEM_NAMES},
    )

    assert "1,Comments,A" in spec
    assert "3,Items,A" in spec
    assert "4,Order\n1=original_order\n2=reverse_order\n*" in spec


def test_parse_order_condition_scores_and_contrast(tmp_path: Path) -> None:
    config_path = _write_order_config(tmp_path, tmp_path / "original.csv", tmp_path / "reverse.csv")
    config = load_order_effect_config(config_path)
    score_path = config.facets_run_dir / "order_scores.4.txt"
    config.facets_run_dir.mkdir(parents=True)
    _write_order_score_file(score_path)

    order_scores = parse_order_condition_scores(config)
    contrast = build_order_condition_contrast(order_scores)

    assert order_scores["order_condition"].tolist() == ["original_order", "reverse_order"]
    assert contrast.loc[0, "order_measure_delta"] == pytest.approx(0.17)


def test_load_order_shift_comparison_requires_same_model_ids(tmp_path: Path) -> None:
    original_path = tmp_path / "original_judges.csv"
    reverse_path = tmp_path / "reverse_judges.csv"
    pd.DataFrame([{"facet_label": "model_a", "measure": 0.1, "s_e": 0.02}]).to_csv(original_path, index=False)
    pd.DataFrame([{"facet_label": "model_b", "measure": 0.2, "s_e": 0.03}]).to_csv(reverse_path, index=False)

    with pytest.raises(ValueError, match="same model ids"):
        load_order_shift_comparison(original_path, reverse_path)


def _annotation_frame(judge_id: str) -> pd.DataFrame:
    """Build one minimal processed LLM annotation row using prompt letters."""

    row = {
        "comment_id": 20001,
        "judge_id": judge_id,
    }
    for item_name in ITEM_NAMES:
        row[item_name] = "B"
    row["hate_speech"] = "A"
    return pd.DataFrame([row])


def _write_order_config(tmp_path: Path, original_path: Path, reverse_path: Path) -> Path:
    """Write a temporary pooled order-effect config for tests."""

    config_path = tmp_path / "order_effect.yaml"
    config_path.write_text(
        f"""
annotations:
  original_paths:
    - {original_path}
  reverse_paths:
    - {reverse_path}

anchors:
  comment_scores_path: {Path("facets/human_baseline/human_facets_scores.1.txt").resolve()}
  item_scores_path: {Path("facets/human_baseline/human_facets_scores.3.txt").resolve()}

output:
  facets_run_dir: {tmp_path / "facets_run"}
  facets_data_filename: order_data.tsv
  facets_spec_filename: order_spec.txt
  facets_score_filename: order_scores.txt
  facets_output_filename: order_output.txt

facets:
  title: Order Effect Test
  model: "?, ?, #, ?, R"
  noncenter: 2
  positive: 1
  arrange: N
  subset_detection: No
  delements:
    - 1N
    - 2N
    - 3N
    - 4N
  csv: Tab
""".strip()
    )
    return config_path


def _write_order_score_file(path: Path) -> None:
    """Write a minimal FACETS score file for the order facet."""

    path.write_text(
        "\n".join(
            [
                "4\tOrder",
                "T.Score\tT.Count\tObs.Avge\tFairMAvge\tMeasure\tS.E.\tInfitMS\tInfitZ\tOutfitMS\tOutfitZ\tPtMea\tPtMeExp\tDiscrim\tDisplace\tStatus\tGroup\tWeight\tSign\tInfChiSqu\tInfitdf\tInfPr2s\tOutfChiSqu\tOutfitdf\tOutfPr2s\t4\tOrder\tF-Number\tF-Label",
                "10.00\t20.00\t.50\t.50\t-.08\t.03\t1.00\t.00\t1.00\t.00\t.00\t.00\t1.00\t.00\t0\t0\t1.00\t1\t1.00\t1.00\t.5000\t1.00\t1.00\t.5000\t1\toriginal_order\t4\tOrder",
                "12.00\t20.00\t.60\t.60\t.09\t.04\t1.00\t.00\t1.00\t.00\t.00\t.00\t1.00\t.00\t0\t0\t1.00\t1\t1.00\t1.00\t.5000\t1.00\t1.00\t.5000\t2\treverse_order\t4\tOrder",
            ]
        )
    )
