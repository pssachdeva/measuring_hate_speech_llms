from pathlib import Path

import pandas as pd

from mhs_llms.config import load_llm_facets_config
from mhs_llms.llm_facets import run_anchored_llm_facets
from mhs_llms.schema import ITEM_NAMES


def test_load_llm_facets_config_resolves_paths(tmp_path: Path) -> None:
    config_path = tmp_path / "reference_llm_facets.yaml"
    config_path.write_text(
        "\n".join(
            [
                "annotations:",
                "  paths:",
                "    - data/reference_set.csv",
                "anchors:",
                "  comment_scores_path: facets/human_baseline/human_facets_scores.1.txt",
                "  item_scores_path: facets/human_baseline/human_facets_scores.3.txt",
                "output:",
                "  facets_run_dir: facets/llm_severities",
                "facets:",
                "  title: Reference LLM FACETS",
            ]
        )
    )

    config = load_llm_facets_config(config_path)

    assert config.annotation_paths[0] == (Path.cwd() / "data" / "reference_set.csv").resolve()
    assert config.comment_scores_path == (
        Path.cwd() / "facets" / "human_baseline" / "human_facets_scores.1.txt"
    ).resolve()
    assert config.item_scores_path == (
        Path.cwd() / "facets" / "human_baseline" / "human_facets_scores.3.txt"
    ).resolve()
    assert config.facets_run_dir == (Path.cwd() / "facets" / "llm_severities").resolve()


def test_load_llm_facets_config_resolves_nested_config_paths_from_cwd(tmp_path: Path) -> None:
    config_dir = tmp_path / "nested" / "configs"
    config_dir.mkdir(parents=True)
    config_path = config_dir / "llm_facets.yaml"
    config_path.write_text(
        "\n".join(
            [
                "annotations:",
                "  paths:",
                "    - data/reference_set.csv",
                "anchors:",
                "  comment_scores_path: facets/human_baseline/human_facets_scores.1.txt",
                "  item_scores_path: facets/human_baseline/human_facets_scores.3.txt",
                "output:",
                "  facets_run_dir: facets/llm_severities",
                "facets:",
                "  title: Nested Paths Test",
            ]
        )
    )

    config = load_llm_facets_config(config_path)

    assert config.annotation_paths[0] == (Path.cwd() / "data" / "reference_set.csv").resolve()
    assert config.comment_scores_path == (
        Path.cwd() / "facets" / "human_baseline" / "human_facets_scores.1.txt"
    ).resolve()
    assert config.item_scores_path == (
        Path.cwd() / "facets" / "human_baseline" / "human_facets_scores.3.txt"
    ).resolve()
    assert config.facets_run_dir == (Path.cwd() / "facets" / "llm_severities").resolve()


def test_run_anchored_llm_facets_writes_outputs(tmp_path: Path) -> None:
    annotation_path = tmp_path / "reference_set.csv"
    _annotation_frame(comment_id=20001, judge_id="model_a").to_csv(annotation_path, index=False)
    config_path = tmp_path / "llm_facets.yaml"
    config_path.write_text(
        "\n".join(
            [
                "annotations:",
                "  paths:",
                f"    - {annotation_path}",
                "anchors:",
                f"  comment_scores_path: {Path('facets/human_baseline/human_facets_scores.1.txt').resolve()}",
                f"  item_scores_path: {Path('facets/human_baseline/human_facets_scores.3.txt').resolve()}",
                "output:",
                f"  facets_run_dir: {tmp_path / 'facets_run'}",
                "  facets_data_filename: linked.tsv",
                "  facets_spec_filename: linked.txt",
                "  facets_score_filename: linked_scores.txt",
                "  facets_output_filename: linked_output.txt",
                "facets:",
                "  title: Linked Test",
            ]
        )
    )

    outputs = run_anchored_llm_facets(config_path)

    assert outputs.facets_data_path.exists()
    assert outputs.facets_spec_path.exists()


def test_run_anchored_llm_facets_assigns_new_judge_ids_automatically(tmp_path: Path) -> None:
    annotations = pd.concat(
        [
            _annotation_frame(comment_id=20001, judge_id="model_a"),
            _annotation_frame(comment_id=20002, judge_id="model_b"),
        ],
        ignore_index=True,
    )
    annotations["judge_id"] = ["openai_gpt-5.4_low", "openai_gpt-5.4_medium"]
    annotation_path = tmp_path / "novel_judges.csv"
    annotations.to_csv(annotation_path, index=False)

    config_path = tmp_path / "llm_facets.yaml"
    config_path.write_text(
        "\n".join(
            [
                "annotations:",
                "  paths:",
                f"    - {annotation_path}",
                "anchors:",
                f"  comment_scores_path: {Path('facets/human_baseline/human_facets_scores.1.txt').resolve()}",
                f"  item_scores_path: {Path('facets/human_baseline/human_facets_scores.3.txt').resolve()}",
                "output:",
                f"  facets_run_dir: {tmp_path / 'facets_run'}",
                "  facets_data_filename: linked.tsv",
                "  facets_spec_filename: linked.txt",
                "  facets_score_filename: linked_scores.txt",
                "  facets_output_filename: linked_output.txt",
                "facets:",
                "  title: Linked Test",
            ]
        )
    )

    outputs = run_anchored_llm_facets(config_path)
    spec_text = outputs.facets_spec_path.read_text()

    assert outputs.facets_data_path.exists()
    assert "openai_gpt-5.4_low" in spec_text
    assert "openai_gpt-5.4_medium" in spec_text


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
