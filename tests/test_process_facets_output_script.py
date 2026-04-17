import subprocess
import sys
from pathlib import Path


def test_process_facets_output_requires_config_only() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/process_facets_output.py",
            "--help",
        ],
        capture_output=True,
        check=True,
        text=True,
    )

    assert "Path to a FACETS YAML config" in result.stdout
    assert "--facets-dir" not in result.stdout
    assert "--output-dir" not in result.stdout


def test_process_facets_output_uses_config_facets_run_dir(tmp_path: Path) -> None:
    facets_dir = tmp_path / "facets_run"
    facets_dir.mkdir()
    (facets_dir / "sample_scores.1.txt").write_text(
        "\n".join(
            [
                "1\tComments",
                "T.Score\tMeasure\t1\tComments\tF-Number\tF-Label\t",
                "57.00\t.45\t1\t1\t1\tComments\t",
            ]
        )
    )
    (facets_dir / "sample_output.txt").write_text(
        "\n".join(
            [
                "Title = Example Run",
                "Total data lines = 1",
                "Valid responses used for estimation = 10",
            ]
        )
    )
    config_path = tmp_path / "facets.yaml"
    config_path.write_text(
        f"""
annotations:
  paths:
    - data/reference_set.csv
anchors:
  comment_scores_path: facets/human_baseline/human_facets_scores.1.txt
  item_scores_path: facets/human_baseline/human_facets_scores.3.txt
output:
  facets_run_dir: {facets_dir}
facets:
  title: Example Run
""".strip()
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/process_facets_output.py",
            str(config_path),
        ],
        capture_output=True,
        check=True,
        text=True,
    )

    assert f"facets_dir={facets_dir}" in result.stdout
    assert (facets_dir / "combined_scores.csv").exists()
    assert (facets_dir / "comments_scores.csv").exists()
    assert (facets_dir / "run_summary.json").exists()
