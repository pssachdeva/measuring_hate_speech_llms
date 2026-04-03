import json

from mhs_llms.facets import extract_facets_run_summary, parse_facets_score_file, process_facets_run


def test_parse_facets_score_file_reads_header_and_rows(tmp_path) -> None:
    score_path = tmp_path / "sample_scores.1.txt"
    score_path.write_text(
        "\n".join(
            [
                "1\tComments",
                "T.Score\tMeasure\t1\tComments\tF-Number\tF-Label\t",
                "57.00\t.45\t1\t1\t1\tComments\t",
                "38.00\t.20\t2\t2\t1\tComments\t",
            ]
        )
    )

    score_frame = parse_facets_score_file(score_path)

    assert score_frame.loc[0, "facet_number"] == 1
    assert score_frame.loc[0, "facet_name"] == "Comments"
    assert score_frame.loc[0, "facet_id"] == 1
    assert score_frame.loc[0, "facet_label"] == 1
    assert score_frame.loc[0, "measure"] == 0.45
    assert "t_score" in score_frame.columns


def test_extract_facets_run_summary_pulls_key_counts(tmp_path) -> None:
    output_path = tmp_path / "sample_output.txt"
    output_path.write_text(
        "\n".join(
            [
                "Title = MHS Human Baseline",
                "Data file = ../../data/human_baseline/human_facets_data.tsv",
                "Scorefile = human_facets_scores.txt",
                "Total lines in data file = 135557",
                "Total data lines = 135556",
                "Responses matched to model: ?,?,#,R,1 = 1355560",
                "    Total non-blank responses found = 1355560",
                "Valid responses used for estimation = 1355560",
                "| JMLE 152       .4688    .0      .1992     -.0002    .0000 |",
                "Warning (2)! Subset checking bypassed",
            ]
        )
    )

    summary = extract_facets_run_summary(output_path)

    assert summary["title"] == "MHS Human Baseline"
    assert summary["total_data_lines"] == 135556
    assert summary["responses_matched"] == 1355560
    assert summary["warnings"] == ["Warning (2)! Subset checking bypassed"]
    assert "JMLE 152" in summary["final_iteration_line"]


def test_process_facets_run_writes_outputs(tmp_path) -> None:
    facets_dir = tmp_path / "facets"
    facets_dir.mkdir()
    (facets_dir / "sample.scores.1.txt").write_text(
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
    output_dir = tmp_path / "processed"

    outputs = process_facets_run(facets_dir=facets_dir, output_dir=output_dir)

    assert outputs.combined_scores_path.exists()
    assert outputs.summary_path.exists()
    summary = json.loads(outputs.summary_path.read_text())
    assert summary["facet_counts"] == {"Comments": 1}
