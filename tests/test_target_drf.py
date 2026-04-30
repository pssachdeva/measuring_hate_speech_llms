import pandas as pd

from mhs_llms.facets.target_drf import (
    filter_target_labels_to_annotations,
    parse_target_pairwise_contrasts,
)


def test_filter_target_labels_to_annotations_reapplies_min_target_count() -> None:
    target_labels = pd.DataFrame(
        [
            {
                "comment_id": 1,
                "n_annotators": 4,
                "raw_target": "target_race_black",
                "target_identity": "target_race_black",
                "target_id": "1",
                "target_share": 0.75,
            },
            {
                "comment_id": 2,
                "n_annotators": 4,
                "raw_target": "target_race_black",
                "target_identity": "target_race_black",
                "target_id": "1",
                "target_share": 1.0,
            },
            {
                "comment_id": 3,
                "n_annotators": 4,
                "raw_target": "target_race_white",
                "target_identity": "target_race_white",
                "target_id": "2",
                "target_share": 0.75,
            },
        ]
    )
    annotations = pd.DataFrame(
        [
            {"comment_id": 1, "judge_id": "model"},
            {"comment_id": 2, "judge_id": "model"},
        ]
    )

    filtered = filter_target_labels_to_annotations(
        target_labels=target_labels,
        annotations=annotations,
        min_comments_per_target=2,
    )

    assert filtered["comment_id"].tolist() == [1, 2]
    assert filtered["target_identity"].unique().tolist() == ["target_race_black"]
    assert filtered["target_id"].unique().tolist() == ["1"]


def test_parse_target_pairwise_contrasts_reads_table_14_rows(tmp_path) -> None:
    report_path = tmp_path / "facets_output.txt"
    report_path.write_text(
        "\n".join(
            [
                "Table 14.1.1.4  Bias/Interaction Pairwise Report (arranged by N).",
                "",
                "Bias/Interaction: 2. Judges, 4. Targets",
                "| Target                      | Target-     Obs-Exp Context                       | Target-     Obs-Exp Context                       | Target- Joint    Rasch-Welch   |",
                "| Num   Judges                | Measr  S.E. Average Nu Targets                    | Measr  S.E. Average Nu Targets                    |Contrast  S.E.   t   d.f. Prob. |",
                "|-----------------------------+---------------------------------------------------+---------------------------------------------------+--------------------------------|",
                "| 20002 openai_gpt-5.4_medium |  -.11   .04    -.02  1 target_gender_men          |  -.10   .07    -.02  2 target_gender_transgender  |   -.01   .08  -.18  1782 .8548 |",
                "+----------------------------------------------------------------------------------------------------------------------------------------------------------------------+",
            ]
        )
    )

    contrasts = parse_target_pairwise_contrasts(report_path)

    assert contrasts.shape == (1, 18)
    assert contrasts.loc[0, "judge_label"] == "openai_gpt-5.4_medium"
    assert contrasts.loc[0, "target_a"] == "target_gender_men"
    assert contrasts.loc[0, "target_b"] == "target_gender_transgender"
    assert contrasts.loc[0, "target_contrast"] == -0.01
    assert contrasts.loc[0, "p_value"] == 0.8548
