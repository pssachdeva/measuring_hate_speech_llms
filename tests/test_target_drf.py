import pandas as pd

from mhs_llms.config import FacetsConfig, TargetDRFConfig
from mhs_llms.facets.target_drf import (
    build_target_drf_facets_spec,
    filter_target_labels_to_annotations,
    parse_target_bias_terms,
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


def test_build_target_drf_facets_spec_can_anchor_targets(tmp_path) -> None:
    facets_frame = pd.DataFrame(
        [
            {
                "comment_id": 1,
                "judge_id": 20002,
                "item_id": "1-10",
                "target_id": "1",
                "sentiment": 1,
            }
        ]
    )
    judge_mapping = pd.DataFrame(
        [{"judge_id": 20002, "external_judge_id": "openai_gpt-5.4_medium"}]
    )
    target_labels = pd.DataFrame(
        [{"comment_id": 1, "target_id": "1", "target_identity": "target_race_black"}]
    )
    config = TargetDRFConfig(
        annotation_paths=(tmp_path / "annotations.csv",),
        dataset_name="dataset",
        split="train",
        min_annotators=4,
        agreement_threshold=0.75,
        min_comments_per_target=1,
        anchor_targets=True,
        collapse_targets={},
        exclude_targets=(),
        comment_scores_path=tmp_path / "comments.txt",
        item_scores_path=tmp_path / "items.txt",
        judge_scores_path=tmp_path / "judges.txt",
        facets_run_dir=tmp_path,
        target_labels_path=tmp_path / "labels.csv",
        facets_data_filename="data.tsv",
        facets_spec_filename="spec.txt",
        facets_score_filename="scores.txt",
        facets_output_filename="output.txt",
        facets=FacetsConfig(
            title="Target DRF",
            model="?, ?B, #, ?B, R",
            delements=("1N", "2N", "3N", "4N"),
            bias="Difficulty",
        ),
    )

    spec = build_target_drf_facets_spec(
        facets_frame=facets_frame,
        config=config,
        judge_mapping=judge_mapping,
        target_labels=target_labels,
        comment_anchors={"1": 0.0},
        item_anchors={
            item_name: 0.0
            for item_name in [
                "sentiment",
                "respect",
                "insult",
                "humiliate",
                "status",
                "dehumanize",
                "violence",
                "genocide",
                "attack_defend",
                "hate_speech",
            ]
        },
        judge_anchors={"openai_gpt-5.4_medium": 0.0},
    )

    assert "4,Targets,A\n1=target_race_black,0\n*" in spec


def test_parse_target_bias_terms_reads_table_13_rows(tmp_path) -> None:
    report_path = tmp_path / "facets_output.txt"
    report_path.write_text(
        "\n".join(
            [
                "Table 13.1.1  Bias/Interaction Report (arranged by N).",
                "",
                "Bias/Interaction: 2. Judges, 4. Targets (lower score = higher bias measure)",
                "|Observd  Expctd  Observd  Obs-Exp| Bias-  Model                    |Infit Outfit|    Judges                                            Targets                              |",
                "|  Score   Score    Count  Average|  Size   S.E.     t   d.f. Prob. | MnSq  MnSq | Sq Num   Judges                               measr- Nu Targets                    measr- |",
                "|---------------------------------+---------------------------------+------------+-------------------------------------------------------------------------------------------|",
                "| 6367    6069.25  5800        .05|   -.21   .03  -6.12  5799 .0000 |  1.1    .8 |  6 20002 openai_gpt-5.4_medium                -.29  6 target_race_black             .00 |",
                "Table 14.1.1.4  Bias/Interaction Pairwise Report (arranged by N).",
            ]
        )
    )
    config = TargetDRFConfig(
        annotation_paths=(tmp_path / "annotations.csv",),
        dataset_name="dataset",
        split="train",
        min_annotators=4,
        agreement_threshold=0.75,
        min_comments_per_target=1,
        anchor_targets=True,
        collapse_targets={},
        exclude_targets=(),
        comment_scores_path=tmp_path / "comments.txt",
        item_scores_path=tmp_path / "items.txt",
        judge_scores_path=tmp_path / "judges.txt",
        facets_run_dir=tmp_path,
        target_labels_path=tmp_path / "labels.csv",
        facets_data_filename="data.tsv",
        facets_spec_filename="spec.txt",
        facets_score_filename="scores.txt",
        facets_output_filename="facets_output.txt",
        facets=FacetsConfig(title="Target DRF"),
    )

    terms = parse_target_bias_terms(config)

    assert terms.loc[0, "judge_label"] == "openai_gpt-5.4_medium"
    assert terms.loc[0, "target_identity"] == "target_race_black"
    assert terms.loc[0, "beta_jm"] == -0.21
    assert terms.loc[0, "p_value"] == 0.0
