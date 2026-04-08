import pandas as pd
import pytest

from mhs_llms.facets.judge_severity_plot import (
    load_reference_openai_judge_severities,
    load_reference_openai_reasoning_severities,
)


def test_load_reference_openai_judge_severities_keeps_requested_order(tmp_path) -> None:
    score_path = tmp_path / "judges_scores.csv"
    pd.DataFrame(
        [
            {"facet_label": "openai_gpt-5.2_medium", "measure": -0.19, "s_e": 0.09},
            {"facet_label": "openai_gpt-4.1", "measure": -0.88, "s_e": 0.09},
            {"facet_label": "openai_gpt-5.4_medium", "measure": -0.19, "s_e": 0.09},
            {"facet_label": "openai_gpt-4o", "measure": -0.94, "s_e": 0.09},
            {"facet_label": "openai_gpt-5.4_low", "measure": -0.23, "s_e": 0.09},
        ]
    ).to_csv(score_path, index=False)

    severity_frame = load_reference_openai_judge_severities(score_path)

    assert severity_frame["facet_label"].tolist() == [
        "openai_gpt-4o",
        "openai_gpt-4.1",
        "openai_gpt-5.2_medium",
        "openai_gpt-5.4_medium",
    ]
    assert severity_frame["display_label"].tolist() == [
        "gpt-4o",
        "gpt-4.1",
        "5.2 medium",
        "5.4 medium",
    ]


def test_load_reference_openai_judge_severities_requires_all_models(tmp_path) -> None:
    score_path = tmp_path / "judges_scores.csv"
    pd.DataFrame(
        [
            {"facet_label": "openai_gpt-4o", "measure": -0.94, "s_e": 0.09},
            {"facet_label": "openai_gpt-4.1", "measure": -0.88, "s_e": 0.09},
        ]
    ).to_csv(score_path, index=False)

    with pytest.raises(ValueError, match="missing requested models"):
        load_reference_openai_judge_severities(score_path)


def test_load_reference_openai_reasoning_severities_keeps_family_and_level_order(tmp_path) -> None:
    score_path = tmp_path / "judges_scores.csv"
    pd.DataFrame(
        [
            {"facet_label": "openai_gpt-5.4_xhigh", "measure": -0.29, "s_e": 0.09},
            {"facet_label": "openai_gpt-5.4_medium", "measure": -0.19, "s_e": 0.09},
            {"facet_label": "openai_gpt-5.4-mini_low", "measure": 0.05, "s_e": 0.09},
            {"facet_label": "openai_gpt-5.2_none", "measure": -0.33, "s_e": 0.09},
            {"facet_label": "openai_gpt-5.4-nano_high", "measure": 0.24, "s_e": 0.08},
            {"facet_label": "openai_gpt-5.4_high", "measure": -0.18, "s_e": 0.09},
            {"facet_label": "openai_gpt-5.2_low", "measure": -0.26, "s_e": 0.09},
            {"facet_label": "openai_gpt-5.2_medium", "measure": -0.19, "s_e": 0.09},
            {"facet_label": "openai_gpt-5.2_high", "measure": -0.13, "s_e": 0.09},
            {"facet_label": "openai_gpt-5.2_xhigh", "measure": -0.19, "s_e": 0.09},
            {"facet_label": "openai_gpt-5.4_none", "measure": -0.32, "s_e": 0.09},
            {"facet_label": "openai_gpt-5.4_low", "measure": -0.23, "s_e": 0.09},
            {"facet_label": "openai_gpt-5.4-mini_none", "measure": -0.01, "s_e": 0.09},
            {"facet_label": "openai_gpt-5.4-mini_medium", "measure": -0.05, "s_e": 0.09},
            {"facet_label": "openai_gpt-5.4-mini_high", "measure": 0.05, "s_e": 0.09},
            {"facet_label": "openai_gpt-5.4-mini_xhigh", "measure": 0.09, "s_e": 0.11},
            {"facet_label": "openai_gpt-5.4-nano_none", "measure": -0.13, "s_e": 0.10},
            {"facet_label": "openai_gpt-5.4-nano_low", "measure": 0.16, "s_e": 0.09},
            {"facet_label": "openai_gpt-5.4-nano_medium", "measure": 0.29, "s_e": 0.08},
            {"facet_label": "openai_gpt-5.4-nano_xhigh", "measure": 0.09, "s_e": 0.09},
        ]
    ).to_csv(score_path, index=False)

    severity_frame = load_reference_openai_reasoning_severities(score_path)

    assert severity_frame["family_label"].drop_duplicates().tolist() == [
        "gpt-5.2",
        "gpt-5.4",
        "gpt-5.4 mini",
        "gpt-5.4 nano",
    ]
    assert severity_frame.loc[severity_frame["family_name"] == "openai_gpt-5.2", "reasoning_level"].tolist() == [
        "none",
        "low",
        "medium",
        "high",
        "xhigh",
    ]


def test_load_reference_openai_reasoning_severities_requires_all_requested_rows(tmp_path) -> None:
    score_path = tmp_path / "judges_scores.csv"
    pd.DataFrame(
        [
            {"facet_label": "openai_gpt-5.2_none", "measure": -0.33, "s_e": 0.09},
            {"facet_label": "openai_gpt-5.2_low", "measure": -0.26, "s_e": 0.09},
        ]
    ).to_csv(score_path, index=False)

    with pytest.raises(ValueError, match="missing requested model"):
        load_reference_openai_reasoning_severities(score_path)
