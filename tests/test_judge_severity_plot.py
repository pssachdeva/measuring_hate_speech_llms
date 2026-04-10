import pandas as pd
import pytest

from mhs_llms.facets.judge_severity_plot import (
    load_reference_anthropic_judge_severities,
    load_reference_anthropic_opus_judge_severities,
    load_reference_reasoning_severities,
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


def test_load_reference_anthropic_judge_severities_keeps_requested_order(tmp_path) -> None:
    score_path = tmp_path / "judges_scores.csv"
    pd.DataFrame(
        [
            {"facet_label": "anthropic_claude-sonnet-4-6_high", "measure": -0.39, "s_e": 0.09},
            {"facet_label": "anthropic_claude-sonnet-4", "measure": -0.58, "s_e": 0.09},
            {"facet_label": "anthropic_claude-sonnet-4-5", "measure": -0.44, "s_e": 0.09},
            {"facet_label": "anthropic_claude-sonnet-4-6_medium", "measure": -0.38, "s_e": 0.09},
        ]
    ).to_csv(score_path, index=False)

    severity_frame = load_reference_anthropic_judge_severities(score_path)

    assert severity_frame["facet_label"].tolist() == [
        "anthropic_claude-sonnet-4",
        "anthropic_claude-sonnet-4-5",
        "anthropic_claude-sonnet-4-6_medium",
        "anthropic_claude-sonnet-4-6_high",
    ]
    assert severity_frame["display_label"].tolist() == [
        "claude sonnet 4",
        "claude sonnet 4.5",
        "claude sonnet 4.6 (medium)",
        "claude sonnet 4.6 (high)",
    ]


def test_load_reference_anthropic_opus_judge_severities_keeps_requested_order(tmp_path) -> None:
    score_path = tmp_path / "judges_scores.csv"
    pd.DataFrame(
        [
            {"facet_label": "anthropic_claude-opus-4-6_medium", "measure": -0.48, "s_e": 0.09},
            {"facet_label": "anthropic_claude-opus-4", "measure": -0.39, "s_e": 0.09},
            {"facet_label": "anthropic_claude-opus-4-1", "measure": -0.47, "s_e": 0.09},
            {"facet_label": "anthropic_claude-opus-4-5_medium", "measure": -0.51, "s_e": 0.09},
        ]
    ).to_csv(score_path, index=False)

    severity_frame = load_reference_anthropic_opus_judge_severities(score_path)

    assert severity_frame["facet_label"].tolist() == [
        "anthropic_claude-opus-4",
        "anthropic_claude-opus-4-1",
        "anthropic_claude-opus-4-5_medium",
        "anthropic_claude-opus-4-6_medium",
    ]
    assert severity_frame["display_label"].tolist() == [
        "opus 4",
        "opus 4.1",
        "opus 4.5 (medium)",
        "opus 4.6 (medium)",
    ]


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


def test_load_reference_reasoning_severities_includes_anthropic_curves(tmp_path) -> None:
    openai_score_path = tmp_path / "openai_judges_scores.csv"
    anthropic_score_path = tmp_path / "anthropic_judges_scores.csv"
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
    ).to_csv(openai_score_path, index=False)
    pd.DataFrame(
        [
            {"facet_label": "anthropic_claude-opus-4-5_low", "measure": -0.48, "s_e": 0.09},
            {"facet_label": "anthropic_claude-opus-4-5_medium", "measure": -0.51, "s_e": 0.09},
            {"facet_label": "anthropic_claude-opus-4-5_high", "measure": -0.38, "s_e": 0.09},
            {"facet_label": "anthropic_claude-opus-4-6_low", "measure": -0.45, "s_e": 0.09},
            {"facet_label": "anthropic_claude-opus-4-6_medium", "measure": -0.48, "s_e": 0.09},
            {"facet_label": "anthropic_claude-opus-4-6_high", "measure": -0.37, "s_e": 0.09},
            {"facet_label": "anthropic_claude-sonnet-4-6_low", "measure": -0.48, "s_e": 0.09},
            {"facet_label": "anthropic_claude-sonnet-4-6_medium", "measure": -0.38, "s_e": 0.09},
            {"facet_label": "anthropic_claude-sonnet-4-6_high", "measure": -0.39, "s_e": 0.09},
        ]
    ).to_csv(anthropic_score_path, index=False)

    severity_frame = load_reference_reasoning_severities(openai_score_path, anthropic_score_path)

    assert severity_frame["family_label"].drop_duplicates().tolist() == [
        "gpt-5.2",
        "gpt-5.4",
        "gpt-5.4 mini",
        "gpt-5.4 nano",
        "Claude Opus 4.5",
        "Claude Opus 4.6",
        "Claude Sonnet 4.6",
    ]
    assert severity_frame.loc[
        severity_frame["family_name"] == "anthropic_claude-opus-4-5",
        "reasoning_level",
    ].tolist() == ["low", "medium", "high"]
    assert severity_frame.loc[
        severity_frame["family_name"] == "anthropic_claude-opus-4-5",
        "provider_label",
    ].drop_duplicates().tolist() == ["Anthropic"]
