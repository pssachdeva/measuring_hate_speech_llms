from pathlib import Path

import pandas as pd

from mhs_llms.facets.model_severity_figure import (
    load_human_judge_severities,
    load_model_judge_severities,
)


def test_load_human_judge_severities_keeps_measure_column(tmp_path: Path) -> None:
    comments_path = tmp_path / "comments_scores.csv"
    pd.DataFrame(
        [
            {"facet_label": "101", "measure": 0.25, "s_e": 0.10},
            {"facet_label": "102", "measure": 0.75, "s_e": 0.12},
        ]
    ).to_csv(comments_path, index=False)

    severity_frame = load_human_judge_severities(comments_path)

    assert severity_frame.columns.tolist() == ["measure"]
    assert severity_frame["measure"].tolist() == [0.25, 0.75]


def test_load_model_judge_severities_labels_and_sorts_descending(tmp_path: Path) -> None:
    openai_path = tmp_path / "openai_judges.csv"
    anthropic_path = tmp_path / "anthropic_judges.csv"

    pd.DataFrame(
        [
            {"facet_label": "openai_gpt-5.4-medium", "measure": -0.70, "s_e": 0.09},
        ]
    ).to_csv(openai_path, index=False)
    pd.DataFrame(
        [
            {"facet_label": "anthropic_claude-sonnet-4-6_medium", "measure": -0.40, "s_e": 0.08},
            {"facet_label": "anthropic_claude-haiku-4-5", "measure": -1.10, "s_e": 0.10},
        ]
    ).to_csv(anthropic_path, index=False)

    severity_frame = load_model_judge_severities([openai_path, anthropic_path])

    assert severity_frame["facet_label"].tolist() == [
        "anthropic_claude-sonnet-4-6_medium",
        "openai_gpt-5.4-medium",
        "anthropic_claude-haiku-4-5",
    ]
    assert severity_frame["provider"].tolist() == ["anthropic", "openai", "anthropic"]
    assert severity_frame["display_label"].tolist() == [
        "Claude Sonnet 4.6 (Medium)",
        "GPT-5.4 Medium",
        "Claude Haiku 4.5",
    ]
