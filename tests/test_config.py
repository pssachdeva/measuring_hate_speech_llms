from pathlib import Path

import pytest

from mhs_llms.config import (
    load_model_batch_config,
    load_model_batch_configs,
    load_severity_decomposition_config,
)


def test_load_model_batch_config_resolves_paths_and_params(tmp_path: Path) -> None:
    config_path = tmp_path / "openai_single.yaml"
    config_path.write_text(
        """
name: openai_gpt-5.4_low
subset: reference_set

prompt:
  system_prompt_path: prompts/mhs_survey_v1.txt

model:
  id: openai_gpt-5.4_low
  provider: openai
  name: gpt-5.4
  max_tokens: 1000
  reasoning:
    effort: low

batches:
  run_dir: batches/openai_gpt-5.4_low
""".strip()
    )

    config = load_model_batch_config(config_path)

    assert config.name == "openai_gpt-5.4_low"
    assert config.model.provider == "openai"
    assert config.model.name == "gpt-5.4"
    assert config.prompt.system_prompt_path == (Path.cwd() / "prompts" / "mhs_survey_v1.txt").resolve()
    assert config.subset == "reference_set"
    assert config.limit is None
    assert config.model.max_tokens == 1000
    assert config.model.reasoning.effort == "low"
    assert config.model.reasoning.budget_tokens is None
    assert config.model.id == "openai_gpt-5.4_low"
    assert config.batches.run_dir == (Path.cwd() / "batches" / "openai_gpt-5.4_low").resolve()


def test_load_model_batch_config_supports_anthropic_shape(tmp_path: Path) -> None:
    config_path = tmp_path / "anthropic_single.yaml"
    config_path.write_text(
        """
name: anthropic_claude-sonnet-4-6_reference
subset: reference_set

prompt:
  system_prompt_path: prompts/mhs_survey_v1.txt

model:
  id: anthropic_claude-sonnet-4-6_reference
  provider: anthropic
  name: claude-sonnet-4-6
  max_tokens: 4096
  reasoning:
    effort: medium

batches:
  run_dir: batches/anthropic_claude-sonnet-4-6_reference
""".strip()
    )

    config = load_model_batch_config(config_path)

    assert config.name == "anthropic_claude-sonnet-4-6_reference"
    assert config.model.provider == "anthropic"
    assert config.model.name == "claude-sonnet-4-6"
    assert config.prompt.system_prompt_path == (Path.cwd() / "prompts" / "mhs_survey_v1.txt").resolve()
    assert config.subset == "reference_set"
    assert config.limit is None
    assert config.model.max_tokens == 4096
    assert config.model.reasoning.effort == "medium"
    assert config.model.reasoning.budget_tokens is None
    assert config.model.id == "anthropic_claude-sonnet-4-6_reference"
    assert config.batches.run_dir == (
        Path.cwd() / "batches" / "anthropic_claude-sonnet-4-6_reference"
    ).resolve()


def test_load_model_batch_configs_resolves_model_list_into_per_model_run_dirs(tmp_path: Path) -> None:
    config_path = tmp_path / "multi_model.yaml"
    config_path.write_text(
        """
name: reference_multi_model
subset: reference_set

prompt:
  system_prompt_path: prompts/mhs_survey_v1.txt

models:
  - id: openai_gpt-5.4_low
    provider: openai
    name: gpt-5.4
    max_tokens: 1000
    reasoning:
      effort: low
  - id: anthropic_claude-sonnet-4-6_thinking
    provider: anthropic
    name: claude-sonnet-4-6
    max_tokens: 2000
    reasoning:
      budget_tokens: 1024

batches:
  run_dir: batches/reference_multi_model
  combined_output_path: data/reference_multi_model_processed.csv
""".strip()
    )

    configs = load_model_batch_configs(config_path)

    assert len(configs) == 2
    assert configs[0].name == "openai_gpt-5.4_low"
    assert configs[0].model.provider == "openai"
    assert configs[0].model.reasoning.effort == "low"
    assert configs[0].batches.run_dir == (
        Path.cwd() / "batches" / "reference_multi_model" / "openai_gpt-5.4_low"
    ).resolve()
    assert configs[0].batches.combined_output_path == (
        Path.cwd() / "data" / "reference_multi_model_processed.csv"
    ).resolve()
    assert configs[1].name == "anthropic_claude-sonnet-4-6_thinking"
    assert configs[1].model.reasoning.budget_tokens == 1024
    assert configs[1].batches.run_dir == (
        Path.cwd() / "batches" / "reference_multi_model" / "anthropic_claude-sonnet-4-6_thinking"
    ).resolve()


def test_load_model_batch_config_rejects_multi_model_config(tmp_path: Path) -> None:
    config_path = tmp_path / "multi_model.yaml"
    config_path.write_text(
        """
name: reference_multi_model
prompt:
  system_prompt_path: prompts/mhs_survey_v1.txt
models:
  - id: one
    provider: openai
    name: gpt-5.4
  - id: two
    provider: openai
    name: gpt-5.4
batches:
  run_dir: batches/reference_multi_model
""".strip()
    )

    with pytest.raises(ValueError, match="more than one model"):
        load_model_batch_config(config_path)


def test_load_model_batch_configs_supports_checked_in_openrouter_smoke_config() -> None:
    configs = load_model_batch_configs(Path("configs/tests/test_openrouter_gemma4_limit1.yaml"))

    assert len(configs) == 1
    config = configs[0]
    assert config.name == "openrouter_google_gemma-4-26b-a4b-it_limit1"
    assert config.model.provider == "openrouter"
    assert config.model.name == "google/gemma-4-26b-a4b-it"
    assert config.model.max_tokens == 512
    assert config.model.params == {"temperature": 0}
    assert config.prompt.system_prompt_path == (Path.cwd() / "prompts" / "mhs_survey_v1.txt").resolve()
    assert config.limit == 10
    assert config.async_retries.max_attempts == 3
    assert config.async_retries.retry_delay_seconds == 0.0
    assert config.batches.run_dir == (
        Path.cwd() / "batches" / "openrouter_gemma4_limit1" / config.model.id
    ).resolve()
    assert config.batches.combined_output_path == (
        Path.cwd() / "data" / "openrouter_gemma4_limit1_processed.csv"
    ).resolve()


def test_load_model_batch_configs_parses_shared_async_retry_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "multi_with_async.yaml"
    config_path.write_text(
        """
name: multi_with_async
prompt:
  system_prompt_path: prompts/mhs_survey_v1.txt

async:
  max_attempts: 5
  retry_delay_seconds: 2.5

models:
  - id: one
    provider: openai
    name: gpt-5.4
  - id: two
    provider: openai
    name: gpt-5.4-mini

batches:
  run_dir: batches/multi_with_async
""".strip()
    )

    configs = load_model_batch_configs(config_path)

    assert len(configs) == 2
    assert all(config.async_retries.max_attempts == 5 for config in configs)
    assert all(config.async_retries.retry_delay_seconds == 2.5 for config in configs)


def test_load_severity_decomposition_config_resolves_paths() -> None:
    config = load_severity_decomposition_config(
        Path("configs/reference_set_all/facets_severity_decomposition.yaml")
    )

    assert config.annotation_paths[0] == (
        Path.cwd() / "data" / "reference_set_openai_processed.csv"
    ).resolve()
    assert config.comment_scores_path == (
        Path.cwd() / "facets" / "human_baseline" / "human_facets_scores.1.txt"
    ).resolve()
    assert config.item_scores_path == (
        Path.cwd() / "facets" / "human_baseline" / "human_facets_scores.3.txt"
    ).resolve()
    assert config.facets.model == "?, ?B, #B, R"
    assert config.facets.bias == "Difficulty"
    assert config.facets.zscore == "0, 0"
