from pathlib import Path

import pytest

from mhs_llms.config import load_model_batch_config, load_model_batch_configs


def test_load_model_batch_config_resolves_paths_and_params() -> None:
    config = load_model_batch_config(Path("configs/queries/reference_oai_gpt54_low.yaml"))

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


def test_load_model_batch_config_supports_anthropic_shape() -> None:
    config = load_model_batch_config(Path("configs/queries/reference_anthropic_claude-sonnet-4-6.yaml"))

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
