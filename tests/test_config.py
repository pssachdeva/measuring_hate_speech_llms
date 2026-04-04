from pathlib import Path

from mhs_llms.config import load_model_batch_config


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

    assert config.name == "reference_anthropic_claude-sonnet-4-6"
    assert config.model.provider == "anthropic"
    assert config.model.name == "claude-sonnet-4-6"
    assert config.prompt.system_prompt_path == (Path.cwd() / "prompts" / "mhs_survey_v1.txt").resolve()
    assert config.subset == "reference_set"
    assert config.limit == 25
    assert config.model.max_tokens == 2000
    assert config.model.reasoning.effort is None
    assert config.model.reasoning.budget_tokens == 1024
    assert config.model.id == "reference_anthropic_claude-sonnet-4-6"
    assert config.batches.run_dir == (
        Path.cwd() / "batches" / "reference_anthropic_claude-sonnet-4-6"
    ).resolve()
