from pathlib import Path

from mhs_llms.config import load_model_batch_config


def test_load_model_batch_config_resolves_paths_and_params() -> None:
    config = load_model_batch_config(Path("configs/exp1_openai.yaml"))

    assert config.name == "exp1_openai"
    assert config.model.provider == "openai"
    assert config.model.name == "gpt-5.4"
    assert config.prompt.system_prompt_path.name == "mhs_survey_v1.txt"
    assert config.subset == "reference_set"
    assert config.limit is None
    assert config.model.max_tokens == 1000
    assert config.model.reasoning.effort == "low"
    assert config.model.reasoning.budget_tokens is None
    assert config.model.id == "openai:gpt-5.4"
    assert config.batches.run_dir.name == "exp1_openai"


def test_load_model_batch_config_supports_anthropic_shape() -> None:
    config = load_model_batch_config(Path("configs/reference_ant_claude46sonnet.yaml"))

    assert config.name == "reference_ant_claude46sonnet"
    assert config.model.provider == "anthropic"
    assert config.model.name == "claude-3-5-sonnet-20241022"
    assert config.prompt.system_prompt_path.name == "mhs_survey_v1.txt"
    assert config.subset == "reference_set"
    assert config.limit is None
    assert config.model.max_tokens == 1000
    assert config.model.reasoning.effort is None
    assert config.model.reasoning.budget_tokens == 512
    assert config.model.id == "anthropic:claude-3-5-sonnet-20241022"
    assert config.batches.run_dir.name == "reference_ant_claude46sonnet"
