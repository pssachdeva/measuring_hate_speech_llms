from pathlib import Path

from mhs_llms.config import load_model_batch_config


def test_load_model_batch_config_resolves_paths_and_params() -> None:
    config = load_model_batch_config(Path("configs/model_batch_openai_example.yaml"))

    assert config.name == "openai_mhs_example"
    assert config.provider.name == "openai"
    assert config.provider.model == "gpt-4.1-mini"
    assert config.prompt.system_prompt_path.name == "mhs_survey_v1.txt"
    assert config.selection.comment_ids_path is not None
    assert config.selection.comment_ids_path.name == "comment_ids.csv"
    assert config.provider.model_params["temperature"] == 0
    assert config.output.run_dir.name == "openai_mhs_example"
