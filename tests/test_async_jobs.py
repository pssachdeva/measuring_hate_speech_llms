import json
from pathlib import Path

import pandas as pd

from mhs_llms import async_jobs
from mhs_llms.async_jobs import (
    _build_async_provider_request,
    _execute_async_request,
    launch_async_for_config,
    process_async,
    process_async_for_config,
)
from mhs_llms.config import (
    BatchModelConfig,
    BatchPromptConfig,
    BatchReasoningConfig,
    BatchStorageConfig,
    ModelBatchConfig,
)


def make_config(tmp_path: Path, provider: str = "openai", model_name: str = "gpt-5.4") -> ModelBatchConfig:
    prompt_path = tmp_path / "system.txt"
    prompt_path.write_text("Return JSON.")
    return ModelBatchConfig(
        name=f"{provider}_{model_name}",
        subset="reference_set",
        limit=None,
        prompt=BatchPromptConfig(
            system_prompt_path=prompt_path,
            user_prompt_template="SOCIAL MEDIA COMMENT:\n{comment_text}",
        ),
        model=BatchModelConfig(
            provider=provider,
            name=model_name,
            id=f"{provider}_{model_name}",
            max_tokens=128,
            params={"temperature": 0},
            reasoning=BatchReasoningConfig(effort="low" if provider == "openai" else None),
        ),
        batches=BatchStorageConfig(
            run_dir=tmp_path / "runs" / f"{provider}_{model_name}",
            request_manifest_filename="request_manifest.jsonl",
            provider_requests_filename="provider_requests.jsonl",
            batch_metadata_filename="batch_job.json",
            raw_results_filename="raw_results.jsonl",
            processed_records_filename="processed_records.jsonl",
            processed_csv_filename="processed_records.csv",
            errors_filename="processing_errors.jsonl",
            combined_output_path=tmp_path / "data" / "combined.csv",
        ),
    )


def test_launch_async_for_config_skips_existing_saved_queries(tmp_path: Path, monkeypatch) -> None:
    config = make_config(tmp_path)
    monkeypatch.setattr(
        async_jobs,
        "_load_batch_comments",
        lambda config: [
            {"comment_id": 1, "text": "first"},
            {"comment_id": 2, "text": "second"},
        ],
    )

    calls: list[str] = []

    def fake_execute(config, request_payload):
        calls.append(request_payload["messages"][1]["content"])
        return {"id": f"resp-{len(calls)}"}, (
            '{"target_groups":["I"],"sentiment":"C","respect":"C","insult":"A","humiliate":"A",'
            '"status":"C","dehumanize":"A","violence":"A","genocide":"A","attack_defend":"C","hate_speech":"B"}'
        )

    monkeypatch.setattr(async_jobs, "_execute_async_request", fake_execute)

    first = launch_async_for_config(config)
    second = launch_async_for_config(config)

    assert first.completed_count == 2
    assert first.skipped_existing_count == 0
    assert second.completed_count == 2
    assert second.skipped_existing_count == 2
    assert calls == ["SOCIAL MEDIA COMMENT:\nfirst", "SOCIAL MEDIA COMMENT:\nsecond"]


def test_execute_async_request_uses_openrouter_openai_client(tmp_path: Path, monkeypatch) -> None:
    config = make_config(tmp_path, provider="openrouter", model_name="qwen/qwen3-max")
    config = ModelBatchConfig(
        name=config.name,
        subset=config.subset,
        limit=config.limit,
        prompt=config.prompt,
        model=BatchModelConfig(
            provider="openrouter",
            name="qwen/qwen3-max",
            id="openrouter_qwen_qwen3-max",
            max_tokens=256,
            params={"temperature": 0},
            reasoning=BatchReasoningConfig(effort="medium", budget_tokens=64),
        ),
        batches=config.batches,
    )
    request_payload = _build_async_provider_request(
        config=config,
        system_prompt="Return JSON.",
        user_prompt="comment text",
    )

    client_kwargs: dict[str, object] = {}
    create_kwargs: dict[str, object] = {}

    class DummyResponse:
        def model_dump(self, mode="json"):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"target_groups":["I"],"sentiment":"C","respect":"C","insult":"A","humiliate":"A","status":"C","dehumanize":"A","violence":"A","genocide":"A","attack_defend":"C","hate_speech":"B"}'
                        }
                    }
                ]
            }

    class DummyCompletions:
        def create(self, **kwargs):
            create_kwargs.update(kwargs)
            return DummyResponse()

    class DummyChat:
        def __init__(self):
            self.completions = DummyCompletions()

    class DummyClient:
        def __init__(self, **kwargs):
            client_kwargs.update(kwargs)
            self.chat = DummyChat()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(async_jobs, "OpenAI", DummyClient)

    _, response_text = _execute_async_request(config=config, request_payload=request_payload)

    assert client_kwargs["base_url"] == async_jobs.OPENROUTER_BASE_URL
    assert client_kwargs["default_headers"] == {"X-Title": async_jobs.OPENROUTER_TITLE}
    assert create_kwargs["extra_body"] == {"reasoning": {"effort": "medium", "max_tokens": 64}}
    assert response_text.startswith('{"target_groups"')


def test_process_async_for_config_writes_partial_outputs(tmp_path: Path, monkeypatch) -> None:
    config = make_config(tmp_path)
    monkeypatch.setattr(
        async_jobs,
        "_load_batch_comments",
        lambda config: [
            {"comment_id": 1, "text": "first"},
            {"comment_id": 2, "text": "second"},
        ],
    )

    paths = async_jobs._async_storage_paths(config)
    paths.responses_dir.mkdir(parents=True, exist_ok=True)
    async_jobs._write_json(
        paths.responses_dir / "comment-1.json",
        {
            "custom_id": "comment-1",
            "comment_id": 1,
            "text": "first",
            "provider_response": {"id": "resp-1"},
            "response_text": '{"target_groups":["I"],"sentiment":"C","respect":"C","insult":"A","humiliate":"A","status":"C","dehumanize":"A","violence":"A","genocide":"A","attack_defend":"C","hate_speech":"B"}',
        },
    )

    outputs = process_async_for_config(config=config)

    dataframe = pd.read_csv(outputs.processed_csv_path)
    assert outputs.completed_count == 1
    assert outputs.total_requests == 2
    assert outputs.is_complete is False
    assert dataframe["comment_id"].tolist() == [1]


def test_process_async_combines_models_into_one_output(tmp_path: Path, monkeypatch) -> None:
    config_path = tmp_path / "multi.yaml"
    config_path.write_text(
        f"""
name: async_multi
subset: reference_set

prompt:
  system_prompt_path: {tmp_path / "system.txt"}

models:
  - id: openai_one
    provider: openai
    name: gpt-5.4
    max_tokens: 128
    reasoning:
      effort: low
  - id: openai_two
    provider: openai
    name: gpt-5.4-mini
    max_tokens: 128
    reasoning:
      effort: low

batches:
  run_dir: {tmp_path / "runs"}
  combined_output_path: {tmp_path / "data" / "combined.csv"}
""".strip()
    )
    (tmp_path / "system.txt").write_text("Return JSON.")

    monkeypatch.setattr(
        async_jobs,
        "_load_batch_comments",
        lambda config: [{"comment_id": 1, "text": "first"}],
    )

    for model_id in ("openai_one", "openai_two"):
        response_dir = tmp_path / "runs" / model_id / async_jobs.ASYNC_RESPONSES_DIRNAME
        response_dir.mkdir(parents=True, exist_ok=True)
        async_jobs._write_json(
            response_dir / "comment-1.json",
            {
                "custom_id": "comment-1",
                "comment_id": 1,
                "text": "first",
                "provider_response": {"id": f"resp-{model_id}"},
                "response_text": '{"target_groups":["I"],"sentiment":"C","respect":"C","insult":"A","humiliate":"A","status":"C","dehumanize":"A","violence":"A","genocide":"A","attack_defend":"C","hate_speech":"B"}',
            },
        )

    outputs = process_async(config_path=config_path)

    assert outputs.all_complete is True
    combined = pd.read_csv(outputs.combined_output_path)
    assert combined["judge_id"].tolist() == ["openai_one", "openai_two"]
