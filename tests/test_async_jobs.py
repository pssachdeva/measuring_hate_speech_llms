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


def test_launch_async_for_config_retries_invalid_outputs_until_valid(tmp_path: Path, monkeypatch) -> None:
    config = make_config(tmp_path)
    monkeypatch.setattr(
        async_jobs,
        "_load_batch_comments",
        lambda config: [{"comment_id": 1, "text": "first"}],
    )

    responses = iter(
        [
            ({"id": "resp-1"}, "not json"),
            (
                {"id": "resp-2"},
                '{"target_groups":["I"],"sentiment":"C","respect":"C","insult":"A","humiliate":"A",'
                '"status":"C","dehumanize":"A","violence":"A","genocide":"A","attack_defend":"C","hate_speech":"B"}',
            ),
        ]
    )

    monkeypatch.setattr(async_jobs, "_execute_async_request", lambda config, request_payload: next(responses))

    outputs = launch_async_for_config(config)
    paths = async_jobs._async_storage_paths(config)

    saved_response = json.loads((paths.responses_dir / "comment-1.json").read_text())
    request_errors = [json.loads(line) for line in paths.request_errors_path.read_text().splitlines() if line.strip()]

    assert outputs.completed_count == 1
    assert outputs.error_count == 1
    assert saved_response["attempt_number"] == 2
    assert request_errors[0]["attempt_number"] == 1
    assert request_errors[0]["error_type"] == "invalid_json_response"


def test_launch_async_for_config_retries_model_refusals(tmp_path: Path, monkeypatch) -> None:
    config = make_config(tmp_path)
    monkeypatch.setattr(
        async_jobs,
        "_load_batch_comments",
        lambda config: [{"comment_id": 1, "text": "first"}],
    )

    responses = iter(
        [
            (
                {"choices": [{"message": {"refusal": "safety", "content": None}}]},
                "",
            ),
            (
                {"id": "resp-2"},
                '{"target_groups":["I"],"sentiment":"C","respect":"C","insult":"A","humiliate":"A",'
                '"status":"C","dehumanize":"A","violence":"A","genocide":"A","attack_defend":"C","hate_speech":"B"}',
            ),
        ]
    )

    monkeypatch.setattr(async_jobs, "_execute_async_request", lambda config, request_payload: next(responses))

    outputs = launch_async_for_config(config)
    paths = async_jobs._async_storage_paths(config)
    request_errors = [json.loads(line) for line in paths.request_errors_path.read_text().splitlines() if line.strip()]

    assert outputs.completed_count == 1
    assert outputs.error_count == 1
    assert request_errors[0]["error_type"] == "model_refusal"


def test_launch_async_for_config_retries_existing_invalid_saved_response(tmp_path: Path, monkeypatch) -> None:
    config = make_config(tmp_path)
    monkeypatch.setattr(
        async_jobs,
        "_load_batch_comments",
        lambda config: [{"comment_id": 1, "text": "first"}],
    )

    paths = async_jobs._async_storage_paths(config)
    paths.responses_dir.mkdir(parents=True, exist_ok=True)
    async_jobs._write_json(
        paths.responses_dir / "comment-1.json",
        {
            "custom_id": "comment-1",
            "comment_id": 1,
            "text": "first",
            "provider_response": {"id": "stale-invalid"},
            "response_text": "still not json",
        },
    )

    calls: list[int] = []

    def fake_execute(config, request_payload):
        calls.append(1)
        return {"id": "resp-1"}, (
            '{"target_groups":["I"],"sentiment":"C","respect":"C","insult":"A","humiliate":"A",'
            '"status":"C","dehumanize":"A","violence":"A","genocide":"A","attack_defend":"C","hate_speech":"B"}'
        )

    monkeypatch.setattr(async_jobs, "_execute_async_request", fake_execute)

    outputs = launch_async_for_config(config)
    saved_response = json.loads((paths.responses_dir / "comment-1.json").read_text())

    assert outputs.completed_count == 1
    assert outputs.skipped_existing_count == 0
    assert calls == [1]
    assert saved_response["provider_response"]["id"] == "resp-1"


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


def test_process_async_for_config_marks_refusals_as_model_refusal(tmp_path: Path, monkeypatch) -> None:
    config = make_config(tmp_path, provider="anthropic", model_name="claude-haiku-4-5-20251001")
    monkeypatch.setattr(
        async_jobs,
        "_load_batch_comments",
        lambda config: [{"comment_id": 1, "text": "first"}],
    )

    paths = async_jobs._async_storage_paths(config)
    paths.responses_dir.mkdir(parents=True, exist_ok=True)
    async_jobs._write_json(
        paths.responses_dir / "comment-1.json",
        {
            "custom_id": "comment-1",
            "comment_id": 1,
            "text": "first",
            "provider_response": {"stop_reason": "end_turn"},
            "response_text": "I can't analyze this content.",
        },
    )

    outputs = process_async_for_config(config=config)

    assert outputs.completed_count == 1
    assert outputs.is_complete is False
    errors = [json.loads(line) for line in paths.processing_errors_path.read_text().splitlines() if line.strip()]
    assert len(errors) == 1
    assert errors[0]["error_type"] == "model_refusal"
    assert errors[0]["provider_stop_reason"] == "end_turn"


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


def test_launch_async_runs_every_model_in_shared_yaml(tmp_path: Path, monkeypatch) -> None:
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
  - id: openai_two
    provider: openai
    name: gpt-5.4-mini
    max_tokens: 128

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
    monkeypatch.setattr(
        async_jobs,
        "_execute_async_request",
        lambda config, request_payload: (
            {"id": f"resp-{config.model.id}"},
            '{"target_groups":["I"],"sentiment":"C","respect":"C","insult":"A","humiliate":"A",'
            '"status":"C","dehumanize":"A","violence":"A","genocide":"A","attack_defend":"C","hate_speech":"B"}',
        ),
    )

    outputs = async_jobs.launch_async(config_path=config_path)

    assert outputs.all_complete is True
    assert [output.model_id for output in outputs.outputs] == ["openai_one", "openai_two"]


def test_launch_async_creates_one_progress_bar_per_model(tmp_path: Path, monkeypatch) -> None:
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
  - id: openai_two
    provider: openai
    name: gpt-5.4-mini
    max_tokens: 128

batches:
  run_dir: {tmp_path / "runs"}
  combined_output_path: {tmp_path / "data" / "combined.csv"}
""".strip()
    )
    (tmp_path / "system.txt").write_text("Return JSON.")

    monkeypatch.setattr(
        async_jobs,
        "_load_batch_comments",
        lambda config: [
            {"comment_id": 1, "text": "first"},
            {"comment_id": 2, "text": "second"},
        ],
    )
    monkeypatch.setattr(
        async_jobs,
        "_execute_async_request",
        lambda config, request_payload: (
            {"id": f"resp-{config.model.id}"},
            '{"target_groups":["I"],"sentiment":"C","respect":"C","insult":"A","humiliate":"A",'
            '"status":"C","dehumanize":"A","violence":"A","genocide":"A","attack_defend":"C","hate_speech":"B"}',
        ),
    )

    progress_events: list[dict[str, object]] = []

    class DummyTqdm:
        def __init__(self, *, total, desc, unit, leave):
            progress_events.append(
                {
                    "event": "create",
                    "total": total,
                    "desc": desc,
                    "unit": unit,
                    "leave": leave,
                }
            )
            self.desc = desc

        def update(self, count):
            progress_events.append({"event": "update", "desc": self.desc, "count": count})

        def close(self):
            progress_events.append({"event": "close", "desc": self.desc})

    monkeypatch.setattr(async_jobs, "tqdm", DummyTqdm)

    outputs = async_jobs.launch_async(config_path=config_path)

    assert outputs.all_complete is True
    assert progress_events == [
        {"event": "create", "total": 2, "desc": "openai_one", "unit": "req", "leave": True},
        {"event": "update", "desc": "openai_one", "count": 1},
        {"event": "update", "desc": "openai_one", "count": 1},
        {"event": "close", "desc": "openai_one"},
        {"event": "create", "total": 2, "desc": "openai_two", "unit": "req", "leave": True},
        {"event": "update", "desc": "openai_two", "count": 1},
        {"event": "update", "desc": "openai_two", "count": 1},
        {"event": "close", "desc": "openai_two"},
    ]


def test_build_async_provider_request_uses_adaptive_thinking_for_claude_46(tmp_path: Path) -> None:
    config = make_config(tmp_path, provider="anthropic", model_name="claude-sonnet-4-6")
    config = ModelBatchConfig(
        name=config.name,
        subset=config.subset,
        limit=config.limit,
        prompt=config.prompt,
        model=BatchModelConfig(
            provider="anthropic",
            name="claude-sonnet-4-6",
            id="anthropic_claude-sonnet-4-6",
            max_tokens=4096,
            params={"temperature": 0},
            reasoning=BatchReasoningConfig(effort="medium"),
        ),
        batches=config.batches,
    )

    payload = _build_async_provider_request(
        config=config,
        system_prompt="Return JSON.",
        user_prompt="comment text",
    )

    assert payload == {
        "model": "claude-sonnet-4-6",
        "system": "Return JSON.",
        "messages": [{"role": "user", "content": "comment text"}],
        "max_tokens": 4096,
        "temperature": 0,
        "output_config": {"effort": "medium"},
        "thinking": {"type": "adaptive"},
    }
