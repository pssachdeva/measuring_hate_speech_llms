from pathlib import Path
import json

import pandas as pd
import pytest

from mhs_llms.batch import (
    _build_requests,
    _create_provider_batch,
    _download_provider_results,
    _select_comment_ids,
    _apply_google_batch_reasoning,
    write_combined_processed_annotations,
    write_processed_annotations,
)
from mhs_llms.batch import (
    _extract_anthropic_result,
    _extract_google_result,
    _extract_moonshot_result,
    _extract_openai_result,
    _extract_together_result,
    _extract_xai_result,
    _batch_status,
    _build_processing_error_record,
    _parse_response_json,
    _provider_api_key,
)
from mhs_llms.config import (
    BatchModelConfig,
    BatchPromptConfig,
    BatchReasoningConfig,
    BatchStorageConfig,
    ModelBatchConfig,
)
from mhs_llms.schema import MHSAnnotationRecord, annotation_record_to_row


def test_parse_response_json_handles_markdown_code_fences() -> None:
    payload = _parse_response_json(
        """```json
{"target_groups":["I"],"sentiment":"C","respect":"C","insult":"A","humiliate":"A","status":"C","dehumanize":"A","violence":"A","genocide":"A","attack_defend":"C","hate_speech":"B"}
```"""
    )

    assert payload["target_groups"] == ["I"]
    assert payload["hate_speech"] == "B"


def test_parse_response_json_extracts_embedded_json_after_reasoning_text() -> None:
    payload = _parse_response_json(
        """thoughtful
The user wants me to analyze this comment carefully.
{
  "target_groups": ["D", "G"],
  "sentiment": "A",
  "respect": "A",
  "insult": "E",
  "humiliate": "E",
  "status": "A",
  "dehumanize": "D",
  "violence": "E",
  "genocide": "A",
  "attack_defend": "D",
  "hate_speech": "A"
}"""
    )

    assert payload["target_groups"] == ["D", "G"]
    assert payload["hate_speech"] == "A"


def test_provider_api_key_reads_xai_env_var(monkeypatch) -> None:
    config = ModelBatchConfig(
        name="test-xai",
        subset="reference_set",
        limit=1,
        prompt=BatchPromptConfig(
            system_prompt_path=Path("prompts/mhs_survey_v1.txt"),
            user_prompt_template="",
        ),
        model=BatchModelConfig(
            provider="xai",
            name="grok-4-fast-reasoning",
            id="xai:grok-4-fast-reasoning",
            max_tokens=256,
            params={},
            reasoning=BatchReasoningConfig(),
        ),
        batches=BatchStorageConfig(
            run_dir=Path("batches/test-xai"),
            request_manifest_filename="request_manifest.jsonl",
            provider_requests_filename="provider_requests.jsonl",
            batch_metadata_filename="batch_job.json",
            raw_results_filename="raw_results.jsonl",
            processed_records_filename="processed_records.jsonl",
            processed_csv_filename="processed_records.csv",
            errors_filename="processing_errors.jsonl",
        ),
    )

    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")

    assert _provider_api_key(config) == "test-xai-key"


def test_provider_api_key_reads_together_env_var(monkeypatch) -> None:
    config = ModelBatchConfig(
        name="test-together",
        subset="reference_set",
        limit=1,
        prompt=BatchPromptConfig(
            system_prompt_path=Path("prompts/mhs_survey_v1.txt"),
            user_prompt_template="",
        ),
        model=BatchModelConfig(
            provider="together",
            name="openai/gpt-oss-120b",
            id="together_openai_gpt-oss-120b",
            max_tokens=256,
            params={},
            reasoning=BatchReasoningConfig(),
        ),
        batches=BatchStorageConfig(
            run_dir=Path("batches/test-together"),
            request_manifest_filename="request_manifest.jsonl",
            provider_requests_filename="provider_requests.jsonl",
            batch_metadata_filename="batch_job.json",
            raw_results_filename="raw_results.jsonl",
            processed_records_filename="processed_records.jsonl",
            processed_csv_filename="processed_records.csv",
            errors_filename="processing_errors.jsonl",
        ),
    )

    monkeypatch.setenv("TOGETHER_API_KEY", "test-together-key")

    assert _provider_api_key(config) == "test-together-key"


def test_provider_api_key_reads_moonshot_env_var(monkeypatch) -> None:
    config = ModelBatchConfig(
        name="test-moonshot",
        subset="reference_set",
        limit=1,
        prompt=BatchPromptConfig(
            system_prompt_path=Path("prompts/mhs_survey_v1.txt"),
            user_prompt_template="",
        ),
        model=BatchModelConfig(
            provider="moonshot",
            name="kimi-k2.5",
            id="moonshot_kimi-k2.5",
            max_tokens=256,
            params={},
            reasoning=BatchReasoningConfig(),
        ),
        batches=BatchStorageConfig(
            run_dir=Path("batches/test-moonshot"),
            request_manifest_filename="request_manifest.jsonl",
            provider_requests_filename="provider_requests.jsonl",
            batch_metadata_filename="batch_job.json",
            raw_results_filename="raw_results.jsonl",
            processed_records_filename="processed_records.jsonl",
            processed_csv_filename="processed_records.csv",
            errors_filename="processing_errors.jsonl",
        ),
    )

    monkeypatch.setenv("MOONSHOT_API_KEY", "test-moonshot-key")

    assert _provider_api_key(config) == "test-moonshot-key"


def test_extract_openai_result_reads_message_text() -> None:
    entry = {
        "custom_id": "comment-1",
        "response": {
            "status_code": 200,
            "body": {
                "choices": [
                    {
                        "message": {
                            "content": '{"target_groups":["I"],"sentiment":"C","respect":"C","insult":"A","humiliate":"A","status":"C","dehumanize":"A","violence":"A","genocide":"A","attack_defend":"C","hate_speech":"B"}'
                        }
                    }
                ]
            },
        },
    }

    custom_id, response_text, _, response_error = _extract_openai_result(entry)

    assert custom_id == "comment-1"
    assert response_error is None
    assert '"hate_speech":"B"' in response_text


def test_extract_anthropic_result_reads_text_blocks() -> None:
    entry = {
        "custom_id": "comment-2",
        "result": {
            "type": "succeeded",
            "message": {
                "content": [
                    {
                        "type": "text",
                        "text": '{"target_groups":["A"],"sentiment":"A","respect":"A","insult":"E","humiliate":"E","status":"A","dehumanize":"E","violence":"E","genocide":"E","attack_defend":"E","hate_speech":"A"}',
                    }
                ]
            },
        },
    }

    custom_id, response_text, _, response_error = _extract_anthropic_result(entry)

    assert custom_id == "comment-2"
    assert response_error is None
    assert '"target_groups":["A"]' in response_text


def test_extract_anthropic_result_handles_refusal_without_text_block() -> None:
    entry = {
        "custom_id": "comment-2",
        "result": {
            "type": "succeeded",
            "message": {
                "content": [],
                "stop_reason": "refusal",
            },
        },
    }

    custom_id, response_text, response_metadata, response_error = _extract_anthropic_result(entry)

    assert custom_id == "comment-2"
    assert response_text == ""
    assert response_metadata["stop_reason"] == "refusal"
    assert response_error == "Anthropic model refusal with no text content"


def test_build_processing_error_record_marks_anthropic_refusal_as_structured_error() -> None:
    error_record = _build_processing_error_record(
        provider_name="anthropic",
        custom_id="comment-20005",
        comment_id=20005,
        response_text=(
            "I can't analyze this content. The provided text describes explicit sexual violence, "
            "which I'm not able to engage with."
        ),
        response_metadata={"stop_reason": "end_turn"},
        exception=json.JSONDecodeError("Expecting value", "", 0),
    )

    assert error_record["error_type"] == "model_refusal"
    assert error_record["error"] == "Model returned an unstructured refusal instead of the expected JSON object"
    assert error_record["provider_stop_reason"] == "end_turn"
    assert "parse_error" in error_record


def test_build_processing_error_record_marks_non_json_text_separately() -> None:
    error_record = _build_processing_error_record(
        provider_name="anthropic",
        custom_id="comment-7",
        comment_id=7,
        response_text="target_groups: [I]",
        response_metadata={"stop_reason": "end_turn"},
        exception=json.JSONDecodeError("Expecting value", "", 0),
    )

    assert error_record["error_type"] == "invalid_json_response"
    assert error_record["error"] == "Model returned non-JSON text instead of the expected JSON object"


def test_extract_google_result_reads_candidate_parts() -> None:
    entry = {
        "key": "comment-3",
        "response": {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": '{"target_groups":["D"],"sentiment":"B","respect":"B","insult":"D","humiliate":"D","status":"B","dehumanize":"D","violence":"D","genocide":"D","attack_defend":"D","hate_speech":"A"}'
                            }
                        ]
                    }
                }
            ]
        },
    }

    custom_id, response_text, _, response_error = _extract_google_result(entry)

    assert custom_id == "comment-3"
    assert response_error is None
    assert '"target_groups":["D"]' in response_text


def test_extract_google_result_reads_top_level_candidates() -> None:
    entry = {
        "key": "comment-9",
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": '{"target_groups":["I"],"sentiment":"C","respect":"C","insult":"A","humiliate":"A","status":"C","dehumanize":"A","violence":"A","genocide":"A","attack_defend":"C","hate_speech":"B"}'
                        }
                    ]
                }
            }
        ],
    }

    custom_id, response_text, _, response_error = _extract_google_result(entry)

    assert custom_id == "comment-9"
    assert response_error is None
    assert '"hate_speech":"B"' in response_text


def test_extract_xai_result_reads_chat_completion_result() -> None:
    entry = {
        "batch_request_id": "comment-5",
        "batch_result": {
            "response": {
                "chat_get_completion": {
                    "choices": [
                        {
                            "message": {
                                "content": '{"target_groups":["I"],"sentiment":"C","respect":"C","insult":"A","humiliate":"A","status":"C","dehumanize":"A","violence":"A","genocide":"A","attack_defend":"C","hate_speech":"B"}'
                            }
                        }
                    ]
                }
            }
        },
    }

    custom_id, response_text, _, response_error = _extract_xai_result(entry)

    assert custom_id == "comment-5"
    assert response_error is None
    assert '"hate_speech":"B"' in response_text


def test_extract_together_result_reads_message_text() -> None:
    entry = {
        "custom_id": "comment-6",
        "response": {
            "status_code": 200,
            "body": {
                "choices": [
                    {
                        "message": {
                            "content": '{"target_groups":["I"],"sentiment":"C","respect":"C","insult":"A","humiliate":"A","status":"C","dehumanize":"A","violence":"A","genocide":"A","attack_defend":"C","hate_speech":"B"}'
                        }
                    }
                ]
            },
        },
    }

    custom_id, response_text, _, response_error = _extract_together_result(entry)

    assert custom_id == "comment-6"
    assert response_error is None
    assert '"hate_speech":"B"' in response_text


def test_extract_together_result_reads_error_file_row() -> None:
    entry = {
        "custom_id": "comment-7",
        "error": {"message": "Invalid model specified", "code": "invalid_model"},
    }

    custom_id, response_text, response_metadata, response_error = _extract_together_result(entry)

    assert custom_id == "comment-7"
    assert response_text == ""
    assert response_metadata["error"]["code"] == "invalid_model"
    assert "invalid_model" in response_error


def test_extract_moonshot_result_reads_message_text() -> None:
    entry = {
        "custom_id": "comment-8",
        "response": {
            "status_code": 0,
            "body": {
                "choices": [
                    {
                        "message": {
                            "content": '{"target_groups":["I"],"sentiment":"C","respect":"C","insult":"A","humiliate":"A","status":"C","dehumanize":"A","violence":"A","genocide":"A","attack_defend":"C","hate_speech":"B"}'
                        }
                    }
                ]
            },
        },
        "error": None,
    }

    custom_id, response_text, _, response_error = _extract_moonshot_result(entry)

    assert custom_id == "comment-8"
    assert response_error is None
    assert '"hate_speech":"B"' in response_text


def test_build_requests_uses_raw_comment_when_user_prompt_template_is_empty() -> None:
    config = ModelBatchConfig(
        name="test",
        subset="reference_set",
        limit=1,
        prompt=BatchPromptConfig(
            system_prompt_path=Path("prompts/mhs_survey_v1.txt"),
            user_prompt_template="",
        ),
        model=BatchModelConfig(
            provider="openai",
            name="gpt-4.1-mini",
            max_tokens=400,
            params={},
            reasoning=BatchReasoningConfig(effort="low"),
        ),
        batches=BatchStorageConfig(
            run_dir=Path("batches/test"),
            request_manifest_filename="request_manifest.jsonl",
            provider_requests_filename="provider_requests.jsonl",
            batch_metadata_filename="batch_job.json",
            raw_results_filename="raw_results.jsonl",
            processed_records_filename="processed_records.jsonl",
            processed_csv_filename="processed_records.csv",
            errors_filename="processing_errors.jsonl",
        ),
    )

    _, provider_requests = _build_requests(
        config=config,
        comments=[{"comment_id": 1, "text": "just the comment"}],
    )

    assert provider_requests[0]["body"]["messages"][1]["content"] == "just the comment"
    assert provider_requests[0]["body"]["max_completion_tokens"] == 400


def test_build_requests_for_anthropic_follow_openai_config_shape() -> None:
    config = ModelBatchConfig(
        name="test-anthropic",
        subset="reference_set",
        limit=1,
        prompt=BatchPromptConfig(
            system_prompt_path=Path("prompts/mhs_survey_v1.txt"),
            user_prompt_template="",
        ),
        model=BatchModelConfig(
            provider="anthropic",
            name="claude-3-5-sonnet-20241022",
            id="anthropic:claude-3-5-sonnet-20241022",
            max_tokens=1000,
            params={},
            reasoning=BatchReasoningConfig(budget_tokens=512),
        ),
        batches=BatchStorageConfig(
            run_dir=Path("batches/test-anthropic"),
            request_manifest_filename="request_manifest.jsonl",
            provider_requests_filename="provider_requests.jsonl",
            batch_metadata_filename="batch_job.json",
            raw_results_filename="raw_results.jsonl",
            processed_records_filename="processed_records.jsonl",
            processed_csv_filename="processed_records.csv",
            errors_filename="processing_errors.jsonl",
        ),
    )

    _, provider_requests = _build_requests(
        config=config,
        comments=[{"comment_id": 1, "text": "just the comment"}],
    )

    assert provider_requests[0] == {
        "custom_id": "comment-1",
        "params": {
            "model": "claude-3-5-sonnet-20241022",
            "system": Path("prompts/mhs_survey_v1.txt").read_text(),
            "messages": [{"role": "user", "content": "just the comment"}],
            "max_tokens": 1000,
            "thinking": {"type": "enabled", "budget_tokens": 512},
        },
    }


def test_build_requests_for_anthropic_use_adaptive_thinking_for_claude_46() -> None:
    config = ModelBatchConfig(
        name="test-anthropic-adaptive",
        subset="reference_set",
        limit=1,
        prompt=BatchPromptConfig(
            system_prompt_path=Path("prompts/mhs_survey_v1.txt"),
            user_prompt_template="",
        ),
        model=BatchModelConfig(
            provider="anthropic",
            name="claude-sonnet-4-6",
            id="anthropic:claude-sonnet-4-6",
            max_tokens=4096,
            params={},
            reasoning=BatchReasoningConfig(effort="medium"),
        ),
        batches=BatchStorageConfig(
            run_dir=Path("batches/test-anthropic-adaptive"),
            request_manifest_filename="request_manifest.jsonl",
            provider_requests_filename="provider_requests.jsonl",
            batch_metadata_filename="batch_job.json",
            raw_results_filename="raw_results.jsonl",
            processed_records_filename="processed_records.jsonl",
            processed_csv_filename="processed_records.csv",
            errors_filename="processing_errors.jsonl",
        ),
    )

    _, provider_requests = _build_requests(
        config=config,
        comments=[{"comment_id": 1, "text": "just the comment"}],
    )

    assert provider_requests[0] == {
        "custom_id": "comment-1",
        "params": {
            "model": "claude-sonnet-4-6",
            "system": Path("prompts/mhs_survey_v1.txt").read_text(),
            "messages": [{"role": "user", "content": "just the comment"}],
            "max_tokens": 4096,
            "output_config": {"effort": "medium"},
            "thinking": {"type": "adaptive"},
        },
    }


def test_build_requests_for_google_use_uploaded_batch_shape() -> None:
    config = ModelBatchConfig(
        name="test-google",
        subset="reference_set",
        limit=1,
        prompt=BatchPromptConfig(
            system_prompt_path=Path("prompts/mhs_survey_v1.txt"),
            user_prompt_template="",
        ),
        model=BatchModelConfig(
            provider="google",
            name="gemini-2.5-pro",
            id="google:gemini-2.5-pro",
            max_tokens=256,
            params={"temperature": 0},
            reasoning=BatchReasoningConfig(budget_tokens=64),
        ),
        batches=BatchStorageConfig(
            run_dir=Path("batches/test-google"),
            request_manifest_filename="request_manifest.jsonl",
            provider_requests_filename="provider_requests.jsonl",
            batch_metadata_filename="batch_job.json",
            raw_results_filename="raw_results.jsonl",
            processed_records_filename="processed_records.jsonl",
            processed_csv_filename="processed_records.csv",
            errors_filename="processing_errors.jsonl",
        ),
    )

    _, provider_requests = _build_requests(
        config=config,
        comments=[{"comment_id": 1, "text": "just the comment"}],
    )

    assert provider_requests[0]["metadata"]["key"] == "comment-1"
    assert provider_requests[0]["request"]["contents"][0]["parts"][0]["text"] == "just the comment"
    assert (
        provider_requests[0]["request"]["systemInstruction"]["parts"][0]["text"]
        == Path("prompts/mhs_survey_v1.txt").read_text()
    )
    assert provider_requests[0]["request"]["generationConfig"]["maxOutputTokens"] == 256
    assert provider_requests[0]["request"]["generationConfig"]["thinkingConfig"] == {
        "thinkingBudget": 64
    }


def test_build_requests_for_xai_follow_native_batch_shape() -> None:
    config = ModelBatchConfig(
        name="test-xai",
        subset="reference_set",
        limit=1,
        prompt=BatchPromptConfig(
            system_prompt_path=Path("prompts/mhs_survey_v1.txt"),
            user_prompt_template="",
        ),
        model=BatchModelConfig(
            provider="xai",
            name="grok-4-fast-reasoning",
            id="xai:grok-4-fast-reasoning",
            max_tokens=256,
            params={},
            reasoning=BatchReasoningConfig(effort="low"),
        ),
        batches=BatchStorageConfig(
            run_dir=Path("batches/test-xai"),
            request_manifest_filename="request_manifest.jsonl",
            provider_requests_filename="provider_requests.jsonl",
            batch_metadata_filename="batch_job.json",
            raw_results_filename="raw_results.jsonl",
            processed_records_filename="processed_records.jsonl",
            processed_csv_filename="processed_records.csv",
            errors_filename="processing_errors.jsonl",
        ),
    )

    _, provider_requests = _build_requests(
        config=config,
        comments=[{"comment_id": 1, "text": "just the comment"}],
    )

    assert provider_requests[0]["batch_request_id"] == "comment-1"
    chat_request = provider_requests[0]["batch_request"]["chat_get_completion"]
    assert chat_request["model"] == "grok-4-fast-reasoning"
    assert chat_request["messages"][1]["content"] == "just the comment"
    assert chat_request["max_tokens"] == 256
    assert chat_request["reasoning_effort"] == "low"


def test_build_requests_for_together_follow_batch_api_shape() -> None:
    config = ModelBatchConfig(
        name="test-together",
        subset="reference_set",
        limit=1,
        prompt=BatchPromptConfig(
            system_prompt_path=Path("prompts/mhs_survey_v1.txt"),
            user_prompt_template="",
        ),
        model=BatchModelConfig(
            provider="together",
            name="openai/gpt-oss-120b",
            id="together_openai_gpt-oss-120b",
            max_tokens=4096,
            params={"temperature": 0.5},
            reasoning=BatchReasoningConfig(),
        ),
        batches=BatchStorageConfig(
            run_dir=Path("batches/test-together"),
            request_manifest_filename="request_manifest.jsonl",
            provider_requests_filename="provider_requests.jsonl",
            batch_metadata_filename="batch_job.json",
            raw_results_filename="raw_results.jsonl",
            processed_records_filename="processed_records.jsonl",
            processed_csv_filename="processed_records.csv",
            errors_filename="processing_errors.jsonl",
        ),
    )

    _, provider_requests = _build_requests(
        config=config,
        comments=[{"comment_id": 1, "text": "just the comment"}],
    )

    assert provider_requests[0]["custom_id"] == "comment-1"
    body = provider_requests[0]["body"]
    assert body["model"] == "openai/gpt-oss-120b"
    assert body["messages"][1]["content"] == "just the comment"
    assert body["max_tokens"] == 4096
    assert body["temperature"] == 0.5


def test_build_requests_for_moonshot_use_openai_batch_shape_with_thinking() -> None:
    config = ModelBatchConfig(
        name="test-moonshot",
        subset="reference_set",
        limit=1,
        prompt=BatchPromptConfig(
            system_prompt_path=Path("prompts/mhs_survey_v1.txt"),
            user_prompt_template="",
        ),
        model=BatchModelConfig(
            provider="moonshot",
            name="kimi-k2.5",
            id="moonshot_kimi-k2.5",
            max_tokens=4096,
            params={"thinking": {"type": "enabled"}},
            reasoning=BatchReasoningConfig(),
        ),
        batches=BatchStorageConfig(
            run_dir=Path("batches/test-moonshot"),
            request_manifest_filename="request_manifest.jsonl",
            provider_requests_filename="provider_requests.jsonl",
            batch_metadata_filename="batch_job.json",
            raw_results_filename="raw_results.jsonl",
            processed_records_filename="processed_records.jsonl",
            processed_csv_filename="processed_records.csv",
            errors_filename="processing_errors.jsonl",
        ),
    )

    _, provider_requests = _build_requests(
        config=config,
        comments=[{"comment_id": 1, "text": "just the comment"}],
    )

    assert provider_requests[0] == {
        "custom_id": "comment-1",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": "kimi-k2.5",
            "messages": [
                {"role": "system", "content": Path("prompts/mhs_survey_v1.txt").read_text()},
                {"role": "user", "content": "just the comment"},
            ],
            "thinking": {"type": "enabled"},
            "max_tokens": 4096,
        },
    }


def test_apply_google_batch_reasoning_maps_effort_to_thinking_level() -> None:
    payload = _apply_google_batch_reasoning(
        generation_config={"max_output_tokens": 80},
        reasoning_effort="medium",
        reasoning_budget_tokens=None,
    )

    assert payload["max_output_tokens"] == 80
    assert payload["thinking_config"]["thinking_level"] == "medium"


def test_apply_google_batch_reasoning_rejects_effort_and_budget_together() -> None:
    import pytest

    with pytest.raises(ValueError, match="cannot set both"):
        _apply_google_batch_reasoning(
            generation_config={},
            reasoning_effort="low",
            reasoning_budget_tokens=64,
        )


def test_create_google_batch_uploads_jsonl_and_uses_uploaded_file_name(
    tmp_path: Path, monkeypatch
) -> None:
    provider_requests_path = tmp_path / "provider_requests.jsonl"
    provider_requests_path.write_text(json.dumps({"key": "comment-1", "request": {}}) + "\n")

    config = ModelBatchConfig(
        name="test-google",
        subset="reference_set",
        limit=1,
        prompt=BatchPromptConfig(
            system_prompt_path=Path("prompts/mhs_survey_v1.txt"),
            user_prompt_template="",
        ),
        model=BatchModelConfig(
            provider="google",
            name="gemini-2.5-pro",
            id="google:gemini-2.5-pro",
            max_tokens=256,
            params={},
            reasoning=BatchReasoningConfig(),
        ),
        batches=BatchStorageConfig(
            run_dir=tmp_path / "batches",
            request_manifest_filename="request_manifest.jsonl",
            provider_requests_filename="provider_requests.jsonl",
            batch_metadata_filename="batch_job.json",
            raw_results_filename="raw_results.jsonl",
            processed_records_filename="processed_records.jsonl",
            processed_csv_filename="processed_records.csv",
            errors_filename="processing_errors.jsonl",
        ),
    )

    create_calls: list[tuple[str, str, object]] = []

    class DummyState:
        name = "JOB_STATE_RUNNING"

    class DummyBatch:
        name = "batches/123"
        state = DummyState()

    class DummyBatches:
        def create(self, model, src, config):
            create_calls.append((model, src, config))
            return DummyBatch()

    class DummyClient:
        def __init__(self, api_key):
            self.batches = DummyBatches()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr("mhs_llms.batch.genai.Client", DummyClient)

    batch = _create_provider_batch(
        config=config,
        provider_requests_path=provider_requests_path,
        provider_requests=[
            {
                "metadata": {"key": "comment-1"},
                "request": {
                    "contents": [{"parts": [{"text": "just the comment"}], "role": "user"}],
                    "generationConfig": {
                        "maxOutputTokens": 256,
                        "thinkingConfig": {"thinkingLevel": "low"},
                    },
                    "systemInstruction": {"parts": [{"text": "system prompt"}]},
                },
            }
        ],
    )

    assert batch.name == "batches/123"
    assert create_calls == [
        (
            "gemini-2.5-pro",
            [
                {
                    "contents": [{"parts": [{"text": "just the comment"}], "role": "user"}],
                    "metadata": {"key": "comment-1"},
                    "config": {
                        "max_output_tokens": 256,
                        "thinking_config": {"thinking_level": "low"},
                        "system_instruction": {"parts": [{"text": "system prompt"}]},
                    },
                }
            ],
            {"display_name": "test-google"},
        )
    ]


def test_create_xai_batch_creates_batch_and_adds_requests(tmp_path: Path, monkeypatch) -> None:
    provider_requests_path = tmp_path / "provider_requests.jsonl"
    provider_requests_path.write_text(
        json.dumps({"batch_request_id": "comment-1", "batch_request": {"chat_get_completion": {}}}) + "\n"
    )

    config = ModelBatchConfig(
        name="test-xai",
        subset="reference_set",
        limit=1,
        prompt=BatchPromptConfig(
            system_prompt_path=Path("prompts/mhs_survey_v1.txt"),
            user_prompt_template="",
        ),
        model=BatchModelConfig(
            provider="xai",
            name="grok-4-fast-reasoning",
            id="xai:grok-4-fast-reasoning",
            max_tokens=256,
            params={},
            reasoning=BatchReasoningConfig(),
        ),
        batches=BatchStorageConfig(
            run_dir=tmp_path / "batches",
            request_manifest_filename="request_manifest.jsonl",
            provider_requests_filename="provider_requests.jsonl",
            batch_metadata_filename="batch_job.json",
            raw_results_filename="raw_results.jsonl",
            processed_records_filename="processed_records.jsonl",
            processed_csv_filename="processed_records.csv",
            errors_filename="processing_errors.jsonl",
        ),
    )

    calls: list[tuple[str, str, dict[str, object] | None]] = []

    def fake_xai_api_request(config, method, path, payload=None, query=None):
        calls.append((method, path, payload))
        if method == "POST" and path == "/v1/batches":
            return {"batch_id": "batch-xai-123", "state": {"num_pending": 0, "num_error": 0, "num_cancelled": 0}}
        if method == "POST" and path == "/v1/batches/batch-xai-123/requests":
            return {"added": 1}
        if method == "GET" and path == "/v1/batches/batch-xai-123":
            return {"batch_id": "batch-xai-123", "state": {"num_pending": 1, "num_error": 0, "num_cancelled": 0}}
        raise AssertionError(f"Unexpected xAI call: {(method, path)}")

    monkeypatch.setattr("mhs_llms.batch._xai_api_request", fake_xai_api_request)

    batch = _create_provider_batch(
        config=config,
        provider_requests_path=provider_requests_path,
        provider_requests=[{"batch_request_id": "comment-1", "batch_request": {"chat_get_completion": {}}}],
    )

    assert batch["batch_id"] == "batch-xai-123"
    assert calls == [
        ("POST", "/v1/batches", {"name": "test-xai"}),
        (
            "POST",
            "/v1/batches/batch-xai-123/requests",
            {"batch_requests": [{"batch_request_id": "comment-1", "batch_request": {"chat_get_completion": {}}}]},
        ),
        ("GET", "/v1/batches/batch-xai-123", None),
    ]


def test_create_together_batch_uploads_file_and_creates_job(tmp_path: Path, monkeypatch) -> None:
    provider_requests_path = tmp_path / "provider_requests.jsonl"
    provider_requests_path.write_text(json.dumps({"custom_id": "comment-1", "body": {}}) + "\n")

    config = ModelBatchConfig(
        name="test-together",
        subset="reference_set",
        limit=1,
        prompt=BatchPromptConfig(
            system_prompt_path=Path("prompts/mhs_survey_v1.txt"),
            user_prompt_template="",
        ),
        model=BatchModelConfig(
            provider="together",
            name="openai/gpt-oss-120b",
            id="together_openai_gpt-oss-120b",
            max_tokens=256,
            params={},
            reasoning=BatchReasoningConfig(),
        ),
        batches=BatchStorageConfig(
            run_dir=tmp_path / "batches",
            request_manifest_filename="request_manifest.jsonl",
            provider_requests_filename="provider_requests.jsonl",
            batch_metadata_filename="batch_job.json",
            raw_results_filename="raw_results.jsonl",
            processed_records_filename="processed_records.jsonl",
            processed_csv_filename="processed_records.csv",
            errors_filename="processing_errors.jsonl",
        ),
    )

    calls: list[tuple[str, str, dict[str, object] | None]] = []

    def fake_together_upload_file(config, path):
        assert path == provider_requests_path
        return {"id": "file-together-123"}

    def fake_together_api_request(config, method, path, payload=None, query=None):
        calls.append((method, path, payload))
        return {
            "job": {
                "id": "batch-together-123",
                "status": "VALIDATING",
                "input_file_id": "file-together-123",
            }
        }

    monkeypatch.setattr("mhs_llms.batch._together_upload_file", fake_together_upload_file)
    monkeypatch.setattr("mhs_llms.batch._together_api_request", fake_together_api_request)

    batch = _create_provider_batch(
        config=config,
        provider_requests_path=provider_requests_path,
        provider_requests=[{"custom_id": "comment-1", "body": {}}],
    )

    assert batch["id"] == "batch-together-123"
    assert calls == [
        (
            "POST",
            "/batches",
            {
                "input_file_id": "file-together-123",
                "endpoint": "/v1/chat/completions",
                "completion_window": "24h",
            },
        )
    ]


def test_download_provider_results_reads_google_output_file(monkeypatch, tmp_path: Path) -> None:
    config = ModelBatchConfig(
        name="test-google",
        subset="reference_set",
        limit=1,
        prompt=BatchPromptConfig(
            system_prompt_path=Path("prompts/mhs_survey_v1.txt"),
            user_prompt_template="",
        ),
        model=BatchModelConfig(
            provider="google",
            name="gemini-2.5-pro",
            id="google:gemini-2.5-pro",
            max_tokens=256,
            params={},
            reasoning=BatchReasoningConfig(),
        ),
        batches=BatchStorageConfig(
            run_dir=tmp_path / "batches",
            request_manifest_filename="request_manifest.jsonl",
            provider_requests_filename="provider_requests.jsonl",
            batch_metadata_filename="batch_job.json",
            raw_results_filename="raw_results.jsonl",
            processed_records_filename="processed_records.jsonl",
            processed_csv_filename="processed_records.csv",
            errors_filename="processing_errors.jsonl",
        ),
    )

    class DummyDest:
        file_name = "files/result-123"

    class DummyBatch:
        dest = DummyDest()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(
        "mhs_llms.batch._download_google_batch_output",
        lambda file_name, api_key: (
            json.dumps(
                {
                    "key": "comment-1",
                    "response": {
                        "candidates": [{"content": {"parts": [{"text": '{"hate_speech":"B"}'}]}}]
                    },
                }
            )
            + "\n"
        ).encode("utf-8"),
    )

    rows = _download_provider_results(config=config, batch_object=DummyBatch())

    assert rows[0]["key"] == "comment-1"
    assert rows[0]["response"]["candidates"][0]["content"]["parts"][0]["text"] == '{"hate_speech":"B"}'


def test_download_provider_results_uses_google_inline_metadata_keys(tmp_path: Path) -> None:
    config = ModelBatchConfig(
        name="test-google",
        subset="reference_set",
        limit=2,
        prompt=BatchPromptConfig(
            system_prompt_path=Path("prompts/mhs_survey_v1.txt"),
            user_prompt_template="",
        ),
        model=BatchModelConfig(
            provider="google",
            name="gemini-2.5-pro",
            id="google:gemini-2.5-pro",
            max_tokens=256,
            params={},
            reasoning=BatchReasoningConfig(),
        ),
        batches=BatchStorageConfig(
            run_dir=tmp_path / "batches",
            request_manifest_filename="request_manifest.jsonl",
            provider_requests_filename="provider_requests.jsonl",
            batch_metadata_filename="batch_job.json",
            raw_results_filename="raw_results.jsonl",
            processed_records_filename="processed_records.jsonl",
            processed_csv_filename="processed_records.csv",
            errors_filename="processing_errors.jsonl",
        ),
    )
    config.batches.run_dir.mkdir(parents=True)
    (config.batches.run_dir / config.batches.request_manifest_filename).write_text(
        "\n".join(
            [
                json.dumps({"comment_id": 1, "custom_id": "comment-1", "text": "first"}),
                json.dumps({"comment_id": 2, "custom_id": "comment-2", "text": "second"}),
            ]
        )
        + "\n"
    )

    class DummyDest:
        inlined_responses = [
            {
                "metadata": {"key": "comment-2"},
                "response": {
                    "candidates": [{"content": {"parts": [{"text": '{"hate_speech":"C"}'}]}}]
                },
            },
            {
                "metadata": {"key": "comment-1"},
                "response": {
                    "candidates": [{"content": {"parts": [{"text": '{"hate_speech":"A"}'}]}}]
                },
            },
        ]

    class DummyBatch:
        dest = DummyDest()

    rows = _download_provider_results(config=config, batch_object=DummyBatch())

    assert [row["custom_id"] for row in rows] == ["comment-2", "comment-1"]
    assert rows[0]["metadata"]["key"] == "comment-2"


def test_download_provider_results_rejects_google_inline_without_metadata_key(
    tmp_path: Path,
) -> None:
    config = ModelBatchConfig(
        name="test-google",
        subset="reference_set",
        limit=1,
        prompt=BatchPromptConfig(
            system_prompt_path=Path("prompts/mhs_survey_v1.txt"),
            user_prompt_template="",
        ),
        model=BatchModelConfig(
            provider="google",
            name="gemini-2.5-pro",
            id="google:gemini-2.5-pro",
            max_tokens=256,
            params={},
            reasoning=BatchReasoningConfig(),
        ),
        batches=BatchStorageConfig(
            run_dir=tmp_path / "batches",
            request_manifest_filename="request_manifest.jsonl",
            provider_requests_filename="provider_requests.jsonl",
            batch_metadata_filename="batch_job.json",
            raw_results_filename="raw_results.jsonl",
            processed_records_filename="processed_records.jsonl",
            processed_csv_filename="processed_records.csv",
            errors_filename="processing_errors.jsonl",
        ),
    )
    config.batches.run_dir.mkdir(parents=True)
    (config.batches.run_dir / config.batches.request_manifest_filename).write_text(
        json.dumps({"comment_id": 1, "custom_id": "comment-1", "text": "first"}) + "\n"
    )

    class DummyDest:
        inlined_responses = [
            {
                "response": {
                    "candidates": [{"content": {"parts": [{"text": '{"hate_speech":"C"}'}]}}]
                },
            },
        ]

    class DummyBatch:
        dest = DummyDest()

    with pytest.raises(ValueError, match="refusing to assign comment ids positionally"):
        _download_provider_results(config=config, batch_object=DummyBatch())


def test_download_provider_results_reads_paginated_xai_results(monkeypatch, tmp_path: Path) -> None:
    config = ModelBatchConfig(
        name="test-xai",
        subset="reference_set",
        limit=1,
        prompt=BatchPromptConfig(
            system_prompt_path=Path("prompts/mhs_survey_v1.txt"),
            user_prompt_template="",
        ),
        model=BatchModelConfig(
            provider="xai",
            name="grok-4-fast-reasoning",
            id="xai:grok-4-fast-reasoning",
            max_tokens=256,
            params={},
            reasoning=BatchReasoningConfig(),
        ),
        batches=BatchStorageConfig(
            run_dir=tmp_path / "batches",
            request_manifest_filename="request_manifest.jsonl",
            provider_requests_filename="provider_requests.jsonl",
            batch_metadata_filename="batch_job.json",
            raw_results_filename="raw_results.jsonl",
            processed_records_filename="processed_records.jsonl",
            processed_csv_filename="processed_records.csv",
            errors_filename="processing_errors.jsonl",
        ),
    )

    calls: list[dict[str, object] | None] = []

    def fake_xai_api_request(config, method, path, payload=None, query=None):
        calls.append(query)
        if query is None or query.get("pagination_token") is None:
            return {
                "results": [{"batch_request_id": "comment-1"}],
                "pagination_token": "next-page",
            }
        return {
            "results": [{"batch_request_id": "comment-2"}],
            "pagination_token": None,
        }

    monkeypatch.setattr("mhs_llms.batch._xai_api_request", fake_xai_api_request)

    rows = _download_provider_results(
        config=config,
        batch_object={"batch_id": "batch-xai-123"},
    )

    assert [row["batch_request_id"] for row in rows] == ["comment-1", "comment-2"]
    assert calls == [
        {"page_size": 1000, "pagination_token": None},
        {"page_size": 1000, "pagination_token": "next-page"},
    ]


def test_download_provider_results_reads_together_output_and_error_files(
    monkeypatch, tmp_path: Path
) -> None:
    config = ModelBatchConfig(
        name="test-together",
        subset="reference_set",
        limit=1,
        prompt=BatchPromptConfig(
            system_prompt_path=Path("prompts/mhs_survey_v1.txt"),
            user_prompt_template="",
        ),
        model=BatchModelConfig(
            provider="together",
            name="openai/gpt-oss-120b",
            id="together_openai_gpt-oss-120b",
            max_tokens=256,
            params={},
            reasoning=BatchReasoningConfig(),
        ),
        batches=BatchStorageConfig(
            run_dir=tmp_path / "batches",
            request_manifest_filename="request_manifest.jsonl",
            provider_requests_filename="provider_requests.jsonl",
            batch_metadata_filename="batch_job.json",
            raw_results_filename="raw_results.jsonl",
            processed_records_filename="processed_records.jsonl",
            processed_csv_filename="processed_records.csv",
            errors_filename="processing_errors.jsonl",
        ),
    )

    def fake_together_download_file(config, file_id):
        if file_id == "file-output":
            return (
                json.dumps(
                    {
                        "custom_id": "comment-1",
                        "response": {"status_code": 200, "body": {"choices": []}},
                    }
                )
                + "\n"
            ).encode("utf-8")
        if file_id == "file-error":
            return (
                json.dumps(
                    {
                        "custom_id": "comment-2",
                        "error": {"message": "timeout", "code": "timeout"},
                    }
                )
                + "\n"
            ).encode("utf-8")
        raise AssertionError(f"Unexpected file_id: {file_id}")

    monkeypatch.setattr("mhs_llms.batch._together_download_file", fake_together_download_file)

    rows = _download_provider_results(
        config=config,
        batch_object={
            "id": "batch-together-123",
            "status": "COMPLETED",
            "output_file_id": "file-output",
            "error_file_id": "file-error",
        },
    )

    assert [row["custom_id"] for row in rows] == ["comment-1", "comment-2"]
    assert rows[1]["error"]["code"] == "timeout"


def test_batch_status_prefers_google_state_name() -> None:
    class DummyState:
        name = "BATCH_STATE_SUCCEEDED"
        value = "JOB_STATE_SUCCEEDED"

    class DummyBatch:
        state = DummyState()

    assert _batch_status("google", DummyBatch()) == "BATCH_STATE_SUCCEEDED"


def test_batch_status_maps_xai_state_counters() -> None:
    assert _batch_status(
        "xai",
        {"state": {"num_pending": 1, "num_error": 0, "num_cancelled": 0}},
    ) == "in_progress"
    assert _batch_status(
        "xai",
        {"state": {"num_pending": 0, "num_error": 0, "num_cancelled": 0}},
    ) == "completed"
    assert _batch_status(
        "xai",
        {"state": {"num_pending": 0, "num_error": 2, "num_cancelled": 0}},
    ) == "completed_with_errors"


def test_batch_status_reads_together_status() -> None:
    assert _batch_status("together", {"id": "batch-together-123", "status": "COMPLETED"}) == "COMPLETED"


def test_select_comment_ids_reference_set_is_in_code_and_sorted() -> None:
    dataframe = pd.DataFrame(
        [
            {"comment_id": 20, "platform": 0, "text": "twenty"},
            {"comment_id": 10, "platform": 1, "text": "ten"},
            {"comment_id": 20, "platform": 1, "text": "twenty duplicate"},
        ]
    )

    comment_ids = _select_comment_ids(dataframe, "reference_set")

    assert comment_ids == [10, 20]


def test_select_comment_ids_all_comments_ignores_platform_filter() -> None:
    dataframe = pd.DataFrame(
        [
            {"comment_id": 20, "platform": 0, "text": "twenty"},
            {"comment_id": 10, "platform": 1, "text": "ten"},
            {"comment_id": 20, "platform": 1, "text": "twenty duplicate"},
        ]
    )

    comment_ids = _select_comment_ids(dataframe, "all_comments")

    assert comment_ids == [10, 20]


def test_select_comment_ids_uses_annotator_count_threshold() -> None:
    dataframe = pd.DataFrame(
        [
            {"comment_id": 10, "platform": 0, "text": "ten"},
            {"comment_id": 10, "platform": 0, "text": "ten"},
            {"comment_id": 20, "platform": 0, "text": "twenty"},
            {"comment_id": 20, "platform": 0, "text": "twenty"},
            {"comment_id": 20, "platform": 0, "text": "twenty"},
            {"comment_id": 20, "platform": 0, "text": "twenty"},
            {"comment_id": 30, "platform": 0, "text": "thirty"},
            {"comment_id": 30, "platform": 0, "text": "thirty"},
            {"comment_id": 30, "platform": 0, "text": "thirty"},
            {"comment_id": 30, "platform": 0, "text": "thirty"},
            {"comment_id": 30, "platform": 0, "text": "thirty"},
            {"comment_id": 30, "platform": 0, "text": "thirty"},
            {"comment_id": 30, "platform": 0, "text": "thirty"},
            {"comment_id": 30, "platform": 0, "text": "thirty"},
        ]
    )

    comment_ids = _select_comment_ids(
        dataframe,
        {"type": "annotator_count_threshold", "min": 4, "max": 7},
    )

    assert comment_ids == [20]


def test_select_comment_ids_reads_comment_ids_file(tmp_path: Path) -> None:
    comment_ids_path = tmp_path / "comment_ids.csv"
    pd.DataFrame({"comment_id": [30, 10, 30, 20]}).to_csv(comment_ids_path, index=False)

    comment_ids = _select_comment_ids(
        pd.DataFrame([{"comment_id": 999, "platform": 0, "text": "ignored"}]),
        {"type": "comment_ids_file", "path": str(comment_ids_path)},
    )

    assert comment_ids == [30, 10, 20]


def test_write_processed_annotations_rewrites_csv_without_duplicates(tmp_path: Path) -> None:
    processed_csv = tmp_path / "processed.csv"
    processed_jsonl = tmp_path / "processed.jsonl"
    output_csv = tmp_path / "aggregate.csv"

    pd.DataFrame(
        [
            {"comment_id": 1, "judge_id": "openai:test", "hate_speech": "B"},
            {"comment_id": 2, "judge_id": "openai:test", "hate_speech": "A"},
        ]
    ).to_csv(processed_csv, index=False)
    processed_jsonl.write_text(
        '{"comment_id": 1, "judge_id": "openai:test", "hate_speech": "B"}\n'
        '{"comment_id": 2, "judge_id": "openai:test", "hate_speech": "A"}\n'
    )

    write_processed_annotations(processed_jsonl, processed_csv, output_csv)
    write_processed_annotations(processed_jsonl, processed_csv, output_csv)

    aggregated = pd.read_csv(output_csv)
    assert aggregated["comment_id"].tolist() == [1, 2]


def test_write_processed_annotations_rewrites_jsonl_without_duplicates(tmp_path: Path) -> None:
    processed_csv = tmp_path / "processed.csv"
    processed_jsonl = tmp_path / "processed.jsonl"
    output_jsonl = tmp_path / "aggregate.jsonl"

    pd.DataFrame([{"comment_id": 1, "judge_id": "openai:test", "hate_speech": "B"}]).to_csv(
        processed_csv, index=False
    )
    processed_jsonl.write_text('{"comment_id": 1, "judge_id": "openai:test", "hate_speech": "B"}\n')

    write_processed_annotations(processed_jsonl, processed_csv, output_jsonl)
    write_processed_annotations(processed_jsonl, processed_csv, output_jsonl)

    lines = output_jsonl.read_text().splitlines()
    assert len(lines) == 1


def test_write_combined_processed_annotations_combines_multiple_models_into_one_csv(
    tmp_path: Path,
) -> None:
    processed_csv_one = tmp_path / "processed_one.csv"
    processed_jsonl_one = tmp_path / "processed_one.jsonl"
    processed_csv_two = tmp_path / "processed_two.csv"
    processed_jsonl_two = tmp_path / "processed_two.jsonl"
    output_csv = tmp_path / "aggregate.csv"

    pd.DataFrame([{"comment_id": 1, "judge_id": "model-one", "hate_speech": "B"}]).to_csv(
        processed_csv_one, index=False
    )
    processed_jsonl_one.write_text('{"comment_id": 1, "judge_id": "model-one", "hate_speech": "B"}\n')
    pd.DataFrame([{"comment_id": 1, "judge_id": "model-two", "hate_speech": "A"}]).to_csv(
        processed_csv_two, index=False
    )
    processed_jsonl_two.write_text('{"comment_id": 1, "judge_id": "model-two", "hate_speech": "A"}\n')

    write_combined_processed_annotations(
        processed_paths=[
            (processed_jsonl_one, processed_csv_one),
            (processed_jsonl_two, processed_csv_two),
        ],
        output_path=output_csv,
    )

    aggregated = pd.read_csv(output_csv)
    assert aggregated["judge_id"].tolist() == ["model-one", "model-two"]


def test_annotation_record_to_row_uses_provider_model_order_without_metadata_by_default() -> None:
    record = MHSAnnotationRecord(
        comment_id=1,
        judge_id="openai:gpt-5.4",
        source_type="model",
        text="test comment",
        target_groups=["I"],
        sentiment="C",
        respect="C",
        insult="A",
        humiliate="A",
        status="C",
        dehumanize="A",
        violence="A",
        genocide="A",
        attack_defend="C",
        hate_speech="B",
        metadata={"provider": "openai", "model": "gpt-5.4", "extra": "value"},
    )

    row = annotation_record_to_row(record)

    assert list(row.keys())[:4] == ["comment_id", "judge_id", "provider", "model"]
    assert row["provider"] == "openai"
    assert row["model"] == "gpt-5.4"
    assert "source_type" not in row
    assert "metadata" not in row


def test_annotation_record_to_row_includes_metadata_when_requested() -> None:
    record = MHSAnnotationRecord(
        comment_id=1,
        judge_id="openai:gpt-5.4",
        source_type="model",
        text="test comment",
        target_groups=["I"],
        sentiment="C",
        respect="C",
        insult="A",
        humiliate="A",
        status="C",
        dehumanize="A",
        violence="A",
        genocide="A",
        attack_defend="C",
        hate_speech="B",
        metadata={"provider": "openai", "model": "gpt-5.4", "extra": "value"},
    )

    row = annotation_record_to_row(record, include_metadata=True)

    assert "metadata" in row
