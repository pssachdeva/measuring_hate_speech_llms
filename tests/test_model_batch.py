from pathlib import Path

import pandas as pd

from mhs_llms.batch import _build_requests, _select_comment_ids, write_processed_annotations
from mhs_llms.batch import (
    _extract_anthropic_result,
    _extract_google_result,
    _extract_openai_result,
    _parse_response_json,
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


def test_extract_google_result_reads_candidate_parts() -> None:
    entry = {
        "metadata": {"custom_id": "comment-3", "comment_id": "3"},
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


def test_write_processed_annotations_creates_and_appends_csv(tmp_path: Path) -> None:
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
    assert aggregated["comment_id"].tolist() == [1, 2, 1, 2]


def test_write_processed_annotations_appends_jsonl(tmp_path: Path) -> None:
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
    assert len(lines) == 2


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
