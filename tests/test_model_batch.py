from mhs_llms.model_batch import (
    _extract_anthropic_result,
    _extract_google_result,
    _extract_openai_result,
    _parse_response_json,
)


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
