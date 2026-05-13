from mhs_llms.retry_direct import (
    _parse_openai_compatible_streaming_lines,
    _parse_together_streaming_lines,
    _merge_processed_rows,
    _merge_processing_errors,
    _merge_raw_results,
    _select_retry_manifest_rows,
)


def test_select_retry_manifest_rows_keeps_manifest_order() -> None:
    manifest_rows = [
        {"custom_id": "comment-1", "comment_id": 1, "text": "one"},
        {"custom_id": "comment-2", "comment_id": 2, "text": "two"},
        {"custom_id": "comment-3", "comment_id": 3, "text": "three"},
    ]
    error_rows = [
        {"custom_id": "comment-3", "error": "bad"},
        {"custom_id": "comment-1", "error": "bad"},
    ]

    selected_rows = _select_retry_manifest_rows(
        manifest_rows=manifest_rows,
        error_rows=error_rows,
    )

    assert [row["custom_id"] for row in selected_rows] == ["comment-1", "comment-3"]


def test_merge_processed_rows_prefers_retry_rows_and_preserves_manifest_order() -> None:
    original_rows = [
        {"comment_id": 1, "judge_id": "model-a", "sentiment": "A"},
        {"comment_id": 3, "judge_id": "model-a", "sentiment": "C"},
    ]
    retry_rows = [
        {"comment_id": 2, "judge_id": "model-a", "sentiment": "B"},
        {"comment_id": 3, "judge_id": "model-a", "sentiment": "D"},
    ]
    manifest_rows = [
        {"comment_id": 1, "custom_id": "comment-1", "text": "one"},
        {"comment_id": 2, "custom_id": "comment-2", "text": "two"},
        {"comment_id": 3, "custom_id": "comment-3", "text": "three"},
    ]

    merged_rows = _merge_processed_rows(
        original_rows=original_rows,
        retry_rows=retry_rows,
        manifest_rows=manifest_rows,
    )

    assert [row["comment_id"] for row in merged_rows] == [1, 2, 3]
    assert merged_rows[2]["sentiment"] == "D"


def test_merge_raw_results_prefers_retry_rows_and_preserves_manifest_order() -> None:
    original_rows = [
        {"custom_id": "comment-1", "response_text": "old-1"},
        {"custom_id": "comment-3", "response_text": "old-3"},
    ]
    retry_rows = [
        {"custom_id": "comment-2", "response_text": "new-2"},
        {"custom_id": "comment-3", "response_text": "new-3"},
    ]
    manifest_rows = [
        {"custom_id": "comment-1", "comment_id": 1},
        {"custom_id": "comment-2", "comment_id": 2},
        {"custom_id": "comment-3", "comment_id": 3},
    ]

    merged_rows = _merge_raw_results(
        original_rows=original_rows,
        retry_rows=retry_rows,
        manifest_rows=manifest_rows,
    )

    assert [row["custom_id"] for row in merged_rows] == [
        "comment-1",
        "comment-2",
        "comment-3",
    ]
    assert merged_rows[2]["response_text"] == "new-3"


def test_merge_processing_errors_keeps_only_unresolved_retry_failures() -> None:
    original_error_rows = [
        {"custom_id": "comment-1", "error": "old-1"},
        {"custom_id": "comment-2", "comment_id": 2, "error": "old-2"},
        {"custom_id": "comment-3", "comment_id": 3, "error": "old-3"},
    ]
    retry_rows = [
        {"comment_id": 1, "judge_id": "model-a"},
        {"comment_id": 2, "judge_id": "model-a"},
    ]
    retry_errors = [
        {"custom_id": "comment-3", "comment_id": 3, "error": "still-bad"},
    ]

    remaining_rows = _merge_processing_errors(
        original_error_rows=original_error_rows,
        retry_rows=retry_rows,
        retry_errors=retry_errors,
    )

    assert remaining_rows == retry_errors


def test_parse_together_streaming_lines_combines_delta_content() -> None:
    provider_response, response_text = _parse_together_streaming_lines(
        [
            b'data: {"choices":[{"delta":{"content":"{\\"target_groups\\":"}}]}\n',
            b'data: {"choices":[{"delta":{"content":" [\\"I\\"]}"}}]}\n',
            b"data: [DONE]\n",
        ]
    )

    assert response_text == '{"target_groups": ["I"]}'
    assert provider_response["stream"] is True
    assert provider_response["choices"][0]["message"]["content"] == response_text


def test_parse_openai_compatible_streaming_lines_combines_delta_content() -> None:
    provider_response, response_text = _parse_openai_compatible_streaming_lines(
        [
            b'data: {"choices":[{"delta":{"role":"assistant"}}]}\n',
            b'data: {"choices":[{"delta":{"content":"{\\"hate_speech\\":"}}]}\n',
            b'data: {"choices":[{"delta":{"content":" \\"B\\"}"}}]}\n',
            b"data: [DONE]\n",
        ]
    )

    assert response_text == '{"hate_speech": "B"}'
    assert provider_response["stream"] is True
    assert provider_response["choices"][0]["message"]["content"] == response_text
