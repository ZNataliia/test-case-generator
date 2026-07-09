import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import anthropic
import httpx
import jsonschema
import pytest

from app import (
    COLUMNS,
    TEST_CASE_SCHEMA,
    _accumulate_stream,
    _display_label,
    _run_generation,
    _set_error,
    build_prompt,
    build_results_dataframe,
)


class _FakeStream:
    """Mimics the slice of anthropic's MessageStream interface our code uses."""

    def __init__(self, chunks):
        self.text_stream = iter(chunks)


class _FakeMessageStream:
    """Mimics the context-manager + text_stream + get_final_message() surface
    of anthropic's client.messages.stream(...) that _run_generation uses."""

    def __init__(self, chunks, stop_reason="end_turn"):
        self.text_stream = iter(chunks)
        self._stop_reason = stop_reason

    def get_final_message(self):
        return SimpleNamespace(stop_reason=self._stop_reason)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _client_returning(chunks, stop_reason="end_turn"):
    client = MagicMock()
    client.messages.stream.return_value = _FakeMessageStream(chunks, stop_reason)
    return client


VALID_TEST_CASE_ITEM = {
    "id": "TC-01",
    "title": "Login with valid credentials",
    "type": "Positive",
    "preconditions": "User has a registered account.",
    "steps": "1. Go to login page. 2. Enter valid email/password. 3. Submit.",
    "expected_result": "User is redirected to the dashboard.",
    "priority": "High",
}

VALID_TEST_CASE = {
    "test_cases": [VALID_TEST_CASE_ITEM],
    "ambiguities": [],
}


# --- build_prompt ---------------------------------------------------------


def test_build_prompt_substitutes_requirement_and_count():
    prompt = build_prompt("Users can log in with email and password.", 5)
    assert "Users can log in with email and password." in prompt
    assert "5" in prompt
    assert "{requirement}" not in prompt
    assert "{count}" not in prompt


def test_build_prompt_preserves_literal_braces_in_requirement():
    # A requirement that itself contains the literal placeholder text used
    # to corrupt this via chained .replace() -- must survive untouched.
    requirement = "The field must show exactly {count} recent orders, per the {requirement} spec."
    prompt = build_prompt(requirement, 5)
    assert "The field must show exactly {count} recent orders, per the {requirement} spec." in prompt
    # The template's own {count} placeholder (not the one inside the
    # requirement text) must still have been substituted with the real count.
    assert "Generate exactly 5 test cases in total." in prompt


# --- _accumulate_stream ----------------------------------------------------


def test_accumulate_stream_returns_full_text_when_not_cancelled():
    event = threading.Event()
    text, cancelled = _accumulate_stream(_FakeStream(["ab", "cd", "ef"]), event)
    assert text == "abcdef"
    assert cancelled is False


def test_accumulate_stream_stops_early_when_cancelled_mid_stream():
    event = threading.Event()

    def chunks():
        yield "ab"
        event.set()  # simulate the user clicking Stop partway through
        yield "cd"
        yield "ef"

    text, cancelled = _accumulate_stream(_FakeStream(chunks()), event)
    assert text == "ab"
    assert cancelled is True


def test_accumulate_stream_treats_full_completion_as_not_cancelled():
    # If the stream finishes naturally, the whole response is already
    # generated and billed -- discarding it saves nothing, so it must be
    # reported as a complete result even if Stop was clicked right at the end.
    event = threading.Event()

    def chunks():
        yield "ab"
        event.set()

    text, cancelled = _accumulate_stream(_FakeStream(chunks()), event)
    assert text == "ab"
    assert cancelled is False


# --- schema ------------------------------------------------------------


def test_schema_accepts_a_valid_document():
    jsonschema.validate(VALID_TEST_CASE, TEST_CASE_SCHEMA)


def test_schema_rejects_missing_required_field():
    bad = json.loads(json.dumps(VALID_TEST_CASE))
    del bad["test_cases"][0]["priority"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, TEST_CASE_SCHEMA)


def test_schema_rejects_invalid_type_enum_value():
    bad = json.loads(json.dumps(VALID_TEST_CASE))
    bad["test_cases"][0]["type"] = "Regression"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, TEST_CASE_SCHEMA)


def test_columns_matches_schema_fields():
    item_schema = TEST_CASE_SCHEMA["properties"]["test_cases"]["items"]
    assert set(COLUMNS) == set(item_schema["required"])
    assert set(COLUMNS) == set(item_schema["properties"].keys())


# --- _display_label / build_results_dataframe -------------------------


def test_display_label_special_cases_id():
    assert _display_label("id") == "ID"


def test_display_label_title_cases_snake_case_fields():
    assert _display_label("expected_result") == "Expected Result"
    assert _display_label("title") == "Title"
    assert _display_label("preconditions") == "Preconditions"


def test_build_results_dataframe_success():
    df = build_results_dataframe([VALID_TEST_CASE_ITEM])
    assert list(df.columns) == ["ID", "Title", "Type", "Priority", "Preconditions", "Steps", "Expected Result"]
    assert df.iloc[0]["ID"] == "TC-01"


def test_build_results_dataframe_raises_keyerror_on_missing_field():
    incomplete = dict(VALID_TEST_CASE_ITEM)
    del incomplete["priority"]
    with pytest.raises(KeyError):
        build_results_dataframe([incomplete])


# --- _set_error ----------------------------------------------------------


def test_set_error_writes_when_not_cancelled():
    result_box = {"status": "running", "result": None, "error_message": None}
    _set_error(result_box, threading.Lock(), "boom")
    assert result_box["status"] == "error"
    assert result_box["error_message"] == "boom"


def test_set_error_does_not_overwrite_cancelled_status():
    result_box = {"status": "cancelled", "result": None, "error_message": None}
    _set_error(result_box, threading.Lock(), "boom")
    assert result_box["status"] == "cancelled"
    assert result_box["error_message"] is None


# --- _run_generation -------------------------------------------------------


@patch("app.anthropic.Anthropic")
def test_run_generation_success_writes_done_with_parsed_result(mock_anthropic_cls):
    mock_anthropic_cls.return_value = _client_returning([json.dumps(VALID_TEST_CASE)])

    result_box = {"status": "running", "result": None, "error_message": None}
    _run_generation("some requirement", 2, result_box, threading.Event(), threading.Lock())

    assert result_box["status"] == "done"
    assert result_box["result"] == VALID_TEST_CASE


@patch("app.anthropic.Anthropic")
def test_run_generation_skips_api_call_when_already_cancelled(mock_anthropic_cls):
    result_box = {"status": "running", "result": None, "error_message": None}
    cancel_event = threading.Event()
    cancel_event.set()

    _run_generation("some requirement", 2, result_box, cancel_event, threading.Lock())

    mock_anthropic_cls.assert_not_called()
    assert result_box["status"] == "running"


@patch("app.anthropic.Anthropic")
def test_run_generation_reports_json_parse_failure(mock_anthropic_cls):
    mock_anthropic_cls.return_value = _client_returning(["not valid json"])

    result_box = {"status": "running", "result": None, "error_message": None}
    _run_generation("some requirement", 2, result_box, threading.Event(), threading.Lock())

    assert result_box["status"] == "error"
    assert "could not be parsed as JSON" in result_box["error_message"]


@patch("app.anthropic.Anthropic")
def test_run_generation_reports_max_tokens_truncation_specifically(mock_anthropic_cls):
    # Even though this text happens to be valid JSON, stop_reason=="max_tokens"
    # must short-circuit with a specific, actionable message before parsing.
    mock_anthropic_cls.return_value = _client_returning(
        [json.dumps({"test_cases": [], "ambiguities": []})], stop_reason="max_tokens"
    )

    result_box = {"status": "running", "result": None, "error_message": None}
    _run_generation("some requirement", 15, result_box, threading.Event(), threading.Lock())

    assert result_box["status"] == "error"
    assert "cut off" in result_box["error_message"]
    assert "15" in result_box["error_message"]


@patch("app.anthropic.Anthropic")
def test_run_generation_reports_connection_error(mock_anthropic_cls):
    client = MagicMock()
    client.messages.stream.side_effect = anthropic.APIConnectionError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )
    mock_anthropic_cls.return_value = client

    result_box = {"status": "running", "result": None, "error_message": None}
    _run_generation("some requirement", 2, result_box, threading.Event(), threading.Lock())

    assert result_box["status"] == "error"
    assert "Could not connect" in result_box["error_message"]


@patch("app.anthropic.Anthropic")
def test_run_generation_does_not_clobber_a_deliberate_stop(mock_anthropic_cls):
    # Regression test for the TOCTOU race: an error arriving after the main
    # thread has already recorded a Stop must not overwrite "cancelled".
    mock_anthropic_cls.return_value = _client_returning(["not valid json"])

    result_box = {"status": "cancelled", "result": None, "error_message": None}
    _run_generation("some requirement", 2, result_box, threading.Event(), threading.Lock())

    assert result_box["status"] == "cancelled"
    assert result_box["error_message"] is None
