import json
import threading

import jsonschema
import pytest

from app import TEST_CASE_SCHEMA, build_prompt, _accumulate_stream


class _FakeStream:
    """Mimics the slice of anthropic's MessageStream interface our code uses."""

    def __init__(self, chunks):
        self.text_stream = iter(chunks)


def test_build_prompt_substitutes_requirement_and_count():
    prompt = build_prompt("Users can log in with email and password.", 5)
    assert "Users can log in with email and password." in prompt
    assert "5" in prompt
    assert "{requirement}" not in prompt
    assert "{count}" not in prompt


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


def test_accumulate_stream_detects_cancellation_set_after_last_chunk():
    # Cancellation can land right as the final chunk arrives, after the loop
    # exits normally rather than via break -- caller still needs to know.
    event = threading.Event()

    def chunks():
        yield "ab"
        event.set()

    text, cancelled = _accumulate_stream(_FakeStream(chunks()), event)
    assert text == "ab"
    assert cancelled is True


VALID_TEST_CASE = {
    "test_cases": [
        {
            "id": "TC-01",
            "title": "Login with valid credentials",
            "type": "Positive",
            "preconditions": "User has a registered account.",
            "steps": "1. Go to login page. 2. Enter valid email/password. 3. Submit.",
            "expected_result": "User is redirected to the dashboard.",
            "priority": "High",
        }
    ],
    "ambiguities": [],
}


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
