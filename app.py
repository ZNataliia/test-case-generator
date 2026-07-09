import json
import os
import threading
import time
from pathlib import Path

import anthropic
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-sonnet-5"
PROMPT_TEMPLATE_PATH = Path(__file__).parent / "prompts" / "template.md"

COLUMNS = [
    "id",
    "title",
    "type",
    "priority",
    "preconditions",
    "steps",
    "expected_result",
]

TEST_CASE_SCHEMA = {
    "type": "object",
    "properties": {
        "test_cases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "type": {"type": "string", "enum": ["Positive", "Negative", "Edge"]},
                    "preconditions": {"type": "string"},
                    "steps": {"type": "string"},
                    "expected_result": {"type": "string"},
                    "priority": {"type": "string", "enum": ["High", "Medium", "Low"]},
                },
                "required": [
                    "id",
                    "title",
                    "type",
                    "preconditions",
                    "steps",
                    "expected_result",
                    "priority",
                ],
                "additionalProperties": False,
            },
        },
        "ambiguities": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Parts of the requirement that were unclear and need human clarification.",
        },
    },
    "required": ["test_cases", "ambiguities"],
    "additionalProperties": False,
}


def build_prompt(requirement: str, count: int) -> str:
    template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    return template.replace("{requirement}", requirement).replace("{count}", str(count))


def _accumulate_stream(stream, cancel_event: threading.Event) -> tuple[str, bool]:
    """Reads text deltas from an open message stream, stopping early if
    cancel_event is set. Returns (accumulated_text, was_cancelled).

    Only touches stream.text_stream and the event, so it can be unit tested
    with a trivial fake stream instead of the real Anthropic SDK.
    """
    text_parts: list[str] = []
    for text in stream.text_stream:
        if cancel_event.is_set():
            return "".join(text_parts), True
        text_parts.append(text)
    return "".join(text_parts), cancel_event.is_set()


def _set_error(result_box: dict, message: str) -> None:
    # Don't overwrite a deliberate Stop with a late-arriving error caused by
    # our own cancellation closing the connection.
    if result_box.get("status") != "cancelled":
        result_box["status"] = "error"
        result_box["error_message"] = message


def _run_generation(requirement: str, count: int, result_box: dict, cancel_event: threading.Event) -> None:
    """Runs in a background thread so the main script can keep rendering a Stop button.

    Uses the streaming API (not .create()) so that when Stop closes the
    connection mid-generation, Claude actually stops generating further
    tokens server-side instead of finishing the full response regardless.
    """
    if cancel_event.is_set():
        return  # Stop was clicked before we even started

    try:
        client = anthropic.Anthropic()
        prompt = build_prompt(requirement, count)
        max_tokens = min(8192, max(2000, count * 500))

        with client.messages.stream(
            model=MODEL,
            max_tokens=max_tokens,
            output_config={"format": {"type": "json_schema", "schema": TEST_CASE_SCHEMA}},
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            text, cancelled = _accumulate_stream(stream, cancel_event)

        if cancelled:
            return  # main thread already set status to "cancelled"

        data = json.loads(text)
        if result_box.get("status") != "cancelled":
            result_box["status"] = "done"
            result_box["result"] = data
    except anthropic.AuthenticationError:
        _set_error(result_box, "Authentication failed. Check that your ANTHROPIC_API_KEY in `.env` is valid.")
    except anthropic.RateLimitError:
        _set_error(result_box, "Rate limited by the Anthropic API. Wait a moment and try again.")
    except anthropic.APIStatusError as e:
        _set_error(result_box, f"Anthropic API error ({e.status_code}): {e.message}")
    except anthropic.APIConnectionError:
        # A connection drop while streaming (including one caused by our own
        # cancellation closing it) can also raise this -- the cancelled-status
        # guard above/in _set_error keeps it from clobbering a deliberate Stop.
        _set_error(result_box, "Could not connect to the Anthropic API. Check your internet connection.")
    except (json.JSONDecodeError, StopIteration):
        _set_error(result_box, "Claude's response could not be parsed as JSON. Try generating again.")
    except Exception as e:  # noqa: BLE001 - surface anything unexpected instead of hanging silently
        _set_error(result_box, f"Unexpected error: {e}")


if __name__ == "__main__":
    st.set_page_config(page_title="Test Case Generator", page_icon="✅", layout="wide")
    st.title("✅ Test Case Generator")
    st.caption("Paste a requirement or user story and get structured test cases back from Claude.")

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key or api_key == "your-api-key-here":
        st.error(
            "No Anthropic API key found. Add `ANTHROPIC_API_KEY=<your key>` to the `.env` file "
            "in the project root, then restart the app."
        )
        st.stop()

    if "gen_state" not in st.session_state:
        st.session_state.gen_state = {"status": "idle", "result": None, "error_message": None}

    requirement = st.text_area(
        "Requirement / User Story",
        height=200,
        placeholder=(
            "e.g. As a user, I want to reset my password via email so that I can "
            "regain access to my account if I forget my password."
        ),
    )

    num_test_cases = st.selectbox("Number of test cases", options=[2, 10, 15], index=1)

    is_running = st.session_state.gen_state["status"] == "running"

    col1, col2 = st.columns([1, 1])
    with col1:
        generate_clicked = st.button(
            "Generate Test Cases",
            type="primary",
            disabled=not requirement.strip() or is_running,
        )
    with col2:
        stop_clicked = st.button(
            "Stop",
            disabled=not is_running,
            help=(
                "Stops sending you further tokens. Checked between streamed chunks, so it "
                "typically takes effect within a second or two -- not instantly -- and any "
                "tokens already generated up to that point are still billed."
            ),
        )

    if generate_clicked:
        cancel_event = threading.Event()
        result_box = {"status": "running", "result": None, "error_message": None}
        st.session_state.cancel_event = cancel_event
        st.session_state.gen_state = result_box
        thread = threading.Thread(
            target=_run_generation,
            args=(requirement, num_test_cases, result_box, cancel_event),
            daemon=True,
        )
        thread.start()

    if stop_clicked:
        st.session_state.gen_state["status"] = "cancelled"
        cancel_event = st.session_state.get("cancel_event")
        if cancel_event is not None:
            cancel_event.set()

    state = st.session_state.gen_state

    if state["status"] == "running":
        with st.spinner("Streaming test cases... click Stop to cancel."):
            time.sleep(0.5)
        st.rerun()
    elif state["status"] == "cancelled":
        st.warning("Generation stopped. Tokens generated before Stop was clicked may still have been billed.")
    elif state["status"] == "error":
        st.error(state["error_message"])
    elif state["status"] == "done":
        result = state["result"]
        ambiguities = result.get("ambiguities") or []
        if ambiguities:
            with st.expander(f"⚠️ {len(ambiguities)} ambiguity(ies) flagged in the requirement", expanded=True):
                for item in ambiguities:
                    st.markdown(f"- {item}")

        test_cases = result.get("test_cases") or []
        if test_cases:
            df = pd.DataFrame(test_cases)[COLUMNS]
            df = df.rename(
                columns={
                    "id": "ID",
                    "title": "Title",
                    "type": "Type",
                    "priority": "Priority",
                    "preconditions": "Preconditions",
                    "steps": "Steps",
                    "expected_result": "Expected Result",
                }
            )
            st.dataframe(df, use_container_width=True, hide_index=True)

            csv_bytes = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download as CSV",
                data=csv_bytes,
                file_name="test_cases.csv",
                mime="text/csv",
            )
        else:
            st.info("No test cases were generated.")
