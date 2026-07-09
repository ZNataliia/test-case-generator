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
                    "priority": {"type": "string", "enum": ["High", "Medium", "Low"]},
                    "preconditions": {"type": "string"},
                    "steps": {"type": "string"},
                    "expected_result": {"type": "string"},
                },
                "required": [
                    "id",
                    "title",
                    "type",
                    "priority",
                    "preconditions",
                    "steps",
                    "expected_result",
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

# Single source of truth for the table's column order -- derived from the
# schema so it can never drift out of sync with what Claude is asked to return.
COLUMNS = list(TEST_CASE_SCHEMA["properties"]["test_cases"]["items"]["properties"].keys())


def _display_label(field_name: str) -> str:
    if field_name == "id":
        return "ID"
    return field_name.replace("_", " ").title()


def build_results_dataframe(test_cases: list[dict]) -> pd.DataFrame:
    """Builds the results table. Raises KeyError if a test case is missing a required column."""
    df = pd.DataFrame(test_cases)[COLUMNS]
    df.columns = [_display_label(c) for c in df.columns]
    return df


def build_prompt(requirement: str, count: int) -> str:
    template = PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    # str.format() only scans the template's own {requirement}/{count}
    # markers -- unlike chained .replace() calls, it never re-scans the
    # substituted values themselves, so a requirement that happens to
    # contain the literal text "{count}" or "{requirement}" is inserted
    # verbatim instead of being partially rewritten.
    return template.format(requirement=requirement, count=count)


def _accumulate_stream(stream, cancel_event: threading.Event) -> tuple[str, bool]:
    """Reads text deltas from an open message stream, stopping early if
    cancel_event is set. Returns (accumulated_text, was_cancelled).

    was_cancelled is only True if we actually stopped before the stream
    finished. If the loop exhausts naturally, the full response has already
    been generated (and billed) in its entirety, so it's returned as a
    complete result even if cancel_event happened to be set right at the
    end -- discarding it at that point wouldn't save a single token.

    Only touches stream.text_stream and the event, so it can be unit tested
    with a trivial fake stream instead of the real Anthropic SDK.
    """
    text_parts: list[str] = []
    for text in stream.text_stream:
        if cancel_event.is_set():
            return "".join(text_parts), True
        text_parts.append(text)
    return "".join(text_parts), False


def _set_error(result_box: dict, lock: threading.Lock, message: str) -> None:
    # Locked so this can't race with the main thread's Stop handler (which
    # writes "cancelled" under the same lock) -- without the lock, a
    # check-then-act on result_box["status"] could let a deliberate Stop be
    # silently overwritten by an error that arrives a moment later.
    with lock:
        if result_box.get("status") != "cancelled":
            result_box["status"] = "error"
            result_box["error_message"] = message


def _run_generation(
    requirement: str,
    count: int,
    result_box: dict,
    cancel_event: threading.Event,
    lock: threading.Lock,
) -> None:
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
        max_tokens = min(16000, max(2000, count * 1200))

        with client.messages.stream(
            model=MODEL,
            max_tokens=max_tokens,
            output_config={"format": {"type": "json_schema", "schema": TEST_CASE_SCHEMA}},
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            text, cancelled = _accumulate_stream(stream, cancel_event)
            final_message = None
            if not cancelled:
                # Safe to call now: the stream is already fully drained by
                # _accumulate_stream, so this returns immediately instead of
                # blocking to read more (it would otherwise defeat
                # cancellation by draining the rest of the response).
                final_message = stream.get_final_message()

        if cancelled:
            return  # main thread already set status to "cancelled"

        if final_message is not None and final_message.stop_reason == "max_tokens":
            _set_error(
                result_box,
                lock,
                f"Claude's response was cut off after hitting the {max_tokens}-token limit "
                f"before finishing all {count} test cases. Try a smaller count, or a shorter requirement.",
            )
            return

        data = json.loads(text)
        with lock:
            if result_box.get("status") != "cancelled":
                result_box["status"] = "done"
                result_box["result"] = data
    except anthropic.AuthenticationError:
        _set_error(result_box, lock, "Authentication failed. Check that your ANTHROPIC_API_KEY in `.env` is valid.")
    except anthropic.RateLimitError:
        _set_error(result_box, lock, "Rate limited by the Anthropic API. Wait a moment and try again.")
    except anthropic.APIStatusError as e:
        _set_error(result_box, lock, f"Anthropic API error ({e.status_code}): {e.message}")
    except anthropic.APIConnectionError:
        # A connection drop while streaming (including one caused by our own
        # cancellation closing it) can also raise this -- the cancelled-status
        # guard inside _set_error keeps it from clobbering a deliberate Stop.
        _set_error(result_box, lock, "Could not connect to the Anthropic API. Check your internet connection.")
    except (json.JSONDecodeError, StopIteration):
        _set_error(result_box, lock, "Claude's response could not be parsed as JSON. Try generating again.")
    except Exception as e:  # noqa: BLE001 - surface anything unexpected instead of hanging silently
        _set_error(result_box, lock, f"Unexpected error: {e}")


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
                "Stops sending you further tokens once the next chunk arrives -- if Claude "
                "hasn't started responding yet, Stop won't take effect until it does. Any "
                "tokens already generated by then are still billed."
            ),
        )

    if generate_clicked:
        cancel_event = threading.Event()
        lock = threading.Lock()
        result_box = {"status": "running", "result": None, "error_message": None}
        st.session_state.cancel_event = cancel_event
        st.session_state.status_lock = lock
        st.session_state.gen_state = result_box
        thread = threading.Thread(
            target=_run_generation,
            args=(requirement, num_test_cases, result_box, cancel_event, lock),
            daemon=True,
        )
        thread.start()

    if stop_clicked:
        lock = st.session_state.get("status_lock")
        if lock is not None:
            with lock:
                st.session_state.gen_state["status"] = "cancelled"
        else:
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
            try:
                df = build_results_dataframe(test_cases)
            except KeyError as e:
                st.error(f"Claude's response was missing an expected field ({e}). Try generating again.")
            else:
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
