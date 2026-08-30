import json
import os
import re

import httpx
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")
DEBUG_MODE = os.environ.get("DEBUG_MODE", "false").lower() == "true"

st.title("RAG Q&A Demo")

_UUID_PREFIX = re.compile(r"^[0-9a-f]{32}_")


def display_name(file_name: str) -> str:
    """Strips the UUID-hex collision-prevention prefix (see main.py's /ingest)
    for display purposes only. The prefixed name remains the real identity
    used on disk and in Qdrant — this never touches that value."""
    return _UUID_PREFIX.sub("", file_name)


def render_retrieval_history(history):
    if not history:
        return
    with st.expander(f"🔁 Retrieval history ({len(history)} attempts) ▸"):
        for attempt in history:
            st.markdown(f"**Attempt {attempt['attempt']}** — query used: _{attempt['question_used']}_")
            for i, source in enumerate(attempt["sources"], start=1):
                with st.container(border=True):
                    st.caption(f"{i}. {display_name(source['file_name'])} · {source['content_type']} · score={source['score']:.3f}")
                    st.markdown(source["text"])


NODE_STATUS = {
    "retrieve": "Retrieving...",
    "assess": "Assessing context...",
    "reformulate": "Reformulating question...",
    "generate": "Generating answer...",
    "insufficient_context": "Finalizing response...",
}


def stream_query(question: str, status_placeholder, result: dict):
    """POSTs to /query (SSE) and yields answer text chunks as they arrive,
    updating status_placeholder with step progress. Writes the final 'done'
    event payload into the `result` dict passed in (mutated in place), since
    st.write_stream only consumes yielded values and discards a generator's
    return value."""
    with httpx.stream(
        "POST", f"{API_URL}/query", json={"question": question}, timeout=120.0
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            event = json.loads(line[len("data: "):])
            if event["type"] == "node":
                status_placeholder.caption(NODE_STATUS.get(event["node"], event["node"]))
            elif event["type"] == "token":
                yield event["text"]
            elif event["type"] == "done":
                result.update(event)


if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "uploads" not in st.session_state:
    st.session_state.uploads = []

with st.sidebar:
    debug_mode = st.toggle("Debug mode", value=DEBUG_MODE)

    st.header("Upload documents")
    uploaded_files = st.file_uploader(
        "Choose .txt, .pdf, or .md files",
        type=["txt", "pdf", "md"],
        accept_multiple_files=True,
        key=st.session_state.uploader_key,
    )
    if uploaded_files and st.button("Ingest"):
        status = st.empty()
        total = len(uploaded_files)
        for i, uploaded_file in enumerate(uploaded_files, start=1):
            status.info(f"Ingesting file {i} of {total}: {uploaded_file.name}")
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            response = httpx.post(f"{API_URL}/ingest", files=files, timeout=120.0)
            st.session_state.uploads.append(
                {"name": uploaded_file.name, "ok": response.status_code == 200}
            )
        status.empty()
        st.session_state.uploader_key += 1
        st.rerun()

    if st.session_state.uploads:
        st.subheader("Upload history")
        for upload in st.session_state.uploads:
            if upload["ok"]:
                st.success(upload["name"])
            else:
                st.error(upload["name"])

    st.header("Manage files")
    docs_response = httpx.get(f"{API_URL}/documents", timeout=30.0)
    files = docs_response.json()["files"]

    groups: dict[str, list[dict]] = {}
    for f in files:
        groups.setdefault(display_name(f["file_name"]), []).append(f)

    for name in sorted(groups):
        entries = sorted(groups[name], key=lambda f: f["ingested_at"] or "")
        for entry in entries:
            label = name if len(entries) == 1 else f"{name} — {entry['ingested_at']}"
            col1, col2 = st.columns([4, 1])
            col1.write(label)
            if col2.button("🗑️", key=f"delete_{entry['file_name']}"):
                httpx.delete(f"{API_URL}/documents/{entry['file_name']}", timeout=30.0)
                st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "is_streaming" not in st.session_state:
    st.session_state.is_streaming = False
interrupted = st.session_state.is_streaming
st.session_state.is_streaming = False

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if debug_mode and message["role"] == "assistant" and message.get("path_taken"):
            with st.expander("🐛 Debug info ▸"):
                st.caption(f"Path: {message['path_taken']} · Retries: {message['retry_count']}")
                if message.get("assess_reasoning"):
                    st.caption(f"Assess reasoning: {message['assess_reasoning']}")
        if debug_mode and message["role"] == "assistant" and message.get("retrieval_history"):
            render_retrieval_history(message["retrieval_history"])
        if message["role"] == "assistant" and message.get("sources"):
            for i, source in enumerate(message["sources"], start=1):
                with st.expander(f"Source {i}: {display_name(source['file_name'])} · {source['content_type']} ▸"):
                    st.write(source["text"])

pending_question = st.session_state.pending_question

if pending_question is None:
    # Phase 1: a fresh submission. Stash it, draw the box disabled,
    # and rerun immediately — before doing any slow backend work —
    # so the disabled state has a chance to reach the browser first.
    question = st.chat_input("Ask a question about your documents", disabled=False)
    if question:
        if interrupted:
            st.warning("Please wait for the current response to finish before sending another question.")
        else:
            st.session_state.pending_question = question
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()
else:
    # Phase 2: the disabled box was already flushed on the prior run.
    if interrupted:
        # A previous Phase 2 run was killed mid-stream by this very
        # rerun (Streamlit tears down the running script on new input).
        # Don't silently retry the stale question — drop it and warn.
        st.session_state.pending_question = None
        st.chat_input("Ask a question about your documents", disabled=False)
        st.warning("Please wait for the current response to finish before sending another question.")
    else:
        st.chat_input("Ask a question about your documents", disabled=True)

        st.session_state.is_streaming = True
        st.session_state.pending_question = None
        try:
            with st.chat_message("assistant"):
                status = st.empty()
                result = {}
                answer = st.write_stream(stream_query(pending_question, status, result))
                status.empty()

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": result["sources"],
                    "path_taken": result["path_taken"],
                    "assess_reasoning": result["assess_reasoning"],
                    "retry_count": result["retry_count"],
                    "retrieval_history": result["retrieval_history"],
                }
            )
        finally:
            st.session_state.is_streaming = False
        st.rerun()