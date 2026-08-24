import json

import httpx
import streamlit as st

API_URL = "http://localhost:8000"

st.title("RAG Q&A Demo")


def render_retrieval_history(history):
    if not history:
        return
    with st.expander(f"🔁 Retrieval history ({len(history)} attempts) ▸"):
        for attempt in history:
            st.markdown(f"**Attempt {attempt['attempt']}** — query used: _{attempt['question_used']}_")
            for i, source in enumerate(attempt["sources"], start=1):
                st.write(f"{i}. [{source['file_name']} · {source['content_type']}] {source['text'][:200]}...")


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

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if message["role"] == "assistant" and message.get("path_taken"):
            st.caption(f"Path: {message['path_taken']} · Retries: {message['retry_count']}")
            if message.get("assess_reasoning"):
                st.caption(f"Assess reasoning: {message['assess_reasoning']}")
        if message["role"] == "assistant" and message.get("retrieval_history"):
            render_retrieval_history(message["retrieval_history"])
        if message["role"] == "assistant" and message.get("sources"):
            for i, source in enumerate(message["sources"], start=1):
                with st.expander(f"Source {i}: {source['file_name']} · {source['content_type']} ▸"):
                    st.write(source["text"])

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        status = st.empty()
        result = {}
        answer = st.write_stream(stream_query(question, status, result))
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
    st.rerun()
