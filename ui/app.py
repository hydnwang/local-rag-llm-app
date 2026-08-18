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
                st.write(f"{i}. [{source['file_name']}] {source['text'][:200]}...")


if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

with st.sidebar:
    st.header("Upload a document")
    uploaded_file = st.file_uploader(
        "Choose a .txt file", type=["txt"], key=st.session_state.uploader_key
    )
    if uploaded_file is not None and st.button("Ingest"):
        with st.spinner("Ingesting..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            response = httpx.post(f"{API_URL}/ingest", files=files, timeout=120.0)
            response.raise_for_status()
        st.success(f"Ingested: {uploaded_file.name}")
        st.session_state.uploader_key += 1
        st.rerun()

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
                with st.expander(f"Source {i}: {source['file_name']} ▸"):
                    st.write(source["text"])

if question := st.chat_input("Ask a question about your documents"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = httpx.post(
                f"{API_URL}/query", json={"question": question}, timeout=120.0
            )
            response.raise_for_status()
            data = response.json()

        st.write(data["answer"])
        st.caption(f"Path: {data['path_taken']} · Retries: {data['retry_count']}")
        st.caption(f"Assess reasoning: {data['assess_reasoning']}")
        render_retrieval_history(data["retrieval_history"])
        for i, source in enumerate(data["sources"], start=1):
            with st.expander(f"Source {i}: {source['file_name']} ▸"):
                st.write(source["text"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": data["answer"],
            "sources": data["sources"],
            "path_taken": data["path_taken"],
            "assess_reasoning": data["assess_reasoning"],
            "retry_count": data["retry_count"],
            "retrieval_history": data["retrieval_history"],
        }
    )
