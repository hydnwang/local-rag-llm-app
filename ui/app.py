import httpx
import streamlit as st

API_URL = "http://localhost:8000"

st.title("RAG Q&A Demo")

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
        for i, source in enumerate(data["sources"], start=1):
            with st.expander(f"Source {i}: {source['file_name']} ▸"):
                st.write(source["text"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": data["answer"],
            "sources": data["sources"],
        }
    )
