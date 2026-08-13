import sys
from pathlib import Path

import httpx
from fastapi import FastAPI
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import BaseModel
from qdrant_client import QdrantClient

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    COLLECTION_NAME,
    EMBED_MODEL_NAME,
    OLLAMA_MODEL,
    OLLAMA_URL,
    QDRANT_URL,
    TOP_K,
)

app = FastAPI()

client = QdrantClient(url=QDRANT_URL)
vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
retriever = index.as_retriever(similarity_top_k=TOP_K)


class QueryRequest(BaseModel):
    question: str


class Source(BaseModel):
    file_name: str
    text: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    nodes = retriever.retrieve(request.question)

    context = "\n\n".join(node.get_content() for node in nodes)
    prompt = (
        f"Answer the question using only the context below. "
        f"If the context doesn't contain the answer, say so.\n\n"
        f"Context:\n{context}\n\nQuestion: {request.question}"
    )

    async with httpx.AsyncClient(timeout=120.0) as http_client:
        response = await http_client.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        answer = response.json()["response"]

    sources = [
        Source(
            file_name=node.metadata.get("file_name", "unknown"),
            text=node.get_content(),
        )
        for node in nodes
    ]

    return QueryResponse(answer=answer, sources=sources)
