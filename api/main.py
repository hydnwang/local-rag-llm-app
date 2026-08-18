import shutil
import sys
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from pydantic import BaseModel
from qdrant_client import QdrantClient

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    COLLECTION_NAME,
    DAGSTER_HOST,
    DAGSTER_PORT,
    EMBED_MODEL_NAME,
    QDRANT_URL,
    TOP_K,
)
from api.graph import build_graph

app = FastAPI()

client = QdrantClient(url=QDRANT_URL)
vector_store = QdrantVectorStore(client=client, collection_name=COLLECTION_NAME)
embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
retriever = index.as_retriever(similarity_top_k=TOP_K)
rag_graph = build_graph(retriever)


class QueryRequest(BaseModel):
    question: str


class Source(BaseModel):
    file_name: str
    text: str


class RetrievalAttempt(BaseModel):
    attempt: int
    question_used: str
    sources: list[Source]


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    path_taken: str
    assess_reasoning: str
    retry_count: int
    retrieval_history: list[RetrievalAttempt] = []


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    result = rag_graph.invoke(
        {
            "question": request.question,
            "original_question": request.question,
            "retry_count": 0,
        }
    )
    return QueryResponse(
        answer=result["answer"],
        sources=[Source(**s) for s in result["sources"]],
        path_taken=result["assessment"],
        assess_reasoning=result["assess_reasoning"],
        retry_count=result["retry_count"],
        retrieval_history=(
            [RetrievalAttempt(**a) for a in result["retrieval_history"]]
            if result["retry_count"] > 0
            else []
        ),
    )


@app.post("/ingest")
async def ingest_file(file: UploadFile) -> dict:
    import asyncio

    from dagster_graphql import DagsterGraphQLClient
    from dagster._core.storage.dagster_run import DagsterRunStatus

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / file.filename
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        client = DagsterGraphQLClient(DAGSTER_HOST, port_number=DAGSTER_PORT)
        run_id = client.submit_job_execution(
            "ingest_job",
            run_config={"ops": {"raw_documents": {"config": {"file_path": str(tmp_path)}}}},
        )

        terminal_states = {
            DagsterRunStatus.SUCCESS,
            DagsterRunStatus.FAILURE,
            DagsterRunStatus.CANCELED,
        }
        status = client.get_run_status(run_id)
        while status not in terminal_states:
            await asyncio.sleep(1)
            status = client.get_run_status(run_id)

    if status != DagsterRunStatus.SUCCESS:
        return {"status": "error", "file_name": file.filename, "run_id": run_id}

    return {"status": "ok", "file_name": file.filename, "run_id": run_id}
