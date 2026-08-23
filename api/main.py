import logging
import shutil
import sys
import tempfile
import time
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from fastapi import FastAPI, UploadFile
from fastapi.responses import Response
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
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

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

log_formatter = logging.Formatter("[%(levelname)s][%(name)s] %(asctime)s | %(message)s", "%Y-%m-%d %H:%M:%S")

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)

file_handler = TimedRotatingFileHandler(LOG_DIR / "app.log", when="midnight", backupCount=14)
file_handler.setFormatter(log_formatter)

logging.basicConfig(level=logging.INFO, handlers=[stream_handler, file_handler])

logger = logging.getLogger("api")

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
    content_type: str
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


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    logger.info(f"POST /query | question={request.question}")
    start = time.monotonic()
    result = rag_graph.invoke(
        {
            "question": request.question,
            "original_question": request.question,
            "retry_count": 0,
        }
    )
    elapsed = time.monotonic() - start
    logger.info(
        f"POST /query | status=200 | path={result['assessment']} | "
        f"retries={result['retry_count']} | elapsed={elapsed:.1f}s"
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

    logger.info(f"POST /ingest | file={file.filename}")
    start = time.monotonic()

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

    elapsed = time.monotonic() - start

    if status != DagsterRunStatus.SUCCESS:
        logger.info(f"POST /ingest | status=500 | file={file.filename} | elapsed={elapsed:.1f}s")
        return {"status": "error", "file_name": file.filename, "run_id": run_id}

    logger.info(f"POST /ingest | status=200 | file={file.filename} | elapsed={elapsed:.1f}s")
    return {"status": "ok", "file_name": file.filename, "run_id": run_id}
