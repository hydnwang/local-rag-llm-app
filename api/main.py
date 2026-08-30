import json
import logging
import shutil
import sys
import time
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import httpx
from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
from qdrant_client import QdrantClient
from qdrant_client.http.models import FieldCondition, Filter, FilterSelector, MatchValue

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    COLLECTION_NAME,
    DAGSTER_HOST,
    DAGSTER_PORT,
    EMBED_MODEL_NAME,
    QDRANT_URL,
    TOP_K,
)
from api.graph import NODE_DURATION_SECONDS, build_generate_prompt, build_graph, sources_from_nodes, stream_ollama
from ingestion.naming import make_ingest_name

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

INGEST_DIR = Path(__file__).resolve().parent.parent / "files" / "ingest"
INGEST_DIR.mkdir(parents=True, exist_ok=True)

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


@app.get("/health")
async def health() -> JSONResponse:
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{QDRANT_URL}/collections")
        qdrant_ok = resp.status_code == 200
    except httpx.HTTPError:
        qdrant_ok = False

    status = "ok" if qdrant_ok else "degraded"
    return JSONResponse(
        status_code=200 if qdrant_ok else 503,
        content={"status": status, "qdrant": "ok" if qdrant_ok else "unreachable"},
    )


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/documents")
async def list_documents() -> dict:
    if not client.collection_exists(COLLECTION_NAME):
        return {"files": []}

    seen: dict[str, str] = {}
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            with_payload=["file_name", "ingested_at"],
            with_vectors=False,
            limit=1000,
            offset=offset,
        )
        for point in points:
            file_name = point.payload.get("file_name")
            if file_name and file_name not in seen:
                seen[file_name] = point.payload.get("ingested_at")
        if offset is None:
            break

    files = sorted(
        [{"file_name": name, "ingested_at": ts} for name, ts in seen.items()],
        key=lambda d: (d["file_name"], d["ingested_at"] or ""),
    )
    return {"files": files}


@app.delete("/documents/{file_name}")
async def delete_document(file_name: str) -> dict:
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="file_name", match=MatchValue(value=file_name))])
        ),
    )
    logger.info(f"DELETE /documents/{file_name}")
    return {"status": "ok", "file_name": file_name}


@app.post("/query")
async def query(request: QueryRequest) -> StreamingResponse:
    logger.info(f"POST /query | question={request.question}")

    async def event_stream():
        start = time.monotonic()
        initial_input = {
            "question": request.question,
            "original_question": request.question,
            "retry_count": 0,
        }
        final_state: dict = dict(initial_input)

        async for update in rag_graph.astream(initial_input, stream_mode="updates"):
            node_name, node_output = next(iter(update.items()))
            yield f"data: {json.dumps({'type': 'node', 'node': node_name})}\n\n"
            if "retrieval_history" in node_output:
                final_state["retrieval_history"] = final_state.get("retrieval_history", []) + node_output["retrieval_history"]
                node_output = {k: v for k, v in node_output.items() if k != "retrieval_history"}
            final_state.update(node_output)

        assessment = final_state.get("assessment", "insufficient")

        if assessment.startswith("pass"):
            yield f"data: {json.dumps({'type': 'node', 'node': 'generate'})}\n\n"
            gen_start = time.monotonic()
            prompt = build_generate_prompt(final_state)
            answer_parts = []
            async with httpx.AsyncClient(timeout=120.0) as client:
                async for chunk in stream_ollama(client, prompt):
                    answer_parts.append(chunk)
                    yield f"data: {json.dumps({'type': 'token', 'text': chunk})}\n\n"
            NODE_DURATION_SECONDS.labels(node="generate").observe(time.monotonic() - gen_start)
            answer = "".join(answer_parts)
            # Sources shown are the top-3 by retrieval similarity score, which assess
            # already validated as sufficient to answer the question — not generate's
            # self-report, which proved unreliable (~25% of citations pointed at
            # chunks that didn't support the answer).
            sources = sources_from_nodes(final_state["nodes"][:3])
        else:
            answer = final_state["answer"]
            sources = final_state["sources"]
            yield f"data: {json.dumps({'type': 'token', 'text': answer})}\n\n"

        elapsed = time.monotonic() - start
        logger.info(
            f"POST /query | status=200 | path={assessment} | "
            f"retries={final_state['retry_count']} | elapsed={elapsed:.1f}s"
        )
        yield "data: " + json.dumps(
            {
                "type": "done",
                "answer": answer,
                "sources": sources,
                "path_taken": assessment,
                "assess_reasoning": final_state.get("assess_reasoning", ""),
                "retry_count": final_state["retry_count"],
                "retrieval_history": (
                    final_state["retrieval_history"] if final_state["retry_count"] > 0 else []
                ),
            }
        ) + "\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.post("/ingest")
async def ingest_file(file: UploadFile) -> dict:
    import asyncio

    from dagster_graphql import DagsterGraphQLClient
    from dagster._core.storage.dagster_run import DagsterRunStatus

    logger.info(f"POST /ingest | file={file.filename}")
    start = time.monotonic()

    # Written to a durable directory, not an ephemeral temp dir — Dagster's daemon
    # runs as a separate process and may not pick up the run immediately, so the
    # file's lifetime can't be tied to this request's own control flow. It's only
    # deleted below once Dagster confirms the run reached a terminal state.
    unique_name = make_ingest_name(file.filename)
    ingest_path = INGEST_DIR / unique_name
    with open(ingest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        client = DagsterGraphQLClient(DAGSTER_HOST, port_number=DAGSTER_PORT)
        run_id = client.submit_job_execution(
            "ingest_job",
            run_config={"ops": {"raw_documents": {"config": {"file_path": str(ingest_path)}}}},
        )
    except Exception:
        logger.exception(f"POST /ingest | failed to submit run | file={file.filename}")
        ingest_path.unlink(missing_ok=True)
        return JSONResponse(
            status_code=502,
            content={"status": "error", "file_name": file.filename, "detail": "Failed to submit ingest job"},
        )

    terminal_states = {
        DagsterRunStatus.SUCCESS,
        DagsterRunStatus.FAILURE,
        DagsterRunStatus.CANCELED,
    }
    try:
        status = client.get_run_status(run_id)
        while status not in terminal_states:
            await asyncio.sleep(1)
            status = client.get_run_status(run_id)
    except Exception:
        # We lost the ability to confirm the run's outcome — leave the file in
        # place rather than guess, since Dagster may still be processing it.
        logger.exception(f"POST /ingest | failed to poll run status | file={file.filename} | run_id={run_id}")
        return JSONResponse(
            status_code=502,
            content={"status": "error", "file_name": file.filename, "run_id": run_id, "detail": "Lost contact while polling run status"},
        )
    finally:
        elapsed = time.monotonic() - start

    if status != DagsterRunStatus.SUCCESS:
        logger.info(f"POST /ingest | status=500 | file={file.filename} | elapsed={elapsed:.1f}s | run_id={run_id}")
        ingest_path.unlink(missing_ok=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "file_name": file.filename, "run_id": run_id},
        )

    ingest_path.unlink(missing_ok=True)

    logger.info(f"POST /ingest | status=200 | file={file.filename} | elapsed={elapsed:.1f}s")
    return {"status": "ok", "file_name": file.filename, "run_id": run_id}
