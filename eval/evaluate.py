"""
Unified RAG evaluation: runs each question through the real LangGraph flow
(retrieve -> assess -> generate | reformulate/retry | insufficient_context)
and scores 4 LLM-judge metrics — all in-process (no FastAPI server required).
Only Qdrant and Ollama need to be running.

Replaces run_testset.py + score_testset.py. Uses api/graph.py directly so the
eval measures the actual system, including retry/assess behavior.

Input:  eval/testset.json
Output: eval/eval_<YYYYMMDD>_<HHMMSS>.json

Usage:
    uv run python eval/evaluate.py
"""

import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import COLLECTION_NAME, EMBED_MODEL_NAME, OLLAMA_MODEL, OLLAMA_URL, QDRANT_URL, TOP_K
from api.graph import build_graph

INPUT_PATH = Path(__file__).resolve().parent / "testset.json"

# --- LLM-judge metrics (unchanged from score_testset.py) ---

JUDGE_PROMPT_TEMPLATE = """{instruction}

{content}

Respond with ONLY a JSON object in this exact format, no other text:
{{"score": <float between 0.0 and 1.0>, "reasoning": "<one sentence>"}}"""

METRIC_INSTRUCTIONS = {
    "faithfulness": (
        "You are evaluating whether an AI-generated answer is faithful to the given "
        "context — i.e. every claim in the answer is supported by the context, with no "
        "invented or hallucinated facts. Score 1.0 if fully supported, 0.0 if entirely "
        "unsupported/hallucinated, and a value in between for partial support."
    ),
    "answer_correctness": (
        "You are evaluating whether an AI-generated answer matches the expected reference "
        "answer in factual content. Score 1.0 if the answer conveys the same correct "
        "information as the reference, 0.0 if it is wrong or missing key facts, and a value "
        "in between for partial correctness."
    ),
    "context_recall": (
        "You are evaluating whether the retrieved context contains the information needed "
        "to produce the reference answer. Score 1.0 if all necessary information is present "
        "in the context, 0.0 if none of it is, and a value in between for partial coverage."
    ),
    "context_precision": (
        "You are evaluating how much of the retrieved context is actually relevant to "
        "answering the question (per the reference answer), versus irrelevant padding. "
        "Score 1.0 if all retrieved context is relevant, 0.0 if none of it is, and a value "
        "in between based on the proportion of relevant content."
    ),
}


def call_judge(client: httpx.Client, instruction: str, content: str) -> dict:
    prompt = JUDGE_PROMPT_TEMPLATE.format(instruction=instruction, content=content)

    response = client.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
    )
    response.raise_for_status()
    raw = response.json()["response"]

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"score": None, "reasoning": f"Could not parse judge output: {raw[:200]}"}

    try:
        parsed = json.loads(match.group())
        return {"score": float(parsed["score"]), "reasoning": parsed.get("reasoning", "")}
    except (json.JSONDecodeError, KeyError, ValueError):
        return {"score": None, "reasoning": f"Could not parse judge output: {raw[:200]}"}


def score_metrics(client: httpx.Client, question: str, reference: str, response_text: str, context: str) -> dict:
    return {
        "faithfulness": call_judge(
            client,
            METRIC_INSTRUCTIONS["faithfulness"],
            f"Context:\n{context}\n\nAnswer:\n{response_text}",
        ),
        "answer_correctness": call_judge(
            client,
            METRIC_INSTRUCTIONS["answer_correctness"],
            f"Question: {question}\n\nReference answer:\n{reference}\n\nGiven answer:\n{response_text}",
        ),
        "context_recall": call_judge(
            client,
            METRIC_INSTRUCTIONS["context_recall"],
            f"Question: {question}\n\nReference answer:\n{reference}\n\nRetrieved context:\n{context}",
        ),
        "context_precision": call_judge(
            client,
            METRIC_INSTRUCTIONS["context_precision"],
            f"Question: {question}\n\nReference answer:\n{reference}\n\nRetrieved context:\n{context}",
        ),
    }


def main() -> None:
    testset = json.loads(INPUT_PATH.read_text())

    print("Loading embedding model and building graph...")
    print("----- Configurations -----")
    print(f"{OLLAMA_URL=}")
    print(f"{OLLAMA_MODEL=}")
    print(f"{QDRANT_URL=}")
    print(f"{EMBED_MODEL_NAME=}")
    print(f"{COLLECTION_NAME=}")
    print(f"{TOP_K=}")
    
    qdrant_client = QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name=COLLECTION_NAME)
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    retriever = index.as_retriever(similarity_top_k=TOP_K)
    rag_graph = build_graph(retriever)

    results = []
    print("----- Evaluation Begins -----")
    with httpx.Client(timeout=120.0) as client:
        for i, item in enumerate(testset, start=1):
            question = item["user_input"]
            reference = item["reference"]
            print(f"[{i}/{len(testset)}] {question}")

            start = time.monotonic()
            graph_result = rag_graph.invoke(
                {"question": question, "original_question": question, "retry_count": 0}
            )
            elapsed_seconds = round(time.monotonic() - start, 1)
            print(f"    -> {graph_result['assessment']} in {elapsed_seconds}s")

            final_sources = graph_result["sources"]
            context = "\n\n".join(s["text"] for s in final_sources)

            start_metrics = time.monotonic()
            metrics = score_metrics(client, question, reference, graph_result["answer"], context)
            elapsed_seconds_metrics = round(time.monotonic() - start_metrics, 1)
            print(f"    -> metrics scored in {elapsed_seconds_metrics}s")
            print(f"    -> total time used: {round(time.monotonic() - start, 1)}s")

            results.append(
                {
                    "user_input": question,
                    "reference": reference,
                    "response": graph_result["answer"],
                    "path_taken": graph_result["assessment"],
                    "retry_count": graph_result["retry_count"],
                    "elapsed_seconds": elapsed_seconds,
                    "assess_reasoning": graph_result["assess_reasoning"],
                    "retrieval_history": graph_result["retrieval_history"],
                    "retrieved_contexts": final_sources,
                    "metrics": metrics,
                }
            )

    summary = {}
    for metric_name in METRIC_INSTRUCTIONS:
        scores = [r["metrics"][metric_name]["score"] for r in results if r["metrics"][metric_name]["score"] is not None]
        summary[metric_name] = {
            "average": sum(scores) / len(scores) if scores else None,
            "parsed": f"{len(scores)}/{len(results)}",
        }

    path_counts: dict[str, int] = {}
    path_timings: dict[str, list[float]] = {}
    for r in results:
        path_counts[r["path_taken"]] = path_counts.get(r["path_taken"], 0) + 1
        path_timings.setdefault(r["path_taken"], []).append(r["elapsed_seconds"])
    retried_count = sum(1 for r in results if r["retry_count"] > 0)
    summary["path_breakdown"] = path_counts
    summary["retry_rate"] = f"{retried_count}/{len(results)}"

    all_timings = [r["elapsed_seconds"] for r in results]
    summary["timing"] = {
        "average_seconds": round(sum(all_timings) / len(all_timings), 1) if all_timings else None,
        "by_path_average_seconds": {
            path: round(sum(times) / len(times), 1) for path, times in path_timings.items()
        },
    }

    output = {"summary": summary, "results": results}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(__file__).resolve().parent / f"eval_{timestamp}.json"
    output_path.write_text(json.dumps(output, indent=2))
    print("----- Evaluation Ends -----")

    print(f"\nWrote {len(results)} records to {output_path}\n")
    print("Average scores:")
    for metric_name in METRIC_INSTRUCTIONS:
        s = summary[metric_name]
        avg_str = f"{s['average']:.2f}" if s["average"] is not None else "no valid scores"
        print(f"  {metric_name}: {avg_str}  ({s['parsed']} parsed)")
    print(f"\nPath breakdown: {path_counts}")
    print(f"Retry rate: {summary['retry_rate']}")
    print(f"Average time: {summary['timing']['average_seconds']}s")
    print(f"By path: {summary['timing']['by_path_average_seconds']}")


if __name__ == "__main__":
    try:
        main()
    except httpx.ConnectError:
        print("Could not reach Ollama — is `ollama serve` running? (Qdrant must also be up.)")
        sys.exit(1)
