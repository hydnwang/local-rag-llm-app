"""
Unified RAG evaluation: retrieve, generate, score 4 LLM-judge metrics, and
compute 4 candidate signals for the future LangGraph `assess` node — all
in-process (no FastAPI server required). Only Qdrant and Ollama need to be
running.

Replaces run_testset.py + score_testset.py.

Input:  eval/testset.json
Output: eval/eval_<YYYYMMDD>_<HHMMSS>.json

Usage:
    uv run python eval/evaluate.py
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import httpx
import numpy as np
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (
    COLLECTION_NAME,
    EMBED_MODEL_NAME,
    OLLAMA_MODEL,
    OLLAMA_URL,
    QDRANT_URL,
    TOP_K,
)

INPUT_PATH = Path(__file__).resolve().parent / "testset.json"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "did", "does", "do",
    "what", "who", "whom", "which", "when", "where", "why", "how",
    "and", "or", "for", "in", "on", "of", "to", "with", "by", "at",
    "as", "it", "its", "this", "that", "these", "those", "be", "been",
}

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


# --- Candidate `assess` signals (diagnostic only, not yet used to gate anything) ---

def extract_keywords(question: str) -> list[str]:
    words = re.findall(r"[A-Za-z']+", question.lower())
    return [w for w in words if w not in STOPWORDS and len(w) > 2]


def compute_signals(
    question: str, nodes, node_embeddings: list[list[float]], reranker: CrossEncoder
) -> dict:
    texts = [node.get_content() for node in nodes]
    sim_scores = [node.score for node in nodes]

    # Signal: cross-encoder max relevance
    pairs = [(question, text) for text in texts]
    rerank_scores = reranker.predict(pairs)
    cross_encoder_max = float(max(rerank_scores))

    # Signal: score gap (top-1 minus top-2 similarity)
    sorted_scores = sorted(sim_scores, reverse=True)
    score_gap_top1_top2 = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else None

    # Signal: redundancy (mean pairwise cosine similarity among retrieved chunk embeddings)
    vecs = np.array(node_embeddings)
    norm_vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    sim_matrix = norm_vecs @ norm_vecs.T
    n = len(vecs)
    upper_tri = sim_matrix[np.triu_indices(n, k=1)]
    redundancy_mean_pairwise = float(upper_tri.mean()) if len(upper_tri) > 0 else None

    # Signal: keyword overlap (any question keyword appears in any retrieved chunk)
    keywords = extract_keywords(question)
    combined_text = " ".join(texts).lower()
    keyword_overlap = any(kw in combined_text for kw in keywords) if keywords else None

    return {
        "cross_encoder_max": cross_encoder_max,
        "score_gap_top1_top2": score_gap_top1_top2,
        "redundancy_mean_pairwise": redundancy_mean_pairwise,
        "keyword_overlap": keyword_overlap,
    }


def main() -> None:
    testset = json.loads(INPUT_PATH.read_text())

    print("Loading models (embedding + reranker)...")
    qdrant_client = QdrantClient(url=QDRANT_URL)
    vector_store = QdrantVectorStore(client=qdrant_client, collection_name=COLLECTION_NAME)
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
    retriever = index.as_retriever(similarity_top_k=TOP_K)
    reranker = CrossEncoder(RERANKER_MODEL_NAME)

    results = []
    with httpx.Client(timeout=120.0) as client:
        for i, item in enumerate(testset, start=1):
            question = item["user_input"]
            reference = item["reference"]
            print(f"[{i}/{len(testset)}] {question}")

            nodes = retriever.retrieve(question)
            node_embeddings = [embed_model.get_text_embedding(n.get_content()) for n in nodes]
            context = "\n\n".join(node.get_content() for node in nodes)

            prompt = (
                f"Answer the question using only the context below. "
                f"If the context doesn't contain the answer, say so.\n\n"
                f"Context:\n{context}\n\nQuestion: {question}"
            )
            gen_response = client.post(
                f"{OLLAMA_URL}/api/generate",
                json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            )
            gen_response.raise_for_status()
            answer = gen_response.json()["response"]

            metrics = score_metrics(client, question, reference, answer, context)
            signals = compute_signals(question, nodes, node_embeddings, reranker)

            results.append(
                {
                    "user_input": question,
                    "reference": reference,
                    "response": answer,
                    "retrieved_contexts": [
                        {
                            "file_name": node.metadata.get("file_name", "unknown"),
                            "text": node.get_content(),
                            "similarity_score": node.score,
                        }
                        for node in nodes
                    ],
                    "metrics": metrics,
                    "candidate_signals": signals,
                }
            )

    summary = {}
    for metric_name in METRIC_INSTRUCTIONS:
        scores = [r["metrics"][metric_name]["score"] for r in results if r["metrics"][metric_name]["score"] is not None]
        summary[metric_name] = {
            "average": sum(scores) / len(scores) if scores else None,
            "parsed": f"{len(scores)}/{len(results)}",
        }

    output = {"summary": summary, "results": results}

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(__file__).resolve().parent / f"eval_{timestamp}.json"
    output_path.write_text(json.dumps(output, indent=2))

    print(f"\nWrote {len(results)} records to {output_path}\n")
    print("Average scores:")
    for metric_name, s in summary.items():
        avg_str = f"{s['average']:.2f}" if s["average"] is not None else "no valid scores"
        print(f"  {metric_name}: {avg_str}  ({s['parsed']} parsed)")


if __name__ == "__main__":
    try:
        main()
    except httpx.ConnectError:
        print("Could not reach Ollama — is `ollama serve` running? (Qdrant must also be up.)")
        sys.exit(1)
