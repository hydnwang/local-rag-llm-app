import functools
import json
import logging
import operator
import re
import sys
import time
from pathlib import Path
from typing import Annotated, TypedDict

import httpx
from langgraph.graph import END, StateGraph
from prometheus_client import Histogram

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import OLLAMA_MODEL, OLLAMA_URL

MAX_RETRIES = 1

NODE_DURATION_SECONDS = Histogram(
    "rag_node_duration_seconds",
    "Time spent in each RAG graph node",
    ["node"],
)


def _timed_node(name: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(state):
            start = time.monotonic()
            try:
                return func(state)
            finally:
                NODE_DURATION_SECONDS.labels(node=name).observe(time.monotonic() - start)

        return wrapper

    return decorator

ASSESS_PROMPT = """You are evaluating whether the retrieved context is sufficient to \
answer the question. Choose one of three verdicts:
- "yes": the context contains the specific information needed to answer the question.
- "partial": the question has multiple parts or asks to compare/combine facts, and the \
context contains enough to answer some but not all parts — a real, honest answer noting \
what's missing is still possible and useful.
- "no": the context is missing or unrelated to what's being asked, and no meaningful \
answer can be given. Being topically related is not enough — the context must contain \
the specific fact or reasoning asked for.

Question: {question}

Retrieved context:
{context}

Respond with ONLY a JSON object in this exact format, no other text:
{{"verdict": "yes" or "partial" or "no", "reasoning": "<one sentence>"}}"""

REFORMULATE_PROMPT = """The following question was not sufficiently answered by the \
retrieved context. Rewrite the question to help retrieve the missing information. \
Respond with ONLY the rewritten question, no other text.

Original question: {question}

Retrieved context (insufficient):
{context}"""

INSUFFICIENT_CONTEXT_MESSAGE = (
    "I don't have enough information in the retrieved documents to answer this confidently."
)


class RAGState(TypedDict):
    question: str
    original_question: str
    nodes: list
    retry_count: int
    assessment: str
    assess_reasoning: str
    answer: str
    sources: list[dict]
    retrieval_history: Annotated[list[dict], operator.add]


def _call_ollama(client: httpx.Client, prompt: str) -> str:
    response = client.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
    )
    response.raise_for_status()
    return response.json()["response"]


def _parse_judge_output(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"verdict": "no", "reasoning": f"Could not parse judge output: {raw[:200]}"}
    try:
        parsed = json.loads(match.group())
        verdict = parsed.get("verdict")
        if verdict not in ("yes", "partial", "no"):
            return {"verdict": "no", "reasoning": f"Unexpected verdict value: {raw[:200]}"}
        return {"verdict": verdict, "reasoning": parsed.get("reasoning", "")}
    except (json.JSONDecodeError, KeyError, ValueError):
        return {"verdict": "no", "reasoning": f"Could not parse judge output: {raw[:200]}"}


def build_graph(retriever):
    """retriever: a LlamaIndex retriever (e.g. index.as_retriever(...)), injected
    from main.py so this module doesn't own vector store / embedding setup."""

    retrieve_logger = logging.getLogger("retrieve")
    assess_logger = logging.getLogger("assess")
    reformulate_logger = logging.getLogger("reformulate")
    generate_logger = logging.getLogger("generate")
    insufficient_logger = logging.getLogger("insufficient_context")

    @_timed_node("retrieve")
    def retrieve(state: RAGState) -> dict:
        nodes = retriever.retrieve(state["question"])
        retrieve_logger.info(f"retrieved {len(nodes)} chunks")
        attempt_record = {
            "attempt": state["retry_count"] + 1,
            "question_used": state["question"],
            "sources": [
                {
                    "file_name": n.metadata.get("file_name", "unknown"),
                    "content_type": n.metadata.get("content_type", "text"),
                    "text": n.get_content(),
                }
                for n in nodes
            ],
        }
        return {"nodes": nodes, "retrieval_history": [attempt_record]}

    @_timed_node("assess")
    def assess(state: RAGState) -> dict:
        context = "\n\n".join(node.get_content() for node in state["nodes"][:3])
        with httpx.Client(timeout=120.0) as client:
            raw = _call_ollama(
                client, ASSESS_PROMPT.format(question=state["original_question"], context=context)
            )
        judged = _parse_judge_output(raw)
        verdict = judged["verdict"]

        if verdict in ("yes", "partial"):
            assess_logger.info(f"verdict=pass ({verdict}) | retry_count={state['retry_count']}")
            return {"assessment": f"pass ({verdict})", "assess_reasoning": judged["reasoning"]}
        if state["retry_count"] < MAX_RETRIES:
            assess_logger.info(f"verdict=retry | retry_count={state['retry_count']}")
            return {"assessment": "retry", "assess_reasoning": judged["reasoning"]}
        assess_logger.info(f"verdict=insufficient | retry_count={state['retry_count']}")
        return {"assessment": "insufficient", "assess_reasoning": judged["reasoning"]}

    def route_after_assess(state: RAGState) -> str:
        return "generate" if state["assessment"].startswith("pass") else state["assessment"]

    @_timed_node("reformulate")
    def reformulate(state: RAGState) -> dict:
        context = "\n\n".join(node.get_content() for node in state["nodes"])
        with httpx.Client(timeout=120.0) as client:
            new_question = _call_ollama(
                client,
                REFORMULATE_PROMPT.format(question=state["original_question"], context=context),
            )
        new_question = new_question.strip()
        reformulate_logger.info(f"rewritten question={new_question}")
        return {"question": new_question, "retry_count": state["retry_count"] + 1}

    @_timed_node("generate")
    def generate(state: RAGState) -> dict:
        context = "\n\n".join(node.get_content() for node in state["nodes"])
        prompt = (
            f"Answer the question using only the context below. "
            f"If the context doesn't contain the answer, say so. "
            f"If the context contains numbers or facts that are related to the question but "
            f"do not directly answer it, point that out explicitly rather than substituting "
            f"them as if they were the answer.\n\n"
            f"Context:\n{context}\n\nQuestion: {state['original_question']}"
        )
        with httpx.Client(timeout=120.0) as client:
            answer = _call_ollama(client, prompt)

        generate_logger.info("answer generated")
        sources = [
            {
                "file_name": node.metadata.get("file_name", "unknown"),
                "content_type": node.metadata.get("content_type", "text"),
                "text": node.get_content(),
            }
            for node in state["nodes"]
        ]
        return {"answer": answer, "sources": sources}

    @_timed_node("insufficient_context")
    def insufficient_context(state: RAGState) -> dict:
        insufficient_logger.info("returning insufficient-context response")
        sources = [
            {
                "file_name": node.metadata.get("file_name", "unknown"),
                "content_type": node.metadata.get("content_type", "text"),
                "text": node.get_content(),
            }
            for node in state["nodes"]
        ]
        return {"answer": INSUFFICIENT_CONTEXT_MESSAGE, "sources": sources}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("assess", assess)
    graph.add_node("reformulate", reformulate)
    graph.add_node("generate", generate)
    graph.add_node("insufficient_context", insufficient_context)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "assess")
    graph.add_conditional_edges(
        "assess",
        route_after_assess,
        {"generate": "generate", "retry": "reformulate", "insufficient": "insufficient_context"},
    )
    graph.add_edge("reformulate", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("insufficient_context", END)

    return graph.compile()
