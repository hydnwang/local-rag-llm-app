import json
import operator
import re
import sys
from pathlib import Path
from typing import Annotated, TypedDict

import httpx
from langgraph.graph import END, StateGraph

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import OLLAMA_MODEL, OLLAMA_URL

MAX_RETRIES = 1

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

    def retrieve(state: RAGState) -> dict:
        nodes = retriever.retrieve(state["question"])
        attempt_record = {
            "attempt": state["retry_count"] + 1,
            "question_used": state["question"],
            "sources": [
                {"file_name": n.metadata.get("file_name", "unknown"), "text": n.get_content()}
                for n in nodes
            ],
        }
        return {"nodes": nodes, "retrieval_history": [attempt_record]}

    def assess(state: RAGState) -> dict:
        context = "\n\n".join(node.get_content() for node in state["nodes"])
        with httpx.Client(timeout=120.0) as client:
            raw = _call_ollama(
                client, ASSESS_PROMPT.format(question=state["original_question"], context=context)
            )
        judged = _parse_judge_output(raw)
        verdict = judged["verdict"]

        if verdict in ("yes", "partial"):
            return {"assessment": f"pass ({verdict})", "assess_reasoning": judged["reasoning"]}
        if state["retry_count"] < MAX_RETRIES:
            return {"assessment": "retry", "assess_reasoning": judged["reasoning"]}
        return {"assessment": "insufficient", "assess_reasoning": judged["reasoning"]}

    def route_after_assess(state: RAGState) -> str:
        return "generate" if state["assessment"].startswith("pass") else state["assessment"]

    def reformulate(state: RAGState) -> dict:
        context = "\n\n".join(node.get_content() for node in state["nodes"])
        with httpx.Client(timeout=120.0) as client:
            new_question = _call_ollama(
                client,
                REFORMULATE_PROMPT.format(question=state["original_question"], context=context),
            )
        return {"question": new_question.strip(), "retry_count": state["retry_count"] + 1}

    def generate(state: RAGState) -> dict:
        context = "\n\n".join(node.get_content() for node in state["nodes"])
        prompt = (
            f"Answer the question using only the context below. "
            f"If the context doesn't contain the answer, say so.\n\n"
            f"Context:\n{context}\n\nQuestion: {state['original_question']}"
        )
        with httpx.Client(timeout=120.0) as client:
            answer = _call_ollama(client, prompt)

        sources = [
            {"file_name": node.metadata.get("file_name", "unknown"), "text": node.get_content()}
            for node in state["nodes"]
        ]
        return {"answer": answer, "sources": sources}

    def insufficient_context(state: RAGState) -> dict:
        sources = [
            {"file_name": node.metadata.get("file_name", "unknown"), "text": node.get_content()}
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
