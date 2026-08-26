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


async def stream_ollama(client: httpx.AsyncClient, prompt: str):
    """Yields answer text chunks as they arrive from Ollama's streaming API."""
    async with client.stream(
        "POST",
        f"{OLLAMA_URL}/api/generate",
        json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": True},
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("response"):
                yield chunk["response"]


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


def build_generate_prompt(state: RAGState) -> str:
    nodes = state["nodes"]
    context = "\n\n".join(f"[{i}] {node.get_content()}" for i, node in enumerate(nodes))
    return (
        f"Answer the question using only the context below. Each chunk is labeled with an "
        f"index like [0], [1], etc. — these labels are for your own bookkeeping only. Do "
        f"NOT write these index numbers anywhere inside your answer text; write the answer "
        f"as normal prose with no bracketed numbers in it at all.\n\n"
        f"If the context doesn't contain the answer, say so. "
        f"If the context contains numbers or facts that are related to the question but "
        f"do not directly answer it, point that out explicitly rather than substituting "
        f"them as if they were the answer.\n\n"
        f"Context:\n{context}\n\nQuestion: {state['original_question']}\n\n"
        f"After your answer, on a new line, list the indices of the chunks you actually "
        f"used to answer — not every chunk you read, only the ones your answer is based "
        f"on. Use this exact format, with no other text after it, and no other line "
        f"breaks or bracketed numbers anywhere earlier in your response:\n"
        f"SOURCES: [<comma-separated indices>]\n\n"
        f"Example of a correctly formatted full response, given a question about a "
        f"character's actor and chunks [0] and [2] containing the relevant casting info:\n"
        f"Jane Doe plays the character in the show.\n"
        f"SOURCES: [0, 2]\n\n"
        f"Now answer the actual question above, following that exact format."
    )


def parse_generate_sources(raw_answer: str, num_nodes: int) -> tuple[str, list[int]]:
    """Splits generate's raw output into (answer_text, used_indices). If the trailing
    SOURCES tag is missing or malformed, falls back to all node indices (fail-open —
    same as showing everything, the pre-filtering-feature default).

    Tolerant of two observed model formats: "SOURCES: [0, 2]" (one bracket, comma-
    separated) and "SOURCES: [0], [2]" (multiple separate brackets) — the model doesn't
    reliably use the single format we ask for, so this accepts both rather than treating
    the second as malformed.

    Also strips any stray bracketed integers (e.g. "[1]") from inside the answer text
    itself — a safety net for cases where the model cites a chunk index inline instead
    of only in the trailing tag, which the prompt instructs against but isn't fully
    reliable in practice."""
    match = re.search(r"\nSOURCES:\s*((?:\[[^\]]*\]\s*,?\s*)+)\s*$", raw_answer)
    if not match:
        return _strip_inline_indices(raw_answer), list(range(num_nodes))

    answer_text = raw_answer[: match.start()].rstrip()
    answer_text = _strip_inline_indices(answer_text)

    raw_numbers = re.findall(r"\d+", match.group(1))
    try:
        indices = [int(n) for n in raw_numbers]
    except ValueError:
        return _strip_inline_indices(raw_answer), list(range(num_nodes))

    indices = [i for i in indices if 0 <= i < num_nodes]
    if not indices:
        return answer_text, list(range(num_nodes))
    return answer_text, indices


def _strip_inline_indices(text: str) -> str:
    """Removes stray bracketed integers like '[1]' or '[2, 3]' from inside answer text.
    Safety net for when generate cites a chunk index inline instead of only in the
    trailing SOURCES tag — collapses any resulting extra whitespace left behind."""
    stripped = re.sub(r"\s*\[\s*\d+(?:\s*,\s*\d+)*\s*\]\s*", " ", text)
    return re.sub(r"[ \t]{2,}", " ", stripped).strip()


def sources_from_nodes(nodes: list) -> list[dict]:
    return [
        {
            "file_name": node.metadata.get("file_name", "unknown"),
            "content_type": node.metadata.get("content_type", "text"),
            "text": node.get_content(),
        }
        for node in nodes
    ]



def build_graph(retriever):
    """retriever: a LlamaIndex retriever (e.g. index.as_retriever(...)), injected
    from main.py so this module doesn't own vector store / embedding setup.

    The graph covers retrieve -> assess -> reformulate looping and the
    insufficient_context terminal path. The "generate" step is NOT a graph node:
    it needs to stream tokens to the caller, which LangGraph nodes (which return
    a complete state update) aren't a fit for. main.py calls stream_ollama()
    directly with build_generate_prompt() once the graph signals a "pass" verdict."""

    retrieve_logger = logging.getLogger("retrieve")
    assess_logger = logging.getLogger("assess")
    reformulate_logger = logging.getLogger("reformulate")
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
        if state["assessment"].startswith("pass"):
            return "generate"
        return state["assessment"]

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

    @_timed_node("insufficient_context")
    def insufficient_context(state: RAGState) -> dict:
        insufficient_logger.info("returning insufficient-context response")
        return {
            "answer": INSUFFICIENT_CONTEXT_MESSAGE,
            "sources": sources_from_nodes(state["nodes"][:3]),
        }

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("assess", assess)
    graph.add_node("reformulate", reformulate)
    graph.add_node("insufficient_context", insufficient_context)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "assess")
    graph.add_conditional_edges(
        "assess",
        route_after_assess,
        {"generate": END, "retry": "reformulate", "insufficient": "insufficient_context"},
    )
    graph.add_edge("reformulate", "retrieve")
    graph.add_edge("insufficient_context", END)

    return graph.compile()
