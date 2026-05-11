"""
nodes.py -- All five node functions for the Self-Healing RAG LangGraph pipeline.

Each function receives the current RAGState and returns a *partial* dict
containing only the keys it wishes to update. LangGraph merges the partial
updates into the running state automatically -- nodes never mutate state in place.

Nodes:
  1. retrieve      -- Query the vector store; populate retrieved_chunks
  2. generate      -- Generate a candidate answer from retrieved chunks
  3. critic        -- Evaluate whether the answer is grounded in the chunks
  4. rewrite_query -- Reformulate the query using critic feedback
  5. fallback      -- Return a graceful "no information" response
"""

import json
import re

from langchain_ollama import ChatOllama

from config import (
    CRITIC_MODEL,
    GENERATOR_MODEL,
    LOG_SEPARATOR,
    REWRITER_MODEL,
)
from prompts import critic_prompt, generator_prompt, rewriter_prompt
from state import RAGState


# -- Shared LLM instances (module-level to avoid re-creating on every call) --
_generator_llm = ChatOllama(model=GENERATOR_MODEL, temperature=0.2)
_critic_llm    = ChatOllama(model=CRITIC_MODEL,    temperature=0.0)  # Deterministic grading
_rewriter_llm  = ChatOllama(model=REWRITER_MODEL,  temperature=0.5)


# -- Retriever is injected before graph execution (see main.py) ---------------
_retriever = None


def set_retriever(retriever) -> None:
    """Register the LangChain retriever so nodes can access it at runtime."""
    global _retriever
    _retriever = retriever


# ─────────────────────────────────────────────────────────────────────────────
# NODE 1 -- retrieve
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(state: RAGState) -> dict:
    """
    Query the vector store with the current query and return the top-k chunks.

    Returns:
        Partial state update: {"retrieved_chunks": [...]}
    """
    print(LOG_SEPARATOR)
    query = state["current_query"]

    if _retriever is None:
        raise RuntimeError(
            "Retriever has not been registered. "
            "Call nodes.set_retriever() before invoking the graph."
        )

    docs   = _retriever.invoke(query)
    chunks = [doc.page_content for doc in docs]

    print(f"[RETRIEVE] Found {len(chunks)} chunk(s) for query: '{query}'")
    for i, chunk in enumerate(chunks, 1):
        preview = chunk[:200].replace("\n", " ")
        print(f"  [{i}] {preview}...")

    return {"retrieved_chunks": chunks}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 2 -- generate
# ─────────────────────────────────────────────────────────────────────────────

def generate(state: RAGState) -> dict:
    """
    Generate a candidate answer from the retrieved chunks using the generator LLM.

    Returns:
        Partial state update: {"generated_answer": "..."}
    """
    print(LOG_SEPARATOR)
    query  = state["current_query"]
    chunks = state["retrieved_chunks"]

    if not chunks:
        answer = "INSUFFICIENT CONTEXT"
        print("[GENERATE] No chunks available -- returning INSUFFICIENT CONTEXT.")
        return {"generated_answer": answer}

    chunks_text = "\n\n---\n\n".join(chunks)

    chain    = generator_prompt | _generator_llm
    response = chain.invoke({"chunks": chunks_text, "query": query})
    answer   = response.content.strip()

    print(f"[GENERATE] Answer ({len(answer)} chars):\n{answer}")

    return {"generated_answer": answer}


# ─────────────────────────────────────────────────────────────────────────────
# NODE 3 -- critic
# ─────────────────────────────────────────────────────────────────────────────

def critic(state: RAGState) -> dict:
    """
    Evaluate whether the generated answer is fully grounded in the retrieved chunks.

    Uses a SEPARATE LLM call from the generator to provide an independent verdict.

    Returns:
        Partial state update: {
            "critic_verdict":    "PASS" | "FAIL" | "UNCERTAIN",
            "critic_reasoning":  "<explanation>",
            "critic_confidence": <float>,
            "final_answer":      "<set only on PASS>",
            "is_complete":       <True only on PASS>,
        }
    """
    print(LOG_SEPARATOR)
    chunks = state["retrieved_chunks"]
    answer = state["generated_answer"]

    chunks_text = "\n\n---\n\n".join(chunks) if chunks else "(no chunks retrieved)"

    chain       = critic_prompt | _critic_llm
    response    = chain.invoke({"chunks": chunks_text, "answer": answer})
    raw_content = response.content.strip()

    # Parse the JSON verdict
    verdict, reasoning, confidence = _parse_critic_response(raw_content)

    print(f"[CRITIC] Verdict   : {verdict}")
    print(f"[CRITIC] Confidence: {confidence:.2f}")
    print(f"[CRITIC] Reasoning : {reasoning}")

    update: dict = {
        "critic_verdict":    verdict,
        "critic_reasoning":  reasoning,
        "critic_confidence": confidence,
    }

    # If the critic passes, lock in the final answer immediately.
    if verdict == "PASS":
        update["final_answer"] = answer
        update["is_complete"]  = True

    return update


def _parse_critic_response(raw: str) -> tuple:
    """
    Safely parse the critic's JSON response.

    Falls back to UNCERTAIN with confidence 0.5 if the response is malformed.
    """
    # Strip any accidental markdown fences the LLM may have added
    cleaned = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        data = json.loads(cleaned)

        # If LLM returned a list [{}], take the first item
        if isinstance(data, list) and len(data) > 0:
            data = data[0]

        if not isinstance(data, dict):
            raise ValueError("Parsed JSON is not a dictionary object.")

        verdict = str(data.get("verdict", "UNCERTAIN")).upper()
        if verdict not in {"PASS", "FAIL", "UNCERTAIN"}:
            verdict = "UNCERTAIN"

        reasoning  = str(data.get("reasoning", "No reasoning provided."))
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))  # clamp to [0, 1]

        return verdict, reasoning, confidence

    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        print(f"[CRITIC] WARNING: Failed to parse JSON response: {exc}")
        print(f"[CRITIC] Raw response was: {raw[:200]}")
        return "UNCERTAIN", f"Could not parse critic response: {exc}", 0.5


# ─────────────────────────────────────────────────────────────────────────────
# NODE 4 -- rewrite_query
# ─────────────────────────────────────────────────────────────────────────────

def rewrite_query(state: RAGState) -> dict:
    """
    Reformulate the current query using critic feedback to improve retrieval.

    Increments retry_count before rewriting so the router can gate on MAX_RETRIES.

    Returns:
        Partial state update: {
            "current_query": "<rewritten query>",
            "retry_count":   <incremented>,
        }
    """
    print(LOG_SEPARATOR)
    original_query   = state["original_query"]
    current_query    = state["current_query"]
    critic_reasoning = state["critic_reasoning"]
    new_retry_count  = state["retry_count"] + 1

    chain    = rewriter_prompt | _rewriter_llm
    response = chain.invoke(
        {
            "original_query":   original_query,
            "current_query":    current_query,
            "critic_reasoning": critic_reasoning,
        }
    )
    new_query = response.content.strip().strip('"').strip("'")

    print(f"[REWRITE] Attempt #{new_retry_count}")
    print(f"[REWRITE] Old query : {current_query}")
    print(f"[REWRITE] New query : {new_query}")

    return {
        "current_query": new_query,
        "retry_count":   new_retry_count,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NODE 5 -- fallback
# ─────────────────────────────────────────────────────────────────────────────

def fallback(state: RAGState) -> dict:
    """
    Return a graceful fallback response when max retries are exceeded or the
    critic is certain that no relevant information exists in the corpus.

    Never raises an exception -- always produces a safe, user-facing message.

    Returns:
        Partial state update: {
            "final_answer": "<fallback message>",
            "is_complete":  True,
        }
    """
    print(LOG_SEPARATOR)
    print("[FALLBACK] Max retries exceeded or no grounded answer found.")
    print(f"[FALLBACK] Original query   : {state['original_query']}")
    print(f"[FALLBACK] Last critic note : {state.get('critic_reasoning', 'N/A')}")
    print(f"[FALLBACK] Retry count      : {state['retry_count']}")

    fallback_message = (
        "I was unable to find reliable information to answer your question "
        "based on the available documents. "
        "Please try rephrasing your question or consult additional sources."
    )

    print("[FALLBACK] Returning graceful fallback response.")

    return {
        "final_answer": fallback_message,
        "is_complete":  True,
    }
