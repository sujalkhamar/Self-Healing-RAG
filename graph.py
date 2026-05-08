"""
graph.py — LangGraph StateGraph definition for the Self-Healing RAG pipeline.
"""

from langgraph.graph import END, StateGraph

from config import MAX_RETRIES
from nodes import fallback, generate, critic, retrieve, rewrite_query
from state import RAGState


# ─────────────────────────────────────────────────────────────────────────────
# Conditional edge router
# ─────────────────────────────────────────────────────────────────────────────

def route_after_critic(state: RAGState) -> str:
    """
    Decide the next node after the critic has evaluated the generated answer.
    """
    verdict = state["critic_verdict"]
    retry_count = state["retry_count"]

    if verdict == "PASS":
        return "end"

    if retry_count >= MAX_RETRIES:
        return "fallback"

    # FAIL or UNCERTAIN — still have retries left
    return "rewrite_query"


# ─────────────────────────────────────────────────────────────────────────────
# Graph construction
# ─────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """
    Construct and compile the Self-Healing RAG StateGraph.
    """
    graph = StateGraph(RAGState)

    # ── Register nodes ────────────────────────────────────────────────────────
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_node("critic", critic)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("fallback", fallback)

    # ── Set entry point ───────────────────────────────────────────────────────
    graph.set_entry_point("retrieve")

    # ── Deterministic edges ───────────────────────────────────────────────────
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "critic")
    graph.add_edge("rewrite_query", "retrieve")   # Forms the self-healing cycle
    graph.add_edge("fallback", END)

    # ── Conditional edge (the self-healing router) ────────────────────────────
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "end": END,
            "fallback": "fallback",
            "rewrite_query": "rewrite_query",
        },
    )

    compiled = graph.compile()
    print("[GRAPH] StateGraph compiled successfully.")
    return compiled
