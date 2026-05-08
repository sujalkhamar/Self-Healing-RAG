"""
prompts.py — Centralised prompt templates for every LLM call in the pipeline.

Keeping prompts here (rather than inlined in nodes.py) makes them easy to
iterate on, version-control, and A/B test without touching node logic.
"""

from langchain_core.prompts import ChatPromptTemplate

# ── Generator Prompt ─────────────────────────────────────────────────────────
GENERATOR_SYSTEM = (
    "You are a helpful and precise assistant. "
    "Answer the user's question ONLY using the provided context chunks. "
    "Do NOT use any prior knowledge outside the context. "
    "If the context does not contain enough information to answer the question, "
    "respond exactly with: INSUFFICIENT CONTEXT"
)

GENERATOR_HUMAN = (
    "Context chunks:\n"
    "-----------------------------------------\n"
    "{chunks}\n"
    "-----------------------------------------\n\n"
    "Question: {query}\n\n"
    "Answer:"
)

generator_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", GENERATOR_SYSTEM),
        ("human", GENERATOR_HUMAN),
    ]
)

# ── Critic Prompt ─────────────────────────────────────────────────────────────
CRITIC_SYSTEM = """You are a strict, meticulous fact-checker.
Your sole job is to determine whether an AI-generated answer is FULLY supported
by the source chunks provided — and nothing else.

Grading rules:
  PASS      → Every factual claim in the answer can be directly traced to the
               source chunks. No information was invented or assumed.
  FAIL      → The answer contains at least one claim that is NOT present in the
               source chunks (hallucination), or it contradicts the chunks.
  UNCERTAIN → The answer is vague / non-committal, OR the source chunks are
               ambiguous enough that you cannot definitively grade it.

Special case: if the answer is exactly "INSUFFICIENT CONTEXT", grade it as PASS
(the model correctly identified it had no usable context).

You MUST respond with ONLY valid JSON — no markdown fences, no extra text:
{{
  "verdict": "PASS" | "FAIL" | "UNCERTAIN",
  "reasoning": "<one concise sentence explaining your decision>",
  "confidence": <float between 0.0 and 1.0>
}}"""

CRITIC_HUMAN = (
    "Source chunks:\n"
    "-----------------------------------------\n"
    "{chunks}\n"
    "-----------------------------------------\n\n"
    "Generated answer:\n"
    "{answer}\n\n"
    "Your JSON verdict:"
)

critic_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", CRITIC_SYSTEM),
        ("human", CRITIC_HUMAN),
    ]
)

# ── Query Rewriter Prompt ────────────────────────────────────────────────────
REWRITER_SYSTEM = (
    "You are a search-query optimisation expert. "
    "Your task is to reformulate a query that failed to retrieve a grounded answer "
    "from a document store. Use the critic's feedback as a guide. "
    "The new query should:\n"
    "  • Use different terminology or synonyms\n"
    "  • Be more specific and targeted\n"
    "  • Focus on the aspect the critic found unsupported\n"
    "Return ONLY the new query string — no explanation, no quotes, no punctuation prefix."
)

REWRITER_HUMAN = (
    "Original user question : {original_query}\n"
    "Failed query            : {current_query}\n"
    "Critic feedback         : {critic_reasoning}\n\n"
    "Rewritten query:"
)

rewriter_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", REWRITER_SYSTEM),
        ("human", REWRITER_HUMAN),
    ]
)
