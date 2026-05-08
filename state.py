from typing import TypedDict, List

class RAGState(TypedDict):
    original_query:    str        # User's verbatim question — never mutated
    current_query:     str        # Active query (may be rewritten each retry)
    retrieved_chunks:  List[str]  # Raw text chunks from the vector store
    generated_answer:  str        # Candidate answer from the generator LLM
    critic_verdict:    str        # "PASS" | "FAIL" | "UNCERTAIN"
    critic_reasoning:  str        # One-sentence critic explanation
    critic_confidence: float      # Confidence score 0.0 – 1.0
    retry_count:       int        # Number of retrieve→generate→critic loops
    final_answer:      str        # Authoritative answer returned to the user
    is_complete:       bool       # Termination sentinel
