"""
config.py — Central configuration for the Self-Healing RAG pipeline.
All tunable constants live here so nothing is hard-coded in business logic.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM Settings ────────────────────────────────────────────────────────────
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
GENERATOR_MODEL: str = "tinyllama"
CRITIC_MODEL: str = "tinyllama"
REWRITER_MODEL: str = "tinyllama"
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

# ── Retrieval Settings ───────────────────────────────────────────────────────
TOP_K_CHUNKS: int = 5
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50

# ── Graph / Retry Settings ───────────────────────────────────────────────────
MAX_RETRIES: int = 3                   # Max retrieval-rewrite cycles before fallback

# ── Vector Store Settings ────────────────────────────────────────────────────
CHROMA_PERSIST_DIR: str = "./chroma_db"
CHROMA_COLLECTION_NAME: str = "self_healing_rag"

# ── Document Directory ───────────────────────────────────────────────────────
DOCUMENTS_DIR: str = "./documents"

# ── Critic Thresholds ────────────────────────────────────────────────────────
CRITIC_PASS_CONFIDENCE_THRESHOLD: float = 0.7   # Confidence below this → UNCERTAIN

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_SEPARATOR: str = "-" * 60
