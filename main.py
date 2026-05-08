"""
main.py -- Entry point for the Self-Healing RAG pipeline.

Usage:
    python main.py
    python main.py --query "Your custom question here"
    python main.py --query "..." --reload   # Force re-ingestion of documents

Environment variables required (in .env or exported):
    OPENAI_API_KEY=sk-...
"""

import argparse
import io
import os
import sys
import time
from dotenv import load_dotenv

# Load environment variables IMMEDIATELY
load_dotenv()

# Inform user if key is missing, but continue since we are using Ollama
if not os.getenv("OPENAI_API_KEY"):
    print("[INFO] OPENAI_API_KEY not found. (Not required for Ollama mode)")

# Force UTF-8 on Windows so the banner and separators render correctly
# if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
#     sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
# if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
#     sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from config import LOG_SEPARATOR, MAX_RETRIES
from graph import build_graph
from nodes import set_retriever
from state import RAGState
from vectorstore import build_vectorstore, get_retriever


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-Healing RAG Pipeline -- powered by LangGraph + AI + ChromaDB"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What is the refund policy for digital products?",
        help="The question to ask the RAG pipeline.",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        default=False,
        help="Force re-ingestion of documents even if ChromaDB already exists.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Startup banner
# ─────────────────────────────────────────────────────────────────────────────

def print_banner() -> None:
    banner = (
        "\n"
        "+--------------------------------------------------------------+\n"
        "|        Self-Healing RAG Pipeline                             |\n"
        "|        Powered by LangGraph + AI + ChromaDB                  |\n"
        "+--------------------------------------------------------------+\n"
    )
    print(banner)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    print_banner()

    # -- Step 2: Build vector store + retriever -------------------------------
    print(LOG_SEPARATOR)
    print("[INIT] Building vector store ...")
    try:
        vectorstore = build_vectorstore(force_reload=args.reload)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    retriever = get_retriever(vectorstore)
    set_retriever(retriever)
    print("[INIT] Retriever ready.")

    # -- Step 3: Compile the LangGraph application ----------------------------
    print(LOG_SEPARATOR)
    print("[INIT] Compiling LangGraph StateGraph ...")
    app = build_graph()

    # -- Step 4: Build initial state ------------------------------------------
    query = args.query.strip()
    print(LOG_SEPARATOR)
    print(f'[INIT] Query: "{query}"')
    print(f"[INIT] MAX_RETRIES = {MAX_RETRIES}")

    initial_state: RAGState = {
        "original_query": query,
        "current_query": query,
        "retrieved_chunks": [],
        "generated_answer": "",
        "critic_verdict": "",
        "critic_reasoning": "",
        "critic_confidence": 0.0,
        "retry_count": 0,
        "final_answer": "",
        "is_complete": False,
    }

    # -- Step 5: Run the pipeline ---------------------------------------------
    print(LOG_SEPARATOR)
    print("[PIPELINE] Starting execution ...\n")
    start_time = time.perf_counter()

    try:
        result: RAGState = app.invoke(initial_state)
    except Exception as exc:
        print(f"\n[ERROR] PIPELINE FAILED: {exc}")
        raise

    elapsed = time.perf_counter() - start_time

    # -- Step 6: Print summary ------------------------------------------------
    print("\n" + "=" * 62)
    print("FINAL ANSWER")
    print("=" * 62)
    print(result["final_answer"])
    print("=" * 62)
    print(f"Total retries    : {result['retry_count']} / {MAX_RETRIES}")
    print(f"Critic verdict   : {result.get('critic_verdict', 'N/A')}")
    print(f"Critic confidence: {result.get('critic_confidence', 0.0):.2f}")
    print(f"Wall-clock time  : {elapsed:.2f}s")
    print("=" * 62)


if __name__ == "__main__":
    main()
