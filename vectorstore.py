"""
vectorstore.py -- ChromaDB vector store setup and document ingestion.

Responsibilities:
  1. Load .txt files from the documents/ directory
  2. Chunk them with RecursiveCharacterTextSplitter
  3. Embed with HuggingFace (local) or OpenAI
  4. Persist to a local ChromaDB collection
  5. Expose a ready-to-use LangChain VectorStoreRetriever
"""

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIR,
    EMBEDDING_MODEL,
    TOP_K_CHUNKS,
)


def _load_documents(docs_dir: str) -> list:
    """Load all .txt files from the given directory into LangChain Documents."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {docs_dir!r}. "
            "Create it and add at least one .txt file."
        )

    txt_files = list(docs_path.glob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in {docs_dir!r}.")

    documents = []
    for txt_file in sorted(txt_files):
        print(f"[VECTORSTORE] Loading: {txt_file.name}")
        loader = TextLoader(str(txt_file), encoding="utf-8")
        documents.extend(loader.load())

    print(f"[VECTORSTORE] Loaded {len(documents)} document(s) from {docs_dir!r}.")
    return documents


def _chunk_documents(documents: list) -> list:
    """Split documents into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    print(
        f"[VECTORSTORE] Created {len(chunks)} chunk(s) "
        f"(size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})."
    )
    return chunks


def build_vectorstore(force_reload: bool = False) -> Chroma:
    """
    Build (or load from disk) the ChromaDB vector store.

    Args:
        force_reload: If True, re-ingest documents even if the DB already exists.

    Returns:
        A Chroma instance backed by a persistent local directory.
    """
    embeddings   = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    persist_path = Path(CHROMA_PERSIST_DIR)

    # If the collection already exists and we are not forcing a reload, reuse it.
    if persist_path.exists() and not force_reload:
        print(f"[VECTORSTORE] Loading existing collection from {CHROMA_PERSIST_DIR!r}.")
        vectorstore = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
        )
        count = vectorstore._collection.count()
        if count > 0:
            print(f"[VECTORSTORE] Collection contains {count} vector(s). Ready.")
            return vectorstore
        print("[VECTORSTORE] Collection is empty -- re-ingesting documents.")

    # Fresh ingest
    print("[VECTORSTORE] Ingesting documents into ChromaDB ...")
    documents = _load_documents(DOCUMENTS_DIR)
    chunks    = _chunk_documents(documents)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print(f"[VECTORSTORE] Persisted {len(chunks)} chunk(s) to {CHROMA_PERSIST_DIR!r}.")
    return vectorstore


def get_retriever(vectorstore: Chroma):
    """Return a LangChain VectorStoreRetriever for the given vectorstore."""
    return vectorstore.as_retriever(search_kwargs={"k": TOP_K_CHUNKS})
