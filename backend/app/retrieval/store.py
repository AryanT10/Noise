"""FAISS vector store — persist locally, add documents, search."""

import shutil
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.logging import logger
from app.retrieval.embeddings import get_embeddings

_STORE_DIR = Path(__file__).resolve().parent.parent.parent / "vector_store"

# Module-level singleton — loaded lazily
_store: FAISS | None = None


def _load_store() -> FAISS | None:
    """Load persisted FAISS index from disk if it exists."""
    global _store
    if _store is not None:
        return _store
    index_path = _STORE_DIR / "index.faiss"
    if index_path.exists():
        logger.info("Loading FAISS index from %s", _STORE_DIR)
        _store = FAISS.load_local(
            str(_STORE_DIR),
            get_embeddings(),
            allow_dangerous_deserialization=True,
        )
    return _store


def add_documents(docs: list[Document]) -> int:
    """Embed and add documents to the vector store, then persist."""
    global _store
    if not docs:
        return 0

    embeddings = get_embeddings()

    if _store is None:
        _load_store()

    if _store is None:
        _store = FAISS.from_documents(docs, embeddings)
    else:
        _store.add_documents(docs)

    _STORE_DIR.mkdir(parents=True, exist_ok=True)
    _store.save_local(str(_STORE_DIR))
    logger.info("Added %d chunks to FAISS store (%s)", len(docs), _STORE_DIR)
    return len(docs)


def search(query: str, k: int = 5) -> list[Document]:
    """Return the top-k most relevant chunks for a query."""
    store = _load_store()
    if store is None:
        logger.warning("No FAISS index found — returning empty results")
        return []
    return store.similarity_search(query, k=k)


def clear_store() -> None:
    """Delete the FAISS index from memory and disk."""
    global _store
    _store = None
    if _STORE_DIR.exists():
        shutil.rmtree(_STORE_DIR)
        logger.info("Cleared FAISS store at %s", _STORE_DIR)
