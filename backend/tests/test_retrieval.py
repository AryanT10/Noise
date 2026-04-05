"""Tests for Phase 3 — retrieval (ingest, store, RAG query)."""

import pytest
from httpx import ASGITransport, AsyncClient
from unittest.mock import patch, AsyncMock

from app.main import app


@pytest.fixture(autouse=True)
def _isolate_store(monkeypatch, tmp_path):
    """Point the vector store at a temp dir so tests don't pollute real data."""
    monkeypatch.setattr("app.retrieval.store._STORE_DIR", tmp_path / "vs")
    monkeypatch.setattr("app.retrieval.store._store", None)


# ── Ingest tests ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_ingest_text(monkeypatch):
    """Ingest text and verify chunks are added."""
    # Mock embeddings to avoid real API calls
    fake_embeddings = _FakeEmbeddings()
    monkeypatch.setattr(
        "app.retrieval.store.get_embeddings", lambda: fake_embeddings
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/ingest",
            json={"texts": ["This is a test document about retrieval augmented generation."]},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["chunks_added"] >= 1


@pytest.mark.asyncio
async def test_ingest_url(monkeypatch):
    """Ingest from a URL (mocked scraper)."""
    fake_embeddings = _FakeEmbeddings()
    monkeypatch.setattr(
        "app.retrieval.store.get_embeddings", lambda: fake_embeddings
    )

    async def fake_extract(url, max_chars=50_000):
        return "Scraped web page content about Python programming."

    monkeypatch.setattr("app.retrieval.ingest.extract_page_text", fake_extract)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/ingest",
            json={"urls": ["https://example.com/test"]},
        )

    assert resp.status_code == 200
    assert resp.json()["chunks_added"] >= 1


@pytest.mark.asyncio
async def test_ingest_empty():
    """Ingesting nothing returns 0 chunks."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/ingest", json={"urls": [], "texts": []})

    assert resp.status_code == 200
    assert resp.json()["chunks_added"] == 0


# ── RAG query tests ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_rag_ask_empty_store(monkeypatch):
    """RAG query with no ingested docs returns a helpful message."""
    monkeypatch.setattr("app.config.settings.groq_api_key", "test-key")
    monkeypatch.setattr("app.config.settings.llm_provider", "groq")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/ask/rag", json={"question": "What is Python?"}
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "No documents" in data["answer"]
    assert data["chunks_used"] == 0


@pytest.mark.asyncio
async def test_rag_ask_with_docs(monkeypatch):
    """Ingest → query round-trip using mocked embeddings and LLM."""
    fake_embeddings = _FakeEmbeddings()
    monkeypatch.setattr(
        "app.retrieval.store.get_embeddings", lambda: fake_embeddings
    )
    monkeypatch.setattr("app.config.settings.groq_api_key", "test-key")
    monkeypatch.setattr("app.config.settings.llm_provider", "groq")

    # Mock the LLM so we don't make real API calls
    async def mock_rag_pipeline(question, top_k=5):
        return {
            "answer": "Mocked RAG answer about Python.",
            "chunks_used": 2,
            "sources": ["manual"],
        }

    monkeypatch.setattr("app.api.routes.run_rag_pipeline", mock_rag_pipeline)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/ask/rag", json={"question": "What is Python?"}
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "Mocked RAG answer" in data["answer"]
    assert data["chunks_used"] == 2


@pytest.mark.asyncio
async def test_rag_ask_rejects_without_api_key(monkeypatch):
    """RAG endpoint rejects if no LLM key configured."""
    monkeypatch.setattr("app.config.settings.openai_api_key", "")
    monkeypatch.setattr("app.config.settings.groq_api_key", "")
    monkeypatch.setattr("app.config.settings.llm_provider", "openai")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/api/ask/rag", json={"question": "test"}
        )

    assert resp.status_code == 500


# ── Store management tests ───────────────────────────────────


@pytest.mark.asyncio
async def test_clear_store(monkeypatch):
    """DELETE /api/store clears the vector store."""
    fake_embeddings = _FakeEmbeddings()
    monkeypatch.setattr(
        "app.retrieval.store.get_embeddings", lambda: fake_embeddings
    )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Ingest something first
        await client.post(
            "/api/ingest", json={"texts": ["Test document."]}
        )
        # Clear
        resp = await client.delete("/api/store")

    assert resp.status_code == 200
    assert resp.json()["status"] == "cleared"


# ── Helpers ──────────────────────────────────────────────────


class _FakeEmbeddings:
    """Deterministic embeddings for testing (no API calls)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        import hashlib

        h = hashlib.sha256(text.encode()).digest()
        # 768-dim vector from hash bytes (repeating)
        vec = []
        for i in range(768):
            vec.append((h[i % len(h)] - 128) / 128.0)
        return vec
