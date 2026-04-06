import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.asyncio
async def test_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_ask_rejects_without_api_key(monkeypatch):
    monkeypatch.setattr("app.config.settings.openai_api_key", "")
    monkeypatch.setattr("app.config.settings.groq_api_key", "")
    monkeypatch.setattr("app.config.settings.llm_provider", "openai")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/ask", json={"question": "hi"})
    assert resp.status_code == 500


@pytest.mark.asyncio
async def test_ask_rejects_without_serper_key(monkeypatch):
    monkeypatch.setattr("app.config.settings.groq_api_key", "test-key")
    monkeypatch.setattr("app.config.settings.llm_provider", "groq")
    monkeypatch.setattr("app.config.settings.serper_api_key", "")
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/ask", json={"question": "hi"})
    assert resp.status_code == 500
    assert "SERPER_API_KEY" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_ask_pipeline_returns_sources(monkeypatch):
    """Mock the pipeline to verify the endpoint wires everything correctly."""

    from app.models.schemas import PipelineResult, Source, Snippet

    fake_result = PipelineResult(
        answer="Test answer citing [1].",
        sources=[Source(number=1, title="Test Page", url="https://example.com")],
        snippets=[Snippet(source_number=1, text="Some extracted text")],
    )

    async def mock_pipeline(question: str) -> PipelineResult:
        return fake_result

    monkeypatch.setattr("app.api.routes.run_graph", mock_pipeline)
    monkeypatch.setattr("app.config.settings.groq_api_key", "test-key")
    monkeypatch.setattr("app.config.settings.llm_provider", "groq")
    monkeypatch.setattr("app.config.settings.serper_api_key", "test-key")

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/ask", json={"question": "What is Python?"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer"] == "Test answer citing [1]."
    assert len(data["sources"]) == 1
    assert data["sources"][0]["url"] == "https://example.com"
    assert len(data["snippets"]) == 1
