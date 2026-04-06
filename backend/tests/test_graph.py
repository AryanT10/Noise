"""Tests for Phase 4 — LangGraph workflow nodes, edges, and end-to-end."""

import pytest
from unittest.mock import AsyncMock, patch

from langchain_core.runnables import RunnableLambda

from app.graph.state import GraphState
from app.graph.nodes import (
    analyze_question,
    search_sources,
    retrieve_chunks,
    filter_evidence,
    synthesize_answer,
    format_response,
)
from app.graph.workflow import build_graph, run_graph
from app.models.schemas import PipelineResult


# ── Helpers ──────────────────────────────────────────────────


def _base_state(**overrides) -> GraphState:
    """Minimal valid state with defaults."""
    state: GraphState = {"question": "What is LangGraph?", "errors": []}
    state.update(overrides)
    return state


class _FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content


def _fake_llm(content: str):
    """Return a RunnableLambda that acts like a chat model for chain piping."""
    async def fn(messages):
        return _FakeLLMResponse(content)
    return RunnableLambda(fn)


def _fake_llm_sequential(responses: list[str]):
    """Return a RunnableLambda that returns different responses on each call."""
    call_count = 0

    async def fn(messages):
        nonlocal call_count
        idx = min(call_count, len(responses) - 1)
        call_count += 1
        return _FakeLLMResponse(responses[idx])
    return RunnableLambda(fn)


# ── Node tests ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_analyze_question_generates_queries():
    with patch("app.graph.nodes.get_llm", return_value=_fake_llm("LangGraph overview\nLangGraph tutorial")):
        result = await analyze_question(_base_state())

    assert "search_queries" in result
    assert len(result["search_queries"]) == 2


@pytest.mark.asyncio
async def test_analyze_question_falls_back_on_error(monkeypatch):
    with patch("app.graph.nodes.get_llm", side_effect=Exception("LLM down")):
        result = await analyze_question(_base_state())

    # Should fall back to using the original question
    assert result["search_queries"] == ["What is LangGraph?"]
    assert any("analyze_question" in e for e in result["errors"])


@pytest.mark.asyncio
async def test_search_sources_deduplicates(monkeypatch):
    async def fake_search(query, num_results=5):
        return [
            {"title": "Result 1", "url": "https://a.com", "snippet": "..."},
            {"title": "Result 2", "url": "https://b.com", "snippet": "..."},
        ]

    with patch("app.graph.nodes.web_search", side_effect=fake_search):
        result = await search_sources(
            _base_state(search_queries=["query 1", "query 2"])
        )

    # Same URLs from both queries — should be deduplicated
    assert len(result["search_results"]) == 2


@pytest.mark.asyncio
async def test_search_sources_handles_failure(monkeypatch):
    with patch("app.graph.nodes.web_search", side_effect=Exception("API error")):
        result = await search_sources(_base_state(search_queries=["test"]))

    assert result["search_results"] == []
    assert any("search_sources" in e for e in result["errors"])


@pytest.mark.asyncio
async def test_retrieve_chunks_scrapes_pages():
    search_results = [
        {"title": "Page 1", "url": "https://a.com", "snippet": "snip1"},
        {"title": "Page 2", "url": "https://b.com", "snippet": "snip2"},
        {"title": "Page 3", "url": "https://c.com", "snippet": "snip3"},
        {"title": "Page 4", "url": "https://d.com", "snippet": "snip4"},
    ]

    async def fake_extract(url):
        return f"Full text from {url}"

    with patch("app.graph.nodes.extract_page_text", side_effect=fake_extract):
        result = await retrieve_chunks(
            _base_state(search_results=search_results)
        )

    docs = result["retrieved_docs"]
    assert len(docs) == 4
    # First 3 should have full text, last one should have snippet
    assert "Full text" in docs[0]["text"]
    assert docs[3]["text"] == "snip4"


@pytest.mark.asyncio
async def test_filter_evidence_keeps_relevant():
    docs = [
        {"title": "Relevant", "url": "https://a.com", "text": "Good info"},
        {"title": "Irrelevant", "url": "https://b.com", "text": "Noise"},
        {"title": "Also relevant", "url": "https://c.com", "text": "More info"},
    ]

    with patch("app.graph.nodes.get_llm", return_value=_fake_llm("1,3")):
        result = await filter_evidence(
            _base_state(retrieved_docs=docs)
        )

    assert len(result["filtered_evidence"]) == 2
    assert result["filtered_evidence"][0]["number"] == 1
    assert result["filtered_evidence"][1]["number"] == 3


@pytest.mark.asyncio
async def test_filter_evidence_keeps_all_on_none():
    docs = [
        {"title": "A", "url": "https://a.com", "text": "text"},
    ]

    with patch("app.graph.nodes.get_llm", return_value=_fake_llm("NONE")):
        result = await filter_evidence(_base_state(retrieved_docs=docs))

    # NONE means keep all as fallback
    assert len(result["filtered_evidence"]) == 1


@pytest.mark.asyncio
async def test_synthesize_answer_produces_draft():
    evidence = [
        {"number": 1, "title": "Source", "url": "https://a.com", "text": "Info about LangGraph"},
    ]

    with patch("app.graph.nodes.get_llm", return_value=_fake_llm("LangGraph is a framework [1].")):
        result = await synthesize_answer(
            _base_state(filtered_evidence=evidence)
        )

    assert "LangGraph" in result["draft_answer"]


@pytest.mark.asyncio
async def test_synthesize_answer_handles_empty_evidence():
    result = await synthesize_answer(_base_state(filtered_evidence=[]))
    assert "couldn't find" in result["draft_answer"].lower()


@pytest.mark.asyncio
async def test_format_response_builds_output():
    evidence = [
        {"number": 1, "title": "Src", "url": "https://a.com", "text": "content"},
    ]
    state = _base_state(
        filtered_evidence=evidence,
        draft_answer="Answer [1].",
    )

    result = await format_response(state)

    assert result["answer"] == "Answer [1]."
    assert len(result["sources"]) == 1
    assert result["sources"][0].number == 1
    assert len(result["snippets"]) == 1
    assert result["citations"] == ["[1]"]


# ── Conditional edge tests ───────────────────────────────────


def test_graph_compiles():
    """The graph should compile without errors."""
    compiled = build_graph()
    assert compiled is not None


# ── End-to-end graph test ────────────────────────────────────


@pytest.mark.asyncio
async def test_run_graph_end_to_end():
    """Full workflow with all external calls mocked."""

    async def fake_search(query, num_results=5):
        return [
            {"title": "LangGraph Docs", "url": "https://docs.example.com", "snippet": "LG info"},
        ]

    async def fake_extract(url):
        return "LangGraph is a library for building stateful agents."

    fake_model = _fake_llm_sequential([
        "What is LangGraph framework",  # analyze_question
        "1",                             # filter_evidence
        "LangGraph is a library for building stateful agents [1].",  # synthesize_answer
    ])

    with (
        patch("app.graph.nodes.get_llm", return_value=fake_model),
        patch("app.graph.nodes.web_search", side_effect=fake_search),
        patch("app.graph.nodes.extract_page_text", side_effect=fake_extract),
    ):
        result = await run_graph("What is LangGraph?")

    assert isinstance(result, PipelineResult)
    assert "LangGraph" in result.answer
    assert len(result.sources) >= 1
    assert result.sources[0].url == "https://docs.example.com"


@pytest.mark.asyncio
async def test_run_graph_no_search_results():
    """Graph should handle the case where search returns nothing."""

    async def empty_search(query, num_results=5):
        return []

    with (
        patch("app.graph.nodes.get_llm", return_value=_fake_llm("test query")),
        patch("app.graph.nodes.web_search", side_effect=empty_search),
    ):
        result = await run_graph("Something obscure?")

    assert isinstance(result, PipelineResult)
    assert result.sources == []
