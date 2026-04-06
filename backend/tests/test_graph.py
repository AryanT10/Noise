"""Tests for Phase 5 — controlled tool-calling via reason_and_act node."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.graph.state import GraphState
from app.graph.nodes import (
    analyze_question,
    reason_and_act,
    filter_evidence,
    synthesize_answer,
    format_response,
    MAX_REASONING_ROUNDS,
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
        self.tool_calls = []


def _fake_llm(content: str):
    """Return a RunnableLambda that acts like a chat model for chain piping."""
    async def fn(messages):
        return _FakeLLMResponse(content)
    return RunnableLambda(fn)


def _fake_tool_llm(tool_calls):
    """Return a mock LLM whose bind_tools().ainvoke() returns an AIMessage with tool_calls."""
    response = AIMessage(content="", tool_calls=tool_calls)

    bound = AsyncMock()
    bound.ainvoke = AsyncMock(return_value=response)

    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=bound)
    return llm


def _fake_sequential_llm(responses):
    """Return a RunnableLambda that returns different responses per call.

    Each response is either a str (plain content) or a dict with a
    ``tool_calls`` key (for bind_tools rounds).  The returned object
    supports prompt piping (``| llm``) and ``.bind_tools()``.
    """
    call_count = {"n": 0}

    async def fn(messages):
        idx = min(call_count["n"], len(responses) - 1)
        call_count["n"] += 1
        resp = responses[idx]
        if isinstance(resp, dict) and "tool_calls" in resp:
            return AIMessage(content=resp.get("content", ""), tool_calls=resp["tool_calls"])
        return _FakeLLMResponse(resp)

    runnable = RunnableLambda(fn)
    # Allow bind_tools to pass through — returns self so same fn is called
    runnable.bind_tools = lambda tools: runnable
    return runnable


# ── analyze_question (unchanged from Phase 4) ───────────────


@pytest.mark.asyncio
async def test_analyze_question_generates_queries():
    with patch("app.graph.nodes.get_llm", return_value=_fake_llm("LangGraph overview\nLangGraph tutorial")):
        result = await analyze_question(_base_state())

    assert "search_queries" in result
    assert len(result["search_queries"]) == 2


@pytest.mark.asyncio
async def test_analyze_question_falls_back_on_error():
    with patch("app.graph.nodes.get_llm", side_effect=Exception("LLM down")):
        result = await analyze_question(_base_state())

    assert result["search_queries"] == ["What is LangGraph?"]
    assert any("analyze_question" in e for e in result["errors"])


# ── reason_and_act ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_reason_and_act_calls_search_web():
    """LLM chooses search_web → underlying web_search is invoked."""
    tool_calls = [{"name": "search_web", "args": {"query": "LangGraph tutorial"}, "id": "tc1"}]

    async def fake_search(query, num_results=5):
        return [
            {"title": "LG Docs", "url": "https://docs.example.com", "snippet": "LG info"},
        ]

    with (
        patch("app.graph.nodes.get_llm", return_value=_fake_tool_llm(tool_calls)),
        patch("app.graph.nodes.web_search", side_effect=fake_search),
    ):
        result = await reason_and_act(
            _base_state(search_queries=["LangGraph tutorial"])
        )

    assert len(result["retrieved_docs"]) == 1
    assert result["retrieved_docs"][0]["url"] == "https://docs.example.com"
    assert result["reasoning_rounds"] == 1
    assert not result["needs_more_evidence"]
    assert any(tc["tool"] == "search_web" for tc in result["tool_calls_made"])


@pytest.mark.asyncio
async def test_reason_and_act_calls_fetch_url():
    """LLM chooses fetch_url → extract_page_text is invoked."""
    tool_calls = [{"name": "fetch_url", "args": {"url": "https://example.com/page"}, "id": "tc1"}]

    with (
        patch("app.graph.nodes.get_llm", return_value=_fake_tool_llm(tool_calls)),
        patch("app.graph.nodes.extract_page_text", new_callable=AsyncMock, return_value="Page content here"),
    ):
        result = await reason_and_act(_base_state())

    assert len(result["retrieved_docs"]) == 1
    assert result["retrieved_docs"][0]["text"] == "Page content here"


@pytest.mark.asyncio
async def test_reason_and_act_calls_retrieve_documents():
    """LLM chooses retrieve_documents → FAISS store is searched."""
    tool_calls = [{"name": "retrieve_documents", "args": {"query": "LangGraph"}, "id": "tc1"}]

    fake_doc = MagicMock()
    fake_doc.page_content = "Internal doc about LangGraph"
    fake_doc.metadata = {"source": "uploaded.pdf"}

    with (
        patch("app.graph.nodes.get_llm", return_value=_fake_tool_llm(tool_calls)),
        patch("app.retrieval.store.search", return_value=[fake_doc]),
    ):
        result = await reason_and_act(_base_state())

    assert len(result["retrieved_docs"]) == 1
    assert "Internal" in result["retrieved_docs"][0]["title"]
    assert "LangGraph" in result["retrieved_docs"][0]["text"]


@pytest.mark.asyncio
async def test_reason_and_act_multiple_tools():
    """LLM chooses both search_web and retrieve_documents in one round."""
    tool_calls = [
        {"name": "search_web", "args": {"query": "LangGraph"}, "id": "tc1"},
        {"name": "retrieve_documents", "args": {"query": "LangGraph"}, "id": "tc2"},
    ]

    async def fake_search(query, num_results=5):
        return [{"title": "Web Result", "url": "https://web.com", "snippet": "from web"}]

    fake_doc = MagicMock()
    fake_doc.page_content = "from internal store"
    fake_doc.metadata = {"source": "notes.pdf"}

    with (
        patch("app.graph.nodes.get_llm", return_value=_fake_tool_llm(tool_calls)),
        patch("app.graph.nodes.web_search", side_effect=fake_search),
        patch("app.retrieval.store.search", return_value=[fake_doc]),
    ):
        result = await reason_and_act(_base_state())

    assert len(result["retrieved_docs"]) == 2
    assert len(result["tool_calls_made"]) == 2


@pytest.mark.asyncio
async def test_reason_and_act_request_more_evidence():
    """LLM requests more evidence → needs_more_evidence flag is set."""
    tool_calls = [
        {"name": "search_web", "args": {"query": "LangGraph"}, "id": "tc1"},
        {"name": "request_more_evidence", "args": {"reason": "need more detail"}, "id": "tc2"},
    ]

    async def fake_search(query, num_results=5):
        return [{"title": "R1", "url": "https://a.com", "snippet": "s1"}]

    with (
        patch("app.graph.nodes.get_llm", return_value=_fake_tool_llm(tool_calls)),
        patch("app.graph.nodes.web_search", side_effect=fake_search),
    ):
        result = await reason_and_act(
            _base_state(reasoning_rounds=0)
        )

    assert result["needs_more_evidence"] is True
    assert result["reasoning_rounds"] == 1


@pytest.mark.asyncio
async def test_reason_and_act_caps_rounds():
    """needs_more_evidence is forced False when max rounds reached."""
    tool_calls = [
        {"name": "request_more_evidence", "args": {"reason": "need more"}, "id": "tc1"},
    ]

    with patch("app.graph.nodes.get_llm", return_value=_fake_tool_llm(tool_calls)):
        result = await reason_and_act(
            _base_state(reasoning_rounds=MAX_REASONING_ROUNDS - 1)
        )

    assert result["needs_more_evidence"] is False


@pytest.mark.asyncio
async def test_reason_and_act_deduplicates_urls():
    """Same URL from prior docs and new search should not be duplicated."""
    prior = [{"title": "Existing", "url": "https://a.com", "text": "old"}]
    tool_calls = [{"name": "search_web", "args": {"query": "q"}, "id": "tc1"}]

    async def fake_search(query, num_results=5):
        return [
            {"title": "Dup", "url": "https://a.com", "snippet": "new"},
            {"title": "Fresh", "url": "https://b.com", "snippet": "b"},
        ]

    with (
        patch("app.graph.nodes.get_llm", return_value=_fake_tool_llm(tool_calls)),
        patch("app.graph.nodes.web_search", side_effect=fake_search),
    ):
        result = await reason_and_act(
            _base_state(retrieved_docs=prior)
        )

    urls = [d["url"] for d in result["retrieved_docs"]]
    assert urls.count("https://a.com") == 1
    assert "https://b.com" in urls


@pytest.mark.asyncio
async def test_reason_and_act_no_tool_calls_falls_back():
    """If LLM returns no tool calls, fall back to web search."""
    response = AIMessage(content="I don't know which tool to use", tool_calls=[])

    bound = AsyncMock()
    bound.ainvoke = AsyncMock(return_value=response)

    llm = MagicMock()
    llm.bind_tools = MagicMock(return_value=bound)

    async def fake_search(query, num_results=5):
        return [{"title": "Fallback", "url": "https://fb.com", "snippet": "fb"}]

    with (
        patch("app.graph.nodes.get_llm", return_value=llm),
        patch("app.graph.nodes.web_search", side_effect=fake_search),
    ):
        result = await reason_and_act(
            _base_state(search_queries=["q1", "q2"])
        )

    assert len(result["retrieved_docs"]) >= 1


@pytest.mark.asyncio
async def test_reason_and_act_handles_tool_failure():
    """If a tool raises, the error is logged and execution continues."""
    tool_calls = [
        {"name": "search_web", "args": {"query": "q"}, "id": "tc1"},
        {"name": "fetch_url", "args": {"url": "https://good.com"}, "id": "tc2"},
    ]

    async def exploding_search(query, num_results=5):
        raise RuntimeError("Serper down")

    with (
        patch("app.graph.nodes.get_llm", return_value=_fake_tool_llm(tool_calls)),
        patch("app.graph.nodes.web_search", side_effect=exploding_search),
        patch("app.graph.nodes.extract_page_text", new_callable=AsyncMock, return_value="good text"),
    ):
        result = await reason_and_act(_base_state())

    # search_web failed but fetch_url succeeded
    assert len(result["retrieved_docs"]) == 1
    assert any("search_web" in e for e in result["errors"])


@pytest.mark.asyncio
async def test_reason_and_act_full_fallback_on_llm_error():
    """If LLM itself fails, fallback to basic web search."""
    llm = MagicMock()
    llm.bind_tools = MagicMock(side_effect=Exception("LLM down"))

    async def fake_search(query, num_results=5):
        return [{"title": "FB", "url": "https://fb.com", "snippet": "s"}]

    with (
        patch("app.graph.nodes.get_llm", return_value=llm),
        patch("app.graph.nodes.web_search", side_effect=fake_search),
    ):
        result = await reason_and_act(
            _base_state(search_queries=["q1"])
        )

    assert len(result["retrieved_docs"]) >= 1
    assert any("reason_and_act" in e for e in result["errors"])


# ── filter / synthesize / format (unchanged from Phase 4) ───


@pytest.mark.asyncio
async def test_filter_evidence_keeps_relevant():
    docs = [
        {"title": "Relevant", "url": "https://a.com", "text": "Good info"},
        {"title": "Irrelevant", "url": "https://b.com", "text": "Noise"},
        {"title": "Also relevant", "url": "https://c.com", "text": "More info"},
    ]

    with patch("app.graph.nodes.get_llm", return_value=_fake_llm("1,3")):
        result = await filter_evidence(_base_state(retrieved_docs=docs))

    assert len(result["filtered_evidence"]) == 2
    assert result["filtered_evidence"][0]["number"] == 1
    assert result["filtered_evidence"][1]["number"] == 3


@pytest.mark.asyncio
async def test_synthesize_answer_produces_draft():
    evidence = [
        {"number": 1, "title": "Source", "url": "https://a.com", "text": "Info about LangGraph"},
    ]

    with patch("app.graph.nodes.get_llm", return_value=_fake_llm("LangGraph is a framework [1].")):
        result = await synthesize_answer(_base_state(filtered_evidence=evidence))

    assert "LangGraph" in result["draft_answer"]


@pytest.mark.asyncio
async def test_format_response_includes_tool_calls_in_output():
    evidence = [
        {"number": 1, "title": "Src", "url": "https://a.com", "text": "content"},
    ]
    state = _base_state(
        filtered_evidence=evidence,
        draft_answer="Answer [1].",
        tool_calls_made=[{"tool": "search_web", "args": {"query": "q"}, "round": 1}],
    )

    result = await format_response(state)

    assert result["answer"] == "Answer [1]."
    assert len(result["sources"]) == 1
    assert result["citations"] == ["[1]"]


# ── Graph compilation & routing ──────────────────────────────


def test_graph_compiles():
    """The graph should compile without errors."""
    compiled = build_graph()
    assert compiled is not None


def test_graph_has_reason_and_act_node():
    """The compiled graph should contain the reason_and_act node."""
    compiled = build_graph()
    node_names = set(compiled.get_graph().nodes.keys())
    assert "reason_and_act" in node_names
    # Old nodes should not be present
    assert "search_sources" not in node_names
    assert "retrieve_chunks" not in node_names


# ── End-to-end graph tests ───────────────────────────────────


@pytest.mark.asyncio
async def test_run_graph_end_to_end():
    """Full workflow with tool-calling — LLM decides to search web."""

    async def fake_search(query, num_results=5):
        return [
            {"title": "LG Docs", "url": "https://docs.example.com", "snippet": "LG info"},
        ]

    fake_model = _fake_sequential_llm([
        "What is LangGraph framework",  # analyze_question
        {  # reason_and_act — tool calls
            "tool_calls": [
                {"name": "search_web", "args": {"query": "What is LangGraph framework"}, "id": "1"},
            ],
        },
        "1",  # filter_evidence
        "LangGraph is a library for building stateful agents [1].",  # synthesize_answer
    ])

    with (
        patch("app.graph.nodes.get_llm", return_value=fake_model),
        patch("app.graph.nodes.web_search", side_effect=fake_search),
    ):
        result = await run_graph("What is LangGraph?")

    assert isinstance(result, PipelineResult)
    assert "LangGraph" in result.answer
    assert len(result.sources) >= 1
    assert result.sources[0].url == "https://docs.example.com"


@pytest.mark.asyncio
async def test_run_graph_with_doc_retrieval():
    """Full workflow where LLM uses internal doc retrieval."""

    fake_doc = MagicMock()
    fake_doc.page_content = "Internal content about LangGraph architecture"
    fake_doc.metadata = {"source": "notes.pdf"}

    fake_model = _fake_sequential_llm([
        "LangGraph architecture",  # analyze_question
        {  # reason_and_act — use internal docs
            "tool_calls": [
                {"name": "retrieve_documents", "args": {"query": "LangGraph architecture"}, "id": "1"},
            ],
        },
        "1",  # filter_evidence
        "LangGraph architecture builds on LangChain [1].",  # synthesize_answer
    ])

    with (
        patch("app.graph.nodes.get_llm", return_value=fake_model),
        patch("app.retrieval.store.search", return_value=[fake_doc]),
    ):
        result = await run_graph("Explain LangGraph architecture")

    assert isinstance(result, PipelineResult)
    assert "LangGraph" in result.answer


@pytest.mark.asyncio
async def test_run_graph_no_results():
    """Graph handles the case where no evidence is gathered."""

    async def empty_search(query, num_results=5):
        return []

    fake_model = _fake_sequential_llm([
        "test query",  # analyze_question
        {  # reason_and_act — search returns nothing
            "tool_calls": [
                {"name": "search_web", "args": {"query": "test query"}, "id": "1"},
            ],
        },
    ])

    with (
        patch("app.graph.nodes.get_llm", return_value=fake_model),
        patch("app.graph.nodes.web_search", side_effect=empty_search),
    ):
        result = await run_graph("Something obscure?")

    assert isinstance(result, PipelineResult)
    assert result.sources == []


@pytest.mark.asyncio
async def test_run_graph_with_more_evidence_loop():
    """Graph loops back through reason_and_act when more evidence requested."""

    call_count = {"n": 0}

    async def fake_search(query, num_results=5):
        call_count["n"] += 1
        return [
            {"title": f"Result{call_count['n']}", "url": f"https://r{call_count['n']}.com", "snippet": "info"},
        ]

    fake_model = _fake_sequential_llm([
        "query",  # analyze_question
        {  # reason_and_act round 1 — search + request more
            "tool_calls": [
                {"name": "search_web", "args": {"query": "query"}, "id": "1"},
                {"name": "request_more_evidence", "args": {"reason": "need more"}, "id": "2"},
            ],
        },
        {  # reason_and_act round 2 — just search
            "tool_calls": [
                {"name": "search_web", "args": {"query": "query refined"}, "id": "3"},
            ],
        },
        "1,2",  # filter_evidence
        "Combined answer [1][2].",  # synthesize_answer
    ])

    with (
        patch("app.graph.nodes.get_llm", return_value=fake_model),
        patch("app.graph.nodes.web_search", side_effect=fake_search),
    ):
        result = await run_graph("Multi-round question?")

    assert isinstance(result, PipelineResult)
    assert len(result.sources) >= 2
