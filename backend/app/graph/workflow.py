"""LangGraph workflow — compiles the nodes into an executable graph."""

from langgraph.graph import StateGraph, START, END

from app.graph.nodes import (
    analyze_question,
    search_sources,
    retrieve_chunks,
    filter_evidence,
    synthesize_answer,
    format_response,
)
from app.graph.state import GraphState
from app.logging import logger
from app.models.schemas import PipelineResult


def _has_search_results(state: GraphState) -> str:
    """Route after search_sources: skip scraping if no results found."""
    if state.get("search_results"):
        return "retrieve_chunks"
    return "format_response"


def _has_evidence(state: GraphState) -> str:
    """Route after filter_evidence: skip synthesis if no relevant evidence."""
    if state.get("filtered_evidence"):
        return "synthesize_answer"
    return "format_response"


def build_graph() -> StateGraph:
    """Construct and compile the question-answering graph."""
    graph = StateGraph(GraphState)

    # ── Add nodes ────────────────────────────────────────────
    graph.add_node("analyze_question", analyze_question)
    graph.add_node("search_sources", search_sources)
    graph.add_node("retrieve_chunks", retrieve_chunks)
    graph.add_node("filter_evidence", filter_evidence)
    graph.add_node("synthesize_answer", synthesize_answer)
    graph.add_node("format_response", format_response)

    # ── Add edges ────────────────────────────────────────────
    graph.add_edge(START, "analyze_question")
    graph.add_edge("analyze_question", "search_sources")

    # Conditional: if search returned results → scrape; otherwise → format
    graph.add_conditional_edges(
        "search_sources",
        _has_search_results,
        {"retrieve_chunks": "retrieve_chunks", "format_response": "format_response"},
    )

    graph.add_edge("retrieve_chunks", "filter_evidence")

    # Conditional: if evidence exists → synthesize; otherwise → format
    graph.add_conditional_edges(
        "filter_evidence",
        _has_evidence,
        {"synthesize_answer": "synthesize_answer", "format_response": "format_response"},
    )

    graph.add_edge("synthesize_answer", "format_response")
    graph.add_edge("format_response", END)

    return graph.compile()


# Module-level compiled graph (reused across requests)
qa_graph = build_graph()


async def run_graph(question: str) -> PipelineResult:
    """Run the full LangGraph workflow for a single question."""
    logger.info("Running LangGraph workflow for: %s", question[:80])

    result = await qa_graph.ainvoke({"question": question, "errors": []})

    if result.get("errors"):
        logger.warning("Graph completed with errors: %s", result["errors"])

    return PipelineResult(
        answer=result.get("answer", "No answer generated."),
        sources=result.get("sources", []),
        snippets=result.get("snippets", []),
    )
