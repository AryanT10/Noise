"""LangGraph workflow — compiles the nodes into an executable graph."""

from langgraph.graph import StateGraph, START, END

from app.graph.nodes import (
    analyze_question,
    reason_and_act,
    filter_evidence,
    synthesize_answer,
    format_response,
)
from app.graph.state import GraphState
from app.logging import logger
from app.models.schemas import PipelineResult


def _after_reasoning(state: GraphState) -> str:
    """Route after reason_and_act: loop back, filter, or skip to format."""
    if state.get("needs_more_evidence"):
        return "reason_and_act"
    if state.get("retrieved_docs"):
        return "filter_evidence"
    return "format_response"


def _has_evidence(state: GraphState) -> str:
    """Route after filter_evidence: skip synthesis if no relevant evidence."""
    if state.get("filtered_evidence"):
        return "synthesize_answer"
    return "format_response"


def build_graph() -> StateGraph:
    """Construct and compile the question-answering graph.

    Flow:
        START → analyze_question → reason_and_act ─┐
                                    ↑               │
                                    └── (needs more) ┘
                                         │ (has docs)
                                         ↓
                                  filter_evidence → synthesize_answer → format_response → END
    """
    graph = StateGraph(GraphState)

    # ── Add nodes ────────────────────────────────────────────
    graph.add_node("analyze_question", analyze_question)
    graph.add_node("reason_and_act", reason_and_act)
    graph.add_node("filter_evidence", filter_evidence)
    graph.add_node("synthesize_answer", synthesize_answer)
    graph.add_node("format_response", format_response)

    # ── Add edges ────────────────────────────────────────────
    graph.add_edge(START, "analyze_question")
    graph.add_edge("analyze_question", "reason_and_act")

    # Controlled loop: reason → (loop | filter | skip)
    graph.add_conditional_edges(
        "reason_and_act",
        _after_reasoning,
        {
            "reason_and_act": "reason_and_act",
            "filter_evidence": "filter_evidence",
            "format_response": "format_response",
        },
    )

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
