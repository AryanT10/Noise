"""LangGraph shared state — the single object flowing through every node."""

from typing import TypedDict

from app.models.schemas import Source, Snippet, Claim, EvidenceItem, ConsensusGroup


class GraphState(TypedDict, total=False):
    """Shared state for the Noise question-answering graph.

    Every node reads from and writes to this dict.
    Fields use `total=False` so nodes only need to set the keys they own.
    """

    # ── Input ────────────────────────────────────────────────────
    question: str

    # ── analyze_question → search_sources ────────────────────────
    search_queries: list[str]

    # ── search_sources → retrieve_chunks ─────────────────────────
    search_results: list[dict]  # [{title, url, snippet}, ...]

    # ── retrieve_chunks → filter_evidence ────────────────────────
    retrieved_docs: list[dict]  # [{title, url, text}, ...]

    # ── filter_evidence → synthesize_answer ──────────────────────
    filtered_evidence: list[dict]  # [{number, title, url, text}, ...]

    # ── synthesize_answer → format_response ──────────────────────
    draft_answer: str

    # ── format_response (final output) ───────────────────────────
    answer: str
    sources: list[Source]
    snippets: list[Snippet]
    citations: list[str]

    # ── Phase 5: tool-calling (reason_and_act node) ────────────
    tool_calls_made: list[dict]   # [{tool, args, round}, ...]
    reasoning_rounds: int         # how many reason_and_act rounds ran
    needs_more_evidence: bool     # LLM wants another gathering round

    # ── Phase 6: answer aggregation ──────────────────────────────
    claims: list[Claim]
    ranked_evidence: list[EvidenceItem]
    consensus_groups: list[ConsensusGroup]
    disagreements: list[str]
    uncertainties: list[str]

    # ── Errors (accumulated by any node) ─────────────────────────
    errors: list[str]
