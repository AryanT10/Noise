from fastapi import APIRouter, HTTPException

from app.chains.llm import _resolve_model
from app.chains.pipeline import run_rag_pipeline
from app.config import settings
from app.graph.workflow import run_graph, run_graph_full
from app.logging import logger
from app.models.schemas import (
    AggregatedAnswer,
    AskRequest,
    AskResponse,
    IngestRequest,
    IngestResponse,
    RAGAskRequest,
    RAGAskResponse,
)
from app.observability.trace_store import trace_store
from app.observability.tracing import get_langsmith_url

router = APIRouter()

_KEY_FOR_PROVIDER = {
    "openai": "openai_api_key",
    "gemini": "google_api_key",
    "groq": "groq_api_key",
}


@router.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    key_attr = _KEY_FOR_PROVIDER.get(settings.llm_provider)
    if not key_attr or not getattr(settings, key_attr, ""):
        raise HTTPException(
            status_code=500,
            detail=f"API key not configured for provider '{settings.llm_provider}'",
        )

    if not settings.serper_api_key:
        raise HTTPException(
            status_code=500,
            detail="SERPER_API_KEY not configured — needed for web search",
        )

    logger.info("Received question: %s", request.question[:80])
    result = await run_graph(request.question)
    return AskResponse(
        answer=result.answer,
        model=_resolve_model(),
        sources=result.sources,
        snippets=result.snippets,
        run_id=result.run_id,
    )


@router.post("/ask/full", response_model=AggregatedAnswer)
async def ask_full(request: AskRequest):
    """Return the full structured aggregation result (claims, evidence, consensus)."""
    key_attr = _KEY_FOR_PROVIDER.get(settings.llm_provider)
    if not key_attr or not getattr(settings, key_attr, ""):
        raise HTTPException(
            status_code=500,
            detail=f"API key not configured for provider '{settings.llm_provider}'",
        )

    if not settings.serper_api_key:
        raise HTTPException(
            status_code=500,
            detail="SERPER_API_KEY not configured — needed for web search",
        )

    logger.info("Received full question: %s", request.question[:80])
    return await run_graph_full(request.question)


# ── Phase 3: Retrieval endpoints ────────────────────────────


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """Ingest URLs and/or raw texts into the vector store."""
    from app.retrieval.ingest import ingest_url, ingest_text

    total_chunks = 0

    for url in request.urls:
        total_chunks += await ingest_url(url)

    for text in request.texts:
        total_chunks += await ingest_text(text)

    logger.info("Ingested %d total chunks", total_chunks)
    return IngestResponse(chunks_added=total_chunks)


@router.post("/ask/rag", response_model=RAGAskResponse)
async def ask_rag(request: RAGAskRequest):
    """Answer a question using only the ingested document store (RAG)."""
    key_attr = _KEY_FOR_PROVIDER.get(settings.llm_provider)
    if not key_attr or not getattr(settings, key_attr, ""):
        raise HTTPException(
            status_code=500,
            detail=f"API key not configured for provider '{settings.llm_provider}'",
        )

    logger.info("RAG question: %s", request.question[:80])
    result = await run_rag_pipeline(request.question, top_k=request.top_k)

    return RAGAskResponse(
        answer=result["answer"],
        model=_resolve_model(),
        chunks_used=result["chunks_used"],
        sources=result["sources"],
    )


@router.delete("/store")
async def clear_store():
    """Delete all ingested documents from the vector store."""
    from app.retrieval.store import clear_store as _clear

    _clear()
    return {"status": "cleared"}


# ── Phase 8: Observability / debug endpoints ─────────────────


@router.get("/traces")
async def list_traces(limit: int = 20):
    """List recent run traces (newest first)."""
    runs = trace_store.list_runs(limit=min(limit, 100))
    langsmith_url = get_langsmith_url()
    return {"runs": runs, "langsmith_url": langsmith_url}


@router.get("/traces/{run_id}")
async def get_trace(run_id: str):
    """Get detailed trace for a specific run (node timing, keys, errors)."""
    detail = trace_store.get_run_detail(run_id)
    if not detail:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    return detail


@router.get("/traces/{run_id}/nodes/{node_name}")
async def get_node_trace(run_id: str, node_name: str):
    """Get full input/output for a specific node in a run."""
    entries = trace_store.get_node_detail(run_id, node_name)
    if entries is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id} not found")
    if not entries:
        raise HTTPException(
            status_code=404,
            detail=f"Node '{node_name}' not found in run {run_id}",
        )
    return {"run_id": run_id, "node": node_name, "executions": entries}


# ── Phase 9: Evaluation endpoints ────────────────────────────


@router.get("/eval/dataset")
async def list_eval_dataset():
    """Return the built-in evaluation test cases."""
    from app.evaluation.dataset import EVAL_DATASET

    return {"cases": [c.model_dump() for c in EVAL_DATASET]}


@router.post("/eval/run")
async def run_evaluation(
    label: str = "",
    case_ids: list[str] | None = None,
    use_llm_judge: bool = True,
):
    """Run the eval suite (or a subset) and return the report.

    Query params:
      - label: human name for this run (e.g. "baseline")
      - case_ids: JSON list of case IDs to run; omit for all
      - use_llm_judge: set False for fast deterministic-only scoring
    """
    from app.evaluation.runner import run_eval

    report = await run_eval(
        label=label,
        case_ids=case_ids,
        use_llm_judge=use_llm_judge,
    )
    return report.model_dump()


@router.post("/eval/compare")
async def compare_evaluations(
    baseline_label: str = "baseline",
    candidate_label: str = "candidate",
    baseline_case_ids: list[str] | None = None,
    candidate_case_ids: list[str] | None = None,
    use_llm_judge: bool = True,
):
    """Run two eval suites back-to-back and return a comparison report."""
    from app.evaluation.runner import compare_reports, run_eval

    baseline = await run_eval(
        label=baseline_label,
        case_ids=baseline_case_ids,
        use_llm_judge=use_llm_judge,
    )
    candidate = await run_eval(
        label=candidate_label,
        case_ids=candidate_case_ids,
        use_llm_judge=use_llm_judge,
    )
    comparison = compare_reports(baseline, candidate)
    return {
        "baseline": baseline.model_dump(),
        "candidate": candidate.model_dump(),
        "comparison": comparison.model_dump(),
    }
