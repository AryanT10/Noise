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
