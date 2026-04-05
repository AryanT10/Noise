"""Phase 2 — hardcoded single-question pipeline.

Flow: question → search → extract → summarize → answer + citations

Phase 3 adds: question → retrieve from vector store → synthesize grounded answer
"""

import asyncio

from langchain_core.prompts import ChatPromptTemplate

from app.chains.llm import get_llm
from app.logging import logger
from app.models.schemas import PipelineResult, Source, Snippet
from app.tools.search import web_search
from app.tools.scraper import extract_page_text

_SUMMARIZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a research assistant. Given a user question and source material "
                "extracted from web pages, produce a clear, concise answer.\n\n"
                "Rules:\n"
                "- Ground every claim in the provided sources. Do NOT hallucinate.\n"
                "- Cite sources inline using [1], [2], etc. matching the source numbers.\n"
                "- If the sources don't contain enough info, say so honestly.\n"
                "- Keep the answer readable in under 30 seconds.\n"
            ),
        ),
        (
            "human",
            (
                "Question: {question}\n\n"
                "Sources:\n{sources_block}\n\n"
                "Write your answer with inline citations."
            ),
        ),
    ]
)

NUM_SEARCH_RESULTS = 5
NUM_PAGES_TO_SCRAPE = 3


async def run_pipeline(question: str) -> PipelineResult:
    """Execute the full search → extract → summarize pipeline."""

    # ── Step 1: Search ───────────────────────────────────────────
    logger.info("Pipeline step 1/3: searching")
    search_results = await web_search(question, num_results=NUM_SEARCH_RESULTS)

    if not search_results:
        return PipelineResult(
            answer="I couldn't find any search results for your question.",
            sources=[],
            snippets=[],
        )

    # ── Step 2: Extract text from top pages (parallel) ───────────
    logger.info("Pipeline step 2/3: extracting text from top %d pages", NUM_PAGES_TO_SCRAPE)
    urls_to_scrape = [r["url"] for r in search_results[:NUM_PAGES_TO_SCRAPE]]
    extracted_texts = await asyncio.gather(
        *(extract_page_text(url) for url in urls_to_scrape)
    )

    # Build sources, snippets, and the context block for the LLM
    sources: list[Source] = []
    snippets: list[Snippet] = []
    sources_block_parts: list[str] = []

    for i, (result, text) in enumerate(
        zip(search_results[:NUM_PAGES_TO_SCRAPE], extracted_texts), start=1
    ):
        sources.append(Source(number=i, title=result["title"], url=result["url"]))
        content = text if text else result["snippet"]
        snippets.append(Snippet(source_number=i, text=content[:2000]))
        sources_block_parts.append(
            f"[{i}] {result['title']}\nURL: {result['url']}\n{content[:2000]}\n"
        )

    # Include remaining search results as snippet-only sources
    for i, result in enumerate(
        search_results[NUM_PAGES_TO_SCRAPE:],
        start=NUM_PAGES_TO_SCRAPE + 1,
    ):
        sources.append(Source(number=i, title=result["title"], url=result["url"]))
        snippets.append(Snippet(source_number=i, text=result["snippet"]))
        sources_block_parts.append(
            f"[{i}] {result['title']}\nURL: {result['url']}\n{result['snippet']}\n"
        )

    sources_block = "\n".join(sources_block_parts)

    # ── Step 3: Summarize with LLM ──────────────────────────────
    logger.info("Pipeline step 3/3: summarizing with LLM")
    llm = get_llm()
    chain = _SUMMARIZE_PROMPT | llm
    response = await chain.ainvoke(
        {"question": question, "sources_block": sources_block}
    )

    return PipelineResult(
        answer=response.content,
        sources=sources,
        snippets=snippets,
    )


# ── Phase 3: RAG pipeline ───────────────────────────────────

_RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a research assistant. Answer the user's question using ONLY "
                "the retrieved context below. Do NOT make up information.\n\n"
                "Rules:\n"
                "- Cite sources inline using [source: <source>] when available.\n"
                "- If the context doesn't contain enough info, say so honestly.\n"
                "- Keep the answer concise and readable.\n"
            ),
        ),
        (
            "human",
            (
                "Question: {question}\n\n"
                "Retrieved context:\n{context}\n\n"
                "Answer based on the context above."
            ),
        ),
    ]
)


async def run_rag_pipeline(question: str, top_k: int = 5) -> dict:
    """Retrieve relevant chunks from the vector store and synthesize an answer."""
    from app.retrieval.store import search as vector_search

    # ── Step 1: Retrieve ─────────────────────────────────────
    logger.info("RAG step 1/2: retrieving top %d chunks", top_k)
    chunks = vector_search(question, k=top_k)

    if not chunks:
        return {
            "answer": "No documents have been ingested yet. Use /api/ingest first.",
            "chunks_used": 0,
            "sources": [],
        }

    # ── Step 2: Build context and synthesize ─────────────────
    logger.info("RAG step 2/2: synthesizing answer from %d chunks", len(chunks))
    context_parts = []
    seen_sources: list[str] = []
    for i, doc in enumerate(chunks, start=1):
        source = doc.metadata.get("source", "unknown")
        if source not in seen_sources:
            seen_sources.append(source)
        context_parts.append(
            f"[Chunk {i} | source: {source}]\n{doc.page_content}\n"
        )

    context = "\n".join(context_parts)

    llm = get_llm()
    chain = _RAG_PROMPT | llm
    response = await chain.ainvoke({"question": question, "context": context})

    return {
        "answer": response.content,
        "chunks_used": len(chunks),
        "sources": seen_sources,
    }
