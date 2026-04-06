"""Graph nodes — each function takes GraphState, returns a partial update."""

import asyncio

from langchain_core.prompts import ChatPromptTemplate

from app.chains.llm import get_llm
from app.graph.state import GraphState
from app.logging import logger
from app.models.schemas import Source, Snippet
from app.tools.scraper import extract_page_text
from app.tools.search import web_search

NUM_SEARCH_RESULTS = 5
NUM_PAGES_TO_SCRAPE = 3

# ── Prompts ──────────────────────────────────────────────────

_ANALYZE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a search-query planner. Given a user question, produce 1–3 "
                "concise web search queries that would help answer the question.\n\n"
                "Return ONLY the queries, one per line. No numbering, no commentary."
            ),
        ),
        ("human", "{question}"),
    ]
)

_FILTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a relevance filter. Given a user question and a set of source "
                "excerpts, return ONLY the numbers of sources that are relevant to "
                "answering the question.\n\n"
                "Return a comma-separated list of source numbers (e.g. '1,3,4'). "
                "If none are relevant, return 'NONE'."
            ),
        ),
        (
            "human",
            "Question: {question}\n\nSources:\n{sources_block}\n\nRelevant source numbers:",
        ),
    ]
)

_SYNTHESIZE_PROMPT = ChatPromptTemplate.from_messages(
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


# ── Node functions ───────────────────────────────────────────


async def analyze_question(state: GraphState) -> dict:
    """Analyze the user question and generate search queries."""
    question = state["question"]
    logger.info("Node [analyze_question]: %s", question[:80])

    try:
        llm = get_llm()
        chain = _ANALYZE_PROMPT | llm
        response = await chain.ainvoke({"question": question})
        queries = [q.strip() for q in response.content.strip().splitlines() if q.strip()]

        if not queries:
            queries = [question]

        logger.info("Generated %d search queries", len(queries))
        return {"search_queries": queries, "errors": state.get("errors", [])}
    except Exception as exc:
        logger.error("analyze_question failed: %s", exc)
        return {
            "search_queries": [question],
            "errors": [*state.get("errors", []), f"analyze_question: {exc}"],
        }


async def search_sources(state: GraphState) -> dict:
    """Run web searches for each generated query."""
    queries = state.get("search_queries", [state["question"]])
    logger.info("Node [search_sources]: %d queries", len(queries))

    all_results: list[dict] = []
    seen_urls: set[str] = set()
    errors = list(state.get("errors", []))

    for query in queries:
        try:
            results = await web_search(query, num_results=NUM_SEARCH_RESULTS)
            for r in results:
                if r["url"] not in seen_urls:
                    seen_urls.add(r["url"])
                    all_results.append(r)
        except Exception as exc:
            logger.warning("Search failed for query '%s': %s", query[:40], exc)
            errors.append(f"search_sources ({query[:40]}): {exc}")

    logger.info("Collected %d unique search results", len(all_results))
    return {"search_results": all_results, "errors": errors}


async def retrieve_chunks(state: GraphState) -> dict:
    """Scrape the top pages to get full text content."""
    search_results = state.get("search_results", [])
    logger.info("Node [retrieve_chunks]: scraping top %d pages", NUM_PAGES_TO_SCRAPE)

    urls_to_scrape = [r["url"] for r in search_results[:NUM_PAGES_TO_SCRAPE]]
    extracted_texts = await asyncio.gather(
        *(extract_page_text(url) for url in urls_to_scrape)
    )

    docs: list[dict] = []
    for result, text in zip(search_results[:NUM_PAGES_TO_SCRAPE], extracted_texts):
        docs.append({
            "title": result["title"],
            "url": result["url"],
            "text": text if text else result["snippet"],
        })

    # Include remaining results as snippet-only docs
    for result in search_results[NUM_PAGES_TO_SCRAPE:]:
        docs.append({
            "title": result["title"],
            "url": result["url"],
            "text": result["snippet"],
        })

    logger.info("Retrieved %d documents", len(docs))
    return {"retrieved_docs": docs, "errors": state.get("errors", [])}


async def filter_evidence(state: GraphState) -> dict:
    """Use the LLM to filter retrieved docs for relevance."""
    docs = state.get("retrieved_docs", [])
    question = state["question"]
    logger.info("Node [filter_evidence]: filtering %d docs", len(docs))

    if not docs:
        return {"filtered_evidence": [], "errors": state.get("errors", [])}

    # Build numbered source block for the LLM
    parts = []
    for i, doc in enumerate(docs, start=1):
        parts.append(f"[{i}] {doc['title']}\n{doc['text'][:1500]}")
    sources_block = "\n\n".join(parts)

    try:
        llm = get_llm()
        chain = _FILTER_PROMPT | llm
        response = await chain.ainvoke(
            {"question": question, "sources_block": sources_block}
        )

        raw = response.content.strip()
        if raw.upper() == "NONE":
            relevant_nums = []
        else:
            relevant_nums = []
            for token in raw.replace(" ", "").split(","):
                try:
                    n = int(token)
                    if 1 <= n <= len(docs):
                        relevant_nums.append(n)
                except ValueError:
                    continue

        # If LLM returned nothing useful, keep all docs
        if not relevant_nums:
            relevant_nums = list(range(1, len(docs) + 1))

        filtered = []
        for n in relevant_nums:
            doc = docs[n - 1]
            filtered.append({
                "number": n,
                "title": doc["title"],
                "url": doc["url"],
                "text": doc["text"][:2000],
            })

        logger.info("Kept %d/%d sources after filtering", len(filtered), len(docs))
        return {"filtered_evidence": filtered, "errors": state.get("errors", [])}

    except Exception as exc:
        logger.warning("filter_evidence LLM call failed: %s — keeping all docs", exc)
        filtered = [
            {"number": i, "title": d["title"], "url": d["url"], "text": d["text"][:2000]}
            for i, d in enumerate(docs, start=1)
        ]
        return {
            "filtered_evidence": filtered,
            "errors": [*state.get("errors", []), f"filter_evidence: {exc}"],
        }


async def synthesize_answer(state: GraphState) -> dict:
    """Generate a draft answer from filtered evidence."""
    evidence = state.get("filtered_evidence", [])
    question = state["question"]
    logger.info("Node [synthesize_answer]: %d evidence pieces", len(evidence))

    if not evidence:
        return {
            "draft_answer": "I couldn't find enough relevant information to answer your question.",
            "errors": state.get("errors", []),
        }

    sources_block = "\n\n".join(
        f"[{e['number']}] {e['title']}\nURL: {e['url']}\n{e['text']}"
        for e in evidence
    )

    try:
        llm = get_llm()
        chain = _SYNTHESIZE_PROMPT | llm
        response = await chain.ainvoke(
            {"question": question, "sources_block": sources_block}
        )
        logger.info("Draft answer generated (%d chars)", len(response.content))
        return {"draft_answer": response.content, "errors": state.get("errors", [])}

    except Exception as exc:
        logger.error("synthesize_answer failed: %s", exc)
        return {
            "draft_answer": "Sorry, I encountered an error generating the answer.",
            "errors": [*state.get("errors", []), f"synthesize_answer: {exc}"],
        }


async def format_response(state: GraphState) -> dict:
    """Pack the final response into the output shape."""
    logger.info("Node [format_response]")

    evidence = state.get("filtered_evidence", [])
    draft = state.get("draft_answer", "No answer was generated.")

    sources = [
        Source(number=e["number"], title=e["title"], url=e["url"])
        for e in evidence
    ]
    snippets = [
        Snippet(source_number=e["number"], text=e["text"][:2000])
        for e in evidence
    ]

    # Extract citation markers like [1], [2] from the draft
    import re
    citation_markers = sorted(set(re.findall(r"\[\d+\]", draft)))

    return {
        "answer": draft,
        "sources": sources,
        "snippets": snippets,
        "citations": citation_markers,
        "errors": state.get("errors", []),
    }
