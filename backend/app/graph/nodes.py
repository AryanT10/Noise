"""Graph nodes — each function takes GraphState, returns a partial update."""

import asyncio
import re as _re

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from app.chains.llm import get_llm
from app.graph.state import GraphState
from app.logging import logger
from app.models.schemas import Source, Snippet
from app.tools.definitions import REASONING_TOOLS
from app.tools.scraper import extract_page_text
from app.tools.search import web_search

from app.aggregation.source_reader import read_sources
from app.aggregation.claim_extractor import extract_claims
from app.aggregation.evidence_ranker import rank_evidence
from app.aggregation.consensus_builder import build_consensus
from app.aggregation.final_writer import write_final_answer

NUM_SEARCH_RESULTS = 5
NUM_PAGES_TO_SCRAPE = 3
MAX_REASONING_ROUNDS = 2

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

# ── Phase 5: Reasoning node prompt ──────────────────────────

_REASON_SYSTEM = (
    "You are a research assistant deciding how to gather evidence for answering "
    "a user question. You have access to the following tools:\n\n"
    "- search_web: Search the internet for current information\n"
    "- fetch_url: Get full text content from a specific URL\n"
    "- retrieve_documents: Search the internal document store\n"
    "- format_citations: Format sources into citations\n"
    "- request_more_evidence: Ask for another gathering round if evidence is insufficient\n\n"
    "Choose the most appropriate tool(s) for the question. Be strategic:\n"
    "- For factual/current questions, use search_web\n"
    "- For questions about internal/ingested content, use retrieve_documents\n"
    "- Use fetch_url only when you need full content from a known URL\n"
    "- Use request_more_evidence only if current evidence is clearly insufficient\n\n"
    "You MUST call at least one data-gathering tool. Do not try to answer yourself."
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


# ── Phase 5: controlled tool-calling node ────────────────────


async def reason_and_act(state: GraphState) -> dict:
    """Reasoning node — uses LLM tool-calling to decide how to gather evidence.

    The LLM chooses which tools to invoke (web search, doc retrieval, URL fetch).
    Tool calls are executed in a single controlled round.  The graph caps the
    total number of rounds at MAX_REASONING_ROUNDS to prevent runaway loops.
    """
    question = state["question"]
    queries = state.get("search_queries", [question])
    rounds = state.get("reasoning_rounds", 0)
    prior_docs = state.get("retrieved_docs", [])
    errors = list(state.get("errors", []))
    tool_calls_log = list(state.get("tool_calls_made", []))

    logger.info("Node [reason_and_act] round=%d", rounds + 1)

    try:
        llm = get_llm()
        llm_with_tools = llm.bind_tools(REASONING_TOOLS)

        # Build context from any prior-round docs
        prior_text = ""
        if prior_docs:
            summaries = [f"  - {d['title']}: {d['text'][:200]}…" for d in prior_docs[:5]]
            prior_text = "\nEvidence gathered so far:\n" + "\n".join(summaries)

        messages = [
            SystemMessage(content=_REASON_SYSTEM),
            HumanMessage(
                content=(
                    f"Question: {question}\n"
                    f"Suggested search queries: {', '.join(queries)}"
                    f"{prior_text}"
                )
            ),
        ]

        response = await llm_with_tools.ainvoke(messages)

        new_docs = list(prior_docs)
        seen_urls = {d["url"] for d in new_docs}
        needs_more = False

        if response.tool_calls:
            for tc in response.tool_calls:
                name = tc["name"]
                args = tc["args"]
                tool_calls_log.append({"tool": name, "args": args, "round": rounds + 1})
                logger.info("Tool call: %s(%s)", name, str(args)[:100])

                try:
                    if name == "search_web":
                        results = await web_search(
                            args.get("query", queries[0]),
                            num_results=NUM_SEARCH_RESULTS,
                        )
                        for r in results:
                            if r["url"] not in seen_urls:
                                seen_urls.add(r["url"])
                                new_docs.append({
                                    "title": r["title"],
                                    "url": r["url"],
                                    "text": r["snippet"],
                                })

                    elif name == "fetch_url":
                        url = args.get("url", "")
                        if url:
                            text = await extract_page_text(url)
                            if url not in seen_urls:
                                seen_urls.add(url)
                                new_docs.append({
                                    "title": url,
                                    "url": url,
                                    "text": text or "",
                                })
                            else:
                                # Upgrade existing snippet with full text
                                for d in new_docs:
                                    if d["url"] == url:
                                        d["text"] = text or d["text"]
                                        break

                    elif name == "retrieve_documents":
                        from app.retrieval.store import search as store_search

                        chunks = store_search(args.get("query", question), k=5)
                        for chunk in chunks:
                            src = chunk.metadata.get("source", "internal")
                            new_docs.append({
                                "title": f"Internal: {src}",
                                "url": src,
                                "text": chunk.page_content[:2000],
                            })

                    elif name == "request_more_evidence":
                        needs_more = True
                        logger.info(
                            "LLM requested more evidence: %s",
                            args.get("reason", ""),
                        )

                    elif name == "format_citations":
                        pass  # Formatting-only tool — no docs to gather

                except Exception as exc:
                    logger.warning("Tool %s failed: %s", name, exc)
                    errors.append(f"tool {name}: {exc}")
        else:
            # No tool calls — fall back to basic web search with generated queries
            logger.info("No tool calls from LLM — falling back to web search")
            for query in queries[:2]:
                try:
                    results = await web_search(query, num_results=NUM_SEARCH_RESULTS)
                    for r in results:
                        if r["url"] not in seen_urls:
                            seen_urls.add(r["url"])
                            new_docs.append({
                                "title": r["title"],
                                "url": r["url"],
                                "text": r["snippet"],
                            })
                except Exception as exc:
                    errors.append(f"fallback search: {exc}")

        # Enforce round cap
        if needs_more and (rounds + 1) >= MAX_REASONING_ROUNDS:
            needs_more = False
            logger.info("Max reasoning rounds reached — proceeding with current evidence")

        return {
            "retrieved_docs": new_docs,
            "tool_calls_made": tool_calls_log,
            "reasoning_rounds": rounds + 1,
            "needs_more_evidence": needs_more,
            "errors": errors,
        }

    except Exception as exc:
        logger.error("reason_and_act failed: %s", exc)
        # Full fallback: basic web search
        new_docs = list(prior_docs)
        seen_urls = {d["url"] for d in new_docs}
        for query in queries[:2]:
            try:
                results = await web_search(query, num_results=NUM_SEARCH_RESULTS)
                for r in results:
                    if r["url"] not in seen_urls:
                        seen_urls.add(r["url"])
                        new_docs.append({
                            "title": r["title"],
                            "url": r["url"],
                            "text": r["snippet"],
                        })
            except Exception:
                pass

        return {
            "retrieved_docs": new_docs,
            "tool_calls_made": tool_calls_log,
            "reasoning_rounds": rounds + 1,
            "needs_more_evidence": False,
            "errors": [*errors, f"reason_and_act: {exc}"],
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
    citation_markers = sorted(set(_re.findall(r"\[\d+\]", draft)))

    return {
        "answer": draft,
        "sources": sources,
        "snippets": snippets,
        "citations": citation_markers,
        "errors": state.get("errors", []),
    }


# ── Phase 6: answer aggregation node ────────────────────────


async def aggregate_answer(state: GraphState) -> dict:
    """Run the full aggregation pipeline: read → extract → rank → consensus → write.

    Replaces the simpler synthesize_answer when the graph has enough evidence.
    """
    evidence_dicts = state.get("filtered_evidence", [])
    question = state["question"]
    errors = list(state.get("errors", []))

    logger.info("Node [aggregate_answer]: %d evidence pieces", len(evidence_dicts))

    if not evidence_dicts:
        return {
            "draft_answer": "I couldn't find enough relevant information to answer your question.",
            "claims": [],
            "ranked_evidence": [],
            "consensus_groups": [],
            "disagreements": [],
            "uncertainties": [],
            "errors": errors,
        }

    try:
        # Step 1 — normalise sources
        evidence_items = read_sources(evidence_dicts)

        # Step 2 — extract claims from each source
        claims = await extract_claims(question, evidence_items)

        # Step 3 — rank evidence by quality
        ranked = await rank_evidence(question, evidence_items)

        # Step 4 — build consensus, find disagreements & uncertainties
        groups, disagreements, uncertainties = await build_consensus(
            question, claims
        )

        # Step 5 — write the final aggregated judgment
        answer = await write_final_answer(
            question, groups, disagreements, uncertainties, ranked
        )

        logger.info(
            "aggregate_answer: %d claims, %d groups, %d disagreements",
            len(claims),
            len(groups),
            len(disagreements),
        )

        return {
            "draft_answer": answer,
            "claims": claims,
            "ranked_evidence": ranked,
            "consensus_groups": groups,
            "disagreements": disagreements,
            "uncertainties": uncertainties,
            "errors": errors,
        }

    except Exception as exc:
        logger.error("aggregate_answer failed — falling back to synthesize: %s", exc)
        errors.append(f"aggregate_answer: {exc}")
        # Fall back to the simpler synthesize logic
        return await synthesize_answer({**state, "errors": errors})
