"""LangChain tool definitions for Phase 5 — controlled tool-calling.

Each tool wraps an existing capability (web search, URL fetch, doc retrieval)
behind a LangChain @tool interface so the LLM can choose which to invoke
within a single controlled reasoning node.
"""

from langchain_core.tools import tool

from app.logging import logger


@tool
async def search_web(query: str) -> str:
    """Search the web for current information about a topic.

    Use when you need up-to-date facts or information from the internet.
    Returns titles, URLs, and snippets from search results.
    """
    from app.tools.search import web_search

    results = await web_search(query, num_results=5)
    if not results:
        return "No web search results found."

    parts = []
    for i, r in enumerate(results, 1):
        parts.append(f"[{i}] {r['title']}\nURL: {r['url']}\n{r['snippet']}")
    return "\n\n".join(parts)


@tool
async def fetch_url(url: str) -> str:
    """Fetch and extract the visible text content from a web page URL.

    Use when you have a specific URL and need its full content,
    e.g. to get more detail from a search result.
    """
    from app.tools.scraper import extract_page_text

    text = await extract_page_text(url)
    if not text:
        return f"Could not extract text from {url}."
    return text


@tool
def retrieve_documents(query: str) -> str:
    """Search the internal document store (vector database) for relevant chunks.

    Use when the question might be answered by previously ingested documents
    in the local knowledge base.
    """
    from app.retrieval.store import search

    docs = search(query, k=5)
    if not docs:
        return "No relevant documents found in the internal store."

    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Doc {i}] Source: {source}\n{doc.page_content[:1500]}")
    return "\n\n".join(parts)


@tool
def format_citations(source_list: str) -> str:
    """Format a list of sources into a clean numbered citation block.

    Input should be one source per line, e.g.:
      Title One | https://example.com/one
      Title Two | https://example.com/two
    """
    lines = [line.strip() for line in source_list.strip().splitlines() if line.strip()]
    if not lines:
        return "No sources provided."

    formatted = []
    for i, line in enumerate(lines, 1):
        formatted.append(f"[{i}] {line}")
    return "\n".join(formatted)


@tool
def request_more_evidence(reason: str) -> str:
    """Signal that current evidence is insufficient and another gathering round is needed.

    Only use this if the gathered information clearly cannot answer the question.
    Explain briefly what additional information is needed.
    """
    return f"More evidence requested: {reason}"


# All tools available to the reasoning node
REASONING_TOOLS = [search_web, fetch_url, retrieve_documents, format_citations, request_more_evidence]
