"""Web search tool using Serper API (google.serper.dev)."""

import httpx

from app.config import settings
from app.logging import logger

SERPER_URL = "https://google.serper.dev/search"


async def web_search(query: str, num_results: int = 5) -> list[dict]:
    """Search the web via Serper and return organic results.

    Returns a list of dicts with keys: title, url, snippet.
    """
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            SERPER_URL,
            json={"q": query, "num": num_results},
            headers={
                "X-API-KEY": settings.serper_api_key,
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()

    data = resp.json()
    results = []
    for item in data.get("organic", [])[:num_results]:
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            }
        )

    logger.info("Search returned %d results for: %s", len(results), query[:60])
    return results
