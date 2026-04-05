"""Simple web page text extractor using stdlib html.parser."""

import httpx
from html.parser import HTMLParser

from app.logging import logger


class _HTMLTextExtractor(HTMLParser):
    """Strips HTML tags and extracts visible text."""

    _SKIP_TAGS = frozenset({"script", "style", "noscript", "head", "meta", "link"})

    def __init__(self) -> None:
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        raw = " ".join(self._parts)
        return " ".join(raw.split())  # collapse whitespace


def _html_to_text(html: str) -> str:
    extractor = _HTMLTextExtractor()
    extractor.feed(html)
    return extractor.get_text()


async def extract_page_text(url: str, max_chars: int = 6000) -> str:
    """Fetch a URL and return its visible text content, truncated to max_chars."""
    try:
        async with httpx.AsyncClient(
            timeout=10,
            follow_redirects=True,
            headers={"User-Agent": "Noise/0.1 (research assistant)"},
        ) as client:
            resp = await client.get(url)
            resp.raise_for_status()

        text = _html_to_text(resp.text)
        if len(text) > max_chars:
            text = text[:max_chars] + "…"

        logger.info("Extracted %d chars from %s", len(text), url)
        return text
    except Exception as exc:
        logger.warning("Failed to extract text from %s: %s", url, exc)
        return ""
