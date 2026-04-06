"""source_reader — normalise filtered evidence into uniform EvidenceItem objects."""

from app.logging import logger
from app.models.schemas import EvidenceItem


def read_sources(filtered_evidence: list[dict]) -> list[EvidenceItem]:
    """Convert raw evidence dicts into structured EvidenceItem models.

    Each dict is expected to carry at minimum: number, title, url, text.
    Malformed entries are logged and skipped.
    """
    items: list[EvidenceItem] = []
    for entry in filtered_evidence:
        try:
            items.append(
                EvidenceItem(
                    source_number=entry["number"],
                    title=entry.get("title", ""),
                    url=entry.get("url", ""),
                    text=(entry.get("text") or "")[:4000],
                )
            )
        except (KeyError, TypeError) as exc:
            logger.warning("source_reader skipped malformed entry: %s", exc)

    logger.info("source_reader: %d items normalised", len(items))
    return items
