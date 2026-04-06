"""evidence_ranker — score each source's quality and sort by relevance."""

import json

from langchain_core.prompts import ChatPromptTemplate

from app.chains.llm import get_llm
from app.logging import logger
from app.models.schemas import EvidenceItem

_RANK_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a source-quality assessor. Given a user question and a list of "
                "sources, rate EACH source on a scale of 0.0 to 1.0 for quality and "
                "relevance. Consider:\n"
                "- Directness: does it answer the question?\n"
                "- Specificity: concrete facts vs vague statements?\n"
                "- Credibility: authoritative domain, detailed content?\n\n"
                "Return a JSON array of objects, one per source, each with:\n"
                '  "source_number": int,\n'
                '  "quality_score": float (0.0–1.0),\n'
                '  "quality_reason": short explanation.\n\n'
                "Return ONLY valid JSON — no markdown fences, no commentary."
            ),
        ),
        (
            "human",
            "Question: {question}\n\nSources:\n{sources_block}",
        ),
    ]
)


async def rank_evidence(
    question: str,
    evidence: list[EvidenceItem],
) -> list[EvidenceItem]:
    """Score each evidence item for quality and return sorted (best first)."""
    if not evidence:
        return []

    sources_block = "\n\n".join(
        f"[{e.source_number}] {e.title}\nURL: {e.url}\n{e.text[:1500]}"
        for e in evidence
    )

    try:
        llm = get_llm()
        chain = _RANK_PROMPT | llm
        response = await chain.ainvoke(
            {"question": question, "sources_block": sources_block}
        )

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        scores = json.loads(raw)
        if not isinstance(scores, list):
            scores = [scores]

        score_map: dict[int, tuple[float, str]] = {}
        for entry in scores:
            num = entry.get("source_number")
            score_map[num] = (
                max(0.0, min(1.0, float(entry.get("quality_score", 0.5)))),
                entry.get("quality_reason", ""),
            )

        for item in evidence:
            if item.source_number in score_map:
                item.quality_score, item.quality_reason = score_map[item.source_number]
            else:
                item.quality_score = 0.5
                item.quality_reason = "Not scored by LLM"

    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("evidence_ranker LLM scoring failed: %s — using defaults", exc)
        for item in evidence:
            item.quality_score = 0.5
            item.quality_reason = "Scoring unavailable"

    ranked = sorted(evidence, key=lambda e: e.quality_score, reverse=True)
    logger.info(
        "evidence_ranker: ranked %d items (top score=%.2f)",
        len(ranked),
        ranked[0].quality_score if ranked else 0,
    )
    return ranked
