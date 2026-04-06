"""evidence_ranker — score each source's quality and sort by relevance."""

from langchain_core.prompts import ChatPromptTemplate

from app.chains.llm import get_structured_llm
from app.logging import logger
from app.models.schemas import EvidenceItem, EvidenceScoreList

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
                "Return a score for every source."
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
        llm = get_structured_llm(EvidenceScoreList)
        chain = _RANK_PROMPT | llm
        result: EvidenceScoreList = await chain.ainvoke(
            {"question": question, "sources_block": sources_block}
        )

        score_map: dict[int, tuple[float, str]] = {}
        for entry in result.scores:
            score_map[entry.source_number] = (
                entry.quality_score,
                entry.quality_reason,
            )

        for item in evidence:
            if item.source_number in score_map:
                item.quality_score, item.quality_reason = score_map[item.source_number]
            else:
                item.quality_score = 0.5
                item.quality_reason = "Not scored by LLM"

    except Exception as exc:
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
