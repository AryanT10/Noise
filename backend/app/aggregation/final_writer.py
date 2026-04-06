"""final_writer — compose the aggregated judgment answer from consensus analysis."""

from langchain_core.prompts import ChatPromptTemplate

from app.chains.llm import get_llm
from app.logging import logger
from app.models.schemas import ConsensusGroup, EvidenceItem

_FINAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a research synthesiser producing a final **aggregated judgment** "
                "— not a mere summary. You have been given:\n"
                "- Consensus groups: claims agreed upon by multiple sources\n"
                "- Disagreements: points where sources contradict each other\n"
                "- Uncertainties: aspects that are unclear or speculative\n"
                "- Ranked source material with quality scores\n\n"
                "Rules:\n"
                "1. Lead with the strongest consensus findings, weighted by how many "
                "   sources agree and by source quality.\n"
                "2. Explicitly note where sources disagree and explain both sides.\n"
                "3. Flag remaining uncertainties honestly.\n"
                "4. Cite sources inline using [N] matching source numbers.\n"
                "5. Do NOT hallucinate — every sentence must trace to provided material.\n"
                "6. Keep the answer readable in under 30 seconds.\n"
            ),
        ),
        (
            "human",
            (
                "Question: {question}\n\n"
                "── Consensus Groups ──\n{consensus_block}\n\n"
                "── Disagreements ──\n{disagreements_block}\n\n"
                "── Uncertainties ──\n{uncertainties_block}\n\n"
                "── Ranked Sources ──\n{sources_block}\n\n"
                "Write your aggregated judgment answer with inline citations."
            ),
        ),
    ]
)


async def write_final_answer(
    question: str,
    consensus_groups: list[ConsensusGroup],
    disagreements: list[str],
    uncertainties: list[str],
    ranked_evidence: list[EvidenceItem],
) -> str:
    """Produce the final aggregated-judgment answer."""
    consensus_block = "\n".join(
        f"• {g.canonical_claim} (sources: {g.supporting_sources}, "
        f"agreement: {g.agreement_count})"
        for g in consensus_groups
    ) or "None identified."

    disagreements_block = "\n".join(
        f"• {d}" for d in disagreements
    ) or "No disagreements detected."

    uncertainties_block = "\n".join(
        f"• {u}" for u in uncertainties
    ) or "No uncertainties flagged."

    sources_block = "\n\n".join(
        f"[{e.source_number}] {e.title} (quality: {e.quality_score:.1f})\n"
        f"URL: {e.url}\n{e.text[:1500]}"
        for e in ranked_evidence
    )

    try:
        llm = get_llm()
        chain = _FINAL_PROMPT | llm
        response = await chain.ainvoke(
            {
                "question": question,
                "consensus_block": consensus_block,
                "disagreements_block": disagreements_block,
                "uncertainties_block": uncertainties_block,
                "sources_block": sources_block,
            }
        )
        logger.info("final_writer: answer generated (%d chars)", len(response.content))
        return response.content

    except Exception as exc:
        logger.error("final_writer failed: %s", exc)
        # Degrade gracefully — produce a basic summary from consensus groups
        fallback_parts = [
            f"• {g.canonical_claim} [sources: {', '.join(str(s) for s in g.supporting_sources)}]"
            for g in consensus_groups
        ]
        return (
            "I was unable to produce a fully synthesised answer — here are the "
            "key findings from available sources:\n\n" + "\n".join(fallback_parts)
        )
