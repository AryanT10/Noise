"""consensus_builder — detect agreement/disagreement, merge duplicate claims, flag uncertainty."""

import json

from langchain_core.prompts import ChatPromptTemplate

from app.chains.llm import get_llm
from app.logging import logger
from app.models.schemas import Claim, ConsensusGroup

_CONSENSUS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a consensus analyst. Given a user question and a numbered list "
                "of claims extracted from different sources, perform three tasks:\n\n"
                "1. **Group** claims that express the same fact (even in different words). "
                "For each group, write one canonical claim and list the source numbers that "
                "support it.\n"
                "2. **Disagreements**: list any pairs of claims that directly contradict "
                "each other. Describe each disagreement in one sentence.\n"
                "3. **Uncertainties**: list anything that the sources are unclear or "
                "speculative about.\n\n"
                "Return a JSON object with three keys:\n"
                '  "consensus_groups": [{{"canonical_claim": str, "supporting_sources": [int]}}, ...],\n'
                '  "disagreements": [str, ...],\n'
                '  "uncertainties": [str, ...]\n\n'
                "Return ONLY valid JSON — no markdown fences, no commentary."
            ),
        ),
        (
            "human",
            "Question: {question}\n\nClaims:\n{claims_block}",
        ),
    ]
)


async def build_consensus(
    question: str,
    claims: list[Claim],
) -> tuple[list[ConsensusGroup], list[str], list[str]]:
    """Cluster claims into consensus groups, surface disagreements & uncertainties.

    Returns (consensus_groups, disagreements, uncertainties).
    """
    if not claims:
        return [], [], []

    claims_block = "\n".join(
        f"  [{c.source_number}] {c.claim}" for c in claims
    )

    try:
        llm = get_llm()
        chain = _CONSENSUS_PROMPT | llm
        response = await chain.ainvoke(
            {"question": question, "claims_block": claims_block}
        )

        raw = response.content.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

        parsed = json.loads(raw)

        groups = [
            ConsensusGroup(
                canonical_claim=g["canonical_claim"],
                supporting_sources=g.get("supporting_sources", []),
                agreement_count=len(g.get("supporting_sources", [])),
            )
            for g in parsed.get("consensus_groups", [])
        ]
        disagreements: list[str] = parsed.get("disagreements", [])
        uncertainties: list[str] = parsed.get("uncertainties", [])

        logger.info(
            "consensus_builder: %d groups, %d disagreements, %d uncertainties",
            len(groups),
            len(disagreements),
            len(uncertainties),
        )
        return groups, disagreements, uncertainties

    except (json.JSONDecodeError, Exception) as exc:
        logger.warning("consensus_builder failed: %s — returning raw claims", exc)
        # Fallback: each claim becomes its own group
        fallback_groups = [
            ConsensusGroup(
                canonical_claim=c.claim,
                supporting_sources=[c.source_number],
                agreement_count=1,
            )
            for c in claims
        ]
        return fallback_groups, [], ["Consensus analysis unavailable"]
