"""consensus_builder — detect agreement/disagreement, merge duplicate claims, flag uncertainty."""

from langchain_core.prompts import ChatPromptTemplate

from app.chains.llm import get_structured_llm
from app.logging import logger
from app.models.schemas import Claim, ConsensusGroup, ConsensusResult

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
                "speculative about."
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
        llm = get_structured_llm(ConsensusResult)
        chain = _CONSENSUS_PROMPT | llm
        result: ConsensusResult = await chain.ainvoke(
            {"question": question, "claims_block": claims_block}
        )

        # Backfill agreement_count from supporting_sources length
        for group in result.consensus_groups:
            if group.agreement_count == 0:
                group.agreement_count = len(group.supporting_sources)

        logger.info(
            "consensus_builder: %d groups, %d disagreements, %d uncertainties",
            len(result.consensus_groups),
            len(result.disagreements),
            len(result.uncertainties),
        )
        return result.consensus_groups, result.disagreements, result.uncertainties

    except Exception as exc:
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
