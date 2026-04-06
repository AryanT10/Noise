"""claim_extractor — ask the LLM to pull discrete claims from each source."""

import json

from langchain_core.prompts import ChatPromptTemplate

from app.chains.llm import get_llm
from app.logging import logger
from app.models.schemas import Claim, EvidenceItem

_EXTRACT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a claim extractor. Given a user question and a source excerpt, "
                "extract every distinct factual claim the source makes that is relevant "
                "to the question.\n\n"
                "Return a JSON array of objects, each with:\n"
                '  "claim": a one-sentence factual statement,\n'
                '  "verbatim_quote": the closest verbatim phrase from the source.\n\n'
                "If the source contains no relevant claims, return an empty array [].\n"
                "Return ONLY valid JSON — no markdown fences, no commentary."
            ),
        ),
        (
            "human",
            (
                "Question: {question}\n\n"
                "Source [{source_number}] — {title}\n"
                "{text}"
            ),
        ),
    ]
)


async def extract_claims(
    question: str,
    evidence: list[EvidenceItem],
) -> list[Claim]:
    """Extract claims from every evidence item via the LLM."""
    all_claims: list[Claim] = []

    for item in evidence:
        try:
            llm = get_llm()
            chain = _EXTRACT_PROMPT | llm
            response = await chain.ainvoke(
                {
                    "question": question,
                    "source_number": item.source_number,
                    "title": item.title,
                    "text": item.text[:3000],
                }
            )
            raw = response.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                parsed = [parsed]

            for obj in parsed:
                all_claims.append(
                    Claim(
                        claim=obj.get("claim", ""),
                        source_number=item.source_number,
                        verbatim_quote=obj.get("verbatim_quote", ""),
                    )
                )

        except (json.JSONDecodeError, Exception) as exc:
            logger.warning(
                "claim_extractor failed for source %d: %s",
                item.source_number,
                exc,
            )

    logger.info("claim_extractor: %d claims from %d sources", len(all_claims), len(evidence))
    return all_claims
