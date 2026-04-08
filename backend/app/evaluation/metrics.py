"""Evaluation metrics — score an answer across the five quality dimensions.

Each metric function takes the answer, sources, and eval case, and returns
a DimensionScore.  The LLM-based judge is used for dimensions that require
semantic understanding; deterministic checks handle the rest.
"""

from __future__ import annotations

import re

from langchain_core.prompts import ChatPromptTemplate

from app.chains.llm import get_structured_llm
from app.evaluation.schemas import DimensionScore, EvalCase, EvalDimension
from app.logging import logger
from app.models.schemas import AggregatedAnswer

from pydantic import BaseModel, Field


# ── Structured output for the LLM judge ─────────────────────


class _JudgeScore(BaseModel):
    score: float = Field(ge=0.0, le=1.0, description="Score from 0.0 to 1.0.")
    reason: str = Field(description="One-sentence justification.")


# ── LLM judge prompts ───────────────────────────────────────

_RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an evaluation judge. Score how relevant the answer is to the "
        "question on a scale of 0.0 to 1.0.\n"
        "1.0 = directly and fully addresses the question.\n"
        "0.0 = completely off-topic.\n"
        "Give a one-sentence reason.",
    ),
    ("human", "Question: {question}\n\nAnswer: {answer}\n\nScore the relevance."),
])

_COMPLETENESS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an evaluation judge. Score how complete the answer is on a "
        "scale of 0.0 to 1.0.\n"
        "1.0 = covers all key aspects a good answer should include.\n"
        "0.0 = barely addresses anything.\n"
        "Consider expected aspects: {expected_keywords}\n"
        "Give a one-sentence reason.",
    ),
    ("human", "Question: {question}\n\nAnswer: {answer}\n\nScore the completeness."),
])

_HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an evaluation judge. Score how well-grounded the answer is on "
        "a scale of 0.0 to 1.0.\n"
        "1.0 = every claim is backed by the provided sources, no fabrication.\n"
        "0.0 = entirely fabricated or unsupported.\n"
        "The answer should only contain information from these source snippets.\n"
        "Give a one-sentence reason.",
    ),
    (
        "human",
        "Question: {question}\n\nAnswer: {answer}\n\n"
        "Source snippets:\n{snippets}\n\nScore the factual grounding.",
    ),
])


# ── Deterministic metrics ────────────────────────────────────


def _score_citation_quality(
    answer: str,
    result: AggregatedAnswer,
    case: EvalCase,
) -> DimensionScore:
    """Check inline citation count and source backing."""
    citation_markers = re.findall(r"\[\d+\]", answer)
    unique_citations = len(set(citation_markers))
    sources_count = len(result.sources)

    # Score based on meeting the expected citation count
    if case.expected_citation_count == 0:
        score = 1.0
        reason = "No citations expected."
    elif unique_citations >= case.expected_citation_count:
        # Bonus if citations actually back to real sources
        valid = sum(1 for c in set(citation_markers)
                    if 1 <= int(c.strip("[]")) <= sources_count)
        ratio = valid / max(unique_citations, 1)
        score = min(1.0, 0.6 + 0.4 * ratio)
        reason = (
            f"{unique_citations} citations found ({valid} valid), "
            f"expected >= {case.expected_citation_count}."
        )
    else:
        score = max(0.0, unique_citations / case.expected_citation_count)
        reason = (
            f"Only {unique_citations} citations, "
            f"expected >= {case.expected_citation_count}."
        )

    return DimensionScore(
        dimension=EvalDimension.CITATION_QUALITY,
        score=round(score, 2),
        reason=reason,
    )


def _score_factual_grounding_deterministic(
    answer: str,
    case: EvalCase,
) -> DimensionScore:
    """Quick keyword check for expected and forbidden keywords."""
    answer_lower = answer.lower()

    # Expected keywords present
    expected_hits = sum(
        1 for kw in case.expected_keywords if kw.lower() in answer_lower
    )
    expected_ratio = (
        expected_hits / len(case.expected_keywords)
        if case.expected_keywords
        else 1.0
    )

    # Forbidden keywords absent (each hit is a penalty)
    forbidden_hits = sum(
        1 for kw in case.forbidden_keywords if kw.lower() in answer_lower
    )
    forbidden_penalty = (
        forbidden_hits / len(case.forbidden_keywords)
        if case.forbidden_keywords
        else 0.0
    )

    score = max(0.0, expected_ratio - forbidden_penalty)

    parts = []
    if case.expected_keywords:
        parts.append(f"{expected_hits}/{len(case.expected_keywords)} expected keywords found")
    if case.forbidden_keywords:
        parts.append(f"{forbidden_hits} forbidden keywords found")
    reason = "; ".join(parts) or "No keyword criteria."

    return DimensionScore(
        dimension=EvalDimension.FACTUAL_GROUNDING,
        score=round(score, 2),
        reason=reason,
    )


# ── LLM-based metrics ───────────────────────────────────────


async def _llm_judge(
    prompt: ChatPromptTemplate,
    variables: dict,
    dimension: EvalDimension,
) -> DimensionScore:
    """Run a single LLM judge call and return a DimensionScore."""
    try:
        llm = get_structured_llm(_JudgeScore)
        chain = prompt | llm
        result: _JudgeScore = await chain.ainvoke(variables)
        return DimensionScore(
            dimension=dimension,
            score=round(result.score, 2),
            reason=result.reason,
        )
    except Exception as exc:
        logger.warning("LLM judge failed for %s: %s", dimension.value, exc)
        return DimensionScore(
            dimension=dimension,
            score=0.5,
            reason=f"Judge unavailable: {exc}",
        )


async def score_relevance(answer: str, case: EvalCase) -> DimensionScore:
    return await _llm_judge(
        _RELEVANCE_PROMPT,
        {"question": case.question, "answer": answer},
        EvalDimension.RELEVANCE,
    )


async def score_completeness(answer: str, case: EvalCase) -> DimensionScore:
    return await _llm_judge(
        _COMPLETENESS_PROMPT,
        {
            "question": case.question,
            "answer": answer,
            "expected_keywords": ", ".join(case.expected_keywords) or "N/A",
        },
        EvalDimension.COMPLETENESS,
    )


async def score_hallucination(
    answer: str,
    result: AggregatedAnswer,
    case: EvalCase,
) -> DimensionScore:
    snippets_text = "\n".join(
        f"[{s.source_number}] {s.text[:500]}" for s in result.snippets
    ) or "No snippets available."

    return await _llm_judge(
        _HALLUCINATION_PROMPT,
        {"question": case.question, "answer": answer, "snippets": snippets_text},
        EvalDimension.HALLUCINATION,
    )


# ── Composite scorer ────────────────────────────────────────

# Weights for the final aggregate score
DIMENSION_WEIGHTS: dict[EvalDimension, float] = {
    EvalDimension.RELEVANCE: 0.25,
    EvalDimension.FACTUAL_GROUNDING: 0.25,
    EvalDimension.CITATION_QUALITY: 0.15,
    EvalDimension.COMPLETENESS: 0.20,
    EvalDimension.HALLUCINATION: 0.15,
}

PASS_THRESHOLD = 0.5


async def score_answer(
    result: AggregatedAnswer,
    case: EvalCase,
    *,
    use_llm_judge: bool = True,
) -> list[DimensionScore]:
    """Score an answer across all five dimensions.

    If use_llm_judge=False, only deterministic checks run (fast, no API calls).
    """
    answer = result.answer
    scores: list[DimensionScore] = []

    # Always run deterministic metrics
    scores.append(_score_citation_quality(answer, result, case))
    scores.append(_score_factual_grounding_deterministic(answer, case))

    if use_llm_judge:
        scores.append(await score_relevance(answer, case))
        scores.append(await score_completeness(answer, case))
        scores.append(await score_hallucination(answer, result, case))
    else:
        # Placeholder scores when LLM judge is skipped
        scores.append(DimensionScore(
            dimension=EvalDimension.RELEVANCE, score=0.5, reason="LLM judge skipped.",
        ))
        scores.append(DimensionScore(
            dimension=EvalDimension.COMPLETENESS, score=0.5, reason="LLM judge skipped.",
        ))
        scores.append(DimensionScore(
            dimension=EvalDimension.HALLUCINATION, score=0.5, reason="LLM judge skipped.",
        ))

    return scores


def compute_aggregate_score(scores: list[DimensionScore]) -> float:
    """Compute a weighted aggregate from dimension scores."""
    total_weight = 0.0
    weighted_sum = 0.0
    for s in scores:
        w = DIMENSION_WEIGHTS.get(s.dimension, 0.0)
        weighted_sum += s.score * w
        total_weight += w
    return round(weighted_sum / total_weight, 3) if total_weight > 0 else 0.0
