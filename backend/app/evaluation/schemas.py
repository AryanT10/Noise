"""Evaluation schemas — models for test cases, metric scores, and eval reports."""

from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class EvalDimension(str, Enum):
    """The five quality dimensions we evaluate."""

    RELEVANCE = "relevance"
    FACTUAL_GROUNDING = "factual_grounding"
    CITATION_QUALITY = "citation_quality"
    COMPLETENESS = "completeness"
    HALLUCINATION = "hallucination"


class EvalCase(BaseModel):
    """A single evaluation test case."""

    id: str = Field(description="Unique identifier, e.g. 'factual-01'")
    question: str
    category: str = Field(
        default="general",
        description="Question category: factual, opinion, multi-source, ambiguous, recent-event",
    )
    expected_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords/phrases that SHOULD appear in a good answer.",
    )
    forbidden_keywords: list[str] = Field(
        default_factory=list,
        description="Keywords/phrases that should NOT appear (hallucination markers).",
    )
    min_sources: int = Field(
        default=1,
        description="Minimum number of sources expected in a good answer.",
    )
    expected_citation_count: int = Field(
        default=1,
        description="Minimum inline citations [N] expected.",
    )
    notes: str = Field(
        default="",
        description="Human notes on what a good answer looks like.",
    )


class DimensionScore(BaseModel):
    """Score for one evaluation dimension."""

    dimension: EvalDimension
    score: float = Field(ge=0.0, le=1.0, description="0.0 = worst, 1.0 = best")
    reason: str = ""


class EvalResult(BaseModel):
    """Result of evaluating one test case."""

    case_id: str
    question: str
    answer: str
    run_id: str = ""
    sources_count: int = 0
    citation_count: int = 0
    scores: list[DimensionScore] = Field(default_factory=list)
    aggregate_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Weighted mean of dimension scores."
    )
    passed: bool = True
    failure_reasons: list[str] = Field(default_factory=list)


class EvalReport(BaseModel):
    """Summary report for a full evaluation run."""

    label: str = Field(
        default="",
        description="Human label for this run, e.g. 'baseline' or 'v2-prompt-tweak'.",
    )
    total_cases: int = 0
    passed_cases: int = 0
    failed_cases: int = 0
    mean_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Mean score per dimension across all cases.",
    )
    overall_score: float = Field(
        default=0.0, description="Grand mean across all dimensions and cases."
    )
    results: list[EvalResult] = Field(default_factory=list)


class EvalCompareItem(BaseModel):
    """Per-dimension delta between two eval runs."""

    dimension: str
    baseline: float
    candidate: float
    delta: float


class EvalComparison(BaseModel):
    """Side-by-side comparison of two evaluation runs."""

    baseline_label: str
    candidate_label: str
    baseline_overall: float
    candidate_overall: float
    overall_delta: float
    dimension_deltas: list[EvalCompareItem] = Field(default_factory=list)
    regressions: list[str] = Field(
        default_factory=list,
        description="Dimensions where the candidate is worse than baseline.",
    )
