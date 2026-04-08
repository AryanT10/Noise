"""Tests for Phase 9 — evaluation suite."""

import pytest
from unittest.mock import patch, AsyncMock

from langchain_core.runnables import RunnableLambda

from app.evaluation.schemas import (
    DimensionScore,
    EvalCase,
    EvalComparison,
    EvalDimension,
    EvalReport,
    EvalResult,
)
from app.evaluation.dataset import EVAL_DATASET
from app.evaluation.metrics import (
    PASS_THRESHOLD,
    _score_citation_quality,
    _score_factual_grounding_deterministic,
    compute_aggregate_score,
    score_answer,
)
from app.evaluation.runner import compare_reports, evaluate_case, run_eval
from app.models.schemas import (
    AggregatedAnswer,
    Snippet,
    Source,
)


# ── Helpers ──────────────────────────────────────────────────


def _make_answer(
    answer: str = "Test answer [1].",
    sources: int = 2,
    snippets: int = 2,
) -> AggregatedAnswer:
    return AggregatedAnswer(
        answer=answer,
        sources=[
            Source(number=i, title=f"Source {i}", url=f"https://example.com/{i}")
            for i in range(1, sources + 1)
        ],
        snippets=[
            Snippet(source_number=i, text=f"Snippet text for source {i}")
            for i in range(1, snippets + 1)
        ],
        run_id="eval-test-run",
    )


def _simple_case(**overrides) -> EvalCase:
    defaults = dict(
        id="test-01",
        question="What is X?",
        category="factual",
        expected_keywords=["answer"],
        forbidden_keywords=["wrong"],
        min_sources=1,
        expected_citation_count=1,
    )
    defaults.update(overrides)
    return EvalCase(**defaults)


def _fake_judge_factory(score: float = 0.8):
    """Return a function that mimics get_structured_llm for the _JudgeScore schema."""
    from app.evaluation.metrics import _JudgeScore

    def fake_get_structured_llm(schema):
        fake_result = _JudgeScore(score=score, reason="Fake judge")
        async def fn(messages):
            return fake_result
        return RunnableLambda(fn)

    return fake_get_structured_llm


# ── Dataset ──────────────────────────────────────────────────


def test_dataset_not_empty():
    assert len(EVAL_DATASET) >= 5


def test_dataset_unique_ids():
    ids = [c.id for c in EVAL_DATASET]
    assert len(ids) == len(set(ids))


def test_dataset_all_have_questions():
    for c in EVAL_DATASET:
        assert len(c.question) > 5


def test_dataset_categories():
    categories = {c.category for c in EVAL_DATASET}
    assert "factual" in categories
    assert "multi-source" in categories


# ── Citation quality (deterministic) ─────────────────────────


def test_citation_quality_meets_expected():
    case = _simple_case(expected_citation_count=2)
    result = _make_answer(answer="Answer [1] and [2] agree.", sources=3)
    score = _score_citation_quality(result.answer, result, case)
    assert score.dimension == EvalDimension.CITATION_QUALITY
    assert score.score >= 0.6


def test_citation_quality_below_expected():
    case = _simple_case(expected_citation_count=3)
    result = _make_answer(answer="Only [1] cited.", sources=3)
    score = _score_citation_quality(result.answer, result, case)
    assert score.score < 0.6


def test_citation_quality_no_citations_expected():
    case = _simple_case(expected_citation_count=0)
    result = _make_answer(answer="No citations needed.")
    score = _score_citation_quality(result.answer, result, case)
    assert score.score == 1.0


def test_citation_quality_invalid_refs():
    case = _simple_case(expected_citation_count=1)
    result = _make_answer(answer="Answer [99].", sources=2)
    score = _score_citation_quality(result.answer, result, case)
    # Citation exists but doesn't map to a real source
    assert score.score >= 0.6  # count met, but validity ratio penalised


# ── Factual grounding (deterministic) ────────────────────────


def test_grounding_all_keywords_found():
    case = _simple_case(expected_keywords=["alpha", "beta"], forbidden_keywords=[])
    score = _score_factual_grounding_deterministic("Alpha and Beta are here.", case)
    assert score.score == 1.0


def test_grounding_some_keywords_missing():
    case = _simple_case(expected_keywords=["alpha", "beta", "gamma"], forbidden_keywords=[])
    score = _score_factual_grounding_deterministic("Only alpha mentioned.", case)
    assert 0.3 <= score.score <= 0.4


def test_grounding_forbidden_keyword_penalty():
    case = _simple_case(expected_keywords=["good"], forbidden_keywords=["bad"])
    score = _score_factual_grounding_deterministic("This is good and bad.", case)
    assert score.score == 0.0  # 1.0 - 1.0


def test_grounding_no_criteria():
    case = _simple_case(expected_keywords=[], forbidden_keywords=[])
    score = _score_factual_grounding_deterministic("Anything goes.", case)
    assert score.score == 1.0


# ── Aggregate score ──────────────────────────────────────────


def test_compute_aggregate_perfect():
    scores = [DimensionScore(dimension=d, score=1.0, reason="perfect") for d in EvalDimension]
    assert compute_aggregate_score(scores) == 1.0


def test_compute_aggregate_zero():
    scores = [DimensionScore(dimension=d, score=0.0, reason="zero") for d in EvalDimension]
    assert compute_aggregate_score(scores) == 0.0


def test_compute_aggregate_empty():
    assert compute_aggregate_score([]) == 0.0


# ── score_answer (no LLM judge) ──────────────────────────────


@pytest.mark.asyncio
async def test_score_answer_deterministic_only():
    case = _simple_case(expected_keywords=["test"], expected_citation_count=1)
    result = _make_answer(answer="This is a test answer [1].")
    scores = await score_answer(result, case, use_llm_judge=False)
    assert len(scores) == 5  # all 5 dimensions
    dims = {s.dimension for s in scores}
    assert dims == set(EvalDimension)


@pytest.mark.asyncio
async def test_score_answer_with_llm_judge():
    case = _simple_case(expected_keywords=["test"], expected_citation_count=1)
    result = _make_answer(answer="This is a test answer [1].")

    with patch("app.evaluation.metrics.get_structured_llm", side_effect=_fake_judge_factory()):
        scores = await score_answer(result, case, use_llm_judge=True)

    assert len(scores) == 5
    # LLM judge dimensions should have non-default scores
    llm_dims = {EvalDimension.RELEVANCE, EvalDimension.COMPLETENESS, EvalDimension.HALLUCINATION}
    for s in scores:
        if s.dimension in llm_dims:
            assert s.score == 0.8  # from fake judge


# ── evaluate_case ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_evaluate_case_passing():
    case = _simple_case(
        expected_keywords=["coffee"],
        forbidden_keywords=["poison"],
        min_sources=1,
        expected_citation_count=1,
    )
    fake_result = _make_answer(answer="Coffee is healthy [1].", sources=2)

    with patch("app.evaluation.runner.run_graph_full", return_value=fake_result):
        result = await evaluate_case(case, use_llm_judge=False)

    assert result.passed is True
    assert result.aggregate_score > 0
    assert result.case_id == "test-01"


@pytest.mark.asyncio
async def test_evaluate_case_forbidden_keyword_fail():
    case = _simple_case(
        expected_keywords=[],
        forbidden_keywords=["danger"],
        min_sources=1,
        expected_citation_count=1,
    )
    fake_result = _make_answer(answer="Danger ahead [1].", sources=2)

    with patch("app.evaluation.runner.run_graph_full", return_value=fake_result):
        result = await evaluate_case(case, use_llm_judge=False)

    assert result.passed is False
    assert any("Forbidden" in r for r in result.failure_reasons)


@pytest.mark.asyncio
async def test_evaluate_case_pipeline_error():
    case = _simple_case()

    with patch("app.evaluation.runner.run_graph_full", side_effect=Exception("boom")):
        result = await evaluate_case(case, use_llm_judge=False)

    assert result.passed is False
    assert "PIPELINE ERROR" in result.answer


@pytest.mark.asyncio
async def test_evaluate_case_too_few_sources():
    case = _simple_case(min_sources=5)
    fake_result = _make_answer(answer="Answer [1].", sources=1)

    with patch("app.evaluation.runner.run_graph_full", return_value=fake_result):
        result = await evaluate_case(case, use_llm_judge=False)

    assert result.passed is False
    assert any("sources" in r.lower() for r in result.failure_reasons)


# ── run_eval ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_eval_subset():
    fake_result = _make_answer(answer="Coffee is good [1] [2].", sources=3)

    with patch("app.evaluation.runner.run_graph_full", return_value=fake_result):
        report = await run_eval(
            label="test-run",
            case_ids=["factual-01", "factual-02"],
            use_llm_judge=False,
        )

    assert report.label == "test-run"
    assert report.total_cases == 2
    assert len(report.results) == 2
    assert report.overall_score >= 0.0


@pytest.mark.asyncio
async def test_run_eval_all_cases():
    fake_result = _make_answer(
        answer="Canberra is the capital, George Orwell wrote 1984 [1] [2].",
        sources=3,
    )

    with patch("app.evaluation.runner.run_graph_full", return_value=fake_result):
        report = await run_eval(label="full", use_llm_judge=False)

    assert report.total_cases == len(EVAL_DATASET)
    assert report.total_cases == report.passed_cases + report.failed_cases


# ── compare_reports ──────────────────────────────────────────


def test_compare_reports_no_regression():
    baseline = EvalReport(
        label="baseline",
        total_cases=2,
        passed_cases=2,
        mean_scores={"relevance": 0.8, "factual_grounding": 0.7},
        overall_score=0.75,
    )
    candidate = EvalReport(
        label="candidate",
        total_cases=2,
        passed_cases=2,
        mean_scores={"relevance": 0.85, "factual_grounding": 0.75},
        overall_score=0.80,
    )
    comp = compare_reports(baseline, candidate)
    assert comp.overall_delta > 0
    assert len(comp.regressions) == 0


def test_compare_reports_with_regression():
    baseline = EvalReport(
        label="baseline",
        total_cases=2,
        passed_cases=2,
        mean_scores={"relevance": 0.9, "citation_quality": 0.8},
        overall_score=0.85,
    )
    candidate = EvalReport(
        label="candidate",
        total_cases=2,
        passed_cases=1,
        mean_scores={"relevance": 0.7, "citation_quality": 0.85},
        overall_score=0.775,
    )
    comp = compare_reports(baseline, candidate)
    assert comp.overall_delta < 0
    assert len(comp.regressions) >= 1
    assert "relevance" in comp.regressions[0]


def test_compare_reports_empty_scores():
    baseline = EvalReport(label="a", mean_scores={}, overall_score=0.0)
    candidate = EvalReport(label="b", mean_scores={}, overall_score=0.0)
    comp = compare_reports(baseline, candidate)
    assert comp.overall_delta == 0.0
    assert len(comp.dimension_deltas) == 0
