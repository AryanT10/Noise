"""Eval runner — execute test cases through the pipeline and score results.

Also provides comparison utilities for regression testing between versions.
"""

from __future__ import annotations

import re
import time

from app.evaluation.dataset import EVAL_DATASET
from app.evaluation.metrics import (
    PASS_THRESHOLD,
    compute_aggregate_score,
    score_answer,
)
from app.evaluation.schemas import (
    EvalCase,
    EvalCompareItem,
    EvalComparison,
    EvalReport,
    EvalResult,
)
from app.graph.workflow import run_graph_full
from app.logging import logger
from app.models.schemas import AggregatedAnswer


async def evaluate_case(
    case: EvalCase,
    *,
    use_llm_judge: bool = True,
) -> EvalResult:
    """Run one test case through the pipeline and score it."""
    logger.info("Eval [%s]: running question: %s", case.id, case.question[:60])

    try:
        result: AggregatedAnswer = await run_graph_full(case.question)
    except Exception as exc:
        logger.error("Eval [%s]: pipeline failed: %s", case.id, exc)
        return EvalResult(
            case_id=case.id,
            question=case.question,
            answer=f"PIPELINE ERROR: {exc}",
            passed=False,
            failure_reasons=[f"Pipeline error: {exc}"],
        )

    answer = result.answer
    citation_count = len(set(re.findall(r"\[\d+\]", answer)))
    sources_count = len(result.sources)

    # Score across all dimensions
    dim_scores = await score_answer(result, case, use_llm_judge=use_llm_judge)
    aggregate = compute_aggregate_score(dim_scores)

    # Determine pass/fail
    failure_reasons: list[str] = []

    if aggregate < PASS_THRESHOLD:
        failure_reasons.append(
            f"Aggregate score {aggregate:.2f} below threshold {PASS_THRESHOLD}"
        )

    if sources_count < case.min_sources:
        failure_reasons.append(
            f"Only {sources_count} sources, expected >= {case.min_sources}"
        )

    # Check forbidden keywords
    answer_lower = answer.lower()
    for kw in case.forbidden_keywords:
        if kw.lower() in answer_lower:
            failure_reasons.append(f"Forbidden keyword found: '{kw}'")

    passed = len(failure_reasons) == 0

    return EvalResult(
        case_id=case.id,
        question=case.question,
        answer=answer,
        run_id=result.run_id,
        sources_count=sources_count,
        citation_count=citation_count,
        scores=dim_scores,
        aggregate_score=aggregate,
        passed=passed,
        failure_reasons=failure_reasons,
    )


async def run_eval(
    *,
    label: str = "",
    case_ids: list[str] | None = None,
    use_llm_judge: bool = True,
) -> EvalReport:
    """Run the full evaluation suite (or a subset) and produce a report.

    Args:
        label: Human-readable name for this run (e.g. "baseline", "v2").
        case_ids: If set, only run cases with these IDs. Otherwise run all.
        use_llm_judge: Whether to use LLM-based judges (slower but deeper).
    """
    cases = EVAL_DATASET
    if case_ids:
        id_set = set(case_ids)
        cases = [c for c in cases if c.id in id_set]

    logger.info("Starting eval run '%s' with %d cases", label, len(cases))
    start = time.time()

    results: list[EvalResult] = []
    for case in cases:
        result = await evaluate_case(case, use_llm_judge=use_llm_judge)
        results.append(result)
        status = "PASS" if result.passed else "FAIL"
        logger.info(
            "Eval [%s]: %s (score=%.2f)", case.id, status, result.aggregate_score
        )

    elapsed = round(time.time() - start, 1)
    logger.info("Eval run '%s' completed in %.1fs", label, elapsed)

    # Aggregate dimension means
    dim_sums: dict[str, float] = {}
    dim_counts: dict[str, int] = {}
    for r in results:
        for s in r.scores:
            dim_sums[s.dimension.value] = dim_sums.get(s.dimension.value, 0.0) + s.score
            dim_counts[s.dimension.value] = dim_counts.get(s.dimension.value, 0) + 1

    mean_scores = {
        dim: round(dim_sums[dim] / dim_counts[dim], 3)
        for dim in dim_sums
    }

    overall = (
        round(sum(mean_scores.values()) / len(mean_scores), 3) if mean_scores else 0.0
    )

    passed_count = sum(1 for r in results if r.passed)

    return EvalReport(
        label=label,
        total_cases=len(results),
        passed_cases=passed_count,
        failed_cases=len(results) - passed_count,
        mean_scores=mean_scores,
        overall_score=overall,
        results=results,
    )


def compare_reports(
    baseline: EvalReport,
    candidate: EvalReport,
) -> EvalComparison:
    """Compare two eval reports and surface regressions."""
    deltas: list[EvalCompareItem] = []
    regressions: list[str] = []

    all_dims = sorted(
        set(baseline.mean_scores.keys()) | set(candidate.mean_scores.keys())
    )

    for dim in all_dims:
        b = baseline.mean_scores.get(dim, 0.0)
        c = candidate.mean_scores.get(dim, 0.0)
        d = round(c - b, 3)
        deltas.append(EvalCompareItem(
            dimension=dim, baseline=b, candidate=c, delta=d,
        ))
        if d < -0.05:  # >5% drop = regression
            regressions.append(f"{dim}: {b:.3f} → {c:.3f} (Δ{d:+.3f})")

    return EvalComparison(
        baseline_label=baseline.label,
        candidate_label=candidate.label,
        baseline_overall=baseline.overall_score,
        candidate_overall=candidate.overall_score,
        overall_delta=round(candidate.overall_score - baseline.overall_score, 3),
        dimension_deltas=deltas,
        regressions=regressions,
    )
