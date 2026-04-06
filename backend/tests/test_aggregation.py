"""Tests for Phase 6 — answer aggregation pipeline."""

import json

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from app.aggregation.source_reader import read_sources
from app.aggregation.claim_extractor import extract_claims
from app.aggregation.evidence_ranker import rank_evidence
from app.aggregation.consensus_builder import build_consensus
from app.aggregation.final_writer import write_final_answer
from app.graph.nodes import aggregate_answer
from app.graph.workflow import build_graph, run_graph
from app.models.schemas import (
    Claim,
    EvidenceItem,
    ConsensusGroup,
    PipelineResult,
)


# ── Helpers ──────────────────────────────────────────────────


def _base_state(**overrides) -> dict:
    state = {"question": "Is coffee healthy?", "errors": []}
    state.update(overrides)
    return state


def _sample_evidence_dicts() -> list[dict]:
    return [
        {"number": 1, "title": "Health.gov", "url": "https://health.gov/coffee", "text": "Coffee contains antioxidants and may reduce risk of disease."},
        {"number": 2, "title": "WebMD", "url": "https://webmd.com/coffee", "text": "Moderate coffee consumption is associated with health benefits."},
        {"number": 3, "title": "Skeptic Journal", "url": "https://skeptic.org/coffee", "text": "Excessive coffee intake can lead to anxiety and sleep issues."},
    ]


class _FakeLLMResponse:
    def __init__(self, content: str):
        self.content = content
        self.tool_calls = []


def _fake_llm(content: str):
    async def fn(messages):
        return _FakeLLMResponse(content)
    return RunnableLambda(fn)


def _fake_sequential_llm(responses):
    """Return a RunnableLambda that returns different responses per call."""
    call_count = {"n": 0}

    async def fn(messages):
        idx = min(call_count["n"], len(responses) - 1)
        call_count["n"] += 1
        resp = responses[idx]
        if isinstance(resp, dict) and "tool_calls" in resp:
            return AIMessage(content=resp.get("content", ""), tool_calls=resp["tool_calls"])
        return _FakeLLMResponse(resp)

    runnable = RunnableLambda(fn)
    runnable.bind_tools = lambda tools: runnable
    return runnable


# ── source_reader ────────────────────────────────────────────


def test_read_sources_normalises():
    items = read_sources(_sample_evidence_dicts())
    assert len(items) == 3
    assert all(isinstance(i, EvidenceItem) for i in items)
    assert items[0].source_number == 1
    assert items[2].url == "https://skeptic.org/coffee"


def test_read_sources_skips_malformed():
    bad = [{"number": 1, "title": "OK", "url": "u", "text": "t"}, {"bad_key": True}]
    items = read_sources(bad)
    assert len(items) == 1


def test_read_sources_empty():
    assert read_sources([]) == []


# ── claim_extractor ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_extract_claims_basic():
    llm_response = json.dumps([
        {"claim": "Coffee contains antioxidants", "verbatim_quote": "contains antioxidants"},
        {"claim": "Coffee may reduce disease risk", "verbatim_quote": "reduce risk of disease"},
    ])

    evidence = [EvidenceItem(source_number=1, title="T", url="u", text="text")]

    with patch("app.aggregation.claim_extractor.get_llm", return_value=_fake_llm(llm_response)):
        claims = await extract_claims("Is coffee healthy?", evidence)

    assert len(claims) == 2
    assert claims[0].source_number == 1
    assert "antioxidants" in claims[0].claim


@pytest.mark.asyncio
async def test_extract_claims_handles_markdown_fences():
    llm_response = '```json\n[{"claim": "A claim", "verbatim_quote": "quote"}]\n```'

    evidence = [EvidenceItem(source_number=1, title="T", url="u", text="text")]

    with patch("app.aggregation.claim_extractor.get_llm", return_value=_fake_llm(llm_response)):
        claims = await extract_claims("Q?", evidence)

    assert len(claims) == 1


@pytest.mark.asyncio
async def test_extract_claims_handles_llm_error():
    evidence = [EvidenceItem(source_number=1, title="T", url="u", text="text")]

    with patch("app.aggregation.claim_extractor.get_llm", side_effect=Exception("LLM down")):
        claims = await extract_claims("Q?", evidence)

    assert claims == []


@pytest.mark.asyncio
async def test_extract_claims_empty_evidence():
    claims = await extract_claims("Q?", [])
    assert claims == []


# ── evidence_ranker ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_rank_evidence_scores_and_sorts():
    llm_response = json.dumps([
        {"source_number": 1, "quality_score": 0.6, "quality_reason": "decent"},
        {"source_number": 2, "quality_score": 0.9, "quality_reason": "authoritative"},
        {"source_number": 3, "quality_score": 0.4, "quality_reason": "vague"},
    ])

    evidence = [
        EvidenceItem(source_number=1, title="A", url="u1", text="t1"),
        EvidenceItem(source_number=2, title="B", url="u2", text="t2"),
        EvidenceItem(source_number=3, title="C", url="u3", text="t3"),
    ]

    with patch("app.aggregation.evidence_ranker.get_llm", return_value=_fake_llm(llm_response)):
        ranked = await rank_evidence("Q?", evidence)

    assert ranked[0].source_number == 2  # highest score first
    assert ranked[0].quality_score == 0.9
    assert ranked[-1].source_number == 3


@pytest.mark.asyncio
async def test_rank_evidence_clamps_scores():
    llm_response = json.dumps([
        {"source_number": 1, "quality_score": 1.5, "quality_reason": "over"},
        {"source_number": 2, "quality_score": -0.3, "quality_reason": "under"},
    ])

    evidence = [
        EvidenceItem(source_number=1, title="A", url="u1", text="t1"),
        EvidenceItem(source_number=2, title="B", url="u2", text="t2"),
    ]

    with patch("app.aggregation.evidence_ranker.get_llm", return_value=_fake_llm(llm_response)):
        ranked = await rank_evidence("Q?", evidence)

    assert ranked[0].quality_score == 1.0
    assert ranked[1].quality_score == 0.0


@pytest.mark.asyncio
async def test_rank_evidence_defaults_on_failure():
    evidence = [EvidenceItem(source_number=1, title="A", url="u", text="t")]

    with patch("app.aggregation.evidence_ranker.get_llm", side_effect=Exception("down")):
        ranked = await rank_evidence("Q?", evidence)

    assert len(ranked) == 1
    assert ranked[0].quality_score == 0.5


@pytest.mark.asyncio
async def test_rank_evidence_empty():
    assert await rank_evidence("Q?", []) == []


# ── consensus_builder ────────────────────────────────────────


@pytest.mark.asyncio
async def test_build_consensus_groups_claims():
    llm_response = json.dumps({
        "consensus_groups": [
            {"canonical_claim": "Coffee has antioxidants", "supporting_sources": [1, 2]},
            {"canonical_claim": "Excess coffee harms sleep", "supporting_sources": [3]},
        ],
        "disagreements": ["Sources 1/2 say healthy; source 3 warns of harm"],
        "uncertainties": ["Optimal daily intake unclear"],
    })

    claims = [
        Claim(claim="Coffee has antioxidants", source_number=1),
        Claim(claim="Coffee linked to health benefits", source_number=2),
        Claim(claim="Too much coffee causes anxiety", source_number=3),
    ]

    with patch("app.aggregation.consensus_builder.get_llm", return_value=_fake_llm(llm_response)):
        groups, disagree, uncertain = await build_consensus("Q?", claims)

    assert len(groups) == 2
    assert groups[0].agreement_count == 2
    assert len(disagree) == 1
    assert len(uncertain) == 1


@pytest.mark.asyncio
async def test_build_consensus_fallback_on_error():
    claims = [Claim(claim="A", source_number=1), Claim(claim="B", source_number=2)]

    with patch("app.aggregation.consensus_builder.get_llm", side_effect=Exception("down")):
        groups, disagree, uncertain = await build_consensus("Q?", claims)

    assert len(groups) == 2  # each claim becomes its own group
    assert "Consensus analysis unavailable" in uncertain


@pytest.mark.asyncio
async def test_build_consensus_empty():
    groups, disagree, uncertain = await build_consensus("Q?", [])
    assert groups == []


# ── final_writer ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_write_final_answer_produces_text():
    groups = [ConsensusGroup(canonical_claim="Coffee is healthy", supporting_sources=[1, 2], agreement_count=2)]
    evidence = [EvidenceItem(source_number=1, title="A", url="u", text="t", quality_score=0.9)]

    with patch("app.aggregation.final_writer.get_llm", return_value=_fake_llm("Coffee is generally healthy [1][2].")):
        answer = await write_final_answer("Q?", groups, [], [], evidence)

    assert "Coffee" in answer


@pytest.mark.asyncio
async def test_write_final_answer_fallback_on_error():
    groups = [ConsensusGroup(canonical_claim="Claim A", supporting_sources=[1], agreement_count=1)]

    with patch("app.aggregation.final_writer.get_llm", side_effect=Exception("down")):
        answer = await write_final_answer("Q?", groups, [], [], [])

    assert "Claim A" in answer
    assert "unable" in answer.lower()


# ── aggregate_answer node ────────────────────────────────────


@pytest.mark.asyncio
async def test_aggregate_answer_full_pipeline():
    """The aggregate_answer node runs all five stages end-to-end."""
    claims_json = json.dumps([{"claim": "C1", "verbatim_quote": "q1"}])
    scores_json = json.dumps([{"source_number": 1, "quality_score": 0.8, "quality_reason": "good"}])
    consensus_json = json.dumps({
        "consensus_groups": [{"canonical_claim": "C1", "supporting_sources": [1]}],
        "disagreements": [],
        "uncertainties": [],
    })
    final_answer = "Aggregated answer about coffee [1]."

    fake_model = _fake_sequential_llm([
        claims_json,      # claim_extractor
        scores_json,      # evidence_ranker
        consensus_json,   # consensus_builder
        final_answer,     # final_writer
    ])

    state = _base_state(filtered_evidence=_sample_evidence_dicts()[:1])

    with patch("app.aggregation.claim_extractor.get_llm", return_value=fake_model), \
         patch("app.aggregation.evidence_ranker.get_llm", return_value=fake_model), \
         patch("app.aggregation.consensus_builder.get_llm", return_value=fake_model), \
         patch("app.aggregation.final_writer.get_llm", return_value=fake_model):
        result = await aggregate_answer(state)

    assert "coffee" in result["draft_answer"].lower() or "Aggregated" in result["draft_answer"]
    assert len(result["claims"]) >= 1
    assert len(result["ranked_evidence"]) >= 1


@pytest.mark.asyncio
async def test_aggregate_answer_empty_evidence():
    result = await aggregate_answer(_base_state(filtered_evidence=[]))

    assert "couldn't find" in result["draft_answer"].lower()
    assert result["claims"] == []
    assert result["consensus_groups"] == []


@pytest.mark.asyncio
async def test_aggregate_answer_falls_back_to_synthesize():
    """If aggregation fails entirely, it falls back to synthesize_answer."""
    evidence = _sample_evidence_dicts()[:1]
    state = _base_state(filtered_evidence=evidence)

    synth_answer = "Synthesized fallback answer [1]."

    with patch("app.graph.nodes.read_sources", side_effect=Exception("boom")), \
         patch("app.graph.nodes.get_llm", return_value=_fake_llm(synth_answer)):
        result = await aggregate_answer(state)

    assert "draft_answer" in result
    assert any("aggregate_answer" in e for e in result["errors"])


# ── Graph wiring ─────────────────────────────────────────────


def test_graph_has_aggregate_answer_node():
    compiled = build_graph()
    node_names = set(compiled.get_graph().nodes.keys())
    assert "aggregate_answer" in node_names
    assert "synthesize_answer" not in node_names


# ── End-to-end ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_graph_with_aggregation():
    """Full workflow with Phase 6 aggregation in the loop."""

    async def fake_search(query, num_results=5):
        return [
            {"title": "Health.gov", "url": "https://health.gov/c", "snippet": "Coffee info"},
        ]

    # 6 LLM calls: analyze, reason (tool_calls), filter, claims, scores, consensus, final
    fake_model = _fake_sequential_llm([
        "Is coffee healthy research",           # analyze_question
        {                                        # reason_and_act
            "tool_calls": [
                {"name": "search_web", "args": {"query": "Is coffee healthy"}, "id": "1"},
            ],
        },
        "1",                                     # filter_evidence
        json.dumps([{"claim": "Coffee is healthy", "verbatim_quote": "healthy"}]),  # claims
        json.dumps([{"source_number": 1, "quality_score": 0.8, "quality_reason": "good"}]),  # ranker
        json.dumps({                             # consensus
            "consensus_groups": [{"canonical_claim": "Coffee is healthy", "supporting_sources": [1]}],
            "disagreements": [],
            "uncertainties": [],
        }),
        "Coffee is generally considered healthy based on evidence [1].",  # final_writer
    ])

    with (
        patch("app.graph.nodes.get_llm", return_value=fake_model),
        patch("app.graph.nodes.web_search", side_effect=fake_search),
        patch("app.aggregation.claim_extractor.get_llm", return_value=fake_model),
        patch("app.aggregation.evidence_ranker.get_llm", return_value=fake_model),
        patch("app.aggregation.consensus_builder.get_llm", return_value=fake_model),
        patch("app.aggregation.final_writer.get_llm", return_value=fake_model),
    ):
        result = await run_graph("Is coffee healthy?")

    assert isinstance(result, PipelineResult)
    assert len(result.answer) > 0
    assert len(result.sources) >= 1
