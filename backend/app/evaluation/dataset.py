"""Test dataset — curated questions covering different evaluation dimensions."""

from app.evaluation.schemas import EvalCase

EVAL_DATASET: list[EvalCase] = [
    # ── Factual questions ────────────────────────────────────
    EvalCase(
        id="factual-01",
        question="What is the speed of light in a vacuum?",
        category="factual",
        expected_keywords=["299,792", "kilometers", "per second", "speed of light"],
        forbidden_keywords=[],
        min_sources=1,
        expected_citation_count=1,
        notes="Should cite a physics reference. Answer must include ~299,792 km/s.",
    ),
    EvalCase(
        id="factual-02",
        question="Who wrote the novel '1984'?",
        category="factual",
        expected_keywords=["George Orwell"],
        forbidden_keywords=["Aldous Huxley", "Ray Bradbury"],
        min_sources=1,
        expected_citation_count=1,
        notes="Unambiguous. Must attribute to Orwell, not confuse with other dystopian authors.",
    ),
    EvalCase(
        id="factual-03",
        question="What is the capital of Australia?",
        category="factual",
        expected_keywords=["Canberra"],
        forbidden_keywords=["Sydney", "Melbourne"],
        min_sources=1,
        expected_citation_count=1,
        notes="Common misconception test. Must say Canberra, NOT Sydney.",
    ),
    # ── Multi-source / consensus questions ───────────────────
    EvalCase(
        id="multi-01",
        question="Is coffee good or bad for your health?",
        category="multi-source",
        expected_keywords=["antioxidant", "caffeine", "moderate"],
        forbidden_keywords=[],
        min_sources=2,
        expected_citation_count=2,
        notes="Should present both benefits and risks. Must cite multiple sources.",
    ),
    EvalCase(
        id="multi-02",
        question="What are the pros and cons of remote work?",
        category="multi-source",
        expected_keywords=["flexibility", "isolation", "productivity"],
        forbidden_keywords=[],
        min_sources=2,
        expected_citation_count=2,
        notes="Balanced answer covering both sides, sourced from multiple references.",
    ),
    # ── Opinion / nuanced questions ──────────────────────────
    EvalCase(
        id="opinion-01",
        question="Is artificial intelligence dangerous?",
        category="opinion",
        expected_keywords=["risk", "benefit", "AI"],
        forbidden_keywords=[],
        min_sources=2,
        expected_citation_count=2,
        notes="Should acknowledge multiple viewpoints. Not make absolute claims.",
    ),
    # ── Ambiguous questions ──────────────────────────────────
    EvalCase(
        id="ambiguous-01",
        question="How big is a football field?",
        category="ambiguous",
        expected_keywords=["yard", "meter", "field"],
        forbidden_keywords=[],
        min_sources=1,
        expected_citation_count=1,
        notes="Should address American vs soccer field or clarify the ambiguity.",
    ),
    # ── Recent events (tests freshness of web search) ────────
    EvalCase(
        id="recent-01",
        question="What were the major tech layoffs in 2025?",
        category="recent-event",
        expected_keywords=["layoff", "2025"],
        forbidden_keywords=[],
        min_sources=2,
        expected_citation_count=1,
        notes="Tests that the pipeline uses live web search for recent info.",
    ),
    # ── Hallucination trap ───────────────────────────────────
    EvalCase(
        id="hallucination-01",
        question="What did Albert Einstein say about social media?",
        category="factual",
        expected_keywords=[],
        forbidden_keywords=[],
        min_sources=1,
        expected_citation_count=1,
        notes="Einstein died in 1955 — should note he never commented on social media, or flag misattribution.",
    ),
    EvalCase(
        id="hallucination-02",
        question="What is the population of the underwater city of Atlantis?",
        category="factual",
        expected_keywords=["myth", "fiction", "legend"],
        forbidden_keywords=[],
        min_sources=1,
        expected_citation_count=1,
        notes="Must not fabricate population numbers. Should clarify Atlantis is mythological.",
    ),
]
