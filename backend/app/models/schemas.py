from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    number: int
    title: str
    url: str


class Snippet(BaseModel):
    source_number: int
    text: str


class PipelineResult(BaseModel):
    answer: str
    sources: list[Source]
    snippets: list[Snippet]


class AskResponse(BaseModel):
    answer: str
    model: str
    sources: list[Source] = []
    snippets: list[Snippet] = []


# ── Retrieval (Phase 3) ─────────────────────────────────────

class IngestRequest(BaseModel):
    """Ingest one or more documents into the vector store."""
    urls: list[str] = []
    texts: list[str] = []
    # pdf_paths omitted from API for security — use ingest_pdf() directly


class IngestResponse(BaseModel):
    chunks_added: int


class RAGAskRequest(BaseModel):
    question: str
    top_k: int = 5


class RAGAskResponse(BaseModel):
    answer: str
    model: str
    chunks_used: int
    sources: list[str] = []


# ── Aggregation (Phase 6) ───────────────────────────────────


class Claim(BaseModel):
    """A single factual claim extracted from a source."""
    claim: str
    source_number: int
    verbatim_quote: str = ""


class EvidenceItem(BaseModel):
    """A piece of evidence with quality assessment."""
    source_number: int
    title: str
    url: str
    text: str
    quality_score: float = 0.0        # 0.0–1.0
    quality_reason: str = ""


class ConsensusGroup(BaseModel):
    """A group of claims that say the same thing (agreement cluster)."""
    canonical_claim: str
    supporting_sources: list[int] = []
    agreement_count: int = 0


class AggregationResult(BaseModel):
    """Full output of the answer aggregation pipeline."""
    claims: list[Claim] = []
    evidence: list[EvidenceItem] = []
    consensus_groups: list[ConsensusGroup] = []
    disagreements: list[str] = []
    uncertainties: list[str] = []
    final_answer: str = ""


# ── Phase 7: Structured LLM output schemas ──────────────────


class SearchQueryGeneration(BaseModel):
    """Structured output for search query planning."""
    queries: list[str] = Field(
        description="1–3 concise web search queries to help answer the user question."
    )


class ExtractedClaim(BaseModel):
    """A single claim extracted by the LLM from a source."""
    claim: str = Field(description="A one-sentence factual statement.")
    verbatim_quote: str = Field(
        default="",
        description="The closest verbatim phrase from the source.",
    )


class ExtractedClaimList(BaseModel):
    """Structured output for claim extraction from a single source."""
    claims: list[ExtractedClaim] = Field(
        default_factory=list,
        description="All distinct factual claims relevant to the question.",
    )


class SourceScore(BaseModel):
    """Quality score for a single source."""
    source_number: int
    quality_score: float = Field(
        ge=0.0, le=1.0,
        description="Quality/relevance score from 0.0 to 1.0.",
    )
    quality_reason: str = Field(
        default="",
        description="Short explanation for the score.",
    )


class EvidenceScoreList(BaseModel):
    """Structured output for evidence ranking."""
    scores: list[SourceScore] = Field(
        description="Quality scores for each source.",
    )


class ConsensusResult(BaseModel):
    """Structured output for consensus analysis."""
    consensus_groups: list[ConsensusGroup] = Field(
        default_factory=list,
        description="Groups of claims that express the same fact.",
    )
    disagreements: list[str] = Field(
        default_factory=list,
        description="Descriptions of contradictions between sources.",
    )
    uncertainties: list[str] = Field(
        default_factory=list,
        description="Aspects that are unclear or speculative.",
    )


class RelevantSourceNumbers(BaseModel):
    """Structured output for evidence filtering."""
    source_numbers: list[int] = Field(
        default_factory=list,
        description="Numbers of sources relevant to the question. Empty if none are relevant.",
    )


class AggregatedAnswer(BaseModel):
    """Full structured response from the aggregation pipeline."""
    answer: str = Field(description="The final aggregated judgment with inline citations.")
    claims: list[Claim] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    consensus_groups: list[ConsensusGroup] = Field(default_factory=list)
    disagreements: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    snippets: list[Snippet] = Field(default_factory=list)
