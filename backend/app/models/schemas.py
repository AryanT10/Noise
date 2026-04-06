from pydantic import BaseModel


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
