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
