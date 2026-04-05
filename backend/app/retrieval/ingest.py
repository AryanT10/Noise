"""Document ingestion — load from various sources, chunk, and store."""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.logging import logger
from app.retrieval.store import add_documents
from app.tools.scraper import extract_page_text

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ". ", " ", ""],
)


async def ingest_url(url: str) -> int:
    """Scrape a web page, chunk it, and add to the vector store."""
    logger.info("Ingesting URL: %s", url)
    text = await extract_page_text(url, max_chars=50_000)
    if not text:
        logger.warning("No text extracted from %s", url)
        return 0

    docs = _splitter.create_documents(
        texts=[text],
        metadatas=[{"source": url, "type": "web"}],
    )
    return add_documents(docs)


async def ingest_pdf(path: str) -> int:
    """Load a PDF file, chunk it, and add to the vector store."""
    from langchain_community.document_loaders import PyPDFLoader

    logger.info("Ingesting PDF: %s", path)
    loader = PyPDFLoader(path)
    pages = loader.load()

    docs = _splitter.split_documents(pages)
    for doc in docs:
        doc.metadata["type"] = "pdf"
    return add_documents(docs)


async def ingest_text(text: str, source: str = "manual") -> int:
    """Chunk raw text and add to the vector store."""
    logger.info("Ingesting text from source: %s (%d chars)", source, len(text))
    docs = _splitter.create_documents(
        texts=[text],
        metadatas=[{"source": source, "type": "text"}],
    )
    return add_documents(docs)
