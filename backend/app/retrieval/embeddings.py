"""Embedding model factory — mirrors the llm.py provider pattern."""

from langchain_core.embeddings import Embeddings

from app.config import settings
from app.logging import logger

_DEFAULT_EMBEDDING_MODELS = {
    "openai": "text-embedding-3-small",
    "gemini": "gemini-embedding-001",
}


def get_embeddings() -> Embeddings:
    """Return an embedding model based on the configured provider.

    Groq has no embedding API, so we fall back to whichever key is available.
    """
    provider = settings.embedding_provider or settings.llm_provider

    if provider == "openai" and settings.openai_api_key:
        from langchain_openai import OpenAIEmbeddings

        logger.info("Using OpenAI embeddings (%s)", _DEFAULT_EMBEDDING_MODELS["openai"])
        return OpenAIEmbeddings(
            model=_DEFAULT_EMBEDDING_MODELS["openai"],
            api_key=settings.openai_api_key,
        )

    if provider == "gemini" and settings.google_api_key:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        logger.info("Using Google embeddings (%s)", _DEFAULT_EMBEDDING_MODELS["gemini"])
        return GoogleGenerativeAIEmbeddings(
            model=_DEFAULT_EMBEDDING_MODELS["gemini"],
            google_api_key=settings.google_api_key,
            transport="rest",
        )

    # Groq fallback: try Google first (free), then OpenAI
    if provider == "groq":
        if settings.google_api_key:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings

            logger.info("Groq has no embeddings — falling back to Google")
            return GoogleGenerativeAIEmbeddings(
                model=_DEFAULT_EMBEDDING_MODELS["gemini"],
                google_api_key=settings.google_api_key,
                transport="rest",
            )
        if settings.openai_api_key:
            from langchain_openai import OpenAIEmbeddings

            logger.info("Groq has no embeddings — falling back to OpenAI")
            return OpenAIEmbeddings(
                model=_DEFAULT_EMBEDDING_MODELS["openai"],
                api_key=settings.openai_api_key,
            )

    raise ValueError(
        f"No embedding provider available for '{provider}'. "
        "Set OPENAI_API_KEY or GOOGLE_API_KEY for embeddings."
    )
