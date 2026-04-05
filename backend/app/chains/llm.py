from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage

from app.config import settings
from app.logging import logger

_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "groq": "llama-3.3-70b-versatile",
}


def _resolve_model() -> str:
    return settings.llm_model or _DEFAULT_MODELS[settings.llm_provider]


def get_llm() -> BaseChatModel:
    provider = settings.llm_provider

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=_resolve_model(),
            api_key=settings.openai_api_key,
            temperature=0,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=_resolve_model(),
            google_api_key=settings.google_api_key,
            temperature=0,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=_resolve_model(),
            api_key=settings.groq_api_key,
            temperature=0,
        )

    raise ValueError(f"Unknown LLM provider: {provider}")


async def ask_llm(question: str) -> str:
    model_name = _resolve_model()
    logger.info("Calling LLM provider=%s model=%s", settings.llm_provider, model_name)
    llm = get_llm()
    response = await llm.ainvoke([HumanMessage(content=question)])
    logger.info("LLM responded (%d chars)", len(response.content))
    return response.content
