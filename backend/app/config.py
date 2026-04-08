from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Provider: "openai", "gemini", or "groq"
    llm_provider: str = "groq"

    openai_api_key: str = ""
    google_api_key: str = ""
    groq_api_key: str = ""
    serper_api_key: str = ""

    llm_model: str = ""  # auto-picks per provider if empty
    log_level: str = "INFO"

    # Retrieval / embeddings — defaults to llm_provider if empty
    embedding_provider: str = ""

    # Phase 8: LangSmith observability
    langsmith_api_key: str = ""
    langsmith_project: str = "noise"
    langsmith_endpoint: str = ""  # defaults to https://smith.langchain.com


settings = Settings()
