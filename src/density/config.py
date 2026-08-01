"""Configuração centralizada, carregada de variáveis de ambiente e do arquivo .env."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="DENSITY_",
        extra="ignore",
        populate_by_name=True,
    )

    database_url: str = "postgresql://density:density@localhost:5432/density"

    # sem o prefixo DENSITY_: é o nome que o ecossistema OpenAI já usa
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")

    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536


def get_settings() -> Settings:
    """Ponto único de acesso à configuração (facilita override em testes)."""
    return Settings()
