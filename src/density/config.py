"""Configuração centralizada, carregada de variáveis de ambiente e do arquivo .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="DENSITY_", extra="ignore")

    database_url: str = "postgresql://density:density@localhost:5432/density"


def get_settings() -> Settings:
    """Ponto único de acesso à configuração (facilita override em testes)."""
    return Settings()
