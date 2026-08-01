"""Interface comum a todos os provedores de embedding."""

from typing import Protocol


class EmbeddingProvider(Protocol):
    """Documentos e queries têm métodos separados de propósito: modelos
    assimétricos (ex.: BGE) aplicam prefixos diferentes a cada lado."""

    @property
    def dimensions(self) -> int: ...

    @property
    def model_name(self) -> str: ...

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...
