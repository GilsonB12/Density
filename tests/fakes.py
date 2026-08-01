"""Dublês determinísticos para testes — sem rede, sem banco."""

import hashlib

from density.models import Document, EmbeddedChunk


class FakeEmbeddingProvider:
    """Vetores determinísticos derivados de hash do texto (sem semântica)."""

    dimensions = 8
    model_name = "fake-embedder"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.md5(text.encode("utf-8")).digest()
        return [b / 255 for b in digest[: self.dimensions]]


class InMemoryVectorStore:
    """Implementa o Protocol VectorStore com um dict, semântica replace-by-source."""

    def __init__(self) -> None:
        self.schema_dimensions: int | None = None
        self.by_source: dict[str, tuple[Document, list[EmbeddedChunk]]] = {}

    def ensure_schema(self, dimensions: int) -> None:
        self.schema_dimensions = dimensions

    def store(self, document: Document, chunks: list[EmbeddedChunk]) -> None:
        self.by_source[document.source] = (document, chunks)

    def count_chunks(self) -> int:
        return sum(len(chunks) for _, chunks in self.by_source.values())
