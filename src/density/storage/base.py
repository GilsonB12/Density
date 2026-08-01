"""Interface comum a todos os backends de armazenamento vetorial."""

from typing import Protocol

from density.models import Document, EmbeddedChunk


class VectorStore(Protocol):
    def ensure_schema(self, dimensions: int) -> None: ...

    def store(self, document: Document, chunks: list[EmbeddedChunk]) -> None:
        """Grava documento + chunks; re-gravar o mesmo source substitui (não duplica)."""
        ...

    def count_chunks(self) -> int: ...
