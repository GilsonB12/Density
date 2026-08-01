"""Interface comum a todas as estratégias de chunking."""

from typing import Protocol

from density.models import Chunk, Document


class Chunker(Protocol):
    def chunk(self, document: Document) -> list[Chunk]: ...
