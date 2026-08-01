"""Chunking de tamanho fixo em tokens, com janela deslizante e overlap."""

from dataclasses import dataclass

from density.chunking.tokens import encode, token_char_offsets
from density.models import Chunk, Document


@dataclass
class FixedChunker:
    chunk_size: int = 512
    overlap: int = 64

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size deve ser positivo, recebido {self.chunk_size}")
        if not 0 <= self.overlap < self.chunk_size:
            raise ValueError(f"overlap ({self.overlap}) deve ser < chunk_size ({self.chunk_size})")

    def chunk(self, document: Document) -> list[Chunk]:
        content = document.content
        tokens = encode(content)
        offsets = token_char_offsets(content)
        step = self.chunk_size - self.overlap

        chunks: list[Chunk] = []
        for a in range(0, len(tokens), step):
            b = min(a + self.chunk_size, len(tokens))
            start, end = offsets[a], offsets[b]
            piece = content[start:end]
            if piece.strip():
                chunks.append(
                    Chunk(
                        document_id=document.id,
                        content=piece,
                        index=len(chunks),
                        start=start,
                        end=end,
                        metadata={"token_count": b - a},
                    )
                )
            if b == len(tokens):
                break
        return chunks
