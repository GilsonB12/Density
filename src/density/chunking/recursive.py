"""Chunking recursivo: respeita fronteiras semânticas (parágrafo > linha > frase).

O texto é quebrado pela hierarquia de separadores até cada peça caber no
orçamento de tokens; peças contíguas são então mescladas de forma gulosa.
Cada separador fica anexado à peça anterior, então a concatenação das peças
reconstrói o texto original — o que garante offsets exatos para citações.
"""

from dataclasses import dataclass

from density.chunking.tokens import count_tokens, encode, token_char_offsets
from density.models import Chunk, Document

_SEPARATORS = ("\n\n", "\n", ". ", " ")

# (start, texto) de um trecho contíguo do documento
_Piece = tuple[int, str]


@dataclass
class RecursiveChunker:
    chunk_size: int = 512
    overlap: int = 64

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size deve ser positivo, recebido {self.chunk_size}")
        if not 0 <= self.overlap < self.chunk_size:
            raise ValueError(f"overlap ({self.overlap}) deve ser < chunk_size ({self.chunk_size})")

    def chunk(self, document: Document) -> list[Chunk]:
        pieces = self._split(document.content, 0, list(_SEPARATORS))
        spans = self._merge(pieces)

        chunks: list[Chunk] = []
        for start, end in spans:
            piece = document.content[start:end]
            if piece.strip():
                chunks.append(
                    Chunk(
                        document_id=document.id,
                        content=piece,
                        index=len(chunks),
                        start=start,
                        end=end,
                        metadata={"token_count": count_tokens(piece)},
                    )
                )
        return chunks

    def _split(self, text: str, base: int, separators: list[str]) -> list[_Piece]:
        """Quebra `text` em peças de até chunk_size tokens, preservando offsets."""
        if count_tokens(text) <= self.chunk_size:
            return [(base, text)]
        if not separators:
            return self._hard_split(text, base)

        sep, rest = separators[0], separators[1:]
        segments = text.split(sep)
        if len(segments) == 1:
            return self._split(text, base, rest)

        pieces: list[_Piece] = []
        pos = base
        for i, segment in enumerate(segments):
            piece = segment + sep if i < len(segments) - 1 else segment
            if piece:
                if count_tokens(piece) <= self.chunk_size:
                    pieces.append((pos, piece))
                else:
                    pieces.extend(self._split(piece, pos, rest))
                pos += len(piece)
        return pieces

    def _hard_split(self, text: str, base: int) -> list[_Piece]:
        """Último recurso: corte seco por tokens (texto sem nenhum separador)."""
        tokens = encode(text)
        offsets = token_char_offsets(text)
        pieces: list[_Piece] = []
        for a in range(0, len(tokens), self.chunk_size):
            b = min(a + self.chunk_size, len(tokens))
            pieces.append((base + offsets[a], text[offsets[a] : offsets[b]]))
        return pieces

    def _merge(self, pieces: list[_Piece]) -> list[tuple[int, int]]:
        """Mescla peças contíguas em janelas de até chunk_size tokens."""
        spans: list[tuple[int, int]] = []
        window: list[_Piece] = []
        window_tokens = 0

        for piece in pieces:
            piece_tokens = count_tokens(piece[1])
            if window and window_tokens + piece_tokens > self.chunk_size:
                spans.append(self._window_span(window))
                window, window_tokens = self._carry_overlap(window)
            window.append(piece)
            window_tokens += piece_tokens

        if window:
            spans.append(self._window_span(window))
        return spans

    @staticmethod
    def _window_span(window: list[_Piece]) -> tuple[int, int]:
        last_start, last_text = window[-1]
        return window[0][0], last_start + len(last_text)

    def _carry_overlap(self, window: list[_Piece]) -> tuple[list[_Piece], int]:
        """Peças finais da janela (até `overlap` tokens) que abrem a próxima."""
        kept: list[_Piece] = []
        kept_tokens = 0
        for piece in reversed(window):
            piece_tokens = count_tokens(piece[1])
            if kept_tokens + piece_tokens > self.overlap:
                break
            kept.insert(0, piece)
            kept_tokens += piece_tokens
        if len(kept) == len(window):  # janela inteira caberia: não haveria avanço
            kept = kept[1:]
            kept_tokens -= count_tokens(window[0][1])
        return kept, kept_tokens
