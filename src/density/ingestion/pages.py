"""Mapeamento de offsets de caractere para páginas de origem."""

from bisect import bisect_right

from density.models import Chunk, Document


def page_for_offset(page_offsets: list[dict[str, int]], offset: int) -> int:
    """Página que contém o caractere na posição `offset` do content."""
    starts = [p["start"] for p in page_offsets]
    i = bisect_right(starts, offset) - 1
    return page_offsets[max(i, 0)]["page"]


def annotate_pages(document: Document, chunks: list[Chunk]) -> None:
    """Grava page_start/page_end no metadata de cada chunk (no-op sem page_offsets)."""
    page_offsets = document.metadata.get("page_offsets")
    if not page_offsets:
        return
    for chunk in chunks:
        chunk.metadata["page_start"] = page_for_offset(page_offsets, chunk.start)
        last_char = max(chunk.start, chunk.end - 1)
        chunk.metadata["page_end"] = page_for_offset(page_offsets, last_char)
