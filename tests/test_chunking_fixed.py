import pytest

from density.chunking import FixedChunker
from density.models import Document


def make_doc(content: str) -> Document:
    return Document(source="test.txt", content=content)


def long_text(words: int = 400) -> str:
    return " ".join(f"palavra{i}" for i in range(words))


def test_short_document_yields_single_chunk():
    doc = make_doc("Um parágrafo curto.")
    chunks = FixedChunker(chunk_size=128, overlap=16).chunk(doc)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.content == doc.content
    assert (chunk.start, chunk.end) == (0, len(doc.content))
    assert chunk.document_id == doc.id


def test_chunks_are_exact_substrings_and_cover_document():
    text = long_text()
    chunks = FixedChunker(chunk_size=64, overlap=16).chunk(make_doc(text))
    assert len(chunks) > 3
    for chunk in chunks:
        assert chunk.content == text[chunk.start : chunk.end]
        assert chunk.metadata["token_count"] <= 64
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_consecutive_chunks_overlap():
    chunks = FixedChunker(chunk_size=64, overlap=16).chunk(make_doc(long_text()))
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.start < prev.end


def test_zero_overlap_produces_contiguous_chunks():
    chunks = FixedChunker(chunk_size=64, overlap=0).chunk(make_doc(long_text(200)))
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        assert nxt.start == prev.end


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        FixedChunker(chunk_size=64, overlap=64)
