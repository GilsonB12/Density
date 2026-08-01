import pytest

from density.chunking import RecursiveChunker
from density.models import Document


def make_doc(content: str) -> Document:
    return Document(source="test.txt", content=content)


def paragraph_text() -> tuple[str, list[str]]:
    paragraphs = [
        f"Parágrafo {i}: " + " ".join(f"palavra{i}{j}" for j in range(10)) + "."
        for i in range(6)
    ]
    return "\n\n".join(paragraphs), paragraphs


def test_respects_paragraph_boundaries_when_possible():
    text, paragraphs = paragraph_text()
    chunks = RecursiveChunker(chunk_size=80, overlap=0).chunk(make_doc(text))
    assert len(chunks) > 1
    paragraph_starts = {text.index(p) for p in paragraphs}
    for chunk in chunks:
        assert chunk.start in paragraph_starts


def test_chunks_are_exact_substrings():
    text, _ = paragraph_text()
    chunks = RecursiveChunker(chunk_size=80, overlap=0).chunk(make_doc(text))
    for chunk in chunks:
        assert chunk.content == text[chunk.start : chunk.end]
        assert chunk.metadata["token_count"] <= 80
    assert chunks[0].start == 0
    assert chunks[-1].end == len(text)


def test_covers_all_meaningful_content_without_overlap():
    text, _ = paragraph_text()
    chunks = RecursiveChunker(chunk_size=80, overlap=0).chunk(make_doc(text))
    covered: set[int] = set()
    for chunk in chunks:
        covered.update(range(chunk.start, chunk.end))
    missing = [i for i in range(len(text)) if i not in covered and not text[i].isspace()]
    assert missing == []


def test_hard_splits_text_without_separators():
    blob = "x" * 5000
    chunks = RecursiveChunker(chunk_size=32, overlap=0).chunk(make_doc(blob))
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.metadata["token_count"] <= 32
        assert chunk.content == blob[chunk.start : chunk.end]


def test_overlap_repeats_trailing_context():
    text = " ".join(f"Frase de numero {i}." for i in range(60))
    chunks = RecursiveChunker(chunk_size=40, overlap=15).chunk(make_doc(text))
    assert len(chunks) > 2
    overlapping = sum(
        1 for prev, nxt in zip(chunks, chunks[1:], strict=False) if nxt.start < prev.end
    )
    assert overlapping > 0


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        RecursiveChunker(chunk_size=32, overlap=32)
