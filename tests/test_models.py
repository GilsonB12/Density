import pytest
from pydantic import ValidationError

from density.models import Chunk, Document


def test_document_generates_unique_ids():
    a = Document(source="a.txt", content="alpha")
    b = Document(source="b.txt", content="beta")
    assert a.id != b.id


def test_document_rejects_empty_content():
    with pytest.raises(ValidationError):
        Document(source="a.txt", content="")


def test_chunk_links_to_document_and_carries_span():
    doc = Document(source="contrato.pdf", content="texto longo do contrato")
    chunk = Chunk(document_id=doc.id, content="texto", index=0, start=0, end=5)
    assert chunk.document_id == doc.id
    assert doc.content[chunk.start : chunk.end] == chunk.content


def test_chunk_rejects_negative_index():
    with pytest.raises(ValidationError):
        Chunk(document_id="x", content="y", index=-1, start=0, end=1)


def test_chunk_rejects_end_before_start():
    with pytest.raises(ValidationError):
        Chunk(document_id="x", content="y", index=0, start=10, end=5)
