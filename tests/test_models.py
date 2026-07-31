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


def test_chunk_links_to_document_and_carries_page_metadata():
    doc = Document(source="contrato.pdf", content="texto longo do contrato")
    chunk = Chunk(document_id=doc.id, content="texto", index=0, metadata={"page": 12})
    assert chunk.document_id == doc.id
    assert chunk.metadata["page"] == 12


def test_chunk_rejects_negative_index():
    with pytest.raises(ValidationError):
        Chunk(document_id="x", content="y", index=-1)
