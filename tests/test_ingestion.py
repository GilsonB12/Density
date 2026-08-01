from pathlib import Path

import pytest

from density.ingestion import (
    EmptyDocumentError,
    UnsupportedFormatError,
    annotate_pages,
    load_document,
)
from density.models import Chunk, Document

FIXTURES = Path(__file__).parent / "fixtures"


def test_load_txt(tmp_path):
    file = tmp_path / "nota.txt"
    file.write_text("Conteúdo simples de teste.", encoding="utf-8")
    doc = load_document(file)
    assert doc.content == "Conteúdo simples de teste."
    assert doc.source.endswith("nota.txt")
    assert doc.metadata["format"] == "txt"


def test_load_md(tmp_path):
    file = tmp_path / "readme.md"
    file.write_text("# Título\n\nParágrafo.", encoding="utf-8")
    doc = load_document(file)
    assert doc.content.startswith("# Título")
    assert doc.metadata["format"] == "md"


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_document(tmp_path / "nao_existe.txt")


def test_unsupported_format_raises(tmp_path):
    file = tmp_path / "dados.csv"
    file.write_text("a,b,c", encoding="utf-8")
    with pytest.raises(UnsupportedFormatError):
        load_document(file)


def test_whitespace_only_file_raises(tmp_path):
    file = tmp_path / "vazio.txt"
    file.write_text("   \n\n  ", encoding="utf-8")
    with pytest.raises(EmptyDocumentError):
        load_document(file)


def test_load_pdf_extracts_text_and_page_offsets():
    doc = load_document(FIXTURES / "sample.pdf")
    assert doc.metadata["format"] == "pdf"
    assert doc.metadata["pages"] == 2
    assert "page one" in doc.content
    assert "Second page" in doc.content

    offsets = doc.metadata["page_offsets"]
    assert [p["page"] for p in offsets] == [1, 2]
    assert offsets[0]["start"] == 0
    assert doc.content.index("Second page") >= offsets[1]["start"]


def test_annotate_pages_stamps_chunk_page_range():
    doc = Document(
        source="x.pdf",
        content="a" * 200,
        metadata={"page_offsets": [{"page": 1, "start": 0}, {"page": 2, "start": 100}]},
    )
    inside = Chunk(document_id=doc.id, content="a", index=0, start=40, end=90)
    spanning = Chunk(document_id=doc.id, content="a", index=1, start=90, end=150)
    annotate_pages(doc, [inside, spanning])
    assert (inside.metadata["page_start"], inside.metadata["page_end"]) == (1, 1)
    assert (spanning.metadata["page_start"], spanning.metadata["page_end"]) == (1, 2)


def test_annotate_pages_is_noop_without_page_offsets():
    doc = Document(source="x.txt", content="abc")
    chunk = Chunk(document_id=doc.id, content="abc", index=0, start=0, end=3)
    annotate_pages(doc, [chunk])
    assert "page_start" not in chunk.metadata
