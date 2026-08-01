import os

import psycopg
import pytest

from density.models import Chunk, Document, EmbeddedChunk
from density.storage.pgvector import PgVectorStore

DATABASE_URL = os.environ.get(
    "DENSITY_DATABASE_URL", "postgresql://density:density@localhost:5432/density"
)
SCHEMA = "density_test"
DIMS = 8

pytestmark = pytest.mark.integration


@pytest.fixture
def store():
    try:
        admin = psycopg.connect(DATABASE_URL, autocommit=True)
    except psycopg.OperationalError:
        pytest.skip("Postgres indisponível — rode: docker compose up -d")
    admin.execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE")
    admin.close()

    s = PgVectorStore(DATABASE_URL, schema=SCHEMA)
    yield s
    s.close()


def make_embedded(document: Document, texts: list[str]) -> list[EmbeddedChunk]:
    out = []
    pos = 0
    for i, text in enumerate(texts):
        chunk = Chunk(
            document_id=document.id,
            content=text,
            index=i,
            start=pos,
            end=pos + len(text),
            metadata={"page_start": i + 1},
        )
        out.append(EmbeddedChunk(chunk=chunk, embedding=[float(i)] * DIMS))
        pos += len(text)
    return out


def test_ensure_schema_is_idempotent_and_starts_empty(store):
    store.ensure_schema(DIMS)
    store.ensure_schema(DIMS)
    assert store.count_chunks() == 0


def test_store_persists_document_and_chunks(store):
    store.ensure_schema(DIMS)
    doc = Document(source="contrato.pdf", content="ab", metadata={"format": "pdf"})
    store.store(doc, make_embedded(doc, ["a", "b"]))
    assert store.count_chunks() == 2


def test_restore_same_source_replaces_instead_of_duplicating(store):
    store.ensure_schema(DIMS)
    doc1 = Document(source="contrato.pdf", content="abc")
    store.store(doc1, make_embedded(doc1, ["a", "b", "c"]))
    doc2 = Document(source="contrato.pdf", content="xy")
    store.store(doc2, make_embedded(doc2, ["x", "y"]))
    assert store.count_chunks() == 2


def test_different_sources_accumulate(store):
    store.ensure_schema(DIMS)
    doc1 = Document(source="a.pdf", content="a")
    store.store(doc1, make_embedded(doc1, ["a"]))
    doc2 = Document(source="b.pdf", content="b")
    store.store(doc2, make_embedded(doc2, ["b"]))
    assert store.count_chunks() == 2


def test_dimension_mismatch_raises_clear_error(store):
    store.ensure_schema(DIMS)
    with pytest.raises(ValueError, match=r"vector\(8\)"):
        store.ensure_schema(DIMS + 1)


def test_chunk_row_roundtrips_content_metadata_and_vector(store):
    store.ensure_schema(DIMS)
    doc = Document(source="c.pdf", content="hello")
    store.store(doc, make_embedded(doc, ["hello"]))

    with psycopg.connect(DATABASE_URL) as conn:
        row = conn.execute(
            f"SELECT content, chunk_index, start_char, end_char, "
            f"metadata->>'page_start', vector_dims(embedding) FROM {SCHEMA}.chunks"
        ).fetchone()
    assert row == ("hello", 0, 0, 5, "1", DIMS)
