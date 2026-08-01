from density.chunking import RecursiveChunker
from density.pipeline import Indexer
from tests.fakes import FakeEmbeddingProvider, InMemoryVectorStore


def make_indexer(store: InMemoryVectorStore) -> Indexer:
    return Indexer(
        chunker=RecursiveChunker(chunk_size=32, overlap=4),
        embedder=FakeEmbeddingProvider(),
        store=store,
    )


def write_sample(tmp_path, name="doc.txt", text="Primeiro parágrafo.\n\nSegundo parágrafo maior."):
    file = tmp_path / name
    file.write_text(text, encoding="utf-8")
    return file


def test_index_stores_document_with_embedded_chunks(tmp_path):
    store = InMemoryVectorStore()
    document, chunks = make_indexer(store).index(write_sample(tmp_path))

    assert len(chunks) >= 1
    assert store.schema_dimensions == FakeEmbeddingProvider.dimensions
    stored_doc, stored_chunks = store.by_source[document.source]
    assert stored_doc.id == document.id
    assert [e.chunk.id for e in stored_chunks] == [c.id for c in chunks]
    for embedded in stored_chunks:
        assert len(embedded.embedding) == FakeEmbeddingProvider.dimensions


def test_embeddings_follow_chunk_content(tmp_path):
    store = InMemoryVectorStore()
    make_indexer(store).index(write_sample(tmp_path))
    provider = FakeEmbeddingProvider()
    for _, stored_chunks in store.by_source.values():
        for embedded in stored_chunks:
            assert embedded.embedding == provider.embed_query(embedded.chunk.content)


def test_reindexing_same_file_replaces_instead_of_duplicating(tmp_path):
    store = InMemoryVectorStore()
    indexer = make_indexer(store)
    file = write_sample(tmp_path)
    indexer.index(file)
    first_count = store.count_chunks()
    indexer.index(file)
    assert store.count_chunks() == first_count
    assert len(store.by_source) == 1
