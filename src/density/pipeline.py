"""Composition root: o único módulo que conhece todos os estágios do pipeline."""

from dataclasses import dataclass
from pathlib import Path

from density.chunking import Chunker
from density.embedding import EmbeddingProvider
from density.ingestion import annotate_pages, load_document
from density.models import Chunk, Document, EmbeddedChunk
from density.storage import VectorStore


@dataclass
class Indexer:
    """arquivo -> Document -> Chunks -> EmbeddedChunks -> VectorStore."""

    chunker: Chunker
    embedder: EmbeddingProvider
    store: VectorStore

    def index(self, path: str | Path) -> tuple[Document, list[Chunk]]:
        document = load_document(path)
        chunks = self.chunker.chunk(document)
        annotate_pages(document, chunks)

        vectors = self.embedder.embed_documents([c.content for c in chunks])
        embedded = [
            EmbeddedChunk(chunk=chunk, embedding=vector)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        self.store.ensure_schema(self.embedder.dimensions)
        self.store.store(document, embedded)
        return document, chunks
