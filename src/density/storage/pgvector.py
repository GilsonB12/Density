"""Backend de armazenamento em Postgres + pgvector."""

import re

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.types.json import Jsonb

from density.models import Document, EmbeddedChunk

_VALID_SCHEMA = re.compile(r"^[a-z_][a-z0-9_]*$")


class PgVectorStore:
    """Documentos e chunks em duas tabelas; o vetor mora junto do chunk.

    Semântica de gravação: re-ingerir o mesmo `source` substitui o documento
    inteiro (DELETE + INSERT na mesma transação) — nunca duplica.
    """

    def __init__(self, database_url: str, schema: str = "public") -> None:
        if not _VALID_SCHEMA.match(schema):
            raise ValueError(f"nome de schema inválido: {schema!r}")
        self._schema = schema
        # autocommit=True: cada execute() commita sozinho, e os blocos
        # `with conn.transaction()` viram transações REAIS onde atomicidade
        # importa. Sem isso, o psycopg3 abre uma transação implícita que nunca
        # commita e os blocos transaction() viram apenas savepoints dela.
        self._conn = psycopg.connect(database_url, autocommit=True)
        if schema != "public":
            self._conn.execute(f"CREATE SCHEMA IF NOT EXISTS {schema}")
        # public fica no path para o tipo `vector` (extensão) resolver
        self._conn.execute(f"SET search_path TO {schema}, public")
        register_vector(self._conn)

    def ensure_schema(self, dimensions: int) -> None:
        existing = self._embedding_dimensions()
        if existing is not None and existing != dimensions:
            raise ValueError(
                f"tabela chunks já existe com vector({existing}), mas o provedor "
                f"produz {dimensions} dimensões — apague a tabela ou use outro schema"
            )
        with self._conn.transaction():
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    source TEXT NOT NULL UNIQUE,
                    content TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'
                )
                """
            )
            self._conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    content TEXT NOT NULL,
                    chunk_index INT NOT NULL,
                    start_char INT NOT NULL,
                    end_char INT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}',
                    embedding vector({int(dimensions)}) NOT NULL
                )
                """
            )
            # sem índice ANN por enquanto: busca exata primeiro (etapa 3),
            # índice HNSW só quando houver números que justifiquem

    def store(self, document: Document, chunks: list[EmbeddedChunk]) -> None:
        with self._conn.transaction():
            self._conn.execute("DELETE FROM documents WHERE source = %s", (document.source,))
            self._conn.execute(
                "INSERT INTO documents (id, source, content, metadata) VALUES (%s, %s, %s, %s)",
                (document.id, document.source, document.content, Jsonb(document.metadata)),
            )
            with self._conn.cursor() as cur:
                cur.executemany(
                    """
                    INSERT INTO chunks
                        (id, document_id, content, chunk_index, start_char, end_char,
                         metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    [
                        (
                            e.chunk.id,
                            e.chunk.document_id,
                            e.chunk.content,
                            e.chunk.index,
                            e.chunk.start,
                            e.chunk.end,
                            Jsonb(e.chunk.metadata),
                            Vector(e.embedding),
                        )
                        for e in chunks
                    ],
                )

    def count_chunks(self) -> int:
        row = self._conn.execute("SELECT count(*) FROM chunks").fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self._conn.close()

    def _embedding_dimensions(self) -> int | None:
        """Dimensão declarada da coluna embedding, ou None se a tabela não existe."""
        row = self._conn.execute(
            """
            SELECT format_type(a.atttypid, a.atttypmod)
            FROM pg_attribute a
            JOIN pg_class c ON a.attrelid = c.oid
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = %s AND c.relname = 'chunks' AND a.attname = 'embedding'
            """,
            (self._schema,),
        ).fetchone()
        if row is None:
            return None
        match = re.search(r"vector\((\d+)\)", row[0])
        return int(match.group(1)) if match else None
