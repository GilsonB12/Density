"""Contratos de dados que fluem entre os estágios do pipeline.

Cada módulo do density é uma função de um contrato para outro:

    ingestion:  arquivo      -> Document
    chunking:   Document     -> list[Chunk]
    embedding:  list[Chunk]  -> list[EmbeddedChunk]   (etapa 2)
    ...

Os contratos das etapas futuras (EmbeddedChunk, RetrievalResult, Answer,
EvalResult) serão adicionados na etapa em que se tornarem necessários.
"""

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


def _new_id() -> str:
    return uuid4().hex


class Document(BaseModel):
    """Um documento ingerido: texto extraído + metadados de origem."""

    id: str = Field(default_factory=_new_id)
    source: str
    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Um trecho de um Document — a unidade de embedding, retrieval e citação."""

    id: str = Field(default_factory=_new_id)
    document_id: str
    content: str = Field(min_length=1)
    index: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
