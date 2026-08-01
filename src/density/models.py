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

from pydantic import BaseModel, Field, model_validator


def _new_id() -> str:
    return uuid4().hex


class Document(BaseModel):
    """Um documento ingerido: texto extraído + metadados de origem."""

    id: str = Field(default_factory=_new_id)
    source: str
    content: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Um trecho de um Document — a unidade de embedding, retrieval e citação.

    `start`/`end` são offsets de caractere no `Document.content` de origem:
    o contrato garante `document.content[start:end] == content`, o que permite
    mapear qualquer chunk de volta à página/posição exata para citações.
    """

    id: str = Field(default_factory=_new_id)
    document_id: str
    content: str = Field(min_length=1)
    index: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _span_is_ordered(self) -> "Chunk":
        if self.end < self.start:
            raise ValueError(f"span invertido: end ({self.end}) < start ({self.start})")
        return self
