"""Ingestão: transforma arquivos (PDF, TXT, MD) em Documents com texto extraído."""

from density.ingestion.loader import (
    EmptyDocumentError,
    UnsupportedFormatError,
    load_document,
)
from density.ingestion.pages import annotate_pages, page_for_offset

__all__ = [
    "EmptyDocumentError",
    "UnsupportedFormatError",
    "annotate_pages",
    "load_document",
    "page_for_offset",
]
