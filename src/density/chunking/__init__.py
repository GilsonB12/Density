"""Chunking: fatia Documents em Chunks prontos para embedding e retrieval."""

from density.chunking.base import Chunker
from density.chunking.fixed import FixedChunker
from density.chunking.recursive import RecursiveChunker

__all__ = ["Chunker", "FixedChunker", "RecursiveChunker"]
