"""Embedding: transforma texto em vetores onde proximidade ≈ similaridade semântica."""

from density.embedding.base import EmbeddingProvider
from density.embedding.openai import OpenAIEmbeddingProvider

__all__ = ["EmbeddingProvider", "OpenAIEmbeddingProvider"]
