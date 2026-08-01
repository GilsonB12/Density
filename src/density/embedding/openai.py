"""Adapter de embeddings da OpenAI (text-embedding-3-*)."""

from typing import Any

from openai import OpenAI

# preço por 1M de tokens de entrada (jul/2026) — usado só para estimativas na CLI
PRICE_PER_MILLION_TOKENS = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}


class OpenAIEmbeddingProvider:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "text-embedding-3-small",
        dimensions: int = 1536,
        batch_size: int = 128,
        client: Any | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size deve ser positivo, recebido {batch_size}")
        if dimensions <= 0:
            raise ValueError(f"dimensions deve ser positivo, recebido {dimensions}")
        # sem api_key explícita, o SDK cai para a env var OPENAI_API_KEY
        # (e levanta OpenAIError na construção se nenhuma existir)
        self._client = client if client is not None else OpenAI(api_key=api_key)
        self._model = model
        self._dimensions = dimensions
        self._batch_size = batch_size

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def price_per_million_tokens(self) -> float | None:
        return PRICE_PER_MILLION_TOKENS.get(self._model)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), self._batch_size):
            batch = texts[i : i + self._batch_size]
            response = self._client.embeddings.create(
                model=self._model,
                input=batch,
                dimensions=self._dimensions,
            )
            # a ordem de response.data não é garantida; .index é o contrato
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend([item.embedding for item in ordered])
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"API devolveu {len(vectors)} embeddings para {len(texts)} textos"
            )
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
