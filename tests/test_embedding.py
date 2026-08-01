import random
from types import SimpleNamespace

import pytest

from density.embedding import OpenAIEmbeddingProvider


class FakeOpenAIClient:
    """Imita client.embeddings.create devolvendo data em ordem embaralhada."""

    def __init__(self, dimensions: int = 4):
        self.calls: list[dict] = []
        self._dimensions = dimensions
        self.embeddings = SimpleNamespace(create=self._create)

    def _create(self, *, model: str, input: list[str], dimensions: int):
        self.calls.append({"model": model, "input": list(input), "dimensions": dimensions})
        data = [
            SimpleNamespace(index=i, embedding=[float(i)] * dimensions)
            for i in range(len(input))
        ]
        random.Random(42).shuffle(data)  # a API não garante ordem; index é o contrato
        return SimpleNamespace(data=data)


def test_returns_one_vector_per_text_in_original_order():
    client = FakeOpenAIClient()
    provider = OpenAIEmbeddingProvider(client=client, dimensions=4)
    vectors = provider.embed_documents(["a", "b", "c"])
    assert vectors == [[0.0] * 4, [1.0] * 4, [2.0] * 4]


def test_batches_requests():
    client = FakeOpenAIClient()
    provider = OpenAIEmbeddingProvider(client=client, dimensions=4, batch_size=128)
    texts = [f"texto {i}" for i in range(300)]
    vectors = provider.embed_documents(texts)
    assert len(vectors) == 300
    assert [len(c["input"]) for c in client.calls] == [128, 128, 44]


def test_forwards_model_and_dimensions():
    client = FakeOpenAIClient()
    provider = OpenAIEmbeddingProvider(
        client=client, model="text-embedding-3-small", dimensions=256
    )
    provider.embed_documents(["a"])
    assert client.calls[0]["model"] == "text-embedding-3-small"
    assert client.calls[0]["dimensions"] == 256
    assert provider.dimensions == 256


def test_embed_query_returns_single_vector():
    provider = OpenAIEmbeddingProvider(client=FakeOpenAIClient(), dimensions=4)
    assert provider.embed_query("pergunta") == [0.0] * 4


def test_empty_input_makes_no_api_call():
    client = FakeOpenAIClient()
    provider = OpenAIEmbeddingProvider(client=client, dimensions=4)
    assert provider.embed_documents([]) == []
    assert client.calls == []


def test_rejects_invalid_batch_size():
    with pytest.raises(ValueError):
        OpenAIEmbeddingProvider(client=FakeOpenAIClient(), batch_size=0)
