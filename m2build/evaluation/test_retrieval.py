import numpy as np

from src.shared.vector_store import FAISSVectorStore


class FakeEmbedder:
    def encode(self, texts):
        return np.asarray(
            [[1.0, 0.0] for _ in texts],
            dtype="float32"
        )


def test_vector_store_retrieval():
    embedder = FakeEmbedder()

    store = FAISSVectorStore(2)

    documents = [
        "heart document",
        "kidney document",
    ]

    store.add(
        embedder.encode(["heart disease", "kidney disease"]),
        documents
    )

    results = store.search(
        embedder.encode(["heart"])[0],
        top_k=1
    )

    assert len(results) >= 1
    assert results[0]["document"] == "heart document"