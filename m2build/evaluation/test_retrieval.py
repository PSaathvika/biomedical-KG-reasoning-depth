import numpy as np

from src.shared.vector_store import FAISSVectorStore


class FakeEmbedder:
    def encode(self, texts):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype="float32")


def main():
    embedder = FakeEmbedder()
    store = FAISSVectorStore(2)
    store.add(embedder.encode(["heart disease", "kidney disease"]), ["heart document", "kidney document"])
    results = store.search(embedder.encode(["heart"])[0], top_k=1)
    assert results[0]["document"] == "heart document"
    print("Retrieval test passed.")


if __name__ == "__main__":
    main()
