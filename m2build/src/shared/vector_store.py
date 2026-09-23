import numpy as np

try:
    import faiss
except ImportError:
    faiss = None


class FAISSVectorStore:
    """FAISS-backed store with a NumPy fallback for local smoke tests."""

    def __init__(self, dimension):
        self.dimension = dimension
        self.documents = []
        self.vectors = np.empty((0, dimension), dtype="float32")
        self.index = faiss.IndexFlatIP(dimension) if faiss is not None else None

    def add(self, embeddings, documents):
        vectors = np.asarray(embeddings, dtype="float32")
        self.documents.extend(documents)
        if self.index is not None:
            self.index.add(vectors)
        else:
            self.vectors = np.vstack([self.vectors, vectors])

    def search(self, query_embedding, top_k=5):
        query = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        limit = min(top_k, len(self.documents))
        if limit == 0:
            return []

        if self.index is not None:
            scores, indices = self.index.search(query, limit)
            scores, indices = scores[0], indices[0]
        else:
            scores_all = self.vectors @ query[0]
            indices = np.argsort(-scores_all)[:limit]
            scores = scores_all[indices]

        return [
            {"document": self.documents[int(idx)], "score": float(score)}
            for score, idx in zip(scores, indices)
            if idx >= 0
        ]
