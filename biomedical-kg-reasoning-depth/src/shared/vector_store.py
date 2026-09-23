import faiss
import numpy as np


class VectorStore:

    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.documents = []

    def add(self, embeddings, documents):
        embeddings = np.asarray(embeddings).astype("float32")
        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(self, query_embedding, top_k=5):
        query_embedding = np.asarray(query_embedding).astype("float32")
        query_embedding = query_embedding.reshape(1, -1)
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, index in zip(scores[0], indices[0]):
            if index != -1:
                results.append({
                    "document": self.documents[index],
                    "score": float(score)
                })
        return results
