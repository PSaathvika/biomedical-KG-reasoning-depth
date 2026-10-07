import numpy as np

try:
    import faiss
except ImportError:
    faiss = None


class FAISSVectorStore:
    """
    FAISS-backed biomedical document store.

    Uses cosine similarity when embeddings are normalized.
    BioMedBERT embeddings produced by the production sweep
    are L2-normalized.
    """

    def __init__(self, dimension):
        self.dimension = int(dimension)

        self.documents = []

        self.vectors = np.empty(
            (0, self.dimension),
            dtype="float32",
        )

        if faiss is not None:
            self.index = faiss.IndexFlatIP(
                self.dimension
            )
        else:
            self.index = None

    def add(
        self,
        embeddings,
        documents,
    ):
        vectors = np.asarray(
            embeddings,
            dtype="float32",
        )

        if vectors.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2-D array."
            )

        if vectors.shape[1] != self.dimension:
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected {self.dimension}, "
                f"got {vectors.shape[1]}"
            )

        if len(vectors) != len(documents):
            raise ValueError(
                "Number of embeddings and documents "
                "must match."
            )

        if len(vectors) == 0:
            return

        self.documents.extend(
            str(document)
            for document in documents
        )

        if self.index is not None:

            self.index.add(vectors)

        else:

            self.vectors = np.vstack(
                [
                    self.vectors,
                    vectors,
                ]
            )

    def search(
        self,
        query_embedding,
        top_k=5,
    ):
        if not self.documents:
            return []

        query = np.asarray(
            query_embedding,
            dtype="float32",
        ).reshape(1, -1)

        if query.shape[1] != self.dimension:
            raise ValueError(
                "Query embedding dimension mismatch."
            )

        limit = min(
            int(top_k),
            len(self.documents),
        )

        if limit <= 0:
            return []

        if self.index is not None:

            scores, indices = (
                self.index.search(
                    query,
                    limit,
                )
            )

            scores = scores[0]
            indices = indices[0]

        else:

            scores_all = (
                self.vectors
                @ query[0]
            )

            indices = np.argsort(
                -scores_all
            )[:limit]

            scores = scores_all[
                indices
            ]

        results = []

        for score, index in zip(
            scores,
            indices,
        ):

            index = int(index)

            if index < 0:
                continue

            results.append(
                {
                    "document": self.documents[
                        index
                    ],
                    "score": float(score),
                }
            )

        return results

    def __len__(self):
        return len(self.documents)