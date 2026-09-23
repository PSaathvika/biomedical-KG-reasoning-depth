import numpy as np

from src.shared.knowledge_graph import KnowledgeGraph
from src.shared.vector_store import FAISSVectorStore
from src.biomedkai import BiomedKAI
from src.kragen import KRAGEN
from src.hypergraph_rag import HyperGraphRAG


class FakeEmbedder:
    def encode(self, texts):
        return np.asarray(
            [[1.0, 0.0] for _ in texts],
            dtype="float32"
        )


def build_components():
    embedder = FakeEmbedder()
    store = FAISSVectorStore(2)

    docs = [
        "Aspirin is related to platelet aggregation and cardiovascular prevention.",
        "Platelet aggregation is affected by cyclooxygenase inhibition.",
        "Cardiovascular prevention can involve antiplatelet therapy.",
    ]

    store.add(embedder.encode(docs), docs)

    kg = KnowledgeGraph([
        ("aspirin", "affects", "platelet aggregation"),
        ("platelet aggregation", "related_to", "cardiovascular prevention"),
        ("cardiovascular prevention", "uses", "antiplatelet therapy"),
    ])

    return embedder, store, kg


def test_methods_support_multiple_reasoning_depths():
    embedder, store, kg = build_components()

    q = "What is aspirin related to?"

    biomedkai = BiomedKAI(embedder, store, kg)
    kragen = KRAGEN(embedder, store)
    hyper = HyperGraphRAG(embedder, store)

    for depth in (1, 2, 3):
        biomedkai_result = biomedkai.retrieve_and_reason(
            q, depth=depth
        )
        kragen_result = kragen.retrieve_and_reason(
            q, depth=depth
        )
        hyper_result = hyper.retrieve_and_reason(
            q, depth=depth
        )

        assert biomedkai_result["depth"] == depth
        assert kragen_result["depth"] == depth
        assert hyper_result["depth"] == depth