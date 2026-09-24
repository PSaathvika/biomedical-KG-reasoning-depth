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
            dtype="float32",
        )


class FakeLLM:
    """
    Records every LLM call so the depth budget can be verified.
    """

    def __init__(self):
        self.calls = []

    def generate(self, prompt, max_new_tokens=128):
        self.calls.append(prompt)

        return (
            f"Reasoning step {len(self.calls)}: "
            "the retrieved biomedical evidence is relevant."
        )


def build_components():
    embedder = FakeEmbedder()

    store = FAISSVectorStore(2)

    docs = [
        (
            "Aspirin is related to platelet aggregation "
            "and cardiovascular prevention."
        ),
        (
            "Platelet aggregation is affected by "
            "cyclooxygenase inhibition."
        ),
        (
            "Cardiovascular prevention can involve "
            "antiplatelet therapy."
        ),
    ]

    store.add(
        embedder.encode(docs),
        docs,
    )

    kg = KnowledgeGraph(
        [
            (
                "aspirin",
                "affects",
                "platelet aggregation",
            ),
            (
                "platelet aggregation",
                "related_to",
                "cardiovascular prevention",
            ),
            (
                "cardiovascular prevention",
                "uses",
                "antiplatelet therapy",
            ),
        ]
    )

    return embedder, store, kg


def test_biomedkai_llm_budget():
    q = "What is aspirin related to?"

    for depth in (1, 2, 3):
        embedder, store, kg = build_components()
        llm = FakeLLM()

        method = BiomedKAI(
            embedder,
            store,
            kg,
            llm=llm,
        )

        result = method.retrieve_and_reason(
            q,
            depth=depth,
        )

        assert result["depth"] == depth
        assert len(result["reasoning_steps"]) == depth
        assert len(llm.calls) == depth


def test_kragen_llm_budget():
    q = "What is aspirin related to?"

    for depth in (1, 2, 3):
        embedder, store, kg = build_components()
        llm = FakeLLM()

        method = KRAGEN(
            embedder,
            store,
            llm=llm,
        )

        result = method.retrieve_and_reason(
            q,
            depth=depth,
        )

        assert result["depth"] == depth
        assert len(result["reasoning_steps"]) == depth
        assert len(llm.calls) == depth


def test_hypergraph_rag_llm_budget():
    q = "What is aspirin related to?"

    for depth in (1, 2, 3):
        embedder, store, kg = build_components()
        llm = FakeLLM()

        method = HyperGraphRAG(
            embedder,
            store,
            llm=llm,
        )

        result = method.retrieve_and_reason(
            q,
            depth=depth,
        )

        assert result["depth"] == depth
        assert len(result["reasoning_steps"]) == depth
        assert len(llm.calls) == depth


def test_all_methods_have_depth_matched_llm_budget():
    """
    Verify the central Milestone 3 requirement:

        depth d -> exactly d LLM reasoning calls
    """

    q = "What is aspirin related to?"

    for depth in (1, 2, 3):

        embedder, store, kg = build_components()
        biomed_llm = FakeLLM()

        BiomedKAI(
            embedder,
            store,
            kg,
            llm=biomed_llm,
        ).retrieve_and_reason(
            q,
            depth=depth,
        )

        assert len(biomed_llm.calls) == depth

        embedder, store, kg = build_components()
        kragen_llm = FakeLLM()

        KRAGEN(
            embedder,
            store,
            llm=kragen_llm,
        ).retrieve_and_reason(
            q,
            depth=depth,
        )

        assert len(kragen_llm.calls) == depth

        embedder, store, kg = build_components()
        hyper_llm = FakeLLM()

        HyperGraphRAG(
            embedder,
            store,
            llm=hyper_llm,
        ).retrieve_and_reason(
            q,
            depth=depth,
        )

        assert len(hyper_llm.calls) == depth


def main():
    test_biomedkai_llm_budget()
    test_kragen_llm_budget()
    test_hypergraph_rag_llm_budget()
    test_all_methods_have_depth_matched_llm_budget()

    print(
        "Milestone 3 iterative graph reasoning tests passed "
        "for BiomedKAI, KRAGEN, and HyperGraphRAG at depths 1, 2, and 3."
    )


if __name__ == "__main__":
    main()