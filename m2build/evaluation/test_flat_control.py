import numpy as np

from src.shared.vector_store import FAISSVectorStore
from src.flat_control import FlatControl


class FakeEmbedder:
    """
    Deterministic embedding model for smoke tests.
    """

    def encode(self, texts):
        return np.asarray(
            [[1.0, 0.0] for _ in texts],
            dtype="float32",
        )


class FakeLLM:
    """
    Fake LLM that records every reasoning call.

    This lets us verify the matched budget without loading
    BioMistral.
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

    documents = [
        (
            "Aspirin is related to platelet aggregation and "
            "cardiovascular prevention."
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
        embedder.encode(documents),
        documents,
    )

    llm = FakeLLM()

    return embedder, store, llm


def test_depth_one_has_one_retrieval_and_one_reasoning_step():
    embedder, store, llm = build_components()

    controller = FlatControl(
        embedder=embedder,
        vector_store=store,
        llm=llm,
    )

    result = controller.retrieve_and_reason(
        "What is aspirin related to?",
        depth=1,
    )

    assert result["depth"] == 1
    assert len(result["retrievals"]) == 1
    assert len(result["reasoning_steps"]) == 1
    assert len(llm.calls) == 1


def test_depth_two_has_two_retrievals_and_two_reasoning_steps():
    embedder, store, llm = build_components()

    controller = FlatControl(
        embedder=embedder,
        vector_store=store,
        llm=llm,
    )

    result = controller.retrieve_and_reason(
        "What is aspirin related to?",
        depth=2,
    )

    assert result["depth"] == 2
    assert len(result["retrievals"]) == 2
    assert len(result["reasoning_steps"]) == 2
    assert len(llm.calls) == 2


def test_depth_three_has_three_retrievals_and_three_reasoning_steps():
    embedder, store, llm = build_components()

    controller = FlatControl(
        embedder=embedder,
        vector_store=store,
        llm=llm,
    )

    result = controller.retrieve_and_reason(
        "What is aspirin related to?",
        depth=3,
    )

    assert result["depth"] == 3
    assert len(result["retrievals"]) == 3
    assert len(result["reasoning_steps"]) == 3
    assert len(llm.calls) == 3


def test_each_step_uses_previous_reasoning_for_next_query():
    embedder, store, llm = build_components()

    controller = FlatControl(
        embedder=embedder,
        vector_store=store,
        llm=llm,
    )

    result = controller.retrieve_and_reason(
        "What is aspirin related to?",
        depth=3,
    )

    first_query = result["retrievals"][0]["query"]
    second_query = result["retrievals"][1]["query"]
    third_query = result["retrievals"][2]["query"]

    assert first_query == "What is aspirin related to?"

    assert "Previous reasoning:" in second_query
    assert "Reasoning step 1" in second_query

    assert "Previous reasoning:" in third_query
    assert "Reasoning step 2" in third_query


def test_flat_control_has_no_graph_structure():
    embedder, store, llm = build_components()

    controller = FlatControl(
        embedder=embedder,
        vector_store=store,
        llm=llm,
    )

    assert not hasattr(controller, "kg")
    assert not hasattr(controller, "knowledge_graph")
    assert not hasattr(controller, "hypergraph")


if __name__ == "__main__":
    test_depth_one_has_one_retrieval_and_one_reasoning_step()
    test_depth_two_has_two_retrievals_and_two_reasoning_steps()
    test_depth_three_has_three_retrievals_and_three_reasoning_steps()
    test_each_step_uses_previous_reasoning_for_next_query()
    test_flat_control_has_no_graph_structure()

    print(
        "Milestone 3 flat-control tests passed "
        "for depths 1, 2 and 3."
    )