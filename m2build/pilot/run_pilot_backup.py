import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import csv
import json
import time

import numpy as np

from src.shared.knowledge_graph import KnowledgeGraph
from src.shared.vector_store import FAISSVectorStore
from src.biomedkai import BiomedKAI
from src.kragen import KRAGEN
from src.hypergraph_rag import HyperGraphRAG
from src.flat_control import FlatControl


RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "milestone4_pilot_results.csv"
QUESTION_FILE = RESULTS_DIR / "milestone4_frozen_questions.json"

DEPTHS = (1, 2, 3)

METHODS = (
    "BiomedKAI",
    "KRAGEN",
    "HyperGraphRAG",
    "FlatControl",
)

QUESTIONS_PER_DATASET = 20


class FakeEmbedder:
    """
    Lightweight deterministic embedder for infrastructure validation.

    This keeps the pilot independent of the large biomedical embedding
    model while verifying the complete experiment controller.
    """

    def encode(self, texts):
        vectors = []

        for text in texts:
            text = text.lower()

            vector = [
                float("aspirin" in text),
                float("platelet" in text),
                float("cardiovascular" in text),
                float("therapy" in text),
                float("cancer" in text),
                float("gene" in text),
                float("protein" in text),
                float("drug" in text),
            ]

            if not any(vector):
                vector[0] = 1.0

            vectors.append(vector)

        return np.asarray(vectors, dtype="float32")


class FakeLLM:
    """
    Deterministic LLM replacement.

    Every call is recorded so the depth budget can be verified.
    """

    def __init__(self):
        self.calls = []

    def generate(self, prompt, max_new_tokens=128):
        self.calls.append(prompt)

        return (
            f"Pilot reasoning step {len(self.calls)}: "
            "the retrieved biomedical evidence is relevant."
        )


def load_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def select_medmcqa_questions():
    """
    Select a deterministic set of 20 MedMCQA questions.

    The source file is already in a fixed order, so selecting the first
    20 records makes the selection reproducible.
    """

    path = ROOT / "data" / "medmcqa" / "dev.json"

    rows = load_json(path)

    if len(rows) < QUESTIONS_PER_DATASET:
        raise ValueError(
            f"MedMCQA contains only {len(rows)} questions."
        )

    selected = []

    for row in rows[:QUESTIONS_PER_DATASET]:
        selected.append(
            {
                "id": str(row["id"]),
                "question": row["question"],
                "options": [
                    row["opa"],
                    row["opb"],
                    row["opc"],
                    row["opd"],
                ],
                "correct_choice": row["cop"],
                "subject_name": row.get("subject_name"),
                "topic_name": row.get("topic_name"),
            }
        )

    return selected


def select_pubmedqa_questions():
    """
    Select a deterministic set of 20 PubMedQA questions.

    The source file is already in a fixed order, so selecting the first
    20 records makes the selection reproducible.
    """

    path = ROOT / "data" / "pubmedqa" / "pqa_labeled.json"

    rows = load_json(path)

    if len(rows) < QUESTIONS_PER_DATASET:
        raise ValueError(
            f"PubMedQA contains only {len(rows)} questions."
        )

    selected = []

    for row in rows[:QUESTIONS_PER_DATASET]:
        selected.append(
            {
                "id": str(row["pubid"]),
                "question": row["question"],
                "context": row["context"],
                "long_answer": row["long_answer"],
                "final_decision": row["final_decision"],
            }
        )

    return selected


def freeze_questions():
    """
    Freeze the exact 20-question subset from each dataset.

    The frozen file is written once and reused on subsequent runs.
    """

    if QUESTION_FILE.exists():
        print(f"Using existing frozen questions: {QUESTION_FILE}")
        return load_json(QUESTION_FILE)

    frozen = {
        "MedMCQA": select_medmcqa_questions(),
        "PubMedQA": select_pubmedqa_questions(),
    }

    with QUESTION_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            frozen,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Frozen questions written to: {QUESTION_FILE}")

    return frozen


def build_knowledge_graph():
    return KnowledgeGraph(
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
            (
                "cancer",
                "related_to",
                "protein",
            ),
            (
                "protein",
                "associated_with",
                "gene",
            ),
            (
                "drug",
                "used_for",
                "therapy",
            ),
        ]
    )


def build_documents():
    return [
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
        (
            "Cancer biology involves proteins and genes "
            "that can influence disease mechanisms."
        ),
        (
            "Proteins are important biological molecules "
            "associated with many cellular processes."
        ),
        (
            "Drug therapy can be used to treat a variety "
            "of biomedical conditions."
        ),
    ]


def build_components():
    embedder = FakeEmbedder()

    documents = build_documents()

    store = FAISSVectorStore(
        dimension=8,
    )

    store.add(
        embedder.encode(documents),
        documents,
    )

    kg = build_knowledge_graph()

    return embedder, store, kg


def build_method(
    method_name,
    embedder,
    store,
    kg,
    llm,
):
    if method_name == "BiomedKAI":
        return BiomedKAI(
            embedder,
            store,
            kg,
            llm=llm,
        )

    if method_name == "KRAGEN":
        return KRAGEN(
            embedder,
            store,
            llm=llm,
        )

    if method_name == "HyperGraphRAG":
        return HyperGraphRAG(
            embedder,
            store,
            llm=llm,
        )

    if method_name == "FlatControl":
        return FlatControl(
            embedder,
            store,
            llm=llm,
        )

    raise ValueError(
        f"Unknown method: {method_name}"
    )


def make_experiment_question(item, dataset_name):
    """
    Convert a frozen dataset record into the question string used by
    the reasoning methods.

    Context/options are included where available so the experiment
    receives the actual biomedical question information.
    """

    if dataset_name == "MedMCQA":
        options = "\n".join(
            [
                f"A. {item['options'][0]}",
                f"B. {item['options'][1]}",
                f"C. {item['options'][2]}",
                f"D. {item['options'][3]}",
            ]
        )

        return (
            f"{item['question']}\n\n"
            f"Options:\n{options}"
        )

    if dataset_name == "PubMedQA":
        contexts = item["context"].get(
            "contexts",
            [],
        )

        context_text = "\n".join(contexts)

        return (
            f"Question: {item['question']}\n\n"
            f"Context:\n{context_text}"
        )

    raise ValueError(
        f"Unknown dataset: {dataset_name}"
    )


def count_retrieval_calls(result):
    """
    Count retrieval operations from the standardized result structures.
    """

    if "retrieved" in result:
        return len(result["retrieved"])

    if "retrievals" in result:
        return len(result["retrievals"])

    if "thoughts" in result:
        return len(result["thoughts"])

    return 0


def count_reasoning_calls(result):
    return len(
        result.get(
            "reasoning_steps",
            [],
        )
    )


def run_pilot(frozen_questions):
    rows = []

    for dataset_name, questions in frozen_questions.items():

        for item in questions:

            question_id = item["id"]

            question = make_experiment_question(
                item,
                dataset_name,
            )

            for method_name in METHODS:

                for depth in DEPTHS:

                    embedder, store, kg = build_components()

                    llm = FakeLLM()

                    method = build_method(
                        method_name=method_name,
                        embedder=embedder,
                        store=store,
                        kg=kg,
                        llm=llm,
                    )

                    start = time.perf_counter()

                    result = method.retrieve_and_reason(
                        question=question,
                        depth=depth,
                        top_k=3,
                    )

                    elapsed = (
                        time.perf_counter()
                        - start
                    )

                    retrieval_calls = count_retrieval_calls(
                        result
                    )

                    reasoning_calls = count_reasoning_calls(
                        result
                    )

                    rows.append(
                        {
                            "dataset": dataset_name,
                            "question_id": question_id,
                            "question": item["question"],
                            "method": method_name,
                            "depth": depth,
                            "retrieval_calls": retrieval_calls,
                            "llm_calls": len(llm.calls),
                            "reasoning_steps": reasoning_calls,
                            "runtime_seconds": round(
                                elapsed,
                                6,
                            ),
                            "answer": result.get(
                                "answer"
                            ),
                        }
                    )

                    print(
                        f"[PASS] "
                        f"{dataset_name} | "
                        f"{question_id} | "
                        f"{method_name} | "
                        f"depth={depth} | "
                        f"retrievals={retrieval_calls} | "
                        f"llm={len(llm.calls)}"
                    )

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        fieldnames = [
            "dataset",
            "question_id",
            "question",
            "method",
            "depth",
            "retrieval_calls",
            "llm_calls",
            "reasoning_steps",
            "runtime_seconds",
            "answer",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    return rows


def validate_results(rows, frozen_questions):
    expected_rows = (
        QUESTIONS_PER_DATASET
        * 2
        * len(METHODS)
        * len(DEPTHS)
    )

    assert len(rows) == expected_rows, (
        f"Expected {expected_rows} rows, "
        f"got {len(rows)}"
    )

    for row in rows:

        expected_depth = int(row["depth"])

        assert int(row["retrieval_calls"]) == expected_depth, (
            f"Retrieval budget mismatch: {row}"
        )

        assert int(row["llm_calls"]) == expected_depth, (
            f"LLM budget mismatch: {row}"
        )

        assert int(row["reasoning_steps"]) == expected_depth, (
            f"Reasoning-step mismatch: {row}"
        )

    combinations = {
        (
            row["dataset"],
            row["question_id"],
            row["method"],
            int(row["depth"]),
        )
        for row in rows
    }

    expected_combinations = {
        (
            dataset,
            question["id"],
            method,
            depth,
        )
        for dataset, questions in frozen_questions.items()
        for question in questions
        for method in METHODS
        for depth in DEPTHS
    }

    assert combinations == expected_combinations, (
        "Not all frozen-question/method/depth combinations were executed."
    )


def main():
    print("=" * 70)
    print("Milestone 4 Real-Question Infrastructure Pilot")
    print("=" * 70)

    frozen_questions = freeze_questions()

    print()
    print(
        f"Frozen MedMCQA questions: "
        f"{len(frozen_questions['MedMCQA'])}"
    )

    print(
        f"Frozen PubMedQA questions: "
        f"{len(frozen_questions['PubMedQA'])}"
    )

    rows = run_pilot(
        frozen_questions
    )

    validate_results(
        rows,
        frozen_questions
    )

    print()
    print("=" * 70)
    print("REAL-QUESTION PILOT PASSED")
    print("=" * 70)

    print(
        f"Rows written: {len(rows)}"
    )

    print(
        f"Frozen questions: {QUESTION_FILE}"
    )

    print(
        f"Results file: {OUTPUT_FILE}"
    )

    print()
    print(
        "Verified:"
    )

    print(
        "  20 MedMCQA questions"
    )

    print(
        "  20 PubMedQA questions"
    )

    print(
        "  4 methods"
    )

    print(
        "  depths 1, 2, and 3"
    )

    print(
        "  depth d -> exactly d retrieval calls"
    )

    print(
        "  depth d -> exactly d LLM calls"
    )

    print(
        "  depth d -> exactly d reasoning steps"
    )

    print(
        "  total expected rows -> 480"
    )


if __name__ == "__main__":
    main()