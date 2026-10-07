import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import csv
import json
import re
import time

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)

from src.shared.knowledge_graph import KnowledgeGraph
from src.shared.vector_store import FAISSVectorStore
from src.biomedkai import BiomedKAI
from src.kragen import KRAGEN
from src.hypergraph_rag import HyperGraphRAG
from src.flat_control import FlatControl


# ============================================================
# PATHS / EXPERIMENT SETTINGS
# ============================================================

RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = RESULTS_DIR / "milestone4_pilot_results.csv"
QUESTION_FILE = RESULTS_DIR / "milestone4_frozen_questions.json"

MEDMCQA_FILE = ROOT / "data" / "medmcqa" / "dev.json"
PUBMEDQA_FILE = ROOT / "data" / "pubmedqa" / "pqa_labeled.json"
PRIMEKG_FILE = ROOT / "data" / "kg" / "primekg.csv"

DEPTHS = (1, 2, 3)

METHODS = (
    "BiomedKAI",
    "KRAGEN",
    "HyperGraphRAG",
    "FlatControl",
)

QUESTIONS_PER_DATASET = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BIOMEDBERT_NAME = (
    "microsoft/"
    "BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
)

BIOMISTRAL_NAME = "BioMistral/BioMistral-7B"

# Batch size for BioMedBERT encoding.
# Keeps memory use much lower than encoding the entire corpus at once.
EMBED_BATCH_SIZE = 16


# ============================================================
# REAL BIOMEDBERT EMBEDDER
# ============================================================

class BioMedBERTEmbedder:
    """
    Real BioMedBERT encoder used for FAISS retrieval.

    Mean-pools the last hidden state and L2-normalizes the
    resulting vectors.
    """

    def __init__(self, model_name=BIOMEDBERT_NAME):
        print()
        print("=" * 70)
        print("Loading BioMedBERT")
        print("=" * 70)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModel.from_pretrained(
            model_name
        ).to(DEVICE)

        self.model.eval()

        self.dimension = int(
            self.model.config.hidden_size
        )

        print(f"BioMedBERT dimension: {self.dimension}")
        print(f"BioMedBERT device: {DEVICE}")

    @torch.no_grad()
    def encode(self, texts, batch_size=EMBED_BATCH_SIZE):
        """
        Encode texts in batches.

        Returns:
            numpy float32 array with shape:
            [number_of_texts, embedding_dimension]
        """

        if not texts:
            return np.empty(
                (0, self.dimension),
                dtype="float32",
            )

        all_embeddings = []

        for start in range(0, len(texts), batch_size):
            batch = texts[
                start:start + batch_size
            ]

            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )

            encoded = {
                key: value.to(DEVICE)
                for key, value in encoded.items()
            }

            outputs = self.model(**encoded)

            hidden = outputs.last_hidden_state

            mask = encoded[
                "attention_mask"
            ].unsqueeze(-1)

            masked_hidden = hidden * mask

            summed = masked_hidden.sum(
                dim=1
            )

            counts = mask.sum(
                dim=1
            ).clamp(min=1)

            embeddings = summed / counts

            embeddings = torch.nn.functional.normalize(
                embeddings,
                p=2,
                dim=1,
            )

            all_embeddings.append(
                embeddings.cpu().numpy().astype(
                    "float32"
                )
            )

        return np.concatenate(
            all_embeddings,
            axis=0,
        )


# ============================================================
# REAL BIOMISTRAL LLM
# ============================================================

class BioMistralLLM:
    """
    Real BioMistral-7B generation wrapper.

    Every generation is recorded so the experiment can report:
      - LLM calls
      - input tokens
      - output tokens
      - total tokens
    """

    def __init__(self, model_name=BIOMISTRAL_NAME):
        print()
        print("=" * 70)
        print("Loading BioMistral-7B")
        print("=" * 70)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        if DEVICE == "cuda":
            model_dtype = torch.float16
            device_map = "auto"
        else:
            model_dtype = torch.float32
            device_map = None

        model_kwargs = {
            "torch_dtype": model_dtype,
        }

        if device_map is not None:
            model_kwargs["device_map"] = device_map

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs,
        )

        if device_map is None:
            self.model = self.model.to(DEVICE)

        self.model.eval()

        self.calls = []

        print("BioMistral loaded")
        print(
            "Input device:",
            next(self.model.parameters()).device,
        )

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=64):
        """
        One LLM call per reasoning step.

        For the final step, score the permitted answer labels in one
        batched forward pass and return the highest-scoring label.
        This avoids relying on instruction-following from the base
        BioMistral model.
        """

        is_final = "FINAL_ANSWER:" in prompt

        if is_final:

            if (
                "Dataset: MedMCQA" in prompt
                or "Options:" in prompt
            ):
                candidates = [
                    "A",
                    "B",
                    "C",
                    "D",
                ]
            else:
                candidates = [
                    "yes",
                    "no",
                    "maybe",
                ]

            scoring_prompt = (
                prompt + "\nFINAL_ANSWER:"
            )

            texts = [
                scoring_prompt + " " + candidate
                for candidate in candidates
            ]

            encoded = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=4096,
            )

            model_device = next(
                self.model.parameters()
            ).device

            encoded = {
                key: value.to(model_device)
                for key, value in encoded.items()
            }

            outputs = self.model(**encoded)

            logits = outputs.logits

            attention = encoded[
                "attention_mask"
            ]

            scores = []

            for i, candidate in enumerate(
                candidates
            ):

                length = int(
                    attention[i].sum().item()
                )

                candidate_ids = self.tokenizer(
                    " " + candidate,
                    add_special_tokens=False,
                )["input_ids"]

                n = len(candidate_ids)

                start_pos = length - n

                logprob = 0.0

                for j, token_id in enumerate(
                    candidate_ids
                ):

                    pos = start_pos + j

                    if pos <= 0:
                        continue

                    next_logits = logits[
                        i,
                        pos - 1,
                    ]

                    log_probs = torch.log_softmax(
                        next_logits,
                        dim=-1,
                    )

                    logprob += float(
                        log_probs[
                            int(token_id)
                        ].item()
                    )

                scores.append(logprob)

            best_index = int(
                np.argmax(scores)
            )

            answer = (
                "FINAL_ANSWER: "
                + candidates[best_index]
            )

            input_tokens = int(
                attention.sum(
                    dim=1
                ).max().item()
            )

            self.calls.append(
                {
                    "input_tokens": input_tokens,
                    "output_tokens": 2,
                    "total_tokens": (
                        input_tokens + 2
                    ),
                }
            )

            return answer

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        )

        model_device = next(
            self.model.parameters()
        ).device

        input_ids = encoded[
            "input_ids"
        ].to(model_device)

        attention_mask = encoded[
            "attention_mask"
        ].to(model_device)

        input_tokens = int(
            input_ids.shape[1]
        )

        output_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=8,
            do_sample=False,
            pad_token_id=(
                self.tokenizer.eos_token_id
            ),
        )

        generated_ids = output_ids[
            0,
            input_tokens:,
        ]

        output_tokens = int(
            generated_ids.shape[0]
        )

        answer = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        ).strip()

        total_tokens = (
            input_tokens
            + output_tokens
        )

        self.calls.append(
            {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
            }
        )

        return answer


# ============================================================
# DATA LOADING
# ============================================================

def load_json(path):
    with path.open(
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def select_medmcqa_questions():
    path = MEDMCQA_FILE

    if not path.exists():
        raise FileNotFoundError(
            f"MedMCQA file not found: {path}"
        )

    rows = load_json(path)

    if len(rows) < QUESTIONS_PER_DATASET:
        raise ValueError(
            f"MedMCQA contains only "
            f"{len(rows)} questions."
        )

    selected = []

    for row in rows[
        :QUESTIONS_PER_DATASET
    ]:

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
                "subject_name": row.get(
                    "subject_name"
                ),
                "topic_name": row.get(
                    "topic_name"
                ),
            }
        )

    return selected


def select_pubmedqa_questions():
    path = PUBMEDQA_FILE

    if not path.exists():
        raise FileNotFoundError(
            f"PubMedQA file not found: {path}"
        )

    rows = load_json(path)

    if len(rows) < QUESTIONS_PER_DATASET:
        raise ValueError(
            f"PubMedQA contains only "
            f"{len(rows)} questions."
        )

    selected = []

    for row in rows[
        :QUESTIONS_PER_DATASET
    ]:

        selected.append(
            {
                "id": str(row["pubid"]),
                "question": row["question"],
                "context": row["context"],
                "long_answer": row["long_answer"],
                "final_decision": row[
                    "final_decision"
                ],
            }
        )

    return selected


def freeze_questions():
    if QUESTION_FILE.exists():

        print(
            f"Using existing frozen questions: "
            f"{QUESTION_FILE}"
        )

        return load_json(
            QUESTION_FILE
        )

    frozen = {
        "MedMCQA": (
            select_medmcqa_questions()
        ),
        "PubMedQA": (
            select_pubmedqa_questions()
        ),
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

    print(
        f"Frozen questions written to: "
        f"{QUESTION_FILE}"
    )

    return frozen


# ============================================================
# KNOWLEDGE GRAPH
# ============================================================

def build_knowledge_graph():
    """
    Load the real PrimeKG once for the entire pilot.
    """

    if not PRIMEKG_FILE.exists():
        raise FileNotFoundError(
            f"PrimeKG file not found: "
            f"{PRIMEKG_FILE}"
        )

    print()
    print("=" * 70)
    print("Loading REAL PrimeKG")
    print("=" * 70)
    print(
        f"PrimeKG path: {PRIMEKG_FILE}"
    )

    kg = KnowledgeGraph(
        csv_path=PRIMEKG_FILE
    )

    print(
        f"PrimeKG edges loaded: "
        f"{len(kg.edges):,}"
    )

    print(
        f"PrimeKG nodes loaded: "
        f"{len(kg.nodes):,}"
    )

    print("=" * 70)

    return kg


# ============================================================
# REAL BIOMEDICAL RETRIEVAL CORPUS
# ============================================================

def extract_pubmed_contexts(context):
    """
    Extract PubMedQA context passages regardless of whether
    the stored context is represented as a dictionary or list.
    """

    if isinstance(context, dict):

        contexts = context.get(
            "contexts",
            [],
        )

        if isinstance(contexts, list):
            return [
                str(x).strip()
                for x in contexts
                if str(x).strip()
            ]

        if isinstance(contexts, str):
            text = contexts.strip()

            return [text] if text else []

    if isinstance(context, list):

        return [
            str(x).strip()
            for x in context
            if str(x).strip()
        ]

    if isinstance(context, str):

        text = context.strip()

        return [text] if text else []

    return []


def build_documents(frozen_questions):
    """
    Build the retrieval corpus from real PubMedQA biomedical
    context passages.

    The frozen evaluation PubMedQA question IDs are excluded
    from the retrieval corpus so their evaluation contexts are
    not directly indexed.
    """

    if not PUBMEDQA_FILE.exists():
        raise FileNotFoundError(
            f"PubMedQA file not found: "
            f"{PUBMEDQA_FILE}"
        )

    rows = load_json(
        PUBMEDQA_FILE
    )

    frozen_pubmed_ids = {
        str(item["id"])
        for item in frozen_questions[
            "PubMedQA"
        ]
    }

    documents = []
    seen = set()

    for row in rows:

        pubid = str(
            row.get("pubid", "")
        )

        if pubid in frozen_pubmed_ids:
            continue

        contexts = extract_pubmed_contexts(
            row.get("context")
        )

        for context_text in contexts:

            normalized = re.sub(
                r"\s+",
                " ",
                context_text,
            ).strip()

            if not normalized:
                continue

            if normalized in seen:
                continue

            seen.add(normalized)
            documents.append(normalized)

    if not documents:
        raise ValueError(
            "No real PubMedQA context documents "
            "were found for the retrieval corpus."
        )

    print()
    print("=" * 70)
    print("Building REAL biomedical retrieval corpus")
    print("=" * 70)
    print(
        "Source: PubMedQA labeled contexts"
    )
    print(
        "Frozen evaluation PubMedQA contexts excluded: "
        f"{len(frozen_pubmed_ids)} question IDs"
    )
    print(
        f"Unique biomedical documents: "
        f"{len(documents):,}"
    )
    print("=" * 70)

    return documents


# ============================================================
# COMPONENT CONSTRUCTION
# ============================================================

def build_components(
    embedder,
    kg,
    frozen_questions,
):
    """
    Build the shared retrieval components once.

    The FAISS store and PrimeKG are read-only during the pilot,
    so the same components are reused across all 480 cells.
    """

    documents = build_documents(
        frozen_questions
    )

    print()
    print("=" * 70)
    print("Encoding biomedical retrieval corpus")
    print("=" * 70)

    document_embeddings = embedder.encode(
        documents,
        batch_size=EMBED_BATCH_SIZE,
    )

    print(
        f"Embeddings shape: "
        f"{document_embeddings.shape}"
    )

    store = FAISSVectorStore(
        dimension=embedder.dimension
    )

    store.add(
        document_embeddings,
        documents,
    )

    print(
        f"FAISS documents indexed: "
        f"{len(documents):,}"
    )

    print("=" * 70)

    return store, kg


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


# ============================================================
# QUESTION FORMATTING
# ============================================================

def make_experiment_question(
    item,
    dataset_name,
):
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
            "Dataset: MedMCQA\n"
            f"{item['question']}\n\n"
            f"Options:\n{options}\n\n"
            "Answer:"
        )

    if dataset_name == "PubMedQA":

        contexts = item[
            "context"
        ].get(
            "contexts",
            [],
        )

        context_text = "\n".join(
            contexts
        )

        return (
            "Dataset: PubMedQA\n"
            f"Question: {item['question']}\n\n"
            f"Context:\n{context_text}\n\n"
            "Answer with exactly one of: "
            "yes, no, or maybe. "
            "Then briefly explain why."
        )

    raise ValueError(
        f"Unknown dataset: {dataset_name}"
    )


# ============================================================
# RESULT COUNTERS
# ============================================================

def count_retrieval_calls(result):

    if "retrieved" in result:

        return len(
            result["retrieved"]
        )

    if "retrievals" in result:

        return len(
            result["retrievals"]
        )

    if "thoughts" in result:

        return len(
            result["thoughts"]
        )

    return 0


def count_reasoning_calls(result):

    return len(
        result.get(
            "reasoning_steps",
            [],
        )
    )


# ============================================================
# ANSWER EXTRACTION
# ============================================================

def extract_medmcqa_answer(text):

    if not text:
        return None

    text = text.strip().upper()

    match = re.search(
        r"FINAL_ANSWER\s*:\s*([ABCD])\b",
        text,
    )

    if match:
        return match.group(1)

    match = re.search(
        r"^\s*([ABCD])\s*[\.\):\-]",
        text,
    )

    if match:
        return match.group(1)

    match = re.search(
        r"FINAL_ANSWER\s*:\s*([1-4])\b",
        text,
    )

    if match:

        return "ABCD"[
            int(match.group(1)) - 1
        ]

    return None


def extract_pubmedqa_answer(text):

    if not text:
        return None

    lower = text.lower()

    match = re.search(
        r"final_answer\s*:\s*(yes|no|maybe)\b",
        lower,
    )

    if match:
        return match.group(1)

    match = re.search(
        r"^\s*(yes|no|maybe)\b",
        lower,
    )

    if match:
        return match.group(1)

    match = re.search(
        r"\b(yes|no|maybe)\b",
        lower,
    )

    if match:
        return match.group(1)

    return None


def score_answer(
    dataset_name,
    item,
    answer,
):

    if dataset_name == "MedMCQA":

        predicted = extract_medmcqa_answer(
            answer
        )

        correct_raw = str(
            item["correct_choice"]
        ).strip()

        if correct_raw in {
            "1",
            "2",
            "3",
            "4",
        }:

            correct = "ABCD"[
                int(correct_raw) - 1
            ]

        else:

            correct = correct_raw.upper()

        return (
            predicted,
            int(
                predicted is not None
                and predicted == correct
            ),
        )

    if dataset_name == "PubMedQA":

        predicted = extract_pubmedqa_answer(
            answer
        )

        correct = str(
            item["final_decision"]
        ).strip().lower()

        return (
            predicted,
            int(
                predicted is not None
                and predicted == correct
            ),
        )

    return None, 0


# ============================================================
# REAL-MODEL PILOT
# ============================================================

def run_pilot(
    frozen_questions,
    embedder,
    llm,
    store,
    kg,
):
    """
    Execute all frozen question × method × depth combinations.

    The real PrimeKG and biomedical FAISS store are shared across
    all experiments. LLM call accounting is reset for every cell.
    """

    rows = []

    for dataset_name, questions in (
        frozen_questions.items()
    ):

        for item in questions:

            question_id = item["id"]

            question = make_experiment_question(
                item,
                dataset_name,
            )

            for method_name in METHODS:

                for depth in DEPTHS:

                    print()
                    print(
                        f"RUNNING | "
                        f"{dataset_name} | "
                        f"{question_id} | "
                        f"{method_name} | "
                        f"depth={depth}"
                    )

                    # The real PrimeKG and FAISS retrieval
                    # store were constructed once in main().
                    #
                    # They are reused here rather than rebuilding
                    # a 1 GB+ KG or re-encoding the corpus 480 times.

                    llm.calls = []

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

                    retrieval_calls = (
                        count_retrieval_calls(
                            result
                        )
                    )

                    reasoning_calls = (
                        count_reasoning_calls(
                            result
                        )
                    )

                    llm_calls = len(
                        llm.calls
                    )

                    input_tokens = sum(
                        call["input_tokens"]
                        for call in llm.calls
                    )

                    output_tokens = sum(
                        call["output_tokens"]
                        for call in llm.calls
                    )

                    total_tokens = sum(
                        call["total_tokens"]
                        for call in llm.calls
                    )

                    answer = result.get(
                        "answer"
                    )

                    predicted, correct = (
                        score_answer(
                            dataset_name,
                            item,
                            answer,
                        )
                    )

                    row = {
                        "dataset": dataset_name,
                        "question_id": question_id,
                        "question": item[
                            "question"
                        ],
                        "method": method_name,
                        "depth": depth,
                        "retrieval_calls": (
                            retrieval_calls
                        ),
                        "llm_calls": llm_calls,
                        "reasoning_steps": (
                            reasoning_calls
                        ),
                        "input_tokens": (
                            input_tokens
                        ),
                        "output_tokens": (
                            output_tokens
                        ),
                        "total_tokens": (
                            total_tokens
                        ),
                        "runtime_seconds": round(
                            elapsed,
                            6,
                        ),
                        "predicted_answer": (
                            predicted
                        ),
                        "correct_answer": (
                            item[
                                "correct_choice"
                            ]
                            if dataset_name
                            == "MedMCQA"
                            else item[
                                "final_decision"
                            ]
                        ),
                        "correct": correct,
                        "answer": answer,
                    }

                    rows.append(row)

                    print(
                        f"[PASS] "
                        f"retrievals="
                        f"{retrieval_calls} | "
                        f"llm="
                        f"{llm_calls} | "
                        f"reasoning="
                        f"{reasoning_calls} | "
                        f"tokens="
                        f"{total_tokens} | "
                        f"correct="
                        f"{correct} | "
                        f"time="
                        f"{elapsed:.2f}s"
                    )

    fieldnames = [
        "dataset",
        "question_id",
        "question",
        "method",
        "depth",
        "retrieval_calls",
        "llm_calls",
        "reasoning_steps",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "runtime_seconds",
        "predicted_answer",
        "correct_answer",
        "correct",
        "answer",
    ]

    with OUTPUT_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    return rows


# ============================================================
# VALIDATION
# ============================================================

def validate_results(
    rows,
    frozen_questions,
):

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

        expected_depth = int(
            row["depth"]
        )

        assert int(
            row["retrieval_calls"]
        ) == expected_depth, (
            f"Retrieval budget mismatch: "
            f"{row}"
        )

        assert int(
            row["llm_calls"]
        ) == expected_depth, (
            f"LLM budget mismatch: "
            f"{row}"
        )

        assert int(
            row["reasoning_steps"]
        ) == expected_depth, (
            f"Reasoning-step budget mismatch: "
            f"{row}"
        )

    expected_combinations = {
        (
            dataset,
            question["id"],
            method,
            depth,
        )
        for dataset, questions
        in frozen_questions.items()
        for question in questions
        for method in METHODS
        for depth in DEPTHS
    }

    combinations = {
        (
            row["dataset"],
            row["question_id"],
            row["method"],
            int(row["depth"]),
        )
        for row in rows
    }

    assert combinations == expected_combinations, (
        "Not all frozen-question/method/depth "
        "combinations were executed."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "Milestone 4 Real-Question Infrastructure Pilot"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Verify required files before loading large models.
    # --------------------------------------------------------

    required_files = [
        MEDMCQA_FILE,
        PUBMEDQA_FILE,
        PRIMEKG_FILE,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Required file not found: {path}"
            )

        print(
            f"Found: {path}"
        )

    # --------------------------------------------------------
    # Freeze evaluation questions.
    # --------------------------------------------------------

    frozen_questions = freeze_questions()

    print(
        f"Frozen MedMCQA questions: "
        f"{len(frozen_questions['MedMCQA'])}"
    )

    print(
        f"Frozen PubMedQA questions: "
        f"{len(frozen_questions['PubMedQA'])}"
    )

    # --------------------------------------------------------
    # Load real pretrained models.
    # --------------------------------------------------------

    embedder = BioMedBERTEmbedder()

    llm = BioMistralLLM()

    # --------------------------------------------------------
    # Load real PrimeKG ONCE.
    # --------------------------------------------------------

    kg = build_knowledge_graph()

    # --------------------------------------------------------
    # Build real biomedical retrieval corpus and FAISS
    # index ONCE.
    # --------------------------------------------------------

    store, kg = build_components(
        embedder=embedder,
        kg=kg,
        frozen_questions=frozen_questions,
    )

    # --------------------------------------------------------
    # Run all 480 experiment cells.
    # --------------------------------------------------------

    rows = run_pilot(
        frozen_questions=frozen_questions,
        embedder=embedder,
        llm=llm,
        store=store,
        kg=kg,
    )

    # --------------------------------------------------------
    # Validate complete experiment.
    # --------------------------------------------------------

    validate_results(
        rows,
        frozen_questions,
    )

    print()
    print("=" * 70)
    print("REAL-QUESTION PILOT PASSED")
    print("=" * 70)

    print(
        f"Rows written: {len(rows)}"
    )

    print(
        f"Frozen questions: "
        f"{QUESTION_FILE}"
    )

    print(
        f"Results file: "
        f"{OUTPUT_FILE}"
    )

    print()
    print("Verified:")
    print(" 20 MedMCQA questions")
    print(" 20 PubMedQA questions")
    print(" 4 methods")
    print(" depths 1, 2, and 3")
    print(" depth d -> exactly d retrieval calls")
    print(" depth d -> exactly d LLM calls")
    print(" depth d -> exactly d reasoning steps")
    print(" real PrimeKG loaded once")
    print(" real PubMedQA biomedical retrieval corpus")
    print(" total expected rows -> 480")


if __name__ == "__main__":
    main()