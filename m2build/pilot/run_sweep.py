import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(ROOT),
)

import csv
import json
import re
import time
import random

import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
)

from src.shared.knowledge_graph import (
    KnowledgeGraph,
)

from src.shared.vector_store import (
    FAISSVectorStore,
)

from src.biomedkai import (
    BiomedKAI,
)

from src.kragen import (
    KRAGEN,
)

from src.hypergraph_rag import (
    HyperGraphRAG,
)

from src.flat_control import (
    FlatControl,
)


# ================================================================
# PATHS
# ================================================================

RESULTS_DIR = (
    ROOT / "results"
)

RESULTS_DIR.mkdir(
    exist_ok=True
)

OUTPUT_FILE = (
    RESULTS_DIR
    / "results_sweep.csv"
)

CHECKPOINT_FILE = (
    RESULTS_DIR
    / "results_sweep_checkpoint.csv"
)

MEDMCQA_FILE = (
    ROOT
    / "data"
    / "medmcqa"
    / "dev.json"
)

PUBMEDQA_FILE = (
    ROOT
    / "data"
    / "pubmedqa"
    / "pqa_labeled.json"
)

PRIMEKG_FILE = (
    ROOT
    / "data"
    / "kg"
    / "primekg.csv"
)


# ================================================================
# EXPERIMENT DESIGN
# ================================================================

SEEDS = (
    42,
    123,
    2026,
)

QUESTIONS_PER_DATASET = 100

DEPTHS = (
    1,
    2,
    3,
)

METHODS = (
    "BiomedKAI",
    "KRAGEN",
    "HyperGraphRAG",
    "FlatControl",
)

TOP_K = 3

CHECKPOINT_EVERY_QUESTIONS = 10


# ================================================================
# MODEL CONFIGURATION
# ================================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

BIOMEDBERT_NAME = (
    "microsoft/"
    "BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
)

BIOMISTRAL_NAME = (
    "BioMistral/BioMistral-7B"
)

EMBED_BATCH_SIZE = 16

MAX_INPUT_TOKENS = 4096

MAX_NEW_TOKENS = 64


# ================================================================
# CSV SCHEMA
# ================================================================

FIELDNAMES = [
    "seed",
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


# ================================================================
# REPRODUCIBILITY
# ================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ================================================================
# BIOMEDBERT
# ================================================================

class BioMedBERTEmbedder:

    def __init__(
        self,
        model_name=BIOMEDBERT_NAME,
    ):

        print()
        print("=" * 70)
        print("Loading BioMedBERT")
        print("=" * 70)

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

        self.model = (
            AutoModel.from_pretrained(
                model_name
            ).to(DEVICE)
        )

        self.model.eval()

        self.dimension = int(
            self.model.config.hidden_size
        )

        print(
            f"BioMedBERT dimension: "
            f"{self.dimension}"
        )

        print(
            f"BioMedBERT device: "
            f"{DEVICE}"
        )

    @torch.no_grad()
    def encode(
        self,
        texts,
        batch_size=EMBED_BATCH_SIZE,
    ):

        if not texts:

            return np.empty(
                (
                    0,
                    self.dimension,
                ),
                dtype="float32",
            )

        embeddings_all = []

        for start in range(
            0,
            len(texts),
            batch_size,
        ):

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
                for key, value
                in encoded.items()
            }

            outputs = self.model(
                **encoded
            )

            hidden = (
                outputs.last_hidden_state
            )

            mask = (
                encoded["attention_mask"]
                .unsqueeze(-1)
            )

            masked = (
                hidden * mask
            )

            summed = masked.sum(
                dim=1
            )

            counts = (
                mask.sum(dim=1)
                .clamp(min=1)
            )

            pooled = (
                summed / counts
            )

            pooled = (
                torch.nn.functional.normalize(
                    pooled,
                    p=2,
                    dim=1,
                )
            )

            embeddings_all.append(
                pooled.cpu()
                .numpy()
                .astype("float32")
            )

        return np.concatenate(
            embeddings_all,
            axis=0,
        )


# ================================================================
# BIOMISTRAL
# ================================================================

class BioMistralLLM:

    def __init__(
        self,
        model_name=BIOMISTRAL_NAME,
    ):

        print()
        print("=" * 70)
        print("Loading BioMistral-7B")
        print("=" * 70)

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                model_name
            )
        )

        if (
            self.tokenizer.pad_token
            is None
        ):

            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        self.calls = []

        if DEVICE == "cuda":

            print(
                "Using CUDA for BioMistral."
            )

            model_dtype = torch.float16

            self.model = (
                AutoModelForCausalLM
                .from_pretrained(
                    model_name,
                    torch_dtype=model_dtype,
                    device_map=None,
                )
                .to("cuda")
            )

        else:

            print(
                "WARNING: CUDA unavailable. "
                "BioMistral will run on CPU."
            )

            self.model = (
                AutoModelForCausalLM
                .from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,
                )
                .to("cpu")
            )

        self.model.eval()

        model_device = (
            next(
                self.model.parameters()
            ).device
        )

        print(
            f"BioMistral device: "
            f"{model_device}"
        )

        print(
            f"CUDA available: "
            f"{torch.cuda.is_available()}"
        )

        if torch.cuda.is_available():

            print(
                "GPU:",
                torch.cuda.get_device_name(
                    0
                ),
            )

            print(
                "VRAM GB:",
                round(
                    torch.cuda.get_device_properties(
                        0
                    ).total_memory
                    / 1024**3,
                    2,
                ),
            )

    @torch.no_grad()
    def generate(
        self,
        prompt,
        max_new_tokens=MAX_NEW_TOKENS,
    ):

        is_final = (
            "FINAL_ANSWER:" in prompt
        )

        model_device = (
            next(
                self.model.parameters()
            ).device
        )

        # ------------------------------------------------------------
        # Final answer scoring.
        # Avoid generating long final responses.
        # ------------------------------------------------------------

        if is_final:

            if (
                "Dataset: MedMCQA"
                in prompt
                or "Options:"
                in prompt
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
                prompt
                + "\nFINAL_ANSWER:"
            )

            texts = [
                scoring_prompt
                + " "
                + candidate
                for candidate
                in candidates
            ]

            encoded = self.tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_INPUT_TOKENS,
            )

            encoded = {
                key: value.to(
                    model_device
                )
                for key, value
                in encoded.items()
            }

            outputs = self.model(
                **encoded
            )

            logits = (
                outputs.logits
            )

            attention = (
                encoded[
                    "attention_mask"
                ]
            )

            scores = []

            for i, candidate in enumerate(
                candidates
            ):

                length = int(
                    attention[i]
                    .sum()
                    .item()
                )

                candidate_ids = (
                    self.tokenizer(
                        " " + candidate,
                        add_special_tokens=False,
                    )["input_ids"]
                )

                n = len(
                    candidate_ids
                )

                start_pos = (
                    length - n
                )

                logprob = 0.0

                for j, token_id in enumerate(
                    candidate_ids
                ):

                    pos = (
                        start_pos + j
                    )

                    if pos <= 0:
                        continue

                    next_logits = (
                        logits[i, pos - 1]
                    )

                    log_probs = (
                        torch.log_softmax(
                            next_logits,
                            dim=-1,
                        )
                    )

                    logprob += float(
                        log_probs[
                            int(token_id)
                        ].item()
                    )

                scores.append(
                    logprob
                )

            best_index = int(
                np.argmax(scores)
            )

            answer = (
                "FINAL_ANSWER: "
                + candidates[
                    best_index
                ]
            )

            input_tokens = int(
                attention.sum(
                    dim=1
                ).max().item()
            )

            self.calls.append(
                {
                    "input_tokens":
                        input_tokens,
                    "output_tokens":
                        2,
                    "total_tokens":
                        input_tokens + 2,
                }
            )

            return answer

        # ------------------------------------------------------------
        # Intermediate reasoning generation.
        # ------------------------------------------------------------

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )

        input_ids = (
            encoded["input_ids"]
            .to(model_device)
        )

        attention_mask = (
            encoded["attention_mask"]
            .to(model_device)
        )

        input_tokens = int(
            input_ids.shape[1]
        )

        output_ids = (
            self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=8,
                do_sample=False,
                pad_token_id=(
                    self.tokenizer.eos_token_id
                ),
            )
        )

        generated_ids = (
            output_ids[
                0,
                input_tokens:
            ]
        )

        output_tokens = int(
            generated_ids.shape[0]
        )

        answer = (
            self.tokenizer.decode(
                generated_ids,
                skip_special_tokens=True,
            ).strip()
        )

        total_tokens = (
            input_tokens
            + output_tokens
        )

        self.calls.append(
            {
                "input_tokens":
                    input_tokens,
                "output_tokens":
                    output_tokens,
                "total_tokens":
                    total_tokens,
            }
        )

        return answer


# ================================================================
# DATA LOADING
# ================================================================

def load_json(path):

    with path.open(
        "r",
        encoding="utf-8",
    ) as f:

        return json.load(f)


# ================================================================
# QUESTION SELECTION
# ================================================================

def select_questions(
    rows,
    dataset_name,
    seed,
):
    """
    Deterministically select 100 questions for a seed.

    Different seeds produce different question samples.
    """

    if len(rows) < QUESTIONS_PER_DATASET:

        raise ValueError(
            f"{dataset_name} contains "
            f"only {len(rows)} rows."
        )

    rng = random.Random(seed)

    indices = list(
        range(len(rows))
    )

    rng.shuffle(indices)

    selected_indices = sorted(
        indices[
            :QUESTIONS_PER_DATASET
        ]
    )

    selected = []

    for index in selected_indices:

        row = rows[index]

        if dataset_name == "MedMCQA":

            selected.append(
                {
                    "id": str(
                        row["id"]
                    ),
                    "question":
                        row["question"],
                    "options": [
                        row["opa"],
                        row["opb"],
                        row["opc"],
                        row["opd"],
                    ],
                    "correct_choice":
                        row["cop"],
                    "subject_name":
                        row.get(
                            "subject_name"
                        ),
                    "topic_name":
                        row.get(
                            "topic_name"
                        ),
                }
            )

        elif dataset_name == "PubMedQA":

            selected.append(
                {
                    "id": str(
                        row["pubid"]
                    ),
                    "question":
                        row["question"],
                    "context":
                        row["context"],
                    "long_answer":
                        row["long_answer"],
                    "final_decision":
                        row["final_decision"],
                }
            )

    return selected


def load_all_questions():

    medmcqa_rows = load_json(
        MEDMCQA_FILE
    )

    pubmedqa_rows = load_json(
        PUBMEDQA_FILE
    )

    question_sets = {}

    for seed in SEEDS:

        question_sets[seed] = {
            "MedMCQA":
                select_questions(
                    medmcqa_rows,
                    "MedMCQA",
                    seed,
                ),
            "PubMedQA":
                select_questions(
                    pubmedqa_rows,
                    "PubMedQA",
                    seed + 1000,
                ),
        }

    return question_sets


# ================================================================
# PUBMEDQA CONTEXT EXTRACTION
# ================================================================

def extract_pubmed_contexts(
    context
):

    if isinstance(
        context,
        dict,
    ):

        contexts = context.get(
            "contexts",
            [],
        )

        if isinstance(
            contexts,
            list,
        ):

            return [
                str(x).strip()
                for x in contexts
                if str(x).strip()
            ]

        if isinstance(
            contexts,
            str,
        ):

            text = contexts.strip()

            return (
                [text]
                if text
                else []
            )

    if isinstance(
        context,
        list,
    ):

        return [
            str(x).strip()
            for x in context
            if str(x).strip()
        ]

    if isinstance(
        context,
        str,
    ):

        text = context.strip()

        return (
            [text]
            if text
            else []
        )

    return []


# ================================================================
# REAL RETRIEVAL CORPUS
# ================================================================

def build_documents(
    question_sets
):

    rows = load_json(
        PUBMEDQA_FILE
    )

    # Exclude all PubMedQA questions that can appear
    # in the evaluation.
    evaluation_ids = set()

    for seed in SEEDS:

        for item in question_sets[
            seed
        ]["PubMedQA"]:

            evaluation_ids.add(
                str(item["id"])
            )

    documents = []
    seen = set()

    for row in rows:

        pubid = str(
            row.get(
                "pubid",
                "",
            )
        )

        if pubid in evaluation_ids:
            continue

        contexts = (
            extract_pubmed_contexts(
                row.get(
                    "context"
                )
            )
        )

        for text in contexts:

            normalized = re.sub(
                r"\s+",
                " ",
                text,
            ).strip()

            if not normalized:
                continue

            if normalized in seen:
                continue

            seen.add(
                normalized
            )

            documents.append(
                normalized
            )

    if not documents:

        raise RuntimeError(
            "No PubMedQA context documents "
            "were found."
        )

    print()
    print("=" * 70)
    print("REAL BIOMEDICAL RETRIEVAL CORPUS")
    print("=" * 70)
    print(
        "Source: PubMedQA labeled contexts"
    )
    print(
        "Evaluation PubMedQA IDs excluded:",
        len(evaluation_ids),
    )
    print(
        "Unique documents:",
        f"{len(documents):,}",
    )
    print("=" * 70)

    return documents


# ================================================================
# REAL PRIMEKG
# ================================================================

def build_knowledge_graph():

    if not PRIMEKG_FILE.exists():

        raise FileNotFoundError(
            f"PrimeKG not found: "
            f"{PRIMEKG_FILE}"
        )

    kg = KnowledgeGraph(
        csv_path=PRIMEKG_FILE
    )

    return kg


# ================================================================
# COMPONENT BUILD
# ================================================================

def build_components(
    embedder,
    kg,
    question_sets,
):

    documents = build_documents(
        question_sets
    )

    print()
    print("=" * 70)
    print("ENCODING REAL BIOMEDICAL CORPUS")
    print("=" * 70)

    embeddings = (
        embedder.encode(
            documents,
            batch_size=EMBED_BATCH_SIZE,
        )
    )

    print(
        "Embedding matrix:",
        embeddings.shape,
    )

    store = FAISSVectorStore(
        dimension=embedder.dimension
    )

    store.add(
        embeddings,
        documents,
    )

    print(
        "FAISS documents:",
        len(store),
    )

    print("=" * 70)

    return store


# ================================================================
# METHOD FACTORY
# ================================================================

# ================================================================
# METHOD FACTORY
# ================================================================

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
            llm,
            store,
        )

    if method_name == "HyperGraphRAG":

        return HyperGraphRAG(
            llm,
            store,
        )

    if method_name == "FlatControl":

        return FlatControl(
            llm,
            store,
        )

    raise ValueError(
        f"Unknown method: {method_name}"
    )


# ================================================================
# QUESTION FORMATTING
# ================================================================

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
            f"Question: {item['question']}\n\n"
            f"Options:\n{options}\n\n"
            "Answer the question using the retrieved "
            "biomedical evidence."
        )

    if dataset_name == "PubMedQA":

        return (
            "Dataset: PubMedQA\n"
            f"Question: {item['question']}\n\n"
            "Answer with exactly one of: "
            "yes, no, or maybe."
        )

    raise ValueError(
        f"Unknown dataset: {dataset_name}"
    )


# ================================================================
# QUESTION FORMATTING
# ================================================================

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
            f"Question: {item['question']}\n\n"
            f"Options:\n{options}\n\n"
            "Answer the question using the retrieved "
            "biomedical evidence."
        )

    if dataset_name == "PubMedQA":

        contexts = (
            item["context"]
            .get(
                "contexts",
                [],
            )
        )

        context_text = "\n".join(
            str(x)
            for x in contexts
        )

        return (
            "Dataset: PubMedQA\n"
            f"Question: {item['question']}\n\n"
            f"Original context:\n{context_text}\n\n"
            "Answer with exactly one of: "
            "yes, no, or maybe."
        )

    raise ValueError(
        f"Unknown dataset: {dataset_name}"
    )


# ================================================================
# ANSWER EXTRACTION
# ================================================================

def extract_medmcqa_answer(
    text
):

    if not text:
        return None

    text = str(text).strip().upper()

    match = re.search(
        r"FINAL_ANSWER\s*:\s*([ABCD])\b",
        text,
    )

    if match:
        return match.group(1)

    match = re.search(
        r"\b([ABCD])\b",
        text,
    )

    if match:
        return match.group(1)

    return None


def extract_pubmedqa_answer(
    text
):

    if not text:
        return None

    lower = str(text).lower()

    match = re.search(
        r"FINAL_ANSWER\s*:\s*"
        r"(yes|no|maybe)\b",
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

        predicted = (
            extract_medmcqa_answer(
                answer
            )
        )

        raw = str(
            item["correct_choice"]
        ).strip()

        if raw in {
            "1",
            "2",
            "3",
            "4",
        }:

            correct = "ABCD"[
                int(raw) - 1
            ]

        else:

            correct = raw.upper()

        return (
            predicted,
            int(
                predicted is not None
                and predicted == correct
            ),
        )

    predicted = (
        extract_pubmedqa_answer(
            answer
        )
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


# ================================================================
# RESULT HELPERS
# ================================================================

def count_retrieval_calls(
    result
):

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


def count_reasoning_calls(
    result
):

    return len(
        result.get(
            "reasoning_steps",
            [],
        )
    )


def load_existing_results():

    if not OUTPUT_FILE.exists():

        return []

    with OUTPUT_FILE.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:

        return list(
            csv.DictReader(f)
        )


def result_key(row):

    return (
        int(row["seed"]),
        row["dataset"],
        str(row["question_id"]),
        row["method"],
        int(row["depth"]),
    )


def completed_keys():

    rows = load_existing_results()

    return {
        result_key(row)
        for row in rows
    }


def append_rows(rows):

    if not rows:
        return

    file_exists = (
        OUTPUT_FILE.exists()
    )

    with OUTPUT_FILE.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDNAMES,
        )

        if not file_exists:

            writer.writeheader()

        writer.writerows(rows)


# ================================================================
# ONE EXPERIMENT CELL
# ================================================================

def run_cell(
    seed,
    dataset_name,
    item,
    method_name,
    depth,
    embedder,
    llm,
    store,
    kg,
):

    question = (
        make_experiment_question(
            item,
            dataset_name,
        )
    )

    print()
    print(
        f"RUNNING | "
        f"seed={seed} | "
        f"{dataset_name} | "
        f"{item['id']} | "
        f"{method_name} | "
        f"depth={depth}"
    )

    llm.calls = []

    method = build_method(
        method_name,
        embedder,
        store,
        kg,
        llm,
    )

    start = time.perf_counter()

    result = (
        method.retrieve_and_reason(
            question=question,
            depth=depth,
            top_k=TOP_K,
        )
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

    reasoning_steps = (
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

    if retrieval_calls != depth:

        raise RuntimeError(
            f"Retrieval budget violation: "
            f"expected {depth}, "
            f"got {retrieval_calls}"
        )

    if llm_calls != depth:

        raise RuntimeError(
            f"LLM budget violation: "
            f"expected {depth}, "
            f"got {llm_calls}"
        )

    if reasoning_steps != depth:

        raise RuntimeError(
            f"Reasoning budget violation: "
            f"expected {depth}, "
            f"got {reasoning_steps}"
        )

    row = {
        "seed":
            seed,

        "dataset":
            dataset_name,

        "question_id":
            item["id"],

        "question":
            item["question"],

        "method":
            method_name,

        "depth":
            depth,

        "retrieval_calls":
            retrieval_calls,

        "llm_calls":
            llm_calls,

        "reasoning_steps":
            reasoning_steps,

        "input_tokens":
            input_tokens,

        "output_tokens":
            output_tokens,

        "total_tokens":
            total_tokens,

        "runtime_seconds":
            round(
                elapsed,
                6,
            ),

        "predicted_answer":
            predicted,

        "correct_answer":
            (
                item["correct_choice"]
                if dataset_name
                == "MedMCQA"
                else item[
                    "final_decision"
                ]
            ),

        "correct":
            correct,

        "answer":
            answer,
    }

    print(
        f"[PASS] "
        f"retrieval={retrieval_calls} | "
        f"llm={llm_calls} | "
        f"reasoning={reasoning_steps} | "
        f"correct={correct} | "
        f"time={elapsed:.2f}s"
    )

    return row


# ================================================================
# VALIDATION
# ================================================================

def validate_dataset_sizes(
    question_sets
):

    for seed in SEEDS:

        assert len(
            question_sets[seed][
                "MedMCQA"
            ]
        ) == QUESTIONS_PER_DATASET

        assert len(
            question_sets[seed][
                "PubMedQA"
            ]
        ) == QUESTIONS_PER_DATASET


def validate_final_results(
    question_sets
):

    rows = load_existing_results()

    expected = (
        len(SEEDS)
        * 2
        * QUESTIONS_PER_DATASET
        * len(METHODS)
        * len(DEPTHS)
    )

    print()
    print(
        f"Expected final rows: "
        f"{expected:,}"
    )

    print(
        f"Actual rows: "
        f"{len(rows):,}"
    )

    if len(rows) != expected:

        raise RuntimeError(
            f"Sweep incomplete: "
            f"expected {expected} rows, "
            f"got {len(rows)}."
        )

    keys = {
        result_key(row)
        for row in rows
    }

    if len(keys) != expected:

        raise RuntimeError(
            "Duplicate experiment "
            "combinations detected."
        )

    for row in rows:

        depth = int(
            row["depth"]
        )

        if int(
            row["retrieval_calls"]
        ) != depth:

            raise RuntimeError(
                "Retrieval budget violation."
            )

        if int(
            row["llm_calls"]
        ) != depth:

            raise RuntimeError(
                "LLM budget violation."
            )

        if int(
            row["reasoning_steps"]
        ) != depth:

            raise RuntimeError(
                "Reasoning budget violation."
            )

    print()
    print("=" * 70)
    print("MILESTONE 5 SWEEP COMPLETE")
    print("=" * 70)
    print(
        f"Rows: {len(rows):,}"
    )
    print(
        f"Output: {OUTPUT_FILE}"
    )
    print("=" * 70)


# ================================================================
# MAIN
# ================================================================

def main():

    print("=" * 70)
    print("MILESTONE 5 PRODUCTION SWEEP")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        print(
            "GPU:",
            torch.cuda.get_device_name(
                0
            ),
        )

        print(
            "VRAM:",
            round(
                torch.cuda.get_device_properties(
                    0
                ).total_memory
                / 1024**3,
                2,
            ),
            "GB",
        )

    required_files = [
        MEDMCQA_FILE,
        PUBMEDQA_FILE,
        PRIMEKG_FILE,
    ]

    for path in required_files:

        if not path.exists():

            raise FileNotFoundError(
                f"Missing required file: "
                f"{path}"
            )

        print(
            f"Found: {path}"
        )

    # ------------------------------------------------------------
    # Load questions.
    # ------------------------------------------------------------

    question_sets = (
        load_all_questions()
    )

    validate_dataset_sizes(
        question_sets
    )

    print()
    print(
        "Questions per seed:"
    )

    for seed in SEEDS:

        print(
            f"  Seed {seed}: "
            f"100 MedMCQA + "
            f"100 PubMedQA"
        )

    # ------------------------------------------------------------
    # Load models.
    # ------------------------------------------------------------

    embedder = (
        BioMedBERTEmbedder()
    )

    llm = (
        BioMistralLLM()
    )

    # ------------------------------------------------------------
    # Load real PrimeKG.
    # ------------------------------------------------------------

    kg = (
        build_knowledge_graph()
    )

    # ------------------------------------------------------------
    # Build real biomedical FAISS corpus.
    # ------------------------------------------------------------

    store = (
        build_components(
            embedder=embedder,
            kg=kg,
            question_sets=question_sets,
        )
    )

    # ------------------------------------------------------------
    # Resume support.
    # ------------------------------------------------------------

    done = completed_keys()

    print()
    print(
        f"Previously completed cells: "
        f"{len(done):,}"
    )

    pending_since_checkpoint = []
    completed_questions = 0

    # ------------------------------------------------------------
    # Production sweep.
    # ------------------------------------------------------------

    for seed in SEEDS:

        set_seed(seed)

        for dataset_name in (
            "MedMCQA",
            "PubMedQA",
        ):

            questions = (
                question_sets[
                    seed
                ][dataset_name]
            )

            for item in questions:

                question_had_new_work = False

                for method_name in METHODS:

                    for depth in DEPTHS:

                        key = (
                            seed,
                            dataset_name,
                            str(item["id"]),
                            method_name,
                            depth,
                        )

                        if key in done:

                            continue

                        row = run_cell(
                            seed=seed,
                            dataset_name=dataset_name,
                            item=item,
                            method_name=method_name,
                            depth=depth,
                            embedder=embedder,
                            llm=llm,
                            store=store,
                            kg=kg,
                        )

                        pending_since_checkpoint.append(
                            row
                        )

                        done.add(key)

                        question_had_new_work = True

                if question_had_new_work:

                    completed_questions += 1

                # ------------------------------------------------
                # Checkpoint every 10 completed questions.
                # ------------------------------------------------

                if (
                    completed_questions
                    % CHECKPOINT_EVERY_QUESTIONS
                    == 0
                    and pending_since_checkpoint
                ):

                    append_rows(
                        pending_since_checkpoint
                    )

                    print()
                    print(
                        "=" * 70
                    )
                    print(
                        "CHECKPOINT SAVED"
                    )
                    print(
                        f"Questions completed "
                        f"since launch: "
                        f"{completed_questions}"
                    )
                    print(
                        f"Rows saved this checkpoint: "
                        f"{len(pending_since_checkpoint)}"
                    )
                    print(
                        f"Total saved rows: "
                        f"{len(load_existing_results()):,}"
                    )
                    print(
                        "=" * 70
                    )

                    pending_since_checkpoint = []

    # ------------------------------------------------------------
    # Final checkpoint.
    # ------------------------------------------------------------

    if pending_since_checkpoint:

        append_rows(
            pending_since_checkpoint
        )

        print()
        print(
            "Final checkpoint saved."
        )

    # ------------------------------------------------------------
    # Final validation.
    # ------------------------------------------------------------

    validate_final_results(
        question_sets
    )


if __name__ == "__main__":
    main()