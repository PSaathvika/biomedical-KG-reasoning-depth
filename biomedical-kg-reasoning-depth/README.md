# Biomedical KG Reasoning Depth

A research project that studies how reasoning depth and knowledge-graph structure affect biomedical question answering.

## Research Question

Does graph-structured reasoning improve biomedical question answering when compared with a flat Chain-of-Thought method given the same reasoning-step budget?

## Methods

- BiomedKAI / CARE-RAG
- KRAGEN
- HyperGraphRAG
- Matched Chain-of-Thought (CoT) control

## Experimental Setup

- Backbone: BioMistral-7B
- Fixed retrieval embedding model
- Vector store: FAISS
- Reasoning depths: 1, 2, 3
- Datasets: MedMCQA and PubMedQA
- Metrics: accuracy, wall-clock time, token usage

## Experimental Stages

1. 20-question pilot
2. Full depth sweep
3. Analysis and write-up

## Repository Structure

```text
biomedical-kg-reasoning-depth/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── shared/
│       ├── __init__.py
│       ├── config.py
│       ├── embeddings.py
│       ├── vector_store.py
│       └── llm.py
├── data/
├── evaluation/
│   └── test_retrieval.py
├── results/
└── logs/
```

## Current Status

### Milestone 1 — Read & Digest

- [x] Research question
- [x] Experimental controls
- [x] Matched-step-budget control
- [x] Predicted outcomes
- [x] Milestone 1 verified

### Milestone 2 — Reimplementation

- [ ] Shared Python harness
- [ ] BioMistral-7B
- [ ] Embedding model
- [ ] FAISS
- [ ] BiomedKAI / CARE-RAG
- [ ] KRAGEN
- [ ] HyperGraphRAG
- [ ] Dataset loaders
- [ ] Pilot

### Milestone 3+

- [ ] Matched-budget flat control
- [ ] Full depth sweep
- [ ] Evaluation and analysis
- [ ] Final paper
