# Biomedical KG Reasoning Depth — Milestone 2 Package

This package contains the code and repository structure needed for Milestone 2:

> Reimplement the three graph methods' core retrieval-and-reasoning loop.

Implemented components:
- Shared BioMistral-7B loader
- Shared BioMedBERT embedding model
- Shared FAISS vector store
- BiomedKAI depth-governed traversal loop
- KRAGEN graph-of-thoughts subproblem loop
- HyperGraphRAG hyperedge construction/expansion loop
- MedMCQA and PubMedQA dataset loaders
- Dataset download helper
- Basic retrieval and method smoke tests

## Important dataset note

The full benchmark datasets are intentionally not bundled into this ZIP. MedMCQA is over 194k questions and its training split is about 147 MB; PubMedQA contains expert, artificial and unlabeled data. The repository should contain the loader/download instructions rather than large generated copies.

Run:

```bash
python data/download_datasets.py
```

For the Milestone 2 implementation, the loaders use the MedMCQA validation/dev split and the labeled PubMedQA split by default. These are suitable for development/smoke testing; the later pilot/full milestones should define the exact evaluation sample and keep it fixed across all methods.

## Run

Create/activate your virtual environment and install:

```bash
pip install -r requirements.txt
```

Then test the shared retrieval layer:

```bash
python -m evaluation.test_retrieval
```

Run method smoke tests without loading BioMistral:

```bash
python -m evaluation.test_methods
```

## Repository structure

```text
biomedical-kg-reasoning-depth/
├── README.md
├── MILESTONE2.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── shared/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── llm.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   └── knowledge_graph.py
│   ├── biomedkai.py
│   ├── kragen.py
│   └── hypergraph_rag.py
├── data/
│   ├── README.md
│   ├── download_datasets.py
│   ├── medmcqa/
│   │   └── README.md
│   └── pubmedqa/
│       └── README.md
├── evaluation/
│   ├── test_retrieval.py
│   └── test_methods.py
├── results/
├── logs/
└── .gitkeep files
```
## Milestone 2 — Graph-Based Reasoning Methods

Milestone 2 implements three graph-based biomedical reasoning methods:

- BiomedKAI
- KRAGEN
- HyperGraphRAG

The implementations use the shared retrieval infrastructure, including
the KnowledgeGraph and FAISSVectorStore components.

### Verification


