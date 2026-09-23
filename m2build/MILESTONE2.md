# Milestone 2 checklist

## Shared harness
- [x] BioMistral-7B loader
- [x] BioMedBERT embeddings
- [x] FAISS retrieval
- [x] Common depth configuration

## Method-specific loops
- [x] BiomedKAI: fixed single-agent, depth-governed KG traversal
- [x] KRAGEN: subproblem decomposition + dependency graph + per-subproblem retrieval
- [x] HyperGraphRAG: entity/hyperedge construction + depth-governed expansion

## Data
- [x] MedMCQA loader
- [x] PubMedQA labeled loader
- [x] Reproducible dataset download helper

## Important experimental rule

All methods must use the same backbone, embedding model, questions and evaluation procedure. Only the retrieval/reasoning logic should differ. The depth argument is restricted to 1, 2 or 3.

This package is a Milestone-2 implementation scaffold. It is not a claim that the complete benchmark has already been run. The actual KG source/corpus and final question sample must be fixed before the pilot/full sweep.
