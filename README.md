# Biomedical KG Reasoning Depth

A research project that studies how reasoning depth and knowledge-graph structure affect biomedical question answering.

The project compares three knowledge-graph-based biomedical reasoning methods:

- BiomedKAI / CARE-RAG
- KRAGEN
- HyperGraphRAG

against a matched Chain-of-Thought (CoT) control.

The main goal is to determine whether graph-structured reasoning provides an advantage over ordinary step-by-step reasoning when both methods are given the same reasoning-step budget.

## Research Question

Does graph-structured reasoning improve biomedical question answering when compared with a flat Chain-of-Thought method given the same reasoning-step budget?

The project also studies how the performance of each method changes as the reasoning depth increases.

## Methods

### 1. BiomedKAI / CARE-RAG

BiomedKAI uses a biomedical knowledge graph and adapts its retrieval strategy based on the type of medical query.

In this project, we focus on its depth-governed knowledge graph traversal.

### 2. KRAGEN

KRAGEN combines:

- Knowledge Graphs
- Retrieval-Augmented Generation (RAG)
- Graph-of-Thoughts (GoT)

A complex question is divided into smaller subproblems, and the relationships between the subproblems are represented using a graph.

### 3. HyperGraphRAG

HyperGraphRAG uses a hypergraph to represent relationships involving multiple entities.

It performs retrieval using entities and hyperedges and uses the retrieved knowledge to guide the final response.

### 4. Matched Chain-of-Thought Control

The flat control performs sequential retrieval and reasoning without using graph structure.

The important part is that it receives the same reasoning-step budget as the graph-based methods.

This allows us to investigate whether any improvement comes from graph structure rather than simply giving the model more reasoning steps.

## Experimental Setup

All four methods will use the same:

- Backbone LLM
- Retrieval embedding model
- Dataset/questions
- Reasoning-depth settings
- Evaluation procedure

### Backbone

The initial backbone is:

**BioMistral-7B**

The model will be used consistently across all experimental methods.

### Embedding Model

The shared retrieval embedding model will be kept fixed across all methods.

### Vector Store

The initial implementation will use:

**FAISS**

FAISS will provide the shared vector retrieval layer so that the different reasoning methods can be compared using the same retrieval setup.

## Reasoning Depth

Each method will be evaluated at three reasoning-depth settings:

- Depth 1
- Depth 2
- Depth 3

The experimental design is:

| Method | Depth 1 | Depth 2 | Depth 3 |
|---|---|---|---|
| BiomedKAI / CARE-RAG | ✓ | ✓ | ✓ |
| KRAGEN | ✓ | ✓ | ✓ |
| HyperGraphRAG | ✓ | ✓ | ✓ |
| Matched CoT Control | ✓ | ✓ | ✓ |

This gives a total of:

**4 methods × 3 depths = 12 experimental settings**

The same questions will be used across the methods so that the comparison remains consistent.

## Main Research Questions

The experiments will investigate three main findings.

### 1. Graph-Structure Advantage

We expect the graph-structured methods to perform better than the matched-step-budget flat Chain-of-Thought control.

This would indicate that the structure of the reasoning process may provide an advantage beyond simply increasing the number of reasoning steps.

### 2. Null / Alternative Outcome

If the graph-based methods do not perform better than the flat control at the same reasoning depth, it would suggest that improvements may mainly come from the number of reasoning steps rather than the graph structure itself.

If the graph-based methods perform better, this would support the importance of structured graph reasoning.

### 3. Depth-3 Flattening

We expect performance to improve when moving from depth 1 to depth 2.

At depth 3, we expect the improvement to become smaller or flatten, indicating that additional reasoning steps may provide limited additional benefit.

## Shared Experimental Harness

The project will use a shared harness so that the main experimental settings remain consistent across all four methods.

The shared harness will handle:

- BioMistral-7B
- Embedding generation
- FAISS vector store
- Dataset loading
- Question processing
- Reasoning-depth configuration
- Result collection
- Evaluation

The method-specific code will only implement the retrieval and reasoning logic required by each approach.

## Repository Structure

```text
biomedical-kg-reasoning-depth/
│
├── README.md
├── requirements.txt
│
├── src/
│   ├── shared/
│   │   ├── llm.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   └── config.py
│   │
│   ├── biomedkai.py
│   ├── kragen.py
│   ├── hypergraph_rag.py
│   └── flat_control.py
│
├── data/
│   ├── medmcqa/
│   └── pubmedqa/
│
├── evaluation/
│   ├── run_pilot.py
│   ├── run_experiments.py
│   └── evaluate.py
│
├── results/
│   ├── pilot/
│   └── final/
│
└── logs/
