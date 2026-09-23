from collections import defaultdict
import re
from .shared.config import DEFAULT_TOP_K, validate_depth


class HyperGraphRAG:
    """Hyperedge construction and depth-governed expansion loop."""

    def __init__(self, embedder, vector_store, llm=None):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm

    @staticmethod
    def build_hyperedges(documents):
        hyperedges = []
        for idx, doc in enumerate(documents):
            terms = re.findall(r"\b[A-Za-z][A-Za-z-]{3,}\b", doc)
            entities = list(dict.fromkeys(t.lower() for t in terms))[:8]
            if entities:
                hyperedges.append({"id": idx, "entities": entities, "document": doc})
        return hyperedges

    @staticmethod
    def expand(hyperedges, seed_entities, depth):
        selected = []
        frontier = set(seed_entities)
        seen_edges = set()
        for _ in range(depth):
            next_frontier = set()
            for edge in hyperedges:
                if edge["id"] in seen_edges:
                    continue
                if frontier.intersection(edge["entities"]):
                    selected.append(edge)
                    seen_edges.add(edge["id"])
                    next_frontier.update(edge["entities"])
            frontier = next_frontier
            if not frontier:
                break
        return selected

    def retrieve_and_reason(self, question, depth=1, top_k=DEFAULT_TOP_K):
        validate_depth(depth)
        embedding = self.embedder.encode([question])[0]
        retrieved = self.vector_store.search(embedding, top_k=top_k)
        documents = [x["document"] for x in retrieved]
        hyperedges = self.build_hyperedges(documents)

        question_terms = set(re.findall(r"\b[A-Za-z][A-Za-z-]{3,}\b", question.lower()))
        expanded = self.expand(hyperedges, question_terms, depth)
        evidence = [edge["document"] for edge in expanded]
        if not evidence:
            evidence = documents

        prompt = self._prompt(question, evidence, depth)
        answer = self.llm.generate(prompt) if self.llm else None
        return {
            "method": "HyperGraphRAG",
            "depth": depth,
            "question": question,
            "retrieved": retrieved,
            "hyperedges": expanded,
            "answer": answer,
        }

    @staticmethod
    def _prompt(question, evidence, depth):
        context = "\n".join(f"- {x}" for x in evidence)
        return (
            f"Answer the biomedical question using the expanded hypergraph evidence. "
            f"Reasoning depth budget: {depth}.\n\nQuestion: {question}\n"
            f"Evidence:\n{context}\n\nAnswer:"
        )
