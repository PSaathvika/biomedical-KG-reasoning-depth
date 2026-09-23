from .shared.config import DEFAULT_TOP_K, validate_depth
from .shared.knowledge_graph import seed_entities


class BiomedKAI:
    """Fixed single-agent approximation of BiomedKAI/CARE-RAG traversal."""

    def __init__(self, embedder, vector_store, knowledge_graph, llm=None):
        self.embedder = embedder
        self.vector_store = vector_store
        self.kg = knowledge_graph
        self.llm = llm

    def retrieve_and_reason(self, question, depth=1, top_k=DEFAULT_TOP_K):
        validate_depth(depth)

        query_embedding = self.embedder.encode([question])[0]
        retrieved = self.vector_store.search(query_embedding, top_k=top_k)

        entities = list(self.kg.adj.keys())
        seeds = seed_entities(question, entities)
        if not seeds and entities:
            seeds = entities[:1]

        paths = self.kg.traverse(seeds, depth)
        graph_evidence = [f"{s} --{r}--> {o}" for s, r, o in paths]
        evidence = [item["document"] for item in retrieved] + graph_evidence

        prompt = self._prompt(question, evidence, depth)
        answer = self.llm.generate(prompt) if self.llm else None
        return {
            "method": "BiomedKAI",
            "depth": depth,
            "question": question,
            "retrieved": retrieved,
            "graph_paths": paths,
            "answer": answer,
        }

    @staticmethod
    def _prompt(question, evidence, depth):
        context = "\n".join(f"- {x}" for x in evidence)
        return (
            f"Answer the biomedical question using only the retrieved evidence. "
            f"Reasoning depth budget: {depth}.\n\n"
            f"Question: {question}\nEvidence:\n{context}\n\nAnswer:"
        )
