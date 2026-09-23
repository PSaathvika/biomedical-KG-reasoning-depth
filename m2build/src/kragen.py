from .shared.config import DEFAULT_TOP_K, validate_depth


class KRAGEN:
    """Graph-of-thoughts style subproblem/retrieval loop."""

    def __init__(self, embedder, vector_store, llm=None):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm

    def decompose(self, question, depth):
        # Deterministic decomposition keeps the budget explicit for comparison.
        parts = [p.strip() for p in question.replace("?", ".").split(".") if p.strip()]
        if len(parts) > 1:
            return parts[:depth]
        return [question] * depth

    def retrieve_and_reason(self, question, depth=1, top_k=DEFAULT_TOP_K):
        validate_depth(depth)
        subproblems = self.decompose(question, depth)
        thoughts = []
        for i, subproblem in enumerate(subproblems, start=1):
            embedding = self.embedder.encode([subproblem])[0]
            evidence = self.vector_store.search(embedding, top_k=top_k)
            thoughts.append({
                "id": i,
                "subproblem": subproblem,
                "depends_on": list(range(1, i)),
                "evidence": evidence,
            })

        prompt = self._prompt(question, thoughts, depth)
        answer = self.llm.generate(prompt) if self.llm else None
        return {
            "method": "KRAGEN",
            "depth": depth,
            "question": question,
            "thoughts": thoughts,
            "answer": answer,
        }

    @staticmethod
    def _prompt(question, thoughts, depth):
        blocks = []
        for thought in thoughts:
            evidence = "\n".join(f"- {x['document']}" for x in thought["evidence"])
            blocks.append(
                f"Subproblem {thought['id']}: {thought['subproblem']}\n"
                f"Dependencies: {thought['depends_on']}\nEvidence:\n{evidence}"
            )
        return (
            f"Solve the biomedical question using a graph of dependent subproblems. "
            f"Reasoning depth budget: {depth}.\n\nQuestion: {question}\n\n"
            + "\n\n".join(blocks)
            + "\n\nFinal answer:"
        )
