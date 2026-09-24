import re

from .shared.config import DEFAULT_TOP_K, validate_depth


class HyperGraphRAG:
    """
    Iterative hypergraph retrieval and reasoning.

    At depth d:
        - d vector retrieval calls are performed.
        - d hypergraph expansion steps are performed.
        - d LLM reasoning calls are performed.

    Reasoning from each step is incorporated into the next retrieval
    query.
    """

    def __init__(
        self,
        embedder,
        vector_store,
        llm=None,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm

    @staticmethod
    def build_hyperedges(documents):
        """
        Construct simple document-level hyperedges.

        Each document becomes one hyperedge containing up to eight
        extracted biomedical-style terms.
        """

        hyperedges = []

        for idx, doc in enumerate(documents):
            terms = re.findall(
                r"\b[A-Za-z][A-Za-z-]{3,}\b",
                doc,
            )

            entities = list(
                dict.fromkeys(
                    term.lower()
                    for term in terms
                )
            )[:8]

            if entities:
                hyperedges.append(
                    {
                        "id": idx,
                        "entities": entities,
                        "document": doc,
                    }
                )

        return hyperedges

    @staticmethod
    def expand(
        hyperedges,
        seed_entities,
        depth,
    ):
        """
        Expand the hypergraph for the requested number of levels.

        The iterative controller calls this with depth=1 at each
        reasoning step so that one expansion level corresponds to one
        reasoning step.
        """

        selected = []
        frontier = set(seed_entities)
        seen_edges = set()

        for _ in range(depth):
            next_frontier = set()

            for edge in hyperedges:
                if edge["id"] in seen_edges:
                    continue

                if frontier.intersection(
                    edge["entities"]
                ):
                    selected.append(edge)
                    seen_edges.add(edge["id"])
                    next_frontier.update(
                        edge["entities"]
                    )

            frontier = next_frontier

            if not frontier:
                break

        return selected

    def retrieve_and_reason(
        self,
        question,
        depth=1,
        top_k=DEFAULT_TOP_K,
    ):
        validate_depth(depth)

        # Initial hypergraph frontier comes from the question.
        frontier = set(
            re.findall(
                r"\b[A-Za-z][A-Za-z-]{3,}\b",
                question.lower(),
            )
        )

        current_query = question

        retrieval_history = []
        expanded_edges = []
        reasoning_steps = []

        for step in range(1, depth + 1):

            # -----------------------------------------------------
            # 1. VECTOR RETRIEVAL
            # Exactly one retrieval call per step.
            # -----------------------------------------------------
            query_embedding = self.embedder.encode(
                [current_query]
            )[0]

            retrieved = self.vector_store.search(
                query_embedding,
                top_k=top_k,
            )

            retrieval_history.append(
                {
                    "step": step,
                    "query": current_query,
                    "documents": retrieved,
                }
            )

            documents = [
                item["document"]
                for item in retrieved
            ]

            # -----------------------------------------------------
            # 2. BUILD HYPEREDGES FROM CURRENT RETRIEVAL
            # -----------------------------------------------------
            hyperedges = self.build_hyperedges(
                documents
            )

            # -----------------------------------------------------
            # 3. ONE HYPERGRAPH EXPANSION LEVEL
            # -----------------------------------------------------
            step_edges = self.expand(
                hyperedges=hyperedges,
                seed_entities=frontier,
                depth=1,
            )

            expanded_edges.extend(step_edges)

            # -----------------------------------------------------
            # 4. UPDATE FRONTIER
            # -----------------------------------------------------
            next_frontier = set()

            for edge in step_edges:
                next_frontier.update(
                    edge["entities"]
                )

            if next_frontier:
                frontier = next_frontier

            # -----------------------------------------------------
            # 5. BUILD EVIDENCE
            # -----------------------------------------------------
            vector_evidence = documents

            graph_evidence = [
                edge["document"]
                for edge in expanded_edges
            ]

            evidence = list(
                dict.fromkeys(
                    vector_evidence + graph_evidence
                )
            )

            # If no hyperedge matched, retain retrieved documents.
            if not evidence:
                evidence = documents

            # -----------------------------------------------------
            # 6. ONE LLM REASONING CALL
            # Exactly one LLM call per step.
            # -----------------------------------------------------
            prompt = self._prompt(
                question=question,
                evidence=evidence,
                previous_reasoning=reasoning_steps,
                step=step,
                depth=depth,
            )

            answer = (
                self.llm.generate(prompt)
                if self.llm is not None
                else None
            )

            reasoning_steps.append(
                {
                    "step": step,
                    "reasoning": answer,
                }
            )

            # -----------------------------------------------------
            # 7. UPDATE NEXT SEARCH QUERY
            # -----------------------------------------------------
            current_query = self._update_query(
                question=question,
                reasoning=answer,
            )

        final_answer = (
            reasoning_steps[-1]["reasoning"]
            if reasoning_steps
            else None
        )

        return {
            "method": "HyperGraphRAG",
            "depth": depth,
            "question": question,
            "retrieved": retrieval_history,
            "hyperedges": expanded_edges,
            "reasoning_steps": reasoning_steps,
            "answer": final_answer,
        }

    @staticmethod
    def _prompt(
        question,
        evidence,
        previous_reasoning,
        step,
        depth,
    ):
        context = "\n".join(
            f"- {item}"
            for item in evidence
        )

        previous = "\n".join(
            f"Step {item['step']}: {item['reasoning']}"
            for item in previous_reasoning
            if item["reasoning"] is not None
        )

        if not previous:
            previous = "None"

        if step < depth:
            instruction = (
                "Reason over the current hypergraph evidence and "
                "produce an intermediate reasoning result that will "
                "guide the next retrieval and expansion."
            )
        else:
            instruction = (
                "Reason over the hypergraph evidence and provide the "
                "final answer to the original biomedical question."
            )

        return (
            "You are performing iterative HyperGraphRAG reasoning.\n\n"
            f"Original question:\n{question}\n\n"
            f"Current step: {step}/{depth}\n\n"
            f"Hypergraph and retrieved evidence:\n{context}\n\n"
            f"Previous reasoning:\n{previous}\n\n"
            f"Instruction:\n{instruction}"
        )

    @staticmethod
    def _update_query(
        question,
        reasoning,
    ):
        if not reasoning:
            return question

        return (
            f"{question}\n\n"
            f"Previous reasoning:\n{reasoning}"
        )