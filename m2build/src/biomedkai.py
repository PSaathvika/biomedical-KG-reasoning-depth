from .shared.config import DEFAULT_TOP_K, validate_depth
from .shared.knowledge_graph import seed_entities


class BiomedKAI:
    """
    Iterative BiomedKAI/CARE-RAG-style retrieval and KG reasoning.

    At depth d:
        - d vector retrieval calls are performed.
        - d LLM reasoning calls are performed.
        - one KG expansion hop is performed per step.

    The reasoning produced at each step is incorporated into the query
    used for the next retrieval.
    """

    def __init__(self, embedder, vector_store, knowledge_graph, llm=None):
        self.embedder = embedder
        self.vector_store = vector_store
        self.kg = knowledge_graph
        self.llm = llm

    def retrieve_and_reason(
        self,
        question,
        depth=1,
        top_k=DEFAULT_TOP_K,
    ):
        validate_depth(depth)

        # ---------------------------------------------------------
        # Find starting KG entities from the original question.
        # ---------------------------------------------------------
        entities = list(self.kg.adj.keys())

        seeds = seed_entities(
            question,
            entities,
        )

        if not seeds and entities:
            seeds = entities[:1]

        # The frontier is advanced one KG hop per reasoning step.
        frontier = list(seeds)

        current_query = question

        retrieval_history = []
        reasoning_steps = []
        graph_paths = []

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

            # -----------------------------------------------------
            # 2. ONE KG EXPANSION HOP
            # -----------------------------------------------------
            step_paths = self.kg.traverse(
                frontier,
                depth=1,
            )

            graph_paths.extend(step_paths)

            # Advance the KG frontier using the objects discovered
            # during this step.
            next_frontier = []

            for subject, relation, obj in step_paths:
                if obj not in next_frontier:
                    next_frontier.append(obj)

            if next_frontier:
                frontier = next_frontier

            # -----------------------------------------------------
            # 3. BUILD CURRENT EVIDENCE
            # -----------------------------------------------------
            vector_evidence = [
                item["document"]
                for item in retrieved
            ]

            graph_evidence = [
                f"{subject} --{relation}--> {obj}"
                for subject, relation, obj in graph_paths
            ]

            evidence = vector_evidence + graph_evidence

            # Remove duplicate evidence while preserving order.
            evidence = list(dict.fromkeys(evidence))

            # -----------------------------------------------------
            # 4. ONE LLM REASONING CALL
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
            # 5. UPDATE QUERY FOR THE NEXT RETRIEVAL
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
            "method": "BiomedKAI",
            "depth": depth,
            "question": question,
            "retrieved": retrieval_history,
            "graph_paths": graph_paths,
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
                "Reason about the evidence and produce an intermediate "
                "reasoning result. This result will guide the next "
                "retrieval and KG expansion step."
            )
        else:
            instruction = (
                "Reason about the evidence and provide the final answer "
                "to the original biomedical question."
            )

        return (
            "You are performing iterative biomedical KG reasoning.\n\n"
            f"Original question:\n{question}\n\n"
            f"Current step: {step}/{depth}\n\n"
            f"Retrieved and graph evidence:\n{context}\n\n"
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