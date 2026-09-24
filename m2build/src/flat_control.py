from .shared.config import DEFAULT_TOP_K, validate_depth


class FlatControl:
    """
    Matched-budget flat retrieval-and-reasoning control.

    Unlike BiomedKAI, KRAGEN, and HyperGraphRAG, this controller
    does not construct or traverse any graph or hypergraph.

    At depth d, it performs exactly:
        - d retrieval calls
        - d LLM reasoning calls

    The reasoning from each step is used to construct the query
    for the next retrieval step.
    """

    def __init__(self, embedder, vector_store, llm=None):
        self.embedder = embedder
        self.vector_store = vector_store
        self.llm = llm

    def retrieve_and_reason(
        self,
        question,
        depth=1,
        top_k=DEFAULT_TOP_K,
    ):
        """
        Perform depth sequential retrieval-and-reasoning steps.

        Parameters
        ----------
        question : str
            Original biomedical question.

        depth : int
            Number of retrieval-and-reasoning iterations.

        top_k : int
            Number of documents retrieved at each iteration.

        Returns
        -------
        dict
            Structured result containing retrievals, reasoning steps,
            and the final answer.
        """

        validate_depth(depth)

        current_query = question
        retrievals = []
        reasoning_steps = []

        for step in range(1, depth + 1):

            # ---------------------------------------------------------
            # 1. RETRIEVAL
            # Exactly one retrieval call occurs at every step.
            # ---------------------------------------------------------
            query_embedding = self.embedder.encode([current_query])[0]

            evidence = self.vector_store.search(
                query_embedding,
                top_k=top_k,
            )

            retrievals.append(
                {
                    "step": step,
                    "query": current_query,
                    "evidence": evidence,
                }
            )

            # ---------------------------------------------------------
            # 2. REASONING
            # Exactly one LLM call occurs at every step.
            # ---------------------------------------------------------
            prompt = self._build_prompt(
                question=question,
                current_query=current_query,
                evidence=evidence,
                previous_reasoning=reasoning_steps,
                step=step,
                depth=depth,
            )

            if self.llm is not None:
                reasoning = self.llm.generate(prompt)
            else:
                reasoning = None

            reasoning_steps.append(
                {
                    "step": step,
                    "reasoning": reasoning,
                }
            )

            # ---------------------------------------------------------
            # 3. UPDATE QUERY
            # The reasoning from this step guides the next retrieval.
            # No additional retrieval or LLM call is made here.
            # ---------------------------------------------------------
            current_query = self._update_query(
                question=question,
                reasoning=reasoning,
            )

        final_answer = (
            reasoning_steps[-1]["reasoning"]
            if reasoning_steps
            else None
        )

        return {
            "method": "FlatControl",
            "depth": depth,
            "question": question,
            "retrievals": retrievals,
            "reasoning_steps": reasoning_steps,
            "answer": final_answer,
        }

    @staticmethod
    def _build_prompt(
        question,
        current_query,
        evidence,
        previous_reasoning,
        step,
        depth,
    ):
        """
        Build the prompt for one reasoning step.
        """

        evidence_text = "\n".join(
            f"- {item['document']}"
            for item in evidence
        )

        previous_text = "\n".join(
            f"Step {item['step']}: {item['reasoning']}"
            for item in previous_reasoning
            if item["reasoning"] is not None
        )

        if not previous_text:
            previous_text = "None"

        if step < depth:
            instruction = (
                "Reason over the retrieved evidence and produce an "
                "intermediate reasoning result. This reasoning will "
                "be used to guide the next retrieval step."
            )
        else:
            instruction = (
                "Reason over the retrieved evidence and provide the "
                "final answer to the original biomedical question."
            )

        return (
            "You are performing a sequential biomedical "
            "retrieval-and-reasoning process.\n\n"
            f"Original question:\n{question}\n\n"
            f"Current search query:\n{current_query}\n\n"
            f"Reasoning step: {step}/{depth}\n\n"
            f"Retrieved evidence:\n{evidence_text}\n\n"
            f"Previous reasoning:\n{previous_text}\n\n"
            f"Instruction:\n{instruction}"
        )

    @staticmethod
    def _update_query(question, reasoning):
        """
        Construct the next retrieval query from the original question
        and the previous reasoning.

        This is a deterministic query update and does not make an
        additional LLM call.
        """

        if not reasoning:
            return question

        return (
            f"{question}\n\n"
            f"Previous reasoning:\n{reasoning}"
        )


# Convenience function for callers that prefer a functional API.
def retrieve_and_reason(
    question,
    embedder,
    vector_store,
    llm=None,
    depth=1,
    top_k=DEFAULT_TOP_K,
):
    controller = FlatControl(
        embedder=embedder,
        vector_store=vector_store,
        llm=llm,
    )

    return controller.retrieve_and_reason(
        question=question,
        depth=depth,
        top_k=top_k,
    )