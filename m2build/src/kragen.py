from .shared.config import DEFAULT_TOP_K, validate_depth


class KRAGEN:
    """
    Iterative graph-of-thoughts-style retrieval and reasoning.

    At depth d:
        - d retrieval calls are performed.
        - d LLM reasoning calls are performed.

    Each subproblem is solved at its corresponding step, and the
    previous reasoning is included when constructing the next query.
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

    def decompose(
        self,
        question,
        depth,
    ):
        """
        Deterministically decompose the question into at most `depth`
        subproblems.

        If the question does not contain multiple sentences, the same
        original question is used for each depth step.
        """

        parts = [
            part.strip()
            for part in question.replace("?", ".").split(".")
            if part.strip()
        ]

        if len(parts) > 1:
            return parts[:depth]

        return [question] * depth

    def retrieve_and_reason(
        self,
        question,
        depth=1,
        top_k=DEFAULT_TOP_K,
    ):
        validate_depth(depth)

        subproblems = self.decompose(
            question,
            depth,
        )

        thoughts = []
        reasoning_steps = []

        current_query = question

        for step in range(1, depth + 1):

            # -----------------------------------------------------
            # Select the subproblem associated with this step.
            # -----------------------------------------------------
            subproblem = subproblems[step - 1]

            # Include previous reasoning in the next query.
            if step == 1:
                retrieval_query = subproblem
            else:
                previous_reasoning = reasoning_steps[-1]["reasoning"]

                retrieval_query = (
                    f"{subproblem}\n\n"
                    f"Previous reasoning:\n{previous_reasoning}"
                )

            current_query = retrieval_query

            # -----------------------------------------------------
            # 1. RETRIEVAL
            # Exactly one retrieval call per step.
            # -----------------------------------------------------
            embedding = self.embedder.encode(
                [current_query]
            )[0]

            evidence = self.vector_store.search(
                embedding,
                top_k=top_k,
            )

            # -----------------------------------------------------
            # 2. RECORD THE CURRENT THOUGHT/SUBPROBLEM
            # -----------------------------------------------------
            thought = {
                "id": step,
                "subproblem": subproblem,
                "query": current_query,
                "depends_on": list(range(1, step)),
                "evidence": evidence,
            }

            thoughts.append(thought)

            # -----------------------------------------------------
            # 3. ONE LLM REASONING CALL
            # Exactly one LLM call per step.
            # -----------------------------------------------------
            prompt = self._prompt(
                question=question,
                thoughts=thoughts,
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

        final_answer = (
            reasoning_steps[-1]["reasoning"]
            if reasoning_steps
            else None
        )

        return {
            "method": "KRAGEN",
            "depth": depth,
            "question": question,
            "thoughts": thoughts,
            "reasoning_steps": reasoning_steps,
            "answer": final_answer,
        }

    @staticmethod
    def _prompt(
        question,
        thoughts,
        step,
        depth,
    ):
        blocks = []

        for thought in thoughts:
            evidence = "\n".join(
                f"- {item['document']}"
                for item in thought["evidence"]
            )

            blocks.append(
                f"Subproblem {thought['id']}: "
                f"{thought['subproblem']}\n"
                f"Dependencies: {thought['depends_on']}\n"
                f"Search query: {thought['query']}\n"
                f"Evidence:\n{evidence}"
            )

        previous_reasoning = []

        for thought_step in thoughts[:-1]:
            # Reasoning is not stored inside thoughts, so this section
            # is intentionally summarized by the current prompt.
            previous_reasoning.append(
                f"Completed subproblem step {thought_step['id']}."
            )

        previous_text = (
            "\n".join(previous_reasoning)
            if previous_reasoning
            else "None"
        )

        if step < depth:
            instruction = (
                "Solve the current subproblem using the evidence. "
                "Produce an intermediate reasoning result that can "
                "guide the next subproblem."
            )
        else:
            instruction = (
                "Solve the current subproblem and provide the final "
                "answer to the original biomedical question."
            )

        return (
            "You are performing iterative KRAGEN-style "
            "graph-of-thoughts reasoning.\n\n"
            f"Original question:\n{question}\n\n"
            f"Current step: {step}/{depth}\n\n"
            f"Previous reasoning context:\n{previous_text}\n\n"
            f"Current reasoning structure:\n"
            + "\n\n".join(blocks)
            + "\n\n"
            f"Instruction:\n{instruction}"
        )