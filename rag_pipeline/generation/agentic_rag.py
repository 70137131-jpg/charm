"""Agentic RAG: RAG with autonomous decision-making"""
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum


class AgentAction(Enum):
    """Possible agent actions"""
    RETRIEVE = "retrieve"
    RERANK = "rerank"
    EXPAND_QUERY = "expand_query"
    FILTER = "filter"
    GENERATE = "generate"
    VERIFY = "verify"
    DONE = "done"


@dataclass
class AgentState:
    """State of the agent during execution"""
    query: str
    original_query: str
    retrieved_docs: List[Tuple[str, float, Dict[str, Any]]]
    generated_answer: Optional[str]
    verification_passed: bool
    iteration: int
    max_iterations: int
    metadata: Dict[str, Any]


class AgenticRAG:
    """
    Agentic RAG system that can autonomously decide actions

    The agent can:
    - Retrieve documents
    - Refine queries
    - Rerank results
    - Generate answers
    - Verify answers
    - Iterate if needed
    """

    def __init__(
        self,
        retriever,
        llm,
        reranker=None,
        query_parser=None,
        hallucination_handler=None,
        max_iterations: int = 3
    ):
        """
        Initialize agentic RAG

        Args:
            retriever: Document retriever
            llm: Language model
            reranker: Optional reranker
            query_parser: Optional query parser
            hallucination_handler: Optional hallucination handler
            max_iterations: Maximum number of iterations
        """
        self.retriever = retriever
        self.llm = llm
        self.reranker = reranker
        self.query_parser = query_parser
        self.hallucination_handler = hallucination_handler
        self.max_iterations = max_iterations

    def run(
        self,
        query: str,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Run the agentic RAG pipeline

        Args:
            query: User query
            verbose: Whether to print progress

        Returns:
            Dictionary with answer and metadata
        """
        # Initialize state
        state = AgentState(
            query=query,
            original_query=query,
            retrieved_docs=[],
            generated_answer=None,
            verification_passed=False,
            iteration=0,
            max_iterations=self.max_iterations,
            metadata={"actions": []}
        )

        # Main loop
        while state.iteration < state.max_iterations:
            state.iteration += 1

            if verbose:
                print(f"\n=== Iteration {state.iteration} ===")

            # Decide next action
            action = self._decide_action(state)

            if verbose:
                print(f"Action: {action.value}")

            # Execute action
            if action == AgentAction.RETRIEVE:
                state = self._retrieve(state, verbose)
            elif action == AgentAction.RERANK:
                state = self._rerank(state, verbose)
            elif action == AgentAction.EXPAND_QUERY:
                state = self._expand_query(state, verbose)
            elif action == AgentAction.FILTER:
                state = self._filter_docs(state, verbose)
            elif action == AgentAction.GENERATE:
                state = self._generate(state, verbose)
            elif action == AgentAction.VERIFY:
                state = self._verify(state, verbose)
            elif action == AgentAction.DONE:
                break

            # Record action
            state.metadata["actions"].append(action.value)

        # Return final result
        return {
            "answer": state.generated_answer,
            "documents": state.retrieved_docs,
            "iterations": state.iteration,
            "actions": state.metadata["actions"],
            "verification_passed": state.verification_passed,
            "metadata": state.metadata
        }

    def _decide_action(self, state: AgentState) -> AgentAction:
        """
        Decide next action based on current state

        This is a simple rule-based decision maker.
        For more sophisticated decisions, use the LLM.
        """
        # First iteration: retrieve or expand query
        if state.iteration == 1:
            if self.query_parser:
                return AgentAction.EXPAND_QUERY
            else:
                return AgentAction.RETRIEVE

        # If no documents retrieved, retrieve
        if not state.retrieved_docs:
            return AgentAction.RETRIEVE

        # If documents retrieved but not reranked, rerank
        if self.reranker and "rerank" not in state.metadata.get("actions", []):
            return AgentAction.RERANK

        # If documents ready but no answer, generate
        if state.generated_answer is None:
            return AgentAction.GENERATE

        # If answer generated but not verified, verify
        if not state.verification_passed and self.hallucination_handler:
            return AgentAction.VERIFY

        # Done
        return AgentAction.DONE

    def _retrieve(self, state: AgentState, verbose: bool) -> AgentState:
        """Retrieve documents"""
        if verbose:
            print(f"Retrieving documents for: {state.query}")

        # Use retriever's search method
        state.retrieved_docs = self.retriever.search(state.query, top_k=10)

        if verbose:
            print(f"Retrieved {len(state.retrieved_docs)} documents")

        return state

    def _rerank(self, state: AgentState, verbose: bool) -> AgentState:
        """Rerank documents"""
        if verbose:
            print("Reranking documents")

        if self.reranker and state.retrieved_docs:
            state.retrieved_docs = self.reranker.rerank(
                state.query,
                state.retrieved_docs,
                top_k=5
            )

            if verbose:
                print(f"Reranked to top {len(state.retrieved_docs)} documents")

        return state

    def _expand_query(self, state: AgentState, verbose: bool) -> AgentState:
        """Expand or refine query"""
        if verbose:
            print(f"Expanding query: {state.query}")

        if self.query_parser:
            parsed = self.query_parser.parse(state.query)

            if parsed.expanded:
                # Use first expansion
                state.query = parsed.expanded[0]

                if verbose:
                    print(f"Expanded to: {state.query}")

        return state

    def _filter_docs(self, state: AgentState, verbose: bool) -> AgentState:
        """Filter documents based on relevance"""
        if verbose:
            print("Filtering documents")

        # Simple filtering by score threshold
        threshold = 0.5
        filtered = [
            (doc, score, meta)
            for doc, score, meta in state.retrieved_docs
            if score >= threshold
        ]

        state.retrieved_docs = filtered

        if verbose:
            print(f"Filtered to {len(filtered)} documents")

        return state

    def _generate(self, state: AgentState, verbose: bool) -> AgentState:
        """Generate answer"""
        if verbose:
            print("Generating answer")

        # Create context from documents
        context_parts = []
        for i, (doc, score, meta) in enumerate(state.retrieved_docs[:5], 1):
            context_parts.append(f"[{i}] {doc}")

        context = "\n\n".join(context_parts)

        # Generate prompt
        prompt = f"""Answer the question based on the provided context. Be accurate and concise.

Context:
{context}

Question: {state.query}

Answer:"""

        # Generate
        response = self.llm.generate(prompt, temperature=0.7)
        state.generated_answer = response.text

        if verbose:
            print(f"Generated answer: {state.generated_answer[:100]}...")

        return state

    def _verify(self, state: AgentState, verbose: bool) -> AgentState:
        """Verify answer for hallucinations"""
        if verbose:
            print("Verifying answer")

        if self.hallucination_handler and state.generated_answer:
            # Create context
            context = "\n".join([doc for doc, _, _ in state.retrieved_docs[:5]])

            # Check for hallucinations
            check = self.hallucination_handler.check_factual_consistency(
                state.generated_answer,
                context
            )

            state.verification_passed = not check.is_hallucination

            if verbose:
                print(f"Verification: {'PASSED' if state.verification_passed else 'FAILED'}")
                if not state.verification_passed:
                    print(f"Reasons: {', '.join(check.reasons)}")

            # If verification failed, try to fix
            if not state.verification_passed and state.iteration < state.max_iterations:
                # Regenerate with stricter prompt
                state.generated_answer = None  # Will regenerate in next iteration

        else:
            # No hallucination handler, assume passed
            state.verification_passed = True

        return state

    def self_reflect(
        self,
        state: AgentState
    ) -> str:
        """
        Use LLM to reflect on current state and decide action

        This is more sophisticated than rule-based decisions
        """
        reflection_prompt = f"""You are an AI agent working on answering a question. Reflect on the current state and decide the next action.

Original Question: {state.original_query}
Current Query: {state.query}
Iteration: {state.iteration}/{state.max_iterations}

Documents Retrieved: {len(state.retrieved_docs)}
Answer Generated: {"Yes" if state.generated_answer else "No"}
Verification Passed: {"Yes" if state.verification_passed else "No"}

Actions Taken So Far: {', '.join(state.metadata.get('actions', []))}

Possible Actions:
1. RETRIEVE - Retrieve more documents
2. RERANK - Rerank existing documents
3. EXPAND_QUERY - Refine or expand the query
4. FILTER - Filter documents by relevance
5. GENERATE - Generate an answer
6. VERIFY - Verify the answer
7. DONE - Finish if satisfied with the answer

What should be the next action? Respond with just the action name.

Next Action:"""

        response = self.llm.generate(reflection_prompt, temperature=0.3)
        action_text = response.text.strip().upper()

        # Parse action
        try:
            return AgentAction[action_text]
        except KeyError:
            # Default action if parsing fails
            return self._decide_action(state)


class MultiHopRAG:
    """
    Multi-hop RAG for questions requiring multiple retrieval steps

    Example: "What university did the CEO of Tesla attend?"
    - First hop: Retrieve info about CEO of Tesla
    - Second hop: Retrieve info about that person's education
    """

    def __init__(self, retriever, llm):
        """
        Initialize multi-hop RAG

        Args:
            retriever: Document retriever
            llm: Language model
        """
        self.retriever = retriever
        self.llm = llm

    def run(
        self,
        query: str,
        max_hops: int = 3,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Run multi-hop RAG

        Args:
            query: User query
            max_hops: Maximum number of hops
            verbose: Whether to print progress

        Returns:
            Dictionary with answer and metadata
        """
        all_docs = []
        current_query = query

        for hop in range(max_hops):
            if verbose:
                print(f"\n=== Hop {hop + 1} ===")
                print(f"Query: {current_query}")

            # Retrieve documents
            docs = self.retriever.search(current_query, top_k=5)
            all_docs.extend(docs)

            if verbose:
                print(f"Retrieved {len(docs)} documents")

            # Check if we have enough information
            context = "\n".join([doc for doc, _, _ in all_docs])

            check_prompt = f"""Can the following question be answered with the given context?

Context:
{context}

Question: {query}

Respond with YES or NO, followed by a brief explanation.

Response:"""

            response = self.llm.generate(check_prompt, temperature=0.0)

            if response.text.strip().upper().startswith("YES"):
                # We have enough information
                if verbose:
                    print("Sufficient information found")
                break

            # Generate next query
            if hop < max_hops - 1:
                next_query_prompt = f"""Generate a follow-up query to find missing information.

Original Question: {query}
Current Context:
{context}

What additional information is needed? Generate a specific search query.

Follow-up Query:"""

                response = self.llm.generate(next_query_prompt, temperature=0.3)
                current_query = response.text.strip()

                if verbose:
                    print(f"Next query: {current_query}")

        # Generate final answer
        context = "\n".join([doc for doc, _, _ in all_docs[:10]])

        answer_prompt = f"""Answer the question based on all the information gathered.

Context:
{context}

Question: {query}

Answer:"""

        response = self.llm.generate(answer_prompt, temperature=0.7)

        return {
            "answer": response.text,
            "documents": all_docs,
            "hops": hop + 1,
        }
