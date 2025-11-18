"""Prompt engineering and templating"""
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re


@dataclass
class PromptTemplate:
    """Prompt template with variables"""
    template: str
    variables: List[str]
    description: Optional[str] = None

    def format(self, **kwargs) -> str:
        """Format template with provided variables"""
        # Check if all required variables are provided
        missing = [var for var in self.variables if var not in kwargs]
        if missing:
            raise ValueError(f"Missing required variables: {missing}")

        return self.template.format(**kwargs)


class PromptEngine:
    """Advanced prompt engineering utilities"""

    # Standard RAG prompt templates
    BASIC_RAG_TEMPLATE = PromptTemplate(
        template="""Use the following context to answer the question. If you cannot answer based on the context, say so.

Context:
{context}

Question: {question}

Answer:""",
        variables=["context", "question"],
        description="Basic RAG prompt"
    )

    DETAILED_RAG_TEMPLATE = PromptTemplate(
        template="""You are a helpful AI assistant. Answer the question based on the provided context. Be accurate and cite specific parts of the context when possible.

Context:
{context}

Question: {question}

Instructions:
- Provide a clear and concise answer
- If the context doesn't contain enough information, acknowledge this
- If you're uncertain, express the level of uncertainty
- Cite relevant parts of the context to support your answer

Answer:""",
        variables=["context", "question"],
        description="Detailed RAG prompt with instructions"
    )

    CONVERSATIONAL_RAG_TEMPLATE = PromptTemplate(
        template="""You are a helpful AI assistant engaged in a conversation. Use the provided context and conversation history to answer the current question.

Context:
{context}

Conversation History:
{history}

Current Question: {question}

Answer naturally and reference previous parts of the conversation when relevant.

Answer:""",
        variables=["context", "history", "question"],
        description="Conversational RAG with history"
    )

    CHAIN_OF_THOUGHT_TEMPLATE = PromptTemplate(
        template="""Answer the following question using the provided context. Think step by step and show your reasoning.

Context:
{context}

Question: {question}

Let's approach this step by step:
1. First, identify the relevant information in the context
2. Then, reason through the answer
3. Finally, provide the conclusion

Answer:""",
        variables=["context", "question"],
        description="Chain-of-thought prompting"
    )

    MULTI_DOC_TEMPLATE = PromptTemplate(
        template="""You are given multiple documents. Synthesize information from all relevant documents to answer the question.

Documents:
{documents}

Question: {question}

Instructions:
- Consider information from all documents
- If documents conflict, note the disagreement
- Synthesize a comprehensive answer
- Indicate which documents support your answer

Answer:""",
        variables=["documents", "question"],
        description="Multi-document synthesis"
    )

    @staticmethod
    def create_context_string(
        documents: List[str],
        scores: Optional[List[float]] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        max_length: Optional[int] = None
    ) -> str:
        """
        Create a formatted context string from retrieved documents

        Args:
            documents: List of document texts
            scores: Optional relevance scores
            metadata: Optional metadata for each document
            max_length: Maximum context length in characters

        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0

        for i, doc in enumerate(documents):
            # Format document with metadata if available
            doc_str = f"Document {i+1}:\n"

            if metadata and i < len(metadata):
                meta = metadata[i]
                if 'source' in meta:
                    doc_str += f"Source: {meta['source']}\n"
                if 'title' in meta:
                    doc_str += f"Title: {meta['title']}\n"

            if scores and i < len(scores):
                doc_str += f"Relevance: {scores[i]:.2f}\n"

            doc_str += f"{doc}\n"

            # Check length limit
            if max_length and current_length + len(doc_str) > max_length:
                break

            context_parts.append(doc_str)
            current_length += len(doc_str)

        return "\n".join(context_parts)

    @staticmethod
    def create_augmented_prompt(
        query: str,
        retrieved_docs: List[Tuple[str, float, Dict[str, Any]]],
        template: Optional[PromptTemplate] = None,
        max_context_length: int = 4000,
        **template_kwargs
    ) -> str:
        """
        Create a complete augmented prompt

        Args:
            query: User query
            retrieved_docs: List of (document, score, metadata) tuples
            template: Prompt template to use
            max_context_length: Maximum context length
            **template_kwargs: Additional template variables

        Returns:
            Complete prompt string
        """
        if template is None:
            template = PromptEngine.BASIC_RAG_TEMPLATE

        # Extract documents, scores, and metadata
        docs = [doc for doc, _, _ in retrieved_docs]
        scores = [score for _, score, _ in retrieved_docs]
        metadata = [meta for _, _, meta in retrieved_docs]

        # Create context
        context = PromptEngine.create_context_string(
            docs,
            scores,
            metadata,
            max_context_length
        )

        # Format template
        template_vars = {
            "context": context,
            "question": query,
            **template_kwargs
        }

        return template.format(**template_vars)

    @staticmethod
    def few_shot_prompt(
        query: str,
        examples: List[Tuple[str, str]],
        context: str
    ) -> str:
        """
        Create a few-shot prompt with examples

        Args:
            query: User query
            examples: List of (question, answer) example pairs
            context: Retrieved context

        Returns:
            Few-shot prompt
        """
        prompt = "Answer questions based on the provided context. Here are some examples:\n\n"

        # Add examples
        for i, (q, a) in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Q: {q}\n"
            prompt += f"A: {a}\n\n"

        # Add current query
        prompt += "Now answer this question:\n\n"
        prompt += f"Context:\n{context}\n\n"
        prompt += f"Q: {query}\n"
        prompt += "A:"

        return prompt

    @staticmethod
    def self_ask_prompt(query: str, context: str) -> str:
        """
        Self-ask prompting: break down complex questions

        Args:
            query: User query
            context: Retrieved context

        Returns:
            Self-ask prompt
        """
        prompt = f"""Answer the question by breaking it down into sub-questions if needed.

Context:
{context}

Question: {query}

Let's approach this systematically:
- If the question is complex, break it into simpler sub-questions
- Answer each sub-question using the context
- Combine the answers to form the final answer

Answer:"""
        return prompt

    @staticmethod
    def verify_answer_prompt(query: str, answer: str, context: str) -> str:
        """
        Create a prompt to verify an answer against the context

        Args:
            query: Original query
            answer: Generated answer
            context: Retrieved context

        Returns:
            Verification prompt
        """
        prompt = f"""Verify if the following answer is supported by the context.

Context:
{context}

Question: {query}

Answer: {answer}

Is this answer accurate and supported by the context? Explain your reasoning.

Verification:"""
        return prompt

    @staticmethod
    def extract_entities_prompt(text: str) -> str:
        """Create a prompt to extract entities from text"""
        prompt = f"""Extract all important entities from the following text. Include people, organizations, locations, dates, and key concepts.

Text:
{text}

Entities (in JSON format):"""
        return prompt

    @staticmethod
    def summarize_context_prompt(context: str, max_words: int = 200) -> str:
        """Create a prompt to summarize context"""
        prompt = f"""Summarize the following context in no more than {max_words} words. Focus on the most important information.

Context:
{context}

Summary:"""
        return prompt


class AdvancedPromptTechniques:
    """Advanced prompting techniques"""

    @staticmethod
    def chain_of_thought(query: str, context: str) -> str:
        """Chain-of-thought prompting"""
        return f"""Answer the question step by step. Show your reasoning process.

Context:
{context}

Question: {query}

Let's solve this step by step:"""

    @staticmethod
    def tree_of_thoughts(query: str, context: str, num_paths: int = 3) -> str:
        """Tree of thoughts: explore multiple reasoning paths"""
        return f"""Explore {num_paths} different approaches to answer this question.

Context:
{context}

Question: {query}

Approach 1:
[Explore first reasoning path]

Approach 2:
[Explore second reasoning path]

Approach 3:
[Explore third reasoning path]

Best Answer:
[Synthesize the best answer from all approaches]"""

    @staticmethod
    def react_prompt(query: str, context: str) -> str:
        """ReAct: Reasoning and Acting"""
        return f"""Answer the question using the ReAct framework. Alternate between Thought, Action, and Observation.

Context:
{context}

Question: {query}

Thought 1: [What do I need to know?]
Action 1: [Look for information in context]
Observation 1: [What did I find?]

Thought 2: [What does this mean?]
Action 2: [Analyze the information]
Observation 2: [My analysis]

Final Answer: [Based on the above reasoning]"""

    @staticmethod
    def meta_prompting(query: str, context: str) -> str:
        """Meta-prompting: prompt the model to improve its own prompt"""
        return f"""You are an expert at answering questions. First, analyze what information you need and how to best approach this question, then answer it.

Context:
{context}

Question: {query}

Analysis:
- What type of question is this?
- What information from the context is most relevant?
- What's the best strategy to answer this?

Strategy:
[Your approach]

Answer:
[Your final answer]"""
