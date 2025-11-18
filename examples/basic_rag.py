"""Basic RAG pipeline example"""
from rag_pipeline import RAGPipeline

# Sample documents
documents = [
    """
    Python is a high-level, interpreted programming language known for its simplicity and readability.
    It was created by Guido van Rossum and first released in 1991. Python supports multiple programming
    paradigms including procedural, object-oriented, and functional programming.
    """,
    """
    Machine learning is a subset of artificial intelligence that focuses on building systems that can
    learn from and make decisions based on data. Python has become the de facto language for machine
    learning due to libraries like scikit-learn, TensorFlow, and PyTorch.
    """,
    """
    Natural Language Processing (NLP) is a field of AI that focuses on the interaction between computers
    and human language. It enables computers to understand, interpret, and generate human language in a
    meaningful way. Popular NLP libraries in Python include NLTK, spaCy, and Hugging Face Transformers.
    """,
    """
    RAG (Retrieval-Augmented Generation) is an AI framework that combines information retrieval with
    text generation. It retrieves relevant documents from a knowledge base and uses them to generate
    more accurate and contextual responses. This approach helps reduce hallucinations in LLM outputs.
    """,
]


def main():
    print("=" * 80)
    print("Basic RAG Pipeline Example")
    print("=" * 80)

    # Initialize RAG pipeline
    print("\n1. Initializing RAG pipeline...")
    rag = RAGPipeline(
        vector_db_type="faiss",
        retrieval_mode="hybrid",
        use_reranker=True,
        llm_provider="openai",
        verbose=True
    )

    # Index documents
    print("\n2. Indexing documents...")
    rag.index_documents(documents, chunk=True)

    # Query the pipeline
    print("\n3. Querying the pipeline...")
    questions = [
        "What is Python used for in machine learning?",
        "How does RAG reduce hallucinations?",
        "What are some popular NLP libraries?"
    ]

    for question in questions:
        print(f"\n{'=' * 80}")
        print(f"Q: {question}")
        print(f"{'=' * 80}")

        answer = rag.query(question, return_context=False)
        print(f"\nA: {answer}")


if __name__ == "__main__":
    main()
