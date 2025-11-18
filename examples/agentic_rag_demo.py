"""Agentic RAG example with autonomous decision-making"""
from rag_pipeline import RAGPipeline
from rag_pipeline.generation.agentic_rag import AgenticRAG
from rag_pipeline.retrieval import HybridSearch
from rag_pipeline.advanced_retrieval import CrossEncoderReranker
from rag_pipeline.generation import LLMFactory

# Knowledge base documents
documents = [
    """
    Artificial Intelligence (AI) is the simulation of human intelligence processes by machines,
    especially computer systems. These processes include learning, reasoning, and self-correction.
    """,
    """
    Machine Learning is a subset of AI that provides systems the ability to automatically learn
    and improve from experience without being explicitly programmed. It focuses on the development
    of computer programs that can access data and use it to learn for themselves.
    """,
    """
    Deep Learning is a subset of machine learning that uses neural networks with multiple layers.
    These neural networks attempt to simulate the behavior of the human brain, allowing it to
    learn from large amounts of data.
    """,
    """
    Natural Language Processing (NLP) is a branch of AI that helps computers understand, interpret,
    and manipulate human language. NLP draws from many disciplines, including computer science and
    computational linguistics.
    """,
    """
    Computer Vision is a field of AI that trains computers to interpret and understand the visual
    world. Using digital images from cameras and videos and deep learning models, machines can
    accurately identify and classify objects.
    """,
]


def main():
    print("=" * 80)
    print("Agentic RAG Example")
    print("=" * 80)

    print("\n1. Setting up components...")

    # Initialize components
    retriever = HybridSearch(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        use_rrf=True
    )

    # Add documents
    retriever.add_documents(documents)

    # Initialize LLM (requires API key)
    try:
        llm = LLMFactory.create("openai", model="gpt-3.5-turbo")

        # Initialize reranker
        reranker = CrossEncoderReranker()

        # Initialize agentic RAG
        print("\n2. Initializing Agentic RAG...")
        agent = AgenticRAG(
            retriever=retriever,
            llm=llm,
            reranker=reranker,
            max_iterations=3
        )

        # Run agent
        query = "How is deep learning different from traditional machine learning?"

        print(f"\n3. Running agent with query: {query}")
        print("=" * 80)

        result = agent.run(query, verbose=True)

        print("\n" + "=" * 80)
        print("Final Results:")
        print("=" * 80)
        print(f"\nAnswer: {result['answer']}")
        print(f"\nIterations: {result['iterations']}")
        print(f"Actions taken: {', '.join(result['actions'])}")
        print(f"Verification passed: {result['verification_passed']}")

    except Exception as e:
        print(f"\nNote: This example requires an OpenAI API key.")
        print(f"Error: {e}")
        print("\nTo run this example:")
        print("1. Set OPENAI_API_KEY environment variable")
        print("2. Or add it to a .env file")


if __name__ == "__main__":
    main()
