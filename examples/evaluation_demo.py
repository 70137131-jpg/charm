"""Demonstration of RAG evaluation metrics"""
from rag_pipeline.evaluation import RetrievalMetrics, GenerationMetrics

def demo_retrieval_metrics():
    """Demonstrate retrieval evaluation metrics"""
    print("=" * 80)
    print("Retrieval Metrics Demonstration")
    print("=" * 80)

    # Simulated retrieval results
    retrieved = ["doc1", "doc2", "doc3", "doc4", "doc5"]
    relevant = {"doc1", "doc3", "doc5", "doc7", "doc9"}

    print(f"\nRetrieved: {retrieved}")
    print(f"Relevant: {relevant}")

    # Calculate metrics
    metrics = RetrievalMetrics.evaluate_retrieval(
        retrieved,
        relevant,
        k_values=[1, 3, 5]
    )

    print("\nMetrics:")
    print("-" * 80)
    for metric, value in metrics.items():
        print(f"{metric:25s}: {value:.4f}")


def demo_generation_metrics():
    """Demonstrate generation evaluation metrics"""
    print("\n" + "=" * 80)
    print("Generation Metrics Demonstration")
    print("=" * 80)

    # Sample Q&A
    question = "What is machine learning?"
    context = """
    Machine learning is a subset of artificial intelligence that provides systems
    the ability to automatically learn and improve from experience without being
    explicitly programmed. It focuses on the development of computer programs
    that can access data and use it to learn for themselves.
    """
    generated_answer = """
    Machine learning is a branch of AI that enables systems to learn from data
    and improve their performance over time without explicit programming. It allows
    computers to learn patterns from data automatically.
    """
    reference_answer = """
    Machine learning is a subset of AI that allows systems to learn and improve
    from experience without being programmed explicitly.
    """

    print(f"\nQuestion: {question}")
    print(f"\nGenerated Answer: {generated_answer}")
    print(f"\nReference Answer: {reference_answer}")

    # Calculate metrics
    print("\n" + "-" * 80)
    print("Metrics:")
    print("-" * 80)

    # Token overlap
    overlap = GenerationMetrics.token_overlap(generated_answer, reference_answer)
    print(f"\nToken Overlap:")
    print(f"  Precision: {overlap['precision']:.4f}")
    print(f"  Recall: {overlap['recall']:.4f}")
    print(f"  F1: {overlap['f1']:.4f}")

    # Exact match
    em = GenerationMetrics.exact_match(generated_answer, reference_answer)
    print(f"\nExact Match: {em:.4f}")

    # Answer relevance (requires embedding model)
    try:
        relevance = GenerationMetrics.answer_relevance(generated_answer, question)
        print(f"\nAnswer Relevance: {relevance:.4f}")

        faithfulness = GenerationMetrics.faithfulness(generated_answer, context)
        print(f"Faithfulness: {faithfulness:.4f}")

        context_relevance = GenerationMetrics.context_relevance(context, question)
        print(f"Context Relevance: {context_relevance:.4f}")
    except Exception as e:
        print(f"\nNote: Semantic metrics require sentence-transformers")
        print(f"Error: {e}")


def main():
    demo_retrieval_metrics()
    demo_generation_metrics()


if __name__ == "__main__":
    main()
