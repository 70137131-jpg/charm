"""Example using reranking to improve results"""
from rag_pipeline.retrieval import SemanticSearch
from rag_pipeline.advanced_retrieval import CrossEncoderReranker

# Sample documents about different topics
documents = [
    "Climate change is causing global temperatures to rise, leading to melting ice caps and rising sea levels.",
    "Renewable energy sources like solar and wind power are becoming increasingly cost-effective.",
    "Electric vehicles are gaining popularity as a more environmentally friendly transportation option.",
    "Python is a versatile programming language used in web development, data science, and automation.",
    "Machine learning models require large amounts of data for training to achieve good performance.",
    "Neural networks are inspired by the structure and function of the human brain.",
    "The greenhouse effect is a natural process that warms the Earth's surface.",
    "Deforestation contributes to climate change by reducing the number of trees that absorb CO2.",
]


def main():
    print("=" * 80)
    print("Reranking Example")
    print("=" * 80)

    # Initialize search and reranker
    print("\n1. Initializing components...")
    search = SemanticSearch(model_name="sentence-transformers/all-MiniLM-L6-v2")
    reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # Add documents
    print("\n2. Adding documents...")
    search.add_documents(documents)

    # Query
    query = "How does climate change affect the environment?"

    print(f"\n3. Query: {query}")
    print("\n" + "=" * 80)

    # Initial retrieval
    print("\nInitial Retrieval (Top 5):")
    print("-" * 80)
    initial_results = search.search(query, top_k=5)

    for i, (doc, score, _) in enumerate(initial_results, 1):
        print(f"\n[{i}] Score: {score:.4f}")
        print(f"{doc}")

    # After reranking
    print("\n" + "=" * 80)
    print("\nAfter Reranking (Top 3):")
    print("-" * 80)
    reranked_results = reranker.rerank(query, initial_results, top_k=3)

    for i, (doc, score, _) in enumerate(reranked_results, 1):
        print(f"\n[{i}] Score: {score:.4f}")
        print(f"{doc}")


if __name__ == "__main__":
    main()
