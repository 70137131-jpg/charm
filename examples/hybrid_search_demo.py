"""Demonstration of hybrid search capabilities"""
from rag_pipeline.retrieval import HybridSearch

# Sample documents
documents = [
    "The Eiffel Tower is located in Paris, France. It was built in 1889.",
    "The Statue of Liberty is in New York City. It was a gift from France in 1886.",
    "The Great Wall of China is one of the most famous landmarks in the world.",
    "The Colosseum in Rome, Italy, was built in 80 AD.",
    "Machine learning algorithms can be supervised, unsupervised, or reinforcement learning.",
    "Deep learning is a subset of machine learning using neural networks with multiple layers.",
    "Natural language processing enables computers to understand human language.",
]

metadata = [
    {"type": "landmark", "country": "France"},
    {"type": "landmark", "country": "USA"},
    {"type": "landmark", "country": "China"},
    {"type": "landmark", "country": "Italy"},
    {"type": "ai", "topic": "machine learning"},
    {"type": "ai", "topic": "deep learning"},
    {"type": "ai", "topic": "nlp"},
]


def main():
    print("=" * 80)
    print("Hybrid Search Demonstration")
    print("=" * 80)

    # Initialize hybrid search
    print("\n1. Initializing hybrid search...")
    search = HybridSearch(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        keyword_method="bm25",
        use_rrf=True
    )

    # Add documents
    print("\n2. Adding documents...")
    search.add_documents(documents, metadata)

    # Test queries
    queries = [
        "famous structures in France",
        "AI and neural networks",
        "landmarks built in the 19th century"
    ]

    for query in queries:
        print(f"\n{'=' * 80}")
        print(f"Query: {query}")
        print(f"{'=' * 80}")

        results = search.search(query, top_k=3)

        for i, (doc, score, meta) in enumerate(results, 1):
            print(f"\n[{i}] Score: {score:.4f}")
            print(f"Document: {doc[:100]}...")
            print(f"Metadata: {meta}")


if __name__ == "__main__":
    main()
