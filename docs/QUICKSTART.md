# RAG Pipeline - Quick Start Guide

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd charm

# Install dependencies
pip install -r requirements.txt

# Download spacy model (required for advanced features)
python -m spacy download en_core_web_sm

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Basic Usage

### 1. Simple RAG Pipeline

```python
from rag_pipeline import RAGPipeline

# Initialize pipeline
rag = RAGPipeline(
    vector_db_type="faiss",      # Vector database type
    retrieval_mode="hybrid",      # Retrieval method
    use_reranker=True,            # Enable reranking
    llm_provider="openai"         # LLM provider
)

# Index documents
documents = [
    "Python is a high-level programming language...",
    "Machine learning is a subset of AI...",
]
rag.index_documents(documents)

# Query
answer = rag.query("What is Python?")
print(answer)
```

### 2. Hybrid Search

```python
from rag_pipeline.retrieval import HybridSearch

# Initialize hybrid search (combines keyword + semantic)
search = HybridSearch(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    keyword_method="bm25",
    use_rrf=True  # Use Reciprocal Rank Fusion
)

# Add documents
search.add_documents(documents, metadata=metadata_list)

# Search
results = search.search("your query", top_k=5)
for doc, score, metadata in results:
    print(f"Score: {score:.4f} - {doc}")
```

### 3. With Reranking

```python
from rag_pipeline.retrieval import SemanticSearch
from rag_pipeline.advanced_retrieval import CrossEncoderReranker

# Initial retrieval
search = SemanticSearch()
search.add_documents(documents)
results = search.search(query, top_k=10)

# Rerank
reranker = CrossEncoderReranker()
reranked = reranker.rerank(query, results, top_k=5)
```

### 4. Agentic RAG

```python
from rag_pipeline.generation.agentic_rag import AgenticRAG
from rag_pipeline.retrieval import HybridSearch
from rag_pipeline.generation import LLMFactory

# Set up components
retriever = HybridSearch()
retriever.add_documents(documents)

llm = LLMFactory.create("openai", model="gpt-3.5-turbo")

# Create agent
agent = AgenticRAG(
    retriever=retriever,
    llm=llm,
    max_iterations=3
)

# Run agent
result = agent.run("your complex query", verbose=True)
print(result['answer'])
```

## Configuration

### Environment Variables

```bash
# API Keys
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key

# Models
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
LLM_MODEL=gpt-3.5-turbo

# Retrieval
TOP_K=5
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# Reranking
USE_RERANKER=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Using Config Object

```python
from rag_pipeline.utils import Config

config = Config()
config.top_k = 10
config.chunk_size = 1000

rag = RAGPipeline(config=config)
```

## Advanced Features

### Custom Chunking

```python
from rag_pipeline.advanced_retrieval import DocumentChunker, AdvancedChunker

# Basic chunking
chunker = DocumentChunker()
chunks = chunker.chunk_by_sentences(text, sentences_per_chunk=5)

# Advanced semantic chunking
advanced = AdvancedChunker()
chunks = advanced.chunk_by_semantic_similarity(text)
```

### Query Expansion

```python
from rag_pipeline.advanced_retrieval import QueryParser

parser = QueryParser()
parsed = parser.parse("your query")

print(f"Keywords: {parsed.keywords}")
print(f"Entities: {parsed.entities}")
print(f"Expanded: {parsed.expanded}")
```

### Hallucination Detection

```python
from rag_pipeline.generation import HallucinationHandler

handler = HallucinationHandler(llm)

# Check for hallucinations
check = handler.check_factual_consistency(answer, context)

if check.is_hallucination:
    print(f"Issues: {check.reasons}")

    # Handle it
    corrected = handler.handle_hallucination(
        answer, context, query, strategy="hedge"
    )
```

### Evaluation

```python
from rag_pipeline.evaluation import RetrievalMetrics, GenerationMetrics

# Evaluate retrieval
metrics = RetrievalMetrics.evaluate_retrieval(
    retrieved=["doc1", "doc2", "doc3"],
    relevant={"doc1", "doc3", "doc5"},
    k_values=[1, 3, 5]
)

# Evaluate generation
metrics = GenerationMetrics.evaluate_rag(
    question=question,
    answer=answer,
    reference=reference_answer,
    context=context
)
```

## Examples

See the `examples/` directory for complete working examples:

- `basic_rag.py` - Simple end-to-end RAG
- `hybrid_search_demo.py` - Hybrid search demonstration
- `with_reranking.py` - Using reranking
- `agentic_rag_demo.py` - Agentic RAG example
- `evaluation_demo.py` - Evaluation metrics

## Troubleshooting

### Common Issues

1. **Import errors**: Make sure all dependencies are installed
   ```bash
   pip install -r requirements.txt
   ```

2. **Spacy model not found**: Download the model
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **API key errors**: Set environment variables or add to .env file

4. **CUDA errors**: If you don't have GPU, the pipeline will use CPU automatically

## Next Steps

- Read the full documentation in `docs/`
- Explore different retrieval methods
- Try different LLM providers
- Experiment with chunking strategies
- Evaluate your RAG performance

## Support

For issues and questions, please open an issue on GitHub.
