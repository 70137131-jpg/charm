# RAG Pipeline - Complete Implementation

A comprehensive Retrieval-Augmented Generation (RAG) pipeline with advanced features including multiple retrieval methods, reranking, and evaluation capabilities.

## Features

### Module 2: Retrieval
- ✅ Metadata filtering
- ✅ Keyword search (TF-IDF)
- ✅ Keyword search (BM25)
- ✅ Semantic search with embeddings
- ✅ Hybrid search (combining semantic + keyword)
- ✅ Retrieval evaluation

### Module 3: Advanced Retrieval
- ✅ Approximate Nearest Neighbors (ANN) algorithms
- ✅ Multiple vector database support (FAISS, ChromaDB, Qdrant)
- ✅ Document chunking strategies
- ✅ Advanced chunking techniques
- ✅ Query parsing and expansion
- ✅ Cross-encoders and ColBERT
- ✅ Reranking with multiple strategies

### Module 4: Generation
- ✅ LLM integration (OpenAI, Anthropic)
- ✅ Multiple sampling strategies
- ✅ Prompt engineering with templates
- ✅ Advanced prompt techniques
- ✅ Hallucination detection and handling
- ✅ LLM performance evaluation
- ✅ Agentic RAG capabilities
- ✅ RAG vs Fine-tuning guidance

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd charm

# Install dependencies
pip install -r requirements.txt

# Download required models
python -m spacy download en_core_web_sm

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

## Quick Start

```python
from rag_pipeline import RAGPipeline

# Initialize the pipeline
rag = RAGPipeline(
    vector_db_type="faiss",
    retrieval_mode="hybrid",
    use_reranker=True
)

# Index your documents
documents = [
    "Your document text here...",
    "Another document...",
]
rag.index_documents(documents)

# Query the pipeline
response = rag.query("What is the main topic?")
print(response)
```

## Architecture

```
rag_pipeline/
├── retrieval/          # Module 2: Retrieval components
│   ├── keyword_search.py
│   ├── semantic_search.py
│   ├── hybrid_search.py
│   └── metadata_filter.py
├── advanced_retrieval/ # Module 3: Advanced techniques
│   ├── vector_stores.py
│   ├── chunking.py
│   ├── query_parser.py
│   └── reranking.py
├── generation/         # Module 4: LLM generation
│   ├── llm_interface.py
│   ├── prompt_engineering.py
│   ├── hallucination_handler.py
│   └── agentic_rag.py
├── evaluation/         # Evaluation metrics
│   ├── retrieval_metrics.py
│   └── generation_metrics.py
└── utils/             # Utilities
    └── config.py
```

## Usage Examples

See the `examples/` directory for detailed usage examples:
- `basic_rag.py` - Simple RAG pipeline
- `hybrid_search.py` - Hybrid retrieval example
- `with_reranking.py` - Using reranking
- `agentic_rag.py` - Agentic RAG example
- `evaluation.py` - Evaluation examples

## Documentation

See `docs/` for detailed documentation on each module.

## License

MIT
