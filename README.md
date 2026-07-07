# RAG Pipeline - Complete Implementation

A comprehensive Retrieval-Augmented Generation (RAG) pipeline with advanced features including multiple retrieval methods, reranking, and evaluation capabilities.

## 🚀 Deploy to Vercel

https://charm-sand.vercel.app/

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

### Option 1: Deploy to Vercel (Recommended)

```bash
# Install Vercel CLI
npm install -g vercel

# Clone and deploy
git clone <repository-url>
cd charm
vercel

# Set environment variables
vercel env add OPENAI_API_KEY
```

See [VERCEL_DEPLOYMENT.md](docs/VERCEL_DEPLOYMENT.md) for detailed instructions.

### Option 2: Local Development

```bash
# Clone the repository
git clone <repository-url>
cd charm

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required models
python -m spacy download en_core_web_sm

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Option 3: Local Flask Server

Run a local Flask development server for testing and development:

```bash
# Method 1: Using the run script (easiest)
./run_flask.sh              # On Linux/Mac
run_flask.bat               # On Windows

# Method 2: Direct Python
python app.py               # Default: http://localhost:5000
python app.py --port 8080   # Custom port
python app.py --debug       # Debug mode

# Method 3: Using Flask CLI
flask run                   # Uses .flaskenv configuration
flask run --port 8080       # Custom port
```

The local Flask server provides:
- Full REST API on http://localhost:5000
- Persistent document storage in `./data/`
- Configuration via environment variables
- Additional endpoints (`/api/reset`, `/api/config`)
- Better error messages for development

## Quick Start

### Web Interface (After Deployment)

1. Navigate to your deployed URL: `https://your-project.vercel.app`
2. Use the web interface to index documents and ask questions

### REST API

```bash
# Index documents
curl -X POST https://your-project.vercel.app/api/index \
  -H "Content-Type: application/json" \
  -d '{"documents": ["Doc 1", "Doc 2"]}'

# Query
curl -X POST https://your-project.vercel.app/api/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Your question?", "top_k": 5}'
```

### Python Package

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
charm/
├── api/                    # Vercel serverless API
│   └── index.py           # Flask API endpoints (Vercel)
├── app.py                 # Local Flask development server
├── run_flask.sh          # Flask run script (Linux/Mac)
├── run_flask.bat         # Flask run script (Windows)
├── .flaskenv             # Flask environment config
├── public/                 # Frontend
│   └── index.html         # Web interface
├── rag_pipeline/          # Core package
│   ├── retrieval/         # Module 2: Retrieval
│   ├── advanced_retrieval/# Module 3: Advanced retrieval
│   ├── generation/        # Module 4: Generation
│   ├── evaluation/        # Evaluation metrics
│   └── pipeline.py        # Main pipeline
├── examples/              # Usage examples
├── docs/                  # Documentation
├── data/                  # Local storage (created on first run)
├── vercel.json           # Vercel configuration
└── requirements.txt      # Dependencies
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check and API info |
| `/api/health` | GET | Service health status |
| `/api/index` | POST | Index documents |
| `/api/query` | POST | Query the RAG pipeline |
| `/api/stats` | GET | Get pipeline statistics |
| `/api/config` | GET | Get current configuration (local only) |
| `/api/reset` | POST | Reset pipeline and clear documents (local only) |

## Usage Examples

See the `examples/` directory for detailed usage examples:
- `basic_rag.py` - Simple RAG pipeline
- `hybrid_search.py` - Hybrid retrieval example
- `with_reranking.py` - Using reranking
- `agentic_rag.py` - Agentic RAG example
- `evaluation.py` - Evaluation examples

## Documentation

- [Quick Start Guide](docs/QUICKSTART.md) - Get started quickly
- [Flask Local Development](docs/FLASK_LOCAL.md) - Run Flask server locally
- [Vercel Deployment](docs/VERCEL_DEPLOYMENT.md) - Deploy to production
- [RAG vs Fine-tuning](docs/RAG_VS_FINETUNING.md) - When to use each approach

## Tech Stack

- **Retrieval**: FAISS, ChromaDB, Qdrant, BM25, TF-IDF
- **Generation**: OpenAI GPT, Anthropic Claude
- **Web**: Flask, Vercel, vanilla JavaScript
- **Evaluation**: Custom metrics, ROUGE, BERTScore

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## License

MIT

---

Made with ❤️ for the RAG community | [Deploy to Vercel](https://vercel.com/new/clone?repository-url=https://github.com/YOUR_USERNAME/charm)
