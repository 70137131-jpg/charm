# Flask Local Development Guide

This guide will help you set up and run the RAG Pipeline Flask server locally for development and testing.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Virtual environment (recommended)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd charm
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate on Linux/Mac
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt

# Or install minimal dependencies for API only
pip install -r requirements-api.txt
```

### 4. Download Required Models

```bash
python -m spacy download en_core_web_sm
```

### 5. Configure Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env with your API keys
# Required for generation (choose one or both):
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Optional configuration:
LLM_PROVIDER=openai              # or 'anthropic'
LLM_MODEL=gpt-3.5-turbo          # or 'gpt-4', 'claude-3-sonnet', etc.
RETRIEVAL_MODE=hybrid            # 'semantic', 'keyword', or 'hybrid'
VECTOR_DB_TYPE=faiss             # 'faiss', 'chromadb', or 'qdrant'
USE_RERANKER=false               # 'true' or 'false'
```

## Running the Server

### Method 1: Using Run Scripts (Easiest)

**Linux/Mac:**
```bash
./run_flask.sh
```

**Windows:**
```bash
run_flask.bat
```

### Method 2: Direct Python

```bash
# Default settings (runs on http://localhost:5000)
python app.py

# Custom port
python app.py --port 8080

# Debug mode (auto-reload on code changes)
python app.py --debug

# Custom host and port
python app.py --host 127.0.0.1 --port 3000
```

### Method 3: Flask CLI

```bash
# Uses .flaskenv configuration (runs on port 5000 in debug mode)
flask run

# Custom port
flask run --port 8080

# Production mode
FLASK_ENV=production flask run
```

## Usage Examples

### 1. Check Server Health

```bash
curl http://localhost:5000/api/health
```

Response:
```json
{
  "status": "healthy",
  "documents_indexed": 0,
  "storage_path": "/path/to/charm/data/rag_documents.pkl"
}
```

### 2. Index Documents

```bash
curl -X POST http://localhost:5000/api/index \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      "Machine learning is a subset of artificial intelligence.",
      "Neural networks are inspired by biological neurons.",
      "Deep learning uses multiple layers of neural networks."
    ],
    "metadata": [
      {"source": "ml_intro.txt", "topic": "machine learning"},
      {"source": "ml_intro.txt", "topic": "neural networks"},
      {"source": "ml_intro.txt", "topic": "deep learning"}
    ]
  }'
```

Response:
```json
{
  "status": "success",
  "indexed": 3,
  "total_chunks": 3,
  "saved_to_disk": true,
  "storage_path": "/path/to/charm/data/rag_documents.pkl"
}
```

### 3. Query the Pipeline

```bash
curl -X POST http://localhost:5000/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "top_k": 3,
    "return_context": true
  }'
```

Response:
```json
{
  "answer": "Machine learning is a subset of artificial intelligence...",
  "query": "What is machine learning?",
  "documents": [
    {
      "text": "Machine learning is a subset of artificial intelligence.",
      "full_text": "Machine learning is a subset of artificial intelligence.",
      "score": 0.92,
      "metadata": {
        "source": "ml_intro.txt",
        "topic": "machine learning"
      }
    }
  ],
  "tokens_used": 150,
  "num_documents_retrieved": 1
}
```

### 4. Get Pipeline Statistics

```bash
curl http://localhost:5000/api/stats
```

Response:
```json
{
  "documents_indexed": 3,
  "retrieval_mode": "hybrid",
  "vector_db_type": "faiss",
  "reranker_enabled": false,
  "llm_provider": "openai",
  "storage_path": "/path/to/charm/data/rag_documents.pkl",
  "storage_exists": true
}
```

### 5. Get Configuration

```bash
curl http://localhost:5000/api/config
```

Response:
```json
{
  "vector_db_type": "faiss",
  "retrieval_mode": "hybrid",
  "use_reranker": false,
  "llm_provider": "openai",
  "documents_indexed": 3,
  "storage_directory": "/path/to/charm/data"
}
```

### 6. Reset Pipeline

```bash
curl -X POST http://localhost:5000/api/reset
```

Response:
```json
{
  "status": "success",
  "message": "RAG pipeline reset successfully",
  "documents_indexed": 0
}
```

## Using the Web Interface

The Flask server serves the web interface from the `public/` directory:

1. Start the Flask server
2. Open your browser to http://localhost:5000
3. Use the interactive web UI to:
   - Index documents (left panel)
   - Ask questions (right panel)
   - View source documents with relevance scores

## File Storage

The local Flask server uses persistent file storage:

- **Storage directory**: `./data/`
- **Documents file**: `./data/rag_documents.pkl`
- Documents persist between server restarts
- Use `/api/reset` to clear all documents

## Configuration Options

### Environment Variables

Set these in your `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | - | OpenAI API key (required for OpenAI LLMs) |
| `ANTHROPIC_API_KEY` | - | Anthropic API key (required for Claude) |
| `LLM_PROVIDER` | `openai` | LLM provider: `openai` or `anthropic` |
| `LLM_MODEL` | `gpt-3.5-turbo` | Model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `RETRIEVAL_MODE` | `hybrid` | Retrieval mode: `semantic`, `keyword`, or `hybrid` |
| `VECTOR_DB_TYPE` | `faiss` | Vector DB: `faiss`, `chromadb`, or `qdrant` |
| `USE_RERANKER` | `false` | Enable cross-encoder reranking |
| `VERBOSE` | `true` | Enable verbose logging |

### Flask Environment Variables

Set these in `.flaskenv` or as environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `FLASK_APP` | `app.py` | Flask application file |
| `FLASK_ENV` | `development` | Environment: `development` or `production` |
| `FLASK_DEBUG` | `1` | Enable debug mode (auto-reload) |
| `FLASK_RUN_HOST` | `0.0.0.0` | Server host |
| `FLASK_RUN_PORT` | `5000` | Server port |

## Troubleshooting

### Port Already in Use

If port 5000 is already in use:
```bash
python app.py --port 8080
```

### API Key Errors

If you get authentication errors:
1. Check your `.env` file has valid API keys
2. Ensure the correct `LLM_PROVIDER` is set
3. Verify API keys are active and have sufficient credits

### Module Import Errors

If you get import errors:
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt

# If using minimal dependencies
pip install -r requirements-api.txt
```

### Documents Not Persisting

If documents don't persist between restarts:
- Check that `./data/` directory exists and is writable
- Verify `DOCUMENTS_PATH` in the logs
- Check file permissions

### Model Download Issues

If sentence transformers fail to download:
```bash
# Pre-download the model
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

## Production Deployment

For production use, consider:

1. **Use a production WSGI server** (not Flask's development server):
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:8000 app:app
   ```

2. **Use environment variables** (not `.env` files):
   ```bash
   export OPENAI_API_KEY=your_key
   export FLASK_ENV=production
   python app.py
   ```

3. **Set up proper logging**:
   - Configure logging to files
   - Use log rotation
   - Monitor error logs

4. **Use a reverse proxy** (nginx, Apache):
   - Handle SSL/TLS termination
   - Serve static files
   - Load balancing

5. **Consider containerization**:
   - Create a Dockerfile
   - Use Docker Compose for dependencies
   - Deploy to cloud platforms

## Next Steps

- Read [QUICKSTART.md](QUICKSTART.md) for detailed pipeline usage
- See [VERCEL_DEPLOYMENT.md](VERCEL_DEPLOYMENT.md) for cloud deployment
- Explore `examples/` directory for code examples
- Check out [RAG_VS_FINETUNING.md](RAG_VS_FINETUNING.md) for guidance

## Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the error messages in the console
3. Ensure all dependencies are installed
4. Verify API keys are configured correctly

---

Happy coding! 🚀
