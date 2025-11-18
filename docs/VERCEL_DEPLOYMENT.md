# Deploying RAG Pipeline to Vercel

This guide walks you through deploying your RAG pipeline to Vercel as a serverless API with a web interface.

## Prerequisites

1. **Vercel Account**: Sign up at [vercel.com](https://vercel.com)
2. **Vercel CLI** (optional): `npm install -g vercel`
3. **API Keys**: OpenAI or Anthropic API key

## Quick Deploy (Recommended)

### Method 1: Deploy via GitHub (Easiest)

1. **Push your code to GitHub** (already done if you followed the setup)

2. **Connect to Vercel**:
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your GitHub repository
   - Vercel will auto-detect the configuration

3. **Set Environment Variables**:
   In the Vercel dashboard, go to Settings > Environment Variables and add:
   ```
   OPENAI_API_KEY=your_openai_key_here
   ANTHROPIC_API_KEY=your_anthropic_key_here (optional)
   LLM_PROVIDER=openai
   LLM_MODEL=gpt-3.5-turbo
   ```

4. **Deploy**:
   - Click "Deploy"
   - Wait for deployment to complete
   - Your app will be live at `https://your-project.vercel.app`

### Method 2: Deploy via Vercel CLI

1. **Install Vercel CLI**:
   ```bash
   npm install -g vercel
   ```

2. **Login**:
   ```bash
   vercel login
   ```

3. **Deploy**:
   ```bash
   cd /path/to/charm
   vercel
   ```

4. **Follow prompts**:
   - Set up project name
   - Link to existing project or create new
   - Deploy!

5. **Set environment variables**:
   ```bash
   vercel env add OPENAI_API_KEY
   vercel env add ANTHROPIC_API_KEY
   vercel env add LLM_PROVIDER
   ```

## Configuration

### vercel.json

The deployment is configured via `vercel.json`:

```json
{
  "version": 2,
  "builds": [
    {
      "src": "api/index.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/api/(.*)",
      "dest": "api/index.py"
    }
  ],
  "functions": {
    "api/index.py": {
      "memory": 3008,
      "maxDuration": 60
    }
  }
}
```

### Environment Variables

Required:
- `OPENAI_API_KEY` - Your OpenAI API key

Optional:
- `ANTHROPIC_API_KEY` - Your Anthropic API key
- `LLM_PROVIDER` - LLM provider (default: "openai")
- `LLM_MODEL` - Model to use (default: "gpt-3.5-turbo")
- `EMBEDDING_MODEL` - Embedding model (default: "sentence-transformers/all-MiniLM-L6-v2")

## API Endpoints

Once deployed, your API will be available at:

### `GET /`
Health check and API information

### `GET /api/health`
Service health status

### `POST /api/index`
Index documents

**Request:**
```json
{
  "documents": [
    "Document 1 text...",
    "Document 2 text..."
  ],
  "metadata": [
    {"source": "doc1"},
    {"source": "doc2"}
  ],
  "chunk": true
}
```

**Response:**
```json
{
  "status": "success",
  "indexed": 2,
  "total_chunks": 10
}
```

### `POST /api/query`
Query the RAG pipeline

**Request:**
```json
{
  "query": "What is machine learning?",
  "top_k": 5,
  "return_context": true
}
```

**Response:**
```json
{
  "answer": "Machine learning is...",
  "documents": [
    {
      "text": "Document text...",
      "score": 0.95,
      "metadata": {}
    }
  ],
  "tokens_used": 150
}
```

### `GET /api/stats`
Get pipeline statistics

**Response:**
```json
{
  "documents_indexed": 10,
  "retrieval_mode": "hybrid",
  "vector_db_type": "faiss",
  "reranker_enabled": false
}
```

## Using the Web Interface

1. Navigate to your deployed URL: `https://your-project.vercel.app`

2. **Index Documents**:
   - Enter documents in the left panel (one per line)
   - Click "Index Documents"

3. **Ask Questions**:
   - Enter your question in the right panel
   - Adjust the number of documents to retrieve
   - Click "Ask Question"

4. **View Results**:
   - See the generated answer
   - View source documents with relevance scores
   - Check pipeline statistics at the bottom

## Testing the API

### Using curl

**Index documents:**
```bash
curl -X POST https://your-project.vercel.app/api/index \
  -H "Content-Type: application/json" \
  -d '{
    "documents": ["Python is a programming language", "Machine learning uses algorithms"],
    "chunk": true
  }'
```

**Query:**
```bash
curl -X POST https://your-project.vercel.app/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Python?",
    "top_k": 3,
    "return_context": true
  }'
```

### Using Python

```python
import requests

BASE_URL = "https://your-project.vercel.app"

# Index documents
response = requests.post(f"{BASE_URL}/api/index", json={
    "documents": [
        "Python is a high-level programming language",
        "Machine learning is a subset of AI"
    ]
})
print(response.json())

# Query
response = requests.post(f"{BASE_URL}/api/query", json={
    "query": "What is Python?",
    "top_k": 5
})
print(response.json())
```

## Limitations

### Vercel Free Tier

- **Execution Time**: 10 seconds per request (Hobby plan)
- **Memory**: 1024 MB (Hobby plan), 3008 MB (Pro plan)
- **Bandwidth**: 100 GB/month
- **Cold Starts**: First request after inactivity may be slower

### Recommendations

1. **Use Pro Plan for Production**:
   - Increased memory (3008 MB)
   - Longer execution time (60s)
   - Better performance

2. **Optimize for Cold Starts**:
   - Reranking is disabled by default (faster initialization)
   - Documents stored in `/tmp` (ephemeral)
   - Consider using external vector database for persistence

3. **Persistent Storage**:
   The current implementation uses `/tmp` which is ephemeral. For production:
   - Use external vector database (Pinecone, Weaviate)
   - Store documents in cloud storage (S3, GCS)
   - Use Vercel KV for metadata

## Advanced Configuration

### Using External Vector Database

Modify `api/index.py` to use Pinecone or Weaviate:

```python
# Use Pinecone for persistent storage
from rag_pipeline.advanced_retrieval import QdrantStore

rag = RAGPipeline(
    vector_db_type="qdrant",
    # ... other config
)
```

### Custom Domain

1. Go to Vercel Dashboard > Settings > Domains
2. Add your custom domain
3. Configure DNS records
4. SSL certificate is automatically provisioned

### Monitoring

Vercel provides built-in monitoring:
- Go to your project dashboard
- View Analytics for request stats
- Check Logs for debugging
- Set up alerts for errors

## Troubleshooting

### Issue: "Module not found"
**Solution**: Make sure all dependencies are in `requirements-api.txt`

### Issue: "Timeout"
**Solution**:
- Reduce `top_k` value
- Disable reranking
- Upgrade to Pro plan for longer execution time

### Issue: "Out of memory"
**Solution**:
- Upgrade to Pro plan (3008 MB)
- Reduce chunk size
- Process fewer documents at once

### Issue: "Documents not persisting"
**Solution**: `/tmp` is ephemeral. Implement persistent storage using:
- External vector database
- Cloud storage
- Vercel KV

## Production Checklist

- [ ] Environment variables configured
- [ ] API keys secured
- [ ] Custom domain configured (optional)
- [ ] Monitoring set up
- [ ] Error handling tested
- [ ] Rate limiting implemented (if needed)
- [ ] CORS configured for your domain
- [ ] External storage configured for persistence
- [ ] Pro plan activated (for production)

## Cost Estimation

### Vercel Costs
- **Hobby**: Free (good for testing)
- **Pro**: $20/month (recommended for production)

### LLM API Costs
- **OpenAI GPT-3.5**: ~$0.002 per 1K tokens
- **OpenAI GPT-4**: ~$0.03 per 1K tokens
- **Anthropic Claude**: ~$0.008 per 1K tokens

**Example**: 1000 queries/day with GPT-3.5:
- ~200K tokens/day
- Cost: ~$4-12/month

## Support

For issues:
1. Check Vercel logs in dashboard
2. Review this documentation
3. Open an issue on GitHub
4. Check Vercel documentation: [vercel.com/docs](https://vercel.com/docs)

## Next Steps

- Set up persistent storage
- Add authentication
- Implement rate limiting
- Add caching layer
- Set up monitoring alerts
- Configure custom domain
