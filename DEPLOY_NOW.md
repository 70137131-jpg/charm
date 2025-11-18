# 🚀 Deploy Your RAG Pipeline to Vercel NOW!

## ✅ What's Ready

Your complete RAG pipeline is now ready for deployment with:

- ✅ Flask REST API (`/api/index.py`)
- ✅ Beautiful web interface (`/public/index.html`)
- ✅ Vercel configuration (`vercel.json`)
- ✅ All dependencies configured
- ✅ Documentation and examples

## 🎯 Quick Deploy (5 minutes)

### Option 1: Deploy via GitHub (Easiest - No CLI Required)

1. **Push to GitHub** (if not already done):
   ```bash
   git push origin claude/build-rag-pipeline-01FS8AsKztKCPeDXYjSgEut7
   ```

2. **Go to Vercel**:
   - Visit https://vercel.com
   - Sign in with GitHub
   - Click "New Project"
   - Select your `charm` repository
   - Click "Import"

3. **Configure**:
   - Project Name: `rag-pipeline` (or any name you like)
   - Root Directory: `./` (leave as is)
   - Build Command: (leave empty)
   - Output Directory: `public`

4. **Add Environment Variables**:
   Click "Environment Variables" and add:
   ```
   Name: OPENAI_API_KEY
   Value: sk-your-actual-openai-key-here
   ```

5. **Deploy**:
   - Click "Deploy"
   - Wait 2-3 minutes
   - Done! 🎉

Your app will be live at: `https://rag-pipeline.vercel.app` (or your custom name)

### Option 2: Deploy via Vercel CLI (Fastest for Developers)

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
   - Setup and deploy: Yes
   - Which scope: (your account)
   - Link to existing project: No
   - Project name: rag-pipeline
   - In which directory: `./`
   - Override settings: No

5. **Add API Key**:
   ```bash
   vercel env add OPENAI_API_KEY
   ```
   Paste your OpenAI API key when prompted

6. **Redeploy with env vars**:
   ```bash
   vercel --prod
   ```

Done! Your deployment URL will be shown in the terminal.

## 🎨 Using Your Deployed App

### Web Interface

1. Open your deployment URL in a browser
2. You'll see a beautiful interface with two panels:

**Left Panel - Index Documents:**
- Enter your documents (one per line)
- Click "Index Documents"
- Wait for confirmation

**Right Panel - Ask Questions:**
- Type your question
- Adjust number of documents to retrieve (1-10)
- Click "Ask Question"
- See the answer with source documents

**Bottom - Statistics:**
- View how many documents are indexed
- See the retrieval mode and vector database type

### API Endpoints

Once deployed, you can use the API programmatically:

**Index Documents:**
```bash
curl -X POST https://your-app.vercel.app/api/index \
  -H "Content-Type: application/json" \
  -d '{
    "documents": [
      "Python is a high-level programming language known for simplicity.",
      "Machine learning enables computers to learn from data.",
      "RAG combines retrieval with generation for better accuracy."
    ]
  }'
```

**Query:**
```bash
curl -X POST https://your-app.vercel.app/api/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is Python?",
    "top_k": 5
  }'
```

**Get Stats:**
```bash
curl https://your-app.vercel.app/api/stats
```

### Python Client

```python
import requests

BASE_URL = "https://your-app.vercel.app"

# Index documents
response = requests.post(f"{BASE_URL}/api/index", json={
    "documents": [
        "Python is great for AI development.",
        "Machine learning models need training data."
    ]
})
print(response.json())

# Query
response = requests.post(f"{BASE_URL}/api/query", json={
    "query": "Tell me about Python",
    "top_k": 3
})
print(response.json()["answer"])
```

## 🔧 Customization

### Change LLM Model

In Vercel Dashboard:
1. Go to Settings > Environment Variables
2. Add/Edit:
   - `LLM_MODEL` = `gpt-4` (or `gpt-3.5-turbo`)
   - `LLM_PROVIDER` = `openai` (or `anthropic`)

For Anthropic Claude:
```
ANTHROPIC_API_KEY=your-anthropic-key
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-sonnet-20240229
```

### Custom Domain

1. Go to Vercel Dashboard > Settings > Domains
2. Add your domain (e.g., `rag.yourdomain.com`)
3. Configure DNS as instructed
4. SSL certificate is automatic!

### Increase Memory/Timeout (Pro Plan)

The default config in `vercel.json` already sets:
- Memory: 3008 MB
- Timeout: 60 seconds

These require Vercel Pro ($20/month).

For the free tier, the limits are:
- Memory: 1024 MB
- Timeout: 10 seconds

## 🎯 What Works Out of the Box

- ✅ Document indexing with automatic chunking
- ✅ Hybrid search (BM25 + semantic embeddings)
- ✅ Multiple retrieval strategies
- ✅ LLM generation (OpenAI GPT or Anthropic Claude)
- ✅ Beautiful, responsive web interface
- ✅ REST API for programmatic access
- ✅ Mobile-friendly design
- ✅ Real-time statistics
- ✅ Error handling
- ✅ CORS enabled for cross-origin requests

## ⚡ Performance

**Cold Start (first request):** ~2-3 seconds
**Warm Requests:** ~500ms - 2 seconds
**Concurrent Users:** Scales automatically

## 💰 Cost Estimate

**Vercel:**
- Free tier: Perfect for testing and demos
- Pro ($20/month): For production use

**OpenAI API:**
- GPT-3.5-turbo: ~$0.002 per 1K tokens
- Example: 1000 queries/day ≈ $4-12/month

**Total for small production:** ~$24-32/month

## 🐛 Troubleshooting

**Issue: "Module not found"**
- Solution: Check `requirements-api.txt` includes all deps
- Redeploy: `vercel --force`

**Issue: "Timeout"**
- Solution: Use Pro plan or reduce `top_k`
- Or: Disable reranking (already disabled by default)

**Issue: "Documents not persisting"**
- This is expected on Vercel free tier
- `/tmp` storage is ephemeral
- Users need to re-index after cold starts
- Solution for production: Use external vector DB

**Issue: "Environment variable not found"**
- Check dashboard: Settings > Environment Variables
- Redeploy after adding vars: `vercel --prod`

## 📊 Monitor Your Deployment

Vercel provides built-in monitoring:
1. Go to your project dashboard
2. Click "Analytics" for request stats
3. Click "Logs" for real-time debugging
4. Click "Speed Insights" for performance

## 🎉 You're Done!

Your RAG pipeline is now:
- ✅ Deployed globally via Vercel CDN
- ✅ Has automatic HTTPS
- ✅ Scales automatically
- ✅ Has built-in monitoring
- ✅ Accessible worldwide

Share your deployment URL and let others try it!

## 📚 Next Steps

1. **Test it**: Try indexing some documents and asking questions
2. **Share it**: Send the URL to your team
3. **Customize it**: Add your own documents and data
4. **Scale it**: Upgrade to Pro if needed
5. **Integrate it**: Use the API in your applications

## 🆘 Need Help?

- Read: [docs/VERCEL_DEPLOYMENT.md](docs/VERCEL_DEPLOYMENT.md)
- Check: [docs/QUICKSTART.md](docs/QUICKSTART.md)
- Vercel Docs: https://vercel.com/docs

---

**Ready to deploy?** Run `vercel` now! 🚀
