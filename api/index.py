"""
Vercel Serverless API for RAG Pipeline
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
CORS(app)

# Global RAG instance (will be initialized lazily)
_rag_instance = None


def get_rag_instance():
    """Get or create RAG instance"""
    global _rag_instance

    if _rag_instance is None:
        from rag_pipeline import RAGPipeline

        # Initialize with minimal configuration for serverless
        _rag_instance = RAGPipeline(
            vector_db_type="faiss",
            retrieval_mode="hybrid",
            use_reranker=False,  # Disable for faster cold starts
            llm_provider=os.getenv("LLM_PROVIDER", "openai"),
            verbose=False
        )

        # Load pre-indexed documents if available
        import pickle
        docs_path = "/tmp/rag_documents.pkl"
        if os.path.exists(docs_path):
            with open(docs_path, "rb") as f:
                data = pickle.load(f)
                _rag_instance.documents = data.get("documents", [])
                _rag_instance.metadata = data.get("metadata", [])

    return _rag_instance


@app.route('/')
def index():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "service": "RAG Pipeline API",
        "version": "1.0.0",
        "endpoints": {
            "POST /api/query": "Query the RAG pipeline",
            "POST /api/index": "Index documents",
            "GET /api/health": "Health check",
            "GET /api/stats": "Get pipeline statistics"
        }
    })


@app.route('/api/health')
def health():
    """Health check"""
    return jsonify({"status": "healthy"})


@app.route('/api/query', methods=['POST'])
def query():
    """Query the RAG pipeline"""
    try:
        data = request.get_json()

        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' in request body"
            }), 400

        user_query = data['query']
        top_k = data.get('top_k', 5)
        return_context = data.get('return_context', True)

        # Get RAG instance
        rag = get_rag_instance()

        # Check if documents are indexed
        if not rag.documents:
            return jsonify({
                "error": "No documents indexed. Please index documents first using /api/index"
            }), 400

        # Query the pipeline
        result = rag.query(
            user_query,
            top_k=top_k,
            return_context=return_context
        )

        # Format response
        if return_context:
            response = {
                "answer": result["answer"],
                "documents": [
                    {
                        "text": doc[:200] + "..." if len(doc) > 200 else doc,
                        "score": float(score),
                        "metadata": meta
                    }
                    for doc, score, meta in result["documents"][:3]
                ],
                "tokens_used": result.get("tokens_used")
            }
        else:
            response = {
                "answer": result
            }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "error": str(e),
            "type": type(e).__name__
        }), 500


@app.route('/api/index', methods=['POST'])
def index_documents():
    """Index documents into the RAG pipeline"""
    try:
        data = request.get_json()

        if not data or 'documents' not in data:
            return jsonify({
                "error": "Missing 'documents' in request body"
            }), 400

        documents = data['documents']
        metadata = data.get('metadata', None)
        chunk = data.get('chunk', True)

        # Get RAG instance
        rag = get_rag_instance()

        # Index documents
        rag.index_documents(
            documents,
            metadata=metadata,
            chunk=chunk
        )

        # Save to persistent storage (Vercel has /tmp)
        import pickle
        with open("/tmp/rag_documents.pkl", "wb") as f:
            pickle.dump({
                "documents": rag.documents,
                "metadata": rag.metadata
            }, f)

        return jsonify({
            "status": "success",
            "indexed": len(documents),
            "total_chunks": len(rag.documents)
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "type": type(e).__name__
        }), 500


@app.route('/api/stats')
def stats():
    """Get pipeline statistics"""
    try:
        rag = get_rag_instance()

        return jsonify({
            "documents_indexed": len(rag.documents),
            "retrieval_mode": rag.retrieval_mode,
            "vector_db_type": rag.vector_db_type,
            "reranker_enabled": rag.use_reranker
        })

    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 500


# For Vercel serverless deployment
def handler(request):
    """Vercel serverless handler"""
    with app.request_context(request.environ):
        return app.full_dispatch_request()
