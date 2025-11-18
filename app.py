"""
Flask Application for RAG Pipeline - Local Development Server

This is a standalone Flask application that can run locally for development and testing.
It provides the same REST API as the Vercel deployment but optimized for local use.

Usage:
    python app.py                    # Run with default settings
    python app.py --debug            # Run in debug mode
    python app.py --port 5001        # Run on custom port

    Or use Flask CLI:
    flask run                        # Run on default port 5000
    flask run --port 8080 --debug    # Run on port 8080 in debug mode
"""
import os
import sys
import pickle
import argparse
from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['JSON_SORT_KEYS'] = False
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = True

# Storage configuration
STORAGE_DIR = Path(__file__).parent / "data"
STORAGE_DIR.mkdir(exist_ok=True)
DOCUMENTS_PATH = STORAGE_DIR / "rag_documents.pkl"

# Global RAG instance (lazy initialization)
_rag_instance = None


def get_rag_instance():
    """Get or create RAG instance with configuration from environment"""
    global _rag_instance

    if _rag_instance is None:
        from rag_pipeline import RAGPipeline

        # Get configuration from environment variables
        vector_db_type = os.getenv("VECTOR_DB_TYPE", "faiss")
        retrieval_mode = os.getenv("RETRIEVAL_MODE", "hybrid")
        use_reranker = os.getenv("USE_RERANKER", "false").lower() == "true"
        llm_provider = os.getenv("LLM_PROVIDER", "openai")
        verbose = os.getenv("VERBOSE", "true").lower() == "true"

        print(f"Initializing RAG Pipeline...")
        print(f"  Vector DB: {vector_db_type}")
        print(f"  Retrieval Mode: {retrieval_mode}")
        print(f"  Reranker: {use_reranker}")
        print(f"  LLM Provider: {llm_provider}")

        _rag_instance = RAGPipeline(
            vector_db_type=vector_db_type,
            retrieval_mode=retrieval_mode,
            use_reranker=use_reranker,
            llm_provider=llm_provider,
            verbose=verbose
        )

        # Load pre-indexed documents if available
        if DOCUMENTS_PATH.exists():
            try:
                with open(DOCUMENTS_PATH, "rb") as f:
                    data = pickle.load(f)
                    _rag_instance.documents = data.get("documents", [])
                    _rag_instance.metadata = data.get("metadata", [])
                print(f"  Loaded {len(_rag_instance.documents)} documents from storage")
            except Exception as e:
                print(f"  Warning: Could not load documents: {e}")

    return _rag_instance


def save_documents(rag):
    """Save indexed documents to persistent storage"""
    try:
        with open(DOCUMENTS_PATH, "wb") as f:
            pickle.dump({
                "documents": rag.documents,
                "metadata": rag.metadata
            }, f)
        return True
    except Exception as e:
        print(f"Error saving documents: {e}")
        return False


# =============================================================================
# API ROUTES
# =============================================================================

@app.route('/')
def index():
    """Root endpoint - API information"""
    return jsonify({
        "status": "ok",
        "service": "RAG Pipeline API (Local Development)",
        "version": "1.0.0",
        "description": "Retrieval-Augmented Generation Pipeline with Flask",
        "endpoints": {
            "GET /": "API information",
            "GET /api/health": "Health check",
            "POST /api/index": "Index documents",
            "POST /api/query": "Query the RAG pipeline",
            "GET /api/stats": "Get pipeline statistics",
            "POST /api/reset": "Reset the pipeline (clear all documents)",
            "GET /api/config": "Get current configuration"
        },
        "documentation": "https://github.com/yourusername/charm"
    })


@app.route('/api/health')
def health():
    """Health check endpoint"""
    try:
        rag = get_rag_instance()
        return jsonify({
            "status": "healthy",
            "documents_indexed": len(rag.documents),
            "storage_path": str(DOCUMENTS_PATH)
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500


@app.route('/api/config')
def config():
    """Get current RAG pipeline configuration"""
    try:
        rag = get_rag_instance()
        return jsonify({
            "vector_db_type": rag.vector_db_type,
            "retrieval_mode": rag.retrieval_mode,
            "use_reranker": rag.use_reranker,
            "llm_provider": rag.llm_provider,
            "documents_indexed": len(rag.documents),
            "storage_directory": str(STORAGE_DIR)
        })
    except Exception as e:
        return jsonify({
            "error": str(e),
            "type": type(e).__name__
        }), 500


@app.route('/api/index', methods=['POST'])
def index_documents():
    """
    Index documents into the RAG pipeline

    Request body:
        {
            "documents": ["text1", "text2", ...],  # Required: list of documents
            "metadata": [...],                      # Optional: metadata for each document
            "chunk": true                           # Optional: whether to chunk documents (default: true)
        }
    """
    try:
        data = request.get_json()

        if not data or 'documents' not in data:
            return jsonify({
                "error": "Missing 'documents' in request body",
                "example": {
                    "documents": ["Document text here"],
                    "metadata": [{"source": "file.txt"}],
                    "chunk": True
                }
            }), 400

        documents = data['documents']
        metadata = data.get('metadata', None)
        chunk = data.get('chunk', True)

        if not isinstance(documents, list):
            return jsonify({
                "error": "'documents' must be a list of strings"
            }), 400

        if len(documents) == 0:
            return jsonify({
                "error": "At least one document is required"
            }), 400

        # Get RAG instance
        rag = get_rag_instance()

        # Index documents
        rag.index_documents(
            documents,
            metadata=metadata,
            chunk=chunk
        )

        # Save to persistent storage
        save_success = save_documents(rag)

        return jsonify({
            "status": "success",
            "indexed": len(documents),
            "total_chunks": len(rag.documents),
            "saved_to_disk": save_success,
            "storage_path": str(DOCUMENTS_PATH) if save_success else None
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "type": type(e).__name__
        }), 500


@app.route('/api/query', methods=['POST'])
def query():
    """
    Query the RAG pipeline

    Request body:
        {
            "query": "your question here",  # Required: the query string
            "top_k": 5,                      # Optional: number of documents to retrieve (default: 5)
            "return_context": true           # Optional: include source documents (default: true)
        }
    """
    try:
        data = request.get_json()

        if not data or 'query' not in data:
            return jsonify({
                "error": "Missing 'query' in request body",
                "example": {
                    "query": "What is machine learning?",
                    "top_k": 5,
                    "return_context": True
                }
            }), 400

        user_query = data['query']
        top_k = data.get('top_k', 5)
        return_context = data.get('return_context', True)

        if not isinstance(user_query, str) or len(user_query.strip()) == 0:
            return jsonify({
                "error": "Query must be a non-empty string"
            }), 400

        # Get RAG instance
        rag = get_rag_instance()

        # Check if documents are indexed
        if not rag.documents:
            return jsonify({
                "error": "No documents indexed. Please index documents first using POST /api/index",
                "hint": "Send a POST request to /api/index with documents in the request body"
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
                "query": user_query,
                "documents": [
                    {
                        "text": doc[:300] + "..." if len(doc) > 300 else doc,
                        "full_text": doc,
                        "score": float(score),
                        "metadata": meta
                    }
                    for doc, score, meta in result["documents"][:top_k]
                ],
                "tokens_used": result.get("tokens_used"),
                "num_documents_retrieved": len(result["documents"])
            }
        else:
            response = {
                "answer": result,
                "query": user_query
            }

        return jsonify(response)

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
            "reranker_enabled": rag.use_reranker,
            "llm_provider": rag.llm_provider,
            "storage_path": str(DOCUMENTS_PATH),
            "storage_exists": DOCUMENTS_PATH.exists()
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "type": type(e).__name__
        }), 500


@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset the RAG pipeline (clear all indexed documents)"""
    try:
        global _rag_instance

        # Clear the instance
        if _rag_instance is not None:
            _rag_instance.documents = []
            _rag_instance.metadata = []

        # Remove storage file
        if DOCUMENTS_PATH.exists():
            DOCUMENTS_PATH.unlink()

        return jsonify({
            "status": "success",
            "message": "RAG pipeline reset successfully",
            "documents_indexed": 0
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "type": type(e).__name__
        }), 500


# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "path": request.path,
        "available_endpoints": [
            "GET /",
            "GET /api/health",
            "POST /api/index",
            "POST /api/query",
            "GET /api/stats",
            "POST /api/reset",
            "GET /api/config"
        ]
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors"""
    return jsonify({
        "error": "Method not allowed",
        "path": request.path,
        "method": request.method
    }), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "error": "Internal server error",
        "message": str(error)
    }), 500


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run the Flask development server"""
    parser = argparse.ArgumentParser(description='RAG Pipeline Flask Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()

    print("=" * 80)
    print("RAG Pipeline Flask Server")
    print("=" * 80)
    print(f"Storage directory: {STORAGE_DIR}")
    print(f"Documents file: {DOCUMENTS_PATH}")
    print(f"Server starting on http://{args.host}:{args.port}")
    print(f"Debug mode: {args.debug}")
    print("=" * 80)
    print("\nAvailable endpoints:")
    print("  GET  /                - API information")
    print("  GET  /api/health      - Health check")
    print("  GET  /api/config      - Get configuration")
    print("  POST /api/index       - Index documents")
    print("  POST /api/query       - Query the RAG pipeline")
    print("  GET  /api/stats       - Get statistics")
    print("  POST /api/reset       - Reset pipeline")
    print("\nPress CTRL+C to stop the server")
    print("=" * 80)
    print()

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )


if __name__ == '__main__':
    main()
