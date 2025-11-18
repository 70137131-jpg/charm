"""Main RAG Pipeline - integrating all modules"""
from typing import List, Dict, Any, Optional, Tuple
import os

from .utils.config import Config
from .utils.logger import setup_logger

from .retrieval import HybridSearch, BM25Search, TfidfSearch
from .retrieval.semantic_search import SemanticSearch
from .retrieval.metadata_filter import MetadataFilter

from .advanced_retrieval import DocumentChunker, AdvancedChunker
from .advanced_retrieval.vector_stores import FAISSStore, ChromaStore, QdrantStore
from .advanced_retrieval.query_parser import QueryParser
from .advanced_retrieval.reranking import CrossEncoderReranker

from .generation import OpenAILLM, AnthropicLLM, LLMFactory
from .generation.prompt_engineering import PromptEngine
from .generation.hallucination_handler import HallucinationHandler

from .evaluation import RetrievalMetrics, GenerationMetrics


class RAGPipeline:
    """
    Complete RAG Pipeline integrating all modules

    Features:
    - Multiple retrieval methods (keyword, semantic, hybrid)
    - Multiple vector stores (FAISS, ChromaDB, Qdrant)
    - Document chunking
    - Query parsing and expansion
    - Reranking
    - LLM generation
    - Hallucination handling
    - Evaluation
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        vector_db_type: str = "faiss",
        retrieval_mode: str = "hybrid",
        use_reranker: bool = True,
        llm_provider: str = "openai",
        verbose: bool = False
    ):
        """
        Initialize RAG Pipeline

        Args:
            config: Configuration object
            vector_db_type: Type of vector database ('faiss', 'chromadb', 'qdrant')
            retrieval_mode: Retrieval mode ('semantic', 'keyword', 'hybrid')
            use_reranker: Whether to use reranking
            llm_provider: LLM provider ('openai', 'anthropic')
            verbose: Enable verbose logging
        """
        self.config = config or Config()
        self.verbose = verbose
        self.logger = setup_logger()

        if self.verbose:
            self.logger.info("Initializing RAG Pipeline...")

        # Initialize retrieval components
        self.retrieval_mode = retrieval_mode
        self._init_retrieval()

        # Initialize vector store
        self.vector_db_type = vector_db_type
        self._init_vector_store()

        # Initialize chunker
        self.chunker = DocumentChunker()

        # Initialize query parser
        self.query_parser = QueryParser()

        # Initialize reranker
        self.use_reranker = use_reranker
        if use_reranker:
            if self.verbose:
                self.logger.info(f"Loading reranker: {self.config.reranker_model}")
            self.reranker = CrossEncoderReranker(self.config.reranker_model)
        else:
            self.reranker = None

        # Initialize LLM
        if self.verbose:
            self.logger.info(f"Loading LLM: {llm_provider} - {self.config.llm_model}")

        self.llm = LLMFactory.create(
            provider=llm_provider,
            model=self.config.llm_model,
            api_key=self.config.openai_api_key if llm_provider == "openai" else self.config.anthropic_api_key
        )

        # Initialize prompt engine
        self.prompt_engine = PromptEngine()

        # Initialize hallucination handler
        self.hallucination_handler = HallucinationHandler(self.llm)

        # Document storage
        self.documents: List[str] = []
        self.metadata: List[Dict[str, Any]] = []

        if self.verbose:
            self.logger.info("RAG Pipeline initialized successfully")

    def _init_retrieval(self):
        """Initialize retrieval components"""
        if self.retrieval_mode == "hybrid":
            if self.verbose:
                self.logger.info(f"Initializing hybrid search with {self.config.embedding_model}")
            self.retriever = HybridSearch(
                embedding_model=self.config.embedding_model,
                keyword_method="bm25"
            )
        elif self.retrieval_mode == "semantic":
            if self.verbose:
                self.logger.info(f"Initializing semantic search with {self.config.embedding_model}")
            self.retriever = SemanticSearch(model_name=self.config.embedding_model)
        elif self.retrieval_mode == "keyword":
            if self.verbose:
                self.logger.info("Initializing BM25 search")
            self.retriever = BM25Search()
        else:
            raise ValueError(f"Unknown retrieval mode: {self.retrieval_mode}")

    def _init_vector_store(self):
        """Initialize vector store"""
        if self.vector_db_type == "faiss":
            if self.verbose:
                self.logger.info("Initializing FAISS vector store")
            self.vector_store = FAISSStore(
                embedding_model=self.config.embedding_model
            )
        elif self.vector_db_type == "chromadb":
            if self.verbose:
                self.logger.info("Initializing ChromaDB vector store")
            self.vector_store = ChromaStore(
                embedding_model=self.config.embedding_model,
                persist_directory=self.config.vector_db_path
            )
        elif self.vector_db_type == "qdrant":
            if self.verbose:
                self.logger.info("Initializing Qdrant vector store")
            self.vector_store = QdrantStore(
                embedding_model=self.config.embedding_model,
                path=self.config.vector_db_path
            )
        else:
            raise ValueError(f"Unknown vector DB type: {self.vector_db_type}")

    def index_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        chunk: bool = True,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None
    ):
        """
        Index documents for retrieval

        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
            chunk: Whether to chunk documents
            chunk_size: Custom chunk size (uses config default if None)
            chunk_overlap: Custom chunk overlap (uses config default if None)
        """
        if self.verbose:
            self.logger.info(f"Indexing {len(documents)} documents...")

        chunk_size = chunk_size or self.config.chunk_size
        chunk_overlap = chunk_overlap or self.config.chunk_overlap

        # Chunk documents if requested
        if chunk:
            chunked_docs = []
            chunked_metadata = []

            for i, doc in enumerate(documents):
                chunks = self.chunker.chunk_by_tokens(
                    doc,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )

                doc_meta = metadata[i] if metadata and i < len(metadata) else {}

                for j, chunk_text in enumerate(chunks):
                    chunked_docs.append(chunk_text)
                    chunk_meta = {
                        **doc_meta,
                        "chunk_id": j,
                        "source_doc": i
                    }
                    chunked_metadata.append(chunk_meta)

            documents = chunked_docs
            metadata = chunked_metadata

            if self.verbose:
                self.logger.info(f"Created {len(documents)} chunks")

        # Store documents
        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))

        # Add to retriever
        self.retriever.add_documents(documents, metadata)

        # Add to vector store
        self.vector_store.add_documents(documents, metadata)

        if self.verbose:
            self.logger.info("Indexing complete")

    def query(
        self,
        query: str,
        top_k: Optional[int] = None,
        use_reranking: Optional[bool] = None,
        filters: Optional[Dict[str, Any]] = None,
        return_context: bool = False,
        check_hallucination: bool = True
    ) -> str:
        """
        Query the RAG pipeline

        Args:
            query: User query
            top_k: Number of documents to retrieve
            use_reranking: Override default reranking setting
            filters: Metadata filters
            return_context: Whether to return context with answer
            check_hallucination: Whether to check for hallucinations

        Returns:
            Generated answer (or dict if return_context=True)
        """
        if self.verbose:
            self.logger.info(f"Processing query: {query}")

        top_k = top_k or self.config.top_k
        use_reranking = use_reranking if use_reranking is not None else self.use_reranker

        # Retrieve documents
        if self.verbose:
            self.logger.info(f"Retrieving top {top_k} documents...")

        retrieved_docs = self.retriever.search(
            query,
            top_k=top_k * 2 if use_reranking else top_k
        )

        if self.verbose:
            self.logger.info(f"Retrieved {len(retrieved_docs)} documents")

        # Apply filters if provided
        if filters:
            retrieved_docs = MetadataFilter.filter_results(retrieved_docs, filters)
            if self.verbose:
                self.logger.info(f"After filtering: {len(retrieved_docs)} documents")

        # Rerank if enabled
        if use_reranking and self.reranker and retrieved_docs:
            if self.verbose:
                self.logger.info("Reranking documents...")
            retrieved_docs = self.reranker.rerank(query, retrieved_docs, top_k=top_k)

        # Generate answer
        if self.verbose:
            self.logger.info("Generating answer...")

        prompt = self.prompt_engine.create_augmented_prompt(
            query,
            retrieved_docs,
            max_context_length=3000
        )

        response = self.llm.generate(
            prompt,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.max_tokens
        )

        answer = response.text

        # Check for hallucinations
        if check_hallucination:
            if self.verbose:
                self.logger.info("Checking for hallucinations...")

            context = "\n".join([doc for doc, _, _ in retrieved_docs])
            hallucination_check = self.hallucination_handler.check_factual_consistency(
                answer,
                context
            )

            if hallucination_check.is_hallucination:
                if self.verbose:
                    self.logger.warning(f"Hallucination detected: {hallucination_check.reasons}")

                # Handle hallucination
                answer = self.hallucination_handler.handle_hallucination(
                    answer,
                    context,
                    query,
                    strategy="hedge"
                )

        if return_context:
            return {
                "answer": answer,
                "documents": retrieved_docs,
                "tokens_used": response.tokens_used
            }

        return answer

    def evaluate(
        self,
        question: str,
        answer: str,
        reference: Optional[str] = None,
        retrieved_docs: Optional[List[Tuple[str, float, Dict[str, Any]]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate RAG performance

        Args:
            question: Original question
            answer: Generated answer
            reference: Reference answer
            retrieved_docs: Retrieved documents

        Returns:
            Dictionary of metrics
        """
        if retrieved_docs:
            context = "\n".join([doc for doc, _, _ in retrieved_docs])
        else:
            context = ""

        metrics = GenerationMetrics.evaluate_rag(
            question,
            answer,
            reference,
            context
        )

        return metrics

    def save(self, path: str):
        """Save the pipeline state"""
        os.makedirs(path, exist_ok=True)

        # Save vector store
        self.vector_store.save(os.path.join(path, "vector_store"))

        # Save documents and metadata
        import pickle
        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "metadata": self.metadata
            }, f)

        if self.verbose:
            self.logger.info(f"Pipeline saved to {path}")

    def load(self, path: str):
        """Load the pipeline state"""
        # Load vector store
        self.vector_store.load(os.path.join(path, "vector_store"))

        # Load documents and metadata
        import pickle
        with open(os.path.join(path, "documents.pkl"), "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.metadata = data["metadata"]

        # Re-index in retriever
        self.retriever.add_documents(self.documents, self.metadata)

        if self.verbose:
            self.logger.info(f"Pipeline loaded from {path}")
