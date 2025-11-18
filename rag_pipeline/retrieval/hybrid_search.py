"""Hybrid search combining keyword and semantic search"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from .keyword_search import BM25Search, TfidfSearch
from .semantic_search import SemanticSearch


class HybridSearch:
    """
    Hybrid search combining keyword-based (BM25) and semantic search

    Uses Reciprocal Rank Fusion (RRF) to combine results
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        keyword_method: str = "bm25",  # "bm25" or "tfidf"
        semantic_weight: float = 0.5,
        keyword_weight: float = 0.5,
        use_rrf: bool = True,
        rrf_k: int = 60
    ):
        """
        Initialize hybrid search

        Args:
            embedding_model: Model for semantic search
            keyword_method: Keyword search method ("bm25" or "tfidf")
            semantic_weight: Weight for semantic search (used if not using RRF)
            keyword_weight: Weight for keyword search (used if not using RRF)
            use_rrf: Whether to use Reciprocal Rank Fusion
            rrf_k: RRF parameter (typically 60)
        """
        self.semantic_search = SemanticSearch(model_name=embedding_model)

        if keyword_method == "bm25":
            self.keyword_search = BM25Search()
        elif keyword_method == "tfidf":
            self.keyword_search = TfidfSearch()
        else:
            raise ValueError(f"Unknown keyword method: {keyword_method}")

        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.use_rrf = use_rrf
        self.rrf_k = rrf_k

    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """Add documents to both search indices"""
        self.semantic_search.add_documents(documents, metadata)
        self.keyword_search.add_documents(documents, metadata)

    def _reciprocal_rank_fusion(
        self,
        keyword_results: List[Tuple[str, float, Dict[str, Any]]],
        semantic_results: List[Tuple[str, float, Dict[str, Any]]],
        top_k: int
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Combine results using Reciprocal Rank Fusion

        RRF score for document d: sum(1 / (k + rank_i(d)))
        where rank_i(d) is the rank of d in result list i
        """
        # Create document -> (score, metadata) mapping
        doc_scores: Dict[str, float] = {}
        doc_metadata: Dict[str, Dict[str, Any]] = {}

        # Process keyword results
        for rank, (doc, _, metadata) in enumerate(keyword_results, start=1):
            rrf_score = 1.0 / (self.rrf_k + rank)
            doc_scores[doc] = doc_scores.get(doc, 0) + rrf_score * self.keyword_weight
            doc_metadata[doc] = metadata

        # Process semantic results
        for rank, (doc, _, metadata) in enumerate(semantic_results, start=1):
            rrf_score = 1.0 / (self.rrf_k + rank)
            doc_scores[doc] = doc_scores.get(doc, 0) + rrf_score * self.semantic_weight
            if doc not in doc_metadata:
                doc_metadata[doc] = metadata

        # Sort by combined score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        # Return top-k results
        results = [
            (doc, score, doc_metadata[doc])
            for doc, score in sorted_docs[:top_k]
        ]

        return results

    def _weighted_fusion(
        self,
        keyword_results: List[Tuple[str, float, Dict[str, Any]]],
        semantic_results: List[Tuple[str, float, Dict[str, Any]]],
        top_k: int
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Combine results using weighted score fusion"""
        # Normalize scores to [0, 1]
        def normalize_scores(results: List[Tuple[str, float, Dict[str, Any]]]) -> Dict[str, float]:
            if not results:
                return {}
            scores = [score for _, score, _ in results]
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score

            if score_range == 0:
                return {doc: 1.0 for doc, _, _ in results}

            return {
                doc: (score - min_score) / score_range
                for doc, score, _ in results
            }

        keyword_scores = normalize_scores(keyword_results)
        semantic_scores = normalize_scores(semantic_results)

        # Combine scores
        all_docs = set(keyword_scores.keys()) | set(semantic_scores.keys())
        combined_scores = {}
        doc_metadata = {}

        for doc in all_docs:
            kw_score = keyword_scores.get(doc, 0)
            sem_score = semantic_scores.get(doc, 0)
            combined_scores[doc] = (
                kw_score * self.keyword_weight +
                sem_score * self.semantic_weight
            )

            # Get metadata from either source
            for d, _, meta in keyword_results + semantic_results:
                if d == doc:
                    doc_metadata[doc] = meta
                    break

        # Sort and return top-k
        sorted_docs = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

        results = [
            (doc, score, doc_metadata.get(doc, {}))
            for doc, score in sorted_docs[:top_k]
        ]

        return results

    def search(
        self,
        query: str,
        top_k: int = 5,
        retrieve_k: Optional[int] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Hybrid search combining keyword and semantic search

        Args:
            query: Search query
            top_k: Number of final results to return
            retrieve_k: Number of results to retrieve from each method before fusion
                       (defaults to top_k * 2)

        Returns:
            List of (document, score, metadata) tuples
        """
        if retrieve_k is None:
            retrieve_k = top_k * 2

        # Get results from both methods
        keyword_results = self.keyword_search.search(query, top_k=retrieve_k)
        semantic_results = self.semantic_search.search(query, top_k=retrieve_k)

        # Combine results
        if self.use_rrf:
            return self._reciprocal_rank_fusion(
                keyword_results,
                semantic_results,
                top_k
            )
        else:
            return self._weighted_fusion(
                keyword_results,
                semantic_results,
                top_k
            )

    def clear(self):
        """Clear all documents from both indices"""
        self.semantic_search.clear()
        self.keyword_search.documents = []
        self.keyword_search.metadata = []
        if hasattr(self.keyword_search, 'bm25'):
            self.keyword_search.bm25 = None
