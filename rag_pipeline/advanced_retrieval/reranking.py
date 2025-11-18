"""Reranking with Cross-Encoders and ColBERT"""
from typing import List, Dict, Any, Tuple, Optional
from abc import ABC, abstractmethod
import numpy as np
from sentence_transformers import CrossEncoder


class Reranker(ABC):
    """Abstract base class for rerankers"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[Tuple[str, float, Dict[str, Any]]],
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Rerank search results

        Args:
            query: Search query
            results: List of (document, score, metadata) tuples
            top_k: Number of top results to return

        Returns:
            Reranked list of (document, score, metadata) tuples
        """
        pass


class CrossEncoderReranker(Reranker):
    """
    Reranker using Cross-Encoder models

    Cross-encoders jointly encode query and document for more accurate relevance scoring
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize cross-encoder reranker

        Args:
            model_name: Name of the cross-encoder model
            device: Device to run on ('cuda', 'cpu', or None for auto)

        Popular models:
            - cross-encoder/ms-marco-MiniLM-L-6-v2 (fast, good quality)
            - cross-encoder/ms-marco-MiniLM-L-12-v2 (better quality, slower)
            - cross-encoder/ms-marco-electra-base (high quality)
        """
        self.model_name = model_name
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        query: str,
        results: List[Tuple[str, float, Dict[str, Any]]],
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Rerank results using cross-encoder

        Args:
            query: Search query
            results: List of (document, score, metadata) tuples
            top_k: Number of top results to return

        Returns:
            Reranked list
        """
        if not results:
            return []

        # Prepare pairs for cross-encoder
        pairs = [(query, doc) for doc, _, _ in results]

        # Get cross-encoder scores
        ce_scores = self.model.predict(pairs)

        # Combine with original results
        reranked = [
            (doc, float(ce_score), metadata)
            for (doc, _, metadata), ce_score in zip(results, ce_scores)
        ]

        # Sort by cross-encoder score
        reranked.sort(key=lambda x: x[1], reverse=True)

        # Return top-k if specified
        if top_k:
            reranked = reranked[:top_k]

        return reranked

    def score_pairs(self, pairs: List[Tuple[str, str]]) -> List[float]:
        """
        Score query-document pairs directly

        Args:
            pairs: List of (query, document) tuples

        Returns:
            List of relevance scores
        """
        scores = self.model.predict(pairs)
        return scores.tolist() if hasattr(scores, 'tolist') else list(scores)


class ColBERTReranker(Reranker):
    """
    ColBERT-style reranker

    Uses late interaction between query and document token embeddings
    This is a simplified implementation - for full ColBERT, use the official library
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize ColBERT-style reranker

        Args:
            model_name: Sentence transformer model
            device: Device to run on
        """
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def _maxsim(
        self,
        query_embeddings: np.ndarray,
        doc_embeddings: np.ndarray
    ) -> float:
        """
        Calculate MaxSim score between query and document

        MaxSim = sum of max similarities for each query token
        """
        # Compute cosine similarities between all query and doc tokens
        similarities = np.dot(query_embeddings, doc_embeddings.T)

        # For each query token, find max similarity with any doc token
        max_sims = np.max(similarities, axis=1)

        # Sum and normalize
        score = np.sum(max_sims)

        return float(score)

    def rerank(
        self,
        query: str,
        results: List[Tuple[str, float, Dict[str, Any]]],
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Rerank using ColBERT-style late interaction

        Args:
            query: Search query
            results: List of (document, score, metadata) tuples
            top_k: Number of top results to return

        Returns:
            Reranked list
        """
        if not results:
            return []

        # Tokenize and encode query
        query_tokens = query.split()[:32]  # Limit tokens
        query_embeddings = self.model.encode(
            query_tokens,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # Score each document
        scores = []
        for doc, _, metadata in results:
            # Tokenize and encode document
            doc_tokens = doc.split()[:256]  # Limit tokens
            doc_embeddings = self.model.encode(
                doc_tokens,
                show_progress_bar=False,
                convert_to_numpy=True
            )

            # Calculate MaxSim score
            score = self._maxsim(query_embeddings, doc_embeddings)
            scores.append(score)

        # Create reranked results
        reranked = [
            (doc, score, metadata)
            for (doc, _, metadata), score in zip(results, scores)
        ]

        # Sort by score
        reranked.sort(key=lambda x: x[1], reverse=True)

        # Return top-k if specified
        if top_k:
            reranked = reranked[:top_k]

        return reranked


class HybridReranker(Reranker):
    """
    Hybrid reranker combining multiple reranking strategies
    """

    def __init__(
        self,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        cross_encoder_weight: float = 0.7,
        original_score_weight: float = 0.3
    ):
        """
        Initialize hybrid reranker

        Args:
            cross_encoder_model: Cross-encoder model to use
            cross_encoder_weight: Weight for cross-encoder scores
            original_score_weight: Weight for original retrieval scores
        """
        self.cross_encoder = CrossEncoderReranker(cross_encoder_model)
        self.ce_weight = cross_encoder_weight
        self.orig_weight = original_score_weight

    def rerank(
        self,
        query: str,
        results: List[Tuple[str, float, Dict[str, Any]]],
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Rerank using hybrid approach

        Combines original retrieval scores with cross-encoder scores

        Args:
            query: Search query
            results: List of (document, score, metadata) tuples
            top_k: Number of top results to return

        Returns:
            Reranked list
        """
        if not results:
            return []

        # Get cross-encoder scores
        ce_results = self.cross_encoder.rerank(query, results, top_k=None)

        # Normalize scores to [0, 1]
        def normalize_scores(items: List[Tuple[str, float, Dict[str, Any]]]) -> Dict[str, float]:
            if not items:
                return {}

            scores = [score for _, score, _ in items]
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score

            if score_range == 0:
                return {doc: 1.0 for doc, _, _ in items}

            return {
                doc: (score - min_score) / score_range
                for doc, score, _ in items
            }

        # Normalize both sets of scores
        orig_scores = normalize_scores(results)
        ce_scores = normalize_scores(ce_results)

        # Combine scores
        combined = []
        for doc, _, metadata in results:
            orig_score = orig_scores.get(doc, 0)
            ce_score = ce_scores.get(doc, 0)
            final_score = (
                self.orig_weight * orig_score +
                self.ce_weight * ce_score
            )
            combined.append((doc, final_score, metadata))

        # Sort by combined score
        combined.sort(key=lambda x: x[1], reverse=True)

        # Return top-k if specified
        if top_k:
            combined = combined[:top_k]

        return combined


class ReciprocalRankFusionReranker(Reranker):
    """
    Reranker using Reciprocal Rank Fusion (RRF)

    Useful for combining results from multiple retrievers
    """

    def __init__(self, k: int = 60):
        """
        Initialize RRF reranker

        Args:
            k: RRF constant (typically 60)
        """
        self.k = k

    def rerank_multiple(
        self,
        query: str,
        result_lists: List[List[Tuple[str, float, Dict[str, Any]]]],
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Rerank and fuse multiple result lists

        Args:
            query: Search query
            result_lists: List of result lists from different retrievers
            top_k: Number of top results to return

        Returns:
            Fused and reranked list
        """
        # Calculate RRF scores
        rrf_scores: Dict[str, float] = {}
        doc_metadata: Dict[str, Dict[str, Any]] = {}

        for result_list in result_lists:
            for rank, (doc, _, metadata) in enumerate(result_list, start=1):
                rrf_score = 1.0 / (self.k + rank)
                rrf_scores[doc] = rrf_scores.get(doc, 0) + rrf_score
                if doc not in doc_metadata:
                    doc_metadata[doc] = metadata

        # Sort by RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        # Create result list
        reranked = [
            (doc, score, doc_metadata[doc])
            for doc, score in sorted_docs
        ]

        # Return top-k if specified
        if top_k:
            reranked = reranked[:top_k]

        return reranked

    def rerank(
        self,
        query: str,
        results: List[Tuple[str, float, Dict[str, Any]]],
        top_k: Optional[int] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Rerank a single result list (just returns sorted by original score)

        For RRF, use rerank_multiple with multiple result lists

        Args:
            query: Search query
            results: List of (document, score, metadata) tuples
            top_k: Number of top results to return

        Returns:
            Sorted list
        """
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)

        if top_k:
            sorted_results = sorted_results[:top_k]

        return sorted_results
