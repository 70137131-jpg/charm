"""Metrics for evaluating retrieval quality"""
from typing import List, Dict, Any, Set, Tuple
import numpy as np


class RetrievalMetrics:
    """Metrics for evaluating document retrieval"""

    @staticmethod
    def precision_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """
        Precision@k: Proportion of retrieved documents that are relevant

        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Number of top results to consider

        Returns:
            Precision score (0-1)
        """
        if k == 0 or not retrieved:
            return 0.0

        retrieved_at_k = set(retrieved[:k])
        relevant_retrieved = retrieved_at_k & relevant

        return len(relevant_retrieved) / k

    @staticmethod
    def recall_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """
        Recall@k: Proportion of relevant documents that are retrieved

        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Number of top results to consider

        Returns:
            Recall score (0-1)
        """
        if not relevant or not retrieved:
            return 0.0

        retrieved_at_k = set(retrieved[:k])
        relevant_retrieved = retrieved_at_k & relevant

        return len(relevant_retrieved) / len(relevant)

    @staticmethod
    def f1_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """
        F1@k: Harmonic mean of precision and recall

        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Number of top results to consider

        Returns:
            F1 score (0-1)
        """
        precision = RetrievalMetrics.precision_at_k(retrieved, relevant, k)
        recall = RetrievalMetrics.recall_at_k(retrieved, relevant, k)

        if precision + recall == 0:
            return 0.0

        return 2 * (precision * recall) / (precision + recall)

    @staticmethod
    def average_precision(
        retrieved: List[str],
        relevant: Set[str]
    ) -> float:
        """
        Average Precision: Mean of precision values at each relevant document

        Args:
            retrieved: List of retrieved document IDs (in order)
            relevant: Set of relevant document IDs

        Returns:
            Average precision score (0-1)
        """
        if not relevant or not retrieved:
            return 0.0

        precisions = []
        num_relevant = 0

        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                num_relevant += 1
                precisions.append(num_relevant / (i + 1))

        if not precisions:
            return 0.0

        return sum(precisions) / len(relevant)

    @staticmethod
    def mean_average_precision(
        results: List[Tuple[List[str], Set[str]]]
    ) -> float:
        """
        Mean Average Precision (MAP): Mean of AP across multiple queries

        Args:
            results: List of (retrieved, relevant) tuples for each query

        Returns:
            MAP score (0-1)
        """
        if not results:
            return 0.0

        aps = [
            RetrievalMetrics.average_precision(retrieved, relevant)
            for retrieved, relevant in results
        ]

        return np.mean(aps)

    @staticmethod
    def ndcg_at_k(
        retrieved: List[str],
        relevance_scores: Dict[str, float],
        k: int
    ) -> float:
        """
        Normalized Discounted Cumulative Gain@k

        Args:
            retrieved: List of retrieved document IDs (in order)
            relevance_scores: Dict mapping document IDs to relevance scores
            k: Number of top results to consider

        Returns:
            NDCG score (0-1)
        """
        if k == 0 or not retrieved:
            return 0.0

        # Calculate DCG
        dcg = 0.0
        for i, doc_id in enumerate(retrieved[:k]):
            relevance = relevance_scores.get(doc_id, 0.0)
            dcg += relevance / np.log2(i + 2)  # i+2 because i starts at 0

        # Calculate ideal DCG
        ideal_scores = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum(score / np.log2(i + 2) for i, score in enumerate(ideal_scores))

        if idcg == 0:
            return 0.0

        return dcg / idcg

    @staticmethod
    def reciprocal_rank(
        retrieved: List[str],
        relevant: Set[str]
    ) -> float:
        """
        Reciprocal Rank: 1 / rank of first relevant document

        Args:
            retrieved: List of retrieved document IDs (in order)
            relevant: Set of relevant document IDs

        Returns:
            RR score (0-1)
        """
        for i, doc_id in enumerate(retrieved):
            if doc_id in relevant:
                return 1.0 / (i + 1)

        return 0.0

    @staticmethod
    def mean_reciprocal_rank(
        results: List[Tuple[List[str], Set[str]]]
    ) -> float:
        """
        Mean Reciprocal Rank (MRR): Mean of RR across multiple queries

        Args:
            results: List of (retrieved, relevant) tuples for each query

        Returns:
            MRR score (0-1)
        """
        if not results:
            return 0.0

        rrs = [
            RetrievalMetrics.reciprocal_rank(retrieved, relevant)
            for retrieved, relevant in results
        ]

        return np.mean(rrs)

    @staticmethod
    def hit_rate_at_k(
        retrieved: List[str],
        relevant: Set[str],
        k: int
    ) -> float:
        """
        Hit Rate@k: Whether at least one relevant doc is in top k

        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            k: Number of top results to consider

        Returns:
            1.0 if hit, 0.0 otherwise
        """
        retrieved_at_k = set(retrieved[:k])
        return 1.0 if retrieved_at_k & relevant else 0.0

    @staticmethod
    def evaluate_retrieval(
        retrieved: List[str],
        relevant: Set[str],
        relevance_scores: Optional[Dict[str, float]] = None,
        k_values: List[int] = [1, 3, 5, 10]
    ) -> Dict[str, Any]:
        """
        Comprehensive retrieval evaluation

        Args:
            retrieved: List of retrieved document IDs
            relevant: Set of relevant document IDs
            relevance_scores: Optional dict of relevance scores for NDCG
            k_values: List of k values to evaluate

        Returns:
            Dictionary of metrics
        """
        metrics = {
            "num_retrieved": len(retrieved),
            "num_relevant": len(relevant),
            "average_precision": RetrievalMetrics.average_precision(retrieved, relevant),
            "reciprocal_rank": RetrievalMetrics.reciprocal_rank(retrieved, relevant),
        }

        # Metrics at different k values
        for k in k_values:
            if k <= len(retrieved):
                metrics[f"precision@{k}"] = RetrievalMetrics.precision_at_k(retrieved, relevant, k)
                metrics[f"recall@{k}"] = RetrievalMetrics.recall_at_k(retrieved, relevant, k)
                metrics[f"f1@{k}"] = RetrievalMetrics.f1_at_k(retrieved, relevant, k)
                metrics[f"hit_rate@{k}"] = RetrievalMetrics.hit_rate_at_k(retrieved, relevant, k)

                if relevance_scores:
                    metrics[f"ndcg@{k}"] = RetrievalMetrics.ndcg_at_k(retrieved, relevance_scores, k)

        return metrics
