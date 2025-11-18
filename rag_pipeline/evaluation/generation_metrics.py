"""Metrics for evaluating generation quality"""
from typing import List, Dict, Any, Optional
import re
import numpy as np

try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False

try:
    from bert_score import score as bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False


class GenerationMetrics:
    """Metrics for evaluating generated text"""

    @staticmethod
    def exact_match(prediction: str, reference: str) -> float:
        """
        Exact match score

        Args:
            prediction: Predicted answer
            reference: Reference answer

        Returns:
            1.0 if exact match, 0.0 otherwise
        """
        # Normalize
        pred_norm = prediction.strip().lower()
        ref_norm = reference.strip().lower()

        return 1.0 if pred_norm == ref_norm else 0.0

    @staticmethod
    def token_overlap(prediction: str, reference: str) -> Dict[str, float]:
        """
        Token-based overlap metrics

        Args:
            prediction: Predicted answer
            reference: Reference answer

        Returns:
            Dictionary with precision, recall, and F1
        """
        # Tokenize
        pred_tokens = set(re.findall(r'\b\w+\b', prediction.lower()))
        ref_tokens = set(re.findall(r'\b\w+\b', reference.lower()))

        if not ref_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        # Calculate metrics
        common = pred_tokens & ref_tokens

        precision = len(common) / len(pred_tokens) if pred_tokens else 0.0
        recall = len(common) / len(ref_tokens) if ref_tokens else 0.0

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1
        }

    @staticmethod
    def rouge_scores(
        prediction: str,
        reference: str,
        metrics: List[str] = ["rouge1", "rouge2", "rougeL"]
    ) -> Dict[str, float]:
        """
        ROUGE scores

        Args:
            prediction: Predicted answer
            reference: Reference answer
            metrics: List of ROUGE metrics to compute

        Returns:
            Dictionary of ROUGE scores
        """
        if not ROUGE_AVAILABLE:
            raise ImportError("rouge-score not available. Install with: pip install rouge-score")

        scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)
        scores = scorer.score(reference, prediction)

        return {
            metric: scores[metric].fmeasure
            for metric in metrics
        }

    @staticmethod
    def bert_score_metric(
        predictions: List[str],
        references: List[str],
        lang: str = "en"
    ) -> Dict[str, float]:
        """
        BERTScore

        Args:
            predictions: List of predicted answers
            references: List of reference answers
            lang: Language code

        Returns:
            Dictionary with precision, recall, and F1
        """
        if not BERT_SCORE_AVAILABLE:
            raise ImportError("bert-score not available. Install with: pip install bert-score")

        P, R, F1 = bert_score(predictions, references, lang=lang, verbose=False)

        return {
            "precision": float(P.mean()),
            "recall": float(R.mean()),
            "f1": float(F1.mean())
        }

    @staticmethod
    def answer_relevance(
        answer: str,
        question: str,
        embedding_model=None
    ) -> float:
        """
        Answer relevance: semantic similarity between answer and question

        Args:
            answer: Generated answer
            question: Original question
            embedding_model: Sentence transformer model

        Returns:
            Relevance score (0-1)
        """
        if embedding_model is None:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Get embeddings
        embeddings = embedding_model.encode([answer, question])

        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        return float(similarity)

    @staticmethod
    def faithfulness(
        answer: str,
        context: str,
        embedding_model=None
    ) -> float:
        """
        Faithfulness: whether answer is grounded in context

        Args:
            answer: Generated answer
            context: Source context
            embedding_model: Sentence transformer model

        Returns:
            Faithfulness score (0-1)
        """
        if embedding_model is None:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Extract claims from answer
        answer_sentences = re.split(r'[.!?]+', answer)
        answer_sentences = [s.strip() for s in answer_sentences if s.strip()]

        if not answer_sentences:
            return 1.0

        # Get embeddings
        answer_embeddings = embedding_model.encode(answer_sentences)
        context_embedding = embedding_model.encode([context])[0]

        # Calculate similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(answer_embeddings, [context_embedding])

        # Average similarity
        avg_similarity = float(np.mean(similarities))

        return avg_similarity

    @staticmethod
    def context_relevance(
        context: str,
        question: str,
        embedding_model=None
    ) -> float:
        """
        Context relevance: whether context is relevant to question

        Args:
            context: Retrieved context
            question: Original question
            embedding_model: Sentence transformer model

        Returns:
            Relevance score (0-1)
        """
        if embedding_model is None:
            from sentence_transformers import SentenceTransformer
            embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

        # Get embeddings
        embeddings = embedding_model.encode([context, question])

        # Calculate cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]

        return float(similarity)

    @staticmethod
    def evaluate_rag(
        question: str,
        answer: str,
        reference: Optional[str],
        context: str,
        embedding_model=None
    ) -> Dict[str, Any]:
        """
        Comprehensive RAG evaluation

        Args:
            question: Original question
            answer: Generated answer
            reference: Reference answer (optional)
            context: Retrieved context
            embedding_model: Sentence transformer model

        Returns:
            Dictionary of metrics
        """
        metrics = {
            "answer_relevance": GenerationMetrics.answer_relevance(answer, question, embedding_model),
            "faithfulness": GenerationMetrics.faithfulness(answer, context, embedding_model),
            "context_relevance": GenerationMetrics.context_relevance(context, question, embedding_model),
        }

        # Metrics requiring reference answer
        if reference:
            metrics["token_overlap"] = GenerationMetrics.token_overlap(answer, reference)
            metrics["exact_match"] = GenerationMetrics.exact_match(answer, reference)

            if ROUGE_AVAILABLE:
                try:
                    metrics["rouge"] = GenerationMetrics.rouge_scores(answer, reference)
                except Exception:
                    pass

        return metrics

    @staticmethod
    def answer_length_stats(answer: str) -> Dict[str, int]:
        """Get length statistics for answer"""
        words = re.findall(r'\b\w+\b', answer)
        sentences = re.split(r'[.!?]+', answer)
        sentences = [s for s in sentences if s.strip()]

        return {
            "num_characters": len(answer),
            "num_words": len(words),
            "num_sentences": len(sentences),
        }
