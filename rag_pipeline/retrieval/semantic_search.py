"""Semantic search using embeddings"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class SemanticSearch:
    """Semantic search using embedding models"""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize semantic search

        Args:
            model_name: Name of the sentence transformer model
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.documents: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Add documents and compute their embeddings

        Args:
            documents: List of document texts
            metadata: Optional metadata for each document
        """
        self.documents.extend(documents)

        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))

        # Compute embeddings for new documents
        new_embeddings = self.model.encode(
            documents,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.0
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search for semantically similar documents

        Args:
            query: Search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            List of (document, score, metadata) tuples
        """
        if not self.documents:
            return []

        # Encode query
        query_embedding = self.model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # Calculate similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Filter by threshold and return results
        results = [
            (self.documents[idx], float(similarities[idx]), self.metadata[idx])
            for idx in top_indices
            if similarities[idx] >= similarity_threshold
        ]

        return results

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for arbitrary texts"""
        return self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True
        )

    def similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts"""
        embeddings = self.get_embeddings([text1, text2])
        return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])

    def clear(self):
        """Clear all documents and embeddings"""
        self.documents = []
        self.metadata = []
        self.embeddings = None
