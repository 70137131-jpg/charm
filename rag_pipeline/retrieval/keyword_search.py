"""Keyword search implementations: TF-IDF and BM25"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class KeywordSearch:
    """Base class for keyword-based search"""

    def __init__(self, remove_stopwords: bool = True):
        self.remove_stopwords = remove_stopwords
        self.stop_words = set(stopwords.words('english')) if remove_stopwords else set()
        self.documents: List[str] = []
        self.metadata: List[Dict[str, Any]] = []

    def preprocess(self, text: str) -> List[str]:
        """Tokenize and optionally remove stopwords"""
        tokens = word_tokenize(text.lower())
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words and t.isalnum()]
        return tokens

    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """Add documents to the index"""
        self.documents.extend(documents)
        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for documents. To be implemented by subclasses"""
        raise NotImplementedError


class TfidfSearch(KeywordSearch):
    """TF-IDF based keyword search"""

    def __init__(self, remove_stopwords: bool = True, max_features: Optional[int] = None):
        super().__init__(remove_stopwords)
        self.max_features = max_features
        self.vectorizer = None
        self.document_vectors = None

    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """Add documents and build TF-IDF index"""
        super().add_documents(documents, metadata)
        self._build_index()

    def _build_index(self):
        """Build TF-IDF index"""
        stop_words_list = list(self.stop_words) if self.remove_stopwords else None

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            stop_words=stop_words_list,
            lowercase=True,
            token_pattern=r'\b\w+\b'
        )

        self.document_vectors = self.vectorizer.fit_transform(self.documents)

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search using TF-IDF similarity

        Returns:
            List of (document, score, metadata) tuples
        """
        if not self.documents:
            return []

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.document_vectors)[0]

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = [
            (self.documents[idx], float(similarities[idx]), self.metadata[idx])
            for idx in top_indices
            if similarities[idx] > 0
        ]

        return results


class BM25Search(KeywordSearch):
    """BM25 based keyword search"""

    def __init__(self, remove_stopwords: bool = True, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 search

        Args:
            remove_stopwords: Whether to remove stopwords
            k1: BM25 parameter controlling term frequency saturation
            b: BM25 parameter controlling length normalization
        """
        super().__init__(remove_stopwords)
        self.k1 = k1
        self.b = b
        self.bm25 = None
        self.tokenized_documents = []

    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None
    ):
        """Add documents and build BM25 index"""
        super().add_documents(documents, metadata)
        self._build_index()

    def _build_index(self):
        """Build BM25 index"""
        self.tokenized_documents = [
            self.preprocess(doc) for doc in self.documents
        ]
        self.bm25 = BM25Okapi(self.tokenized_documents)

    def search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Search using BM25 ranking

        Returns:
            List of (document, score, metadata) tuples
        """
        if not self.documents:
            return []

        tokenized_query = self.preprocess(query)
        scores = self.bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = [
            (self.documents[idx], float(scores[idx]), self.metadata[idx])
            for idx in top_indices
            if scores[idx] > 0
        ]

        return results

    def get_top_n(self, query: str, n: int = 5) -> List[str]:
        """Get top n documents for a query (convenience method)"""
        results = self.search(query, top_k=n)
        return [doc for doc, _, _ in results]
