"""Vector database implementations using FAISS, ChromaDB, and Qdrant"""
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import numpy as np
import os
import pickle

# FAISS
import faiss

# ChromaDB
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False

# Qdrant
try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Distance, VectorParams, PointStruct
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

from sentence_transformers import SentenceTransformer


class VectorStore(ABC):
    """Abstract base class for vector stores"""

    @abstractmethod
    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ):
        """Add documents to the vector store"""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar documents"""
        pass

    @abstractmethod
    def delete(self, ids: List[str]):
        """Delete documents by ID"""
        pass

    @abstractmethod
    def save(self, path: str):
        """Save the vector store"""
        pass

    @abstractmethod
    def load(self, path: str):
        """Load the vector store"""
        pass


class FAISSStore(VectorStore):
    """
    FAISS-based vector store for efficient ANN search

    Uses IndexFlatIP for exact search or IndexIVFFlat for approximate search
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_gpu: bool = False,
        index_type: str = "flat"  # "flat" or "ivf"
    ):
        """
        Initialize FAISS store

        Args:
            embedding_model: Sentence transformer model
            use_gpu: Whether to use GPU for FAISS
            index_type: Type of FAISS index ("flat" for exact, "ivf" for approximate)
        """
        self.model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.use_gpu = use_gpu
        self.index_type = index_type

        # Initialize FAISS index
        if index_type == "flat":
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine similarity)
        elif index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
            self.index.nprobe = 10
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        if use_gpu and faiss.get_num_gpus() > 0:
            self.index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, self.index)

        self.documents: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.ids: List[str] = []

    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ):
        """Add documents to FAISS index"""
        # Generate embeddings
        embeddings = self.model.encode(
            documents,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # For cosine similarity
        )

        # Train index if using IVF and not yet trained
        if self.index_type == "ivf" and not self.index.is_trained:
            self.index.train(embeddings)

        # Add to index
        self.index.add(embeddings)

        # Store documents and metadata
        self.documents.extend(documents)

        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{}] * len(documents))

        if ids:
            self.ids.extend(ids)
        else:
            start_id = len(self.ids)
            self.ids.extend([f"doc_{start_id + i}" for i in range(len(documents))])

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar documents"""
        if len(self.documents) == 0:
            return []

        # Encode query
        query_embedding = self.model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Search
        distances, indices = self.index.search(query_embedding, min(top_k * 2, len(self.documents)))

        # Prepare results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                metadata = self.metadata[idx]

                # Apply metadata filtering if specified
                if filter_dict:
                    from ..retrieval.metadata_filter import MetadataFilter
                    if not MetadataFilter._matches_filters(metadata, filter_dict):
                        continue

                results.append((doc, float(dist), metadata))

                if len(results) >= top_k:
                    break

        return results

    def delete(self, ids: List[str]):
        """
        Delete documents by ID

        Note: FAISS doesn't support deletion directly, so we rebuild the index
        """
        # Find indices to keep
        indices_to_keep = [i for i, doc_id in enumerate(self.ids) if doc_id not in ids]

        if not indices_to_keep:
            # Clear everything
            self.__init__(
                embedding_model=self.model.model_name_or_path,
                use_gpu=self.use_gpu,
                index_type=self.index_type
            )
            return

        # Rebuild with remaining documents
        kept_docs = [self.documents[i] for i in indices_to_keep]
        kept_metadata = [self.metadata[i] for i in indices_to_keep]
        kept_ids = [self.ids[i] for i in indices_to_keep]

        # Reset and re-add
        self.__init__(
            embedding_model=self.model.model_name_or_path,
            use_gpu=self.use_gpu,
            index_type=self.index_type
        )
        self.add_documents(kept_docs, kept_metadata, kept_ids)

    def save(self, path: str):
        """Save FAISS index and metadata"""
        os.makedirs(path, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))

        # Save metadata
        with open(os.path.join(path, "metadata.pkl"), "wb") as f:
            pickle.dump({
                "documents": self.documents,
                "metadata": self.metadata,
                "ids": self.ids,
                "embedding_model": self.model.model_name_or_path,
                "index_type": self.index_type
            }, f)

    def load(self, path: str):
        """Load FAISS index and metadata"""
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))

        # Load metadata
        with open(os.path.join(path, "metadata.pkl"), "rb") as f:
            data = pickle.load(f)
            self.documents = data["documents"]
            self.metadata = data["metadata"]
            self.ids = data["ids"]
            self.model = SentenceTransformer(data["embedding_model"])
            self.index_type = data["index_type"]


class ChromaStore(VectorStore):
    """ChromaDB-based vector store"""

    def __init__(
        self,
        collection_name: str = "rag_documents",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: Optional[str] = None
    ):
        """Initialize ChromaDB store"""
        if not CHROMADB_AVAILABLE:
            raise ImportError("ChromaDB not available. Install with: pip install chromadb")

        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)

        # Initialize ChromaDB client
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ):
        """Add documents to ChromaDB"""
        # Generate embeddings
        embeddings = self.model.encode(
            documents,
            show_progress_bar=False,
            convert_to_numpy=True
        ).tolist()

        # Generate IDs if not provided
        if ids is None:
            existing_count = self.collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(documents))]

        # Prepare metadata
        if metadata is None:
            metadata = [{}] * len(documents)

        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadata,
            ids=ids
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search ChromaDB"""
        # Encode query
        query_embedding = self.model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True
        ).tolist()[0]

        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict  # ChromaDB native filtering
        )

        # Format results
        formatted_results = []
        for doc, distance, metadata in zip(
            results['documents'][0],
            results['distances'][0],
            results['metadatas'][0]
        ):
            # Convert distance to similarity score
            similarity = 1 - distance  # ChromaDB returns distance, we want similarity
            formatted_results.append((doc, similarity, metadata))

        return formatted_results

    def delete(self, ids: List[str]):
        """Delete documents by ID"""
        self.collection.delete(ids=ids)

    def save(self, path: str):
        """ChromaDB persists automatically if persist_directory is set"""
        pass

    def load(self, path: str):
        """ChromaDB loads automatically from persist_directory"""
        pass


class QdrantStore(VectorStore):
    """Qdrant-based vector store"""

    def __init__(
        self,
        collection_name: str = "rag_documents",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        url: Optional[str] = None,
        path: Optional[str] = None
    ):
        """
        Initialize Qdrant store

        Args:
            collection_name: Name of the collection
            embedding_model: Sentence transformer model
            url: Qdrant server URL (for remote)
            path: Local path (for in-memory/disk)
        """
        if not QDRANT_AVAILABLE:
            raise ImportError("Qdrant not available. Install with: pip install qdrant-client")

        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()

        # Initialize Qdrant client
        if url:
            self.client = QdrantClient(url=url)
        elif path:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(":memory:")

        # Create collection if it doesn't exist
        try:
            self.client.get_collection(collection_name)
        except:
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )

    def add_documents(
        self,
        documents: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ):
        """Add documents to Qdrant"""
        # Generate embeddings
        embeddings = self.model.encode(
            documents,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        # Generate IDs if not provided
        if ids is None:
            from uuid import uuid4
            ids = [str(uuid4()) for _ in range(len(documents))]

        # Prepare metadata
        if metadata is None:
            metadata = [{}] * len(documents)

        # Add documents
        points = []
        for idx, (doc_id, doc, embedding, meta) in enumerate(
            zip(ids, documents, embeddings, metadata)
        ):
            payload = {"text": doc, **meta}
            points.append(
                PointStruct(
                    id=doc_id,
                    vector=embedding.tolist(),
                    payload=payload
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search Qdrant"""
        # Encode query
        query_embedding = self.model.encode(
            [query],
            show_progress_bar=False,
            convert_to_numpy=True
        )[0].tolist()

        # Search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            query_filter=filter_dict  # Qdrant native filtering
        )

        # Format results
        formatted_results = []
        for result in results:
            doc = result.payload.get("text", "")
            score = result.score
            metadata = {k: v for k, v in result.payload.items() if k != "text"}
            formatted_results.append((doc, score, metadata))

        return formatted_results

    def delete(self, ids: List[str]):
        """Delete documents by ID"""
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids
        )

    def save(self, path: str):
        """Qdrant persists automatically if path is set"""
        pass

    def load(self, path: str):
        """Qdrant loads automatically from path"""
        pass
