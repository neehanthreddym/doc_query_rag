from typing import List, Dict, Any
import numpy as np
from vector_store_interface import VectorStoreInterface

class ANNOYVectorStore(VectorStoreInterface):
    """Vector store implementation using ANNOY (Approximate Nearest Neighbors Oh Yeah)."""
    
    def __init__(self, n_trees: int = 10):
        """
        Initialize ANNOY vector store.
        
        Args:
            n_trees: Number of trees to build (higher = more accurate but slower)
        """
        try:
            from annoy import AnnoyIndex
        except ImportError:
            raise ImportError("annoy library not installed. Install with: pip install annoy")
        
        self.AnnoyIndex = AnnoyIndex
        self.n_trees = n_trees
        self.index = None
        self.documents = None
        self.embedding_dim = None
    
    @property
    def name(self) -> str:
        return "ANNOY"
    
    def build(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]):
        """Builds ANNOY index from embeddings."""
        print(f"Building {self.name} index with {self.n_trees} trees...")
        
        self.embedding_dim = embeddings.shape[1]
        self.index = self.AnnoyIndex(self.embedding_dim, metric='angular')
        
        # Add all embeddings to the index
        for idx, embedding in enumerate(embeddings):
            self.index.add_item(idx, embedding)
        
        # Build the index
        self.index.build(self.n_trees)
        self.documents = documents
        
        print(f"✅ {self.name} index built successfully")
    
    def query(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Query for top_k nearest neighbors."""
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")
        
        # Get nearest neighbor indices
        indices = self.index.get_nns_by_vector(query_embedding, top_k)
        
        # Return corresponding documents
        return [self.documents[idx] for idx in indices]

class HNSWVectorStore(VectorStoreInterface):
    """Vector store implementation using HNSW (Hierarchical Navigable Small Worlds)."""

    def __init__(self, max_connections: int = 16, ef_construction: int = 200):
        """
        Initialize HNSW vetor store
        
        Args:
            max_connections: Maximum number of connections for each node
            ef_construction: Size of the dynamic list for the nearest neighbors during construction
        """
        try:
            import hnswlib
        except ImportError:
            raise ImportError("hnswlib library is not installed. Install with: pip install hnswlib")
        
        self.hnswlib = hnswlib
        self.max_connections = max_connections
        self.ef_construction = ef_construction
        self.index = None
        self.documents = None
        self.embedding_dim = None
        self.ef = None # ef parameter for querying

    @property
    def name(self) -> str:
        return "HNSW"
    
    def buid(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]):
        """Builds HNSW index from embeddings."""
        print(f"Building {self.name} index...")

        self.embedding_dim = embeddings.shape[1]
        num_items = embeddings.shape[0]

        # Create index
        self.index = self.hnswlib.Index(space='cosine', dim=self.embedding_dim)
        self.index.init_index(
            max_items=num_items,
            ef_construction=self.ef_construction,
            M=self.max_connections
        )

        # Add items to index
        self.index.add_items(embeddings, np.arange(num_items))
        
        # Set ef parameter for query time
        self.ef = min(self.ef_construction, num_items)
        self.index.set_ef(self.ef)
        
        self.documents = documents
        
        print(f"✅ {self.name} index built successfully")
    
    def query(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Query for top_k nearest neighbors."""
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")
        
        # Reshape query embedding if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Get nearest neighbor indices
        indices, _ = self.index.knn_query(query_embedding, k=top_k)
        indices = indices[0]  # Get first result

        # Return corresponding documents
        return [self.documents[idx] for idx in indices]

class FAISSVectorStore(VectorStoreInterface):
    """Vector store implementation using FAISS (IVF + Product Quantization)."""
    
    def __init__(self, n_clusters: int = 100, pq_m: int = 8):
        """
        Initialize FAISS vector store with IVF + PQ.
        
        Args:
            n_clusters: Number of clusters for IVF
            pq_m: Number of subvectors for product quantization
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss library not installed. Install with: pip install faiss-cpu")
        
        self.faiss = faiss
        self.n_clusters = n_clusters
        self.pq_m = pq_m
        self.index = None
        self.documents = None
        self.embedding_dim = None
    
    @property
    def name(self) -> str:
        return "FAISS (IVF+PQ)"
    
    def build(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]):
        """Builds FAISS index with IVF + PQ from embeddings."""
        print(f"Building {self.name} index...")
        
        # Normalize embeddings for cosine distance
        embeddings = embeddings.astype(np.float32)
        self.faiss.normalize_L2(embeddings)
        
        self.embedding_dim = embeddings.shape[1]
        num_items = embeddings.shape[0]
        
        # Ensure n_clusters is reasonable
        n_clusters = min(self.n_clusters, max(10, num_items // 10))
        
        # Create IVF + PQ index
        quantizer = self.faiss.IndexFlatL2(self.embedding_dim)
        self.index = self.faiss.IndexIVFPQ(
            quantizer,
            self.embedding_dim,
            n_clusters,
            self.pq_m,
            8  # bits per component
        )
        
        # Train and add items
        self.index.train(embeddings)
        self.index.add(embeddings)
        
        # Set number of probes for query
        self.index.nprobe = max(1, n_clusters // 4)
        
        self.documents = documents
        
        print(f"✅ {self.name} index built successfully")
    
    def query(self, query_embedding: np.ndarray, top_k: int) -> List[Dict[str, Any]]:
        """Query for top_k nearest neighbors."""
        if self.index is None:
            raise ValueError("Index not built. Call build() first.")
        
        # Normalize query embedding
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        self.faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, top_k)
        indices = indices[0]
        
        # Return corresponding documents
        return [self.documents[idx] for idx in indices if idx >= 0]
