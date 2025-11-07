from typing import List, Dict, Any
import numpy as np
from .vector_store_interface import VectorStoreInterface

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
    
    def build(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]):
        """Builds HNSW index from embeddings."""
        print(f"Building {self.name} index...")

        self.embedding_dim = embeddings.shape[1]
        num_items = embeddings.shape[0]

        # Create index
        self.index = self.hnswlib.Index(space='cosine', dim=self.embedding_dim)
        self.index.init_index(
            max_elements=num_items,
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
    """Vector store implementation using FAISS (IVF)."""
    
    def __init__(self):
        """
        Initialize FAISS vector store with IVF .
        
        Args:
            n_clusters: Number of clusters for IVF
            pq_m: Number of subvectors for product quantization
        """
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss library not installed. Install with: pip install faiss-cpu")
        
        self.faiss = faiss
        # self.n_clusters = n_clusters
        self.index = None
        self.documents = None
        self.embedding_dim = None
    
    @property
    def name(self) -> str:
        return "FAISS (Flat)"
    
    def build(self, embeddings: np.ndarray, documents: List[Dict[str, Any]]):
        """Builds FAISS index with Flat from embeddings."""
        print(f"Building {self.name} index...")
        
        # Normalize embeddings
        embeddings = embeddings.astype(np.float32)
        self.faiss.normalize_L2(embeddings)
        
        self.embedding_dim = embeddings.shape[1]
        # num_items = embeddings.shape[0]   

        # Rule: n_clusters should be < num_items but not too large
        # n_clusters = min(self.n_clusters, max(1, num_items // 10))
        
        # Create IVF index
        # quantizer = self.faiss.IndexFlatL2(self.embedding_dim)
        # self.index = self.faiss.IndexIVFFlat(
        #     quantizer, 
        #     self.embedding_dim, 
        #     n_clusters, 
        #     self.faiss.METRIC_L2
        # )

        # Flat Index
        self.index = self.faiss.IndexFlatL2(self.embedding_dim)
        
        # print(f"  Training IVF with {num_items} vectors...")
        # self.index.train(embeddings)
        
        # Add embeddings to the index
        self.index.add(embeddings)

        # Set nprobe (number of clusters to search)
        # Higher nprobe = more accurate but slower
        # self.index.nprobe = max(1, int(np.sqrt(n_clusters)) + 1)
                
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

# Helper function to create vector store instances
def get_vector_store(store_type: str, **kwargs) -> VectorStoreInterface:
    """Factory function to create vector store instances."""
    stores = {
        'annoy': ANNOYVectorStore,
        'hnsw': HNSWVectorStore,
        'faiss': FAISSVectorStore,
    }
    
    if store_type.lower() not in stores:
        raise ValueError(f"Unknown store type: {store_type}. Available: {list(stores.keys())}")
    
    return stores[store_type.lower()](**kwargs)

if __name__ == "__main__":
    print("This module provides implementations (wrappers) for vector stores. "
          "Import its functions to use them.")