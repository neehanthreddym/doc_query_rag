import time
import numpy as np
import psutil
import os
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

@dataclass
class BenchmarkMetrics:
    """Stores all benchmark metrics for a vector store."""
    store_name: str
    
    build_time_seconds: float
    memory_usage_mb: float
    queries_per_second: float
    avg_query_time_ms: float
    recall_at_10: float
    recall_at_100: float

    dataset_size: int
    embedding_dim: int
    num_queries_tested: int

    def print_summary(self):
        """Print detailed summary of metrics."""
        print(f"\n{'='*80}")
        print(f"{self.store_name.upper()} - BENCHMARK METRICS")
        print(f"{'='*80}\n")

        print(f"BUILD TIME (seconds)")
        print(f"   {self.build_time_seconds:.4f}s\n")
        
        print(f"MEMORY USAGE (MB)")
        print(f"   {self.memory_usage_mb:.2f} MB\n")

        print(f"QUERY SPEED (QPS - Queries Per Second)")
        print(f"   {self.queries_per_second:.2f} QPS")
        print(f"   (Average query time: {self.avg_query_time_ms:.4f}ms)\n")
        
        print(f"SEARCH ACCURACY (Recall@k)")
        print(f"   Recall@10:  {self.recall_at_10:.4f} ({self.recall_at_10*100:.2f}%)")
        print(f"   Recall@100: {self.recall_at_100:.4f} ({self.recall_at_100*100:.2f}%)\n")
        
        print(f"DATA INFO")
        print(f"   Size: {self.dataset_size} vectors")
        print(f"   Dimension: {self.embedding_dim}D")
        print(f"   Queries tested: {self.num_queries_tested}")
        print(f"{'='*80}\n")

def calculate_recall_at_k(retrieved_indices: np.ndarray, 
                         ground_truth_indices: np.ndarray, 
                         k: int) -> float:
    """
    Calculate Recall@k metric.
    
    Recall@k measures: What percentage of the true top-k neighbors did we find?
    
    Formula: Recall@k = |{top-k retrieved} ∩ {top-k ground truth}| / |top-k ground truth|
    
    Range: 0.0 to 1.0
    - 1.0 = perfect recall (found all true neighbors)
    - 0.5 = found 50% of true neighbors
    - 0.0 = found none of the true neighbors
    
    Args:
        retrieved_indices: Indices returned by vector store (first k)
        ground_truth_indices: True nearest neighbor indices (first k)
        k: Number of top results to evaluate
    
    Returns:
        Recall score between 0.0 and 1.0
    """
    # Get top-k results
    retrieved_set = set(retrieved_indices[:k])
    ground_truth_set = set(ground_truth_indices[:k])
    
    if len(ground_truth_set) == 0:
        return 1.0
    
    # Count overlapping indices
    overlap = len(retrieved_set & ground_truth_set)
    
    # Calculate recall
    recall = overlap / len(ground_truth_set)
    
    return recall

def measure_query_speed(vector_store,
                       query_embeddings: np.ndarray,
                       top_k: int = 10,
                       num_queries: Optional[int] = None) -> Tuple[float, float, List[List[Dict]]]:
    """
    Measure Query Speed (QPS - Queries Per Second).
    
    This measures how fast the vector store can respond to queries.
    
    Process:
    1. Run multiple queries through the vector store
    2. Measure time for each query
    3. Calculate average query time in milliseconds
    4. Convert to QPS: 1000ms / avg_time_ms = queries per second
    
    Args:
        vector_store: VectorStore instance (already built)
        query_embeddings: Array of query embeddings
        top_k: Number of results to retrieve per query
        num_queries: Number of queries to run (default: all)
    
    Returns:
        Tuple of (avg_query_time_ms, qps, all_responses)
    """
    if num_queries is None:
        num_queries = len(query_embeddings)
    else:
        num_queries = min(num_queries, len(query_embeddings))
    
    query_times = []
    all_responses = []
    
    print(f"  Running {num_queries} queries...")
    
    for i in range(num_queries):
        query_embedding = query_embeddings[i]
        
        # Measure query time
        start_time = time.time()
        responses = vector_store.query(query_embedding, top_k=top_k)
        query_time_ms = (time.time() - start_time) * 1000
        
        query_times.append(query_time_ms)
        all_responses.append(responses)
        
        # Show progress
        if (i + 1) % max(1, num_queries // 5) == 0:
            print(f"    Completed {i + 1}/{num_queries} queries")
    
    # Calculate statistics
    avg_query_time_ms = np.mean(query_times)
    qps = 1000.0 / avg_query_time_ms
    
    return avg_query_time_ms, qps, all_responses

def measure_memory_usage() -> float:
    """
    Measure Memory Usage (MB).
    
    Measures the current RAM used by the Python process.
    
    The difference between memory before and after building
    the index represents the memory overhead of the index.
    
    Returns:
        Memory usage in MB
    """
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / (1024 * 1024)  # Convert bytes to MB
    
    return memory_mb

def calculate_search_accuracy(vector_store,
                            embeddings: np.ndarray,
                            query_embeddings: np.ndarray,
                            store_name: str,
                            top_k: int = 10,
                            num_samples: int = 5) -> Tuple[float, float]:
    """
    Calculate Search Accuracy (Recall@k).
    
    This measures how accurately the vector store finds the true nearest neighbors.

    Process:
    1. For each query, get results from the vector store
    2. Calculate ground truth by brute force (computing distance to all vectors)
    3. Compare vector store results to ground truth
    4. Calculate recall@10 and recall@100

    Args:
        vector_store: VectorStore instance (already built)
        embeddings: All document embeddings (used for ground truth)
        query_embeddings: Query embeddings to test
        top_k: Number of results to evaluate
        num_samples: Number of queries to use (for speed)
    
    Returns:
        Tuple of (recall@10, recall@100)
    """
    recall_scores_10 = []
    recall_scores_100 = []
    
    for query_embedding in query_embeddings[:num_samples]:
        retrieve_k = max(100, top_k)
        store_results = vector_store.query(query_embedding, top_k=retrieve_k)
        retrieved_indices = np.array([doc['id'] for doc in store_results])

        # Choose metric based on store type
        if "ANNOY" in store_name:
            # ANNOY: Cosine Similarity with Angular Distance
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            cosine_similarities = np.dot(embeddings_norm, query_norm)
            cosine_similarities = np.clip(cosine_similarities, -1.0, 1.0)
            angular_distances = np.arccos(cosine_similarities)
            ground_truth = np.argsort(angular_distances)[:retrieve_k]
        elif "HNSW" in store_name:
            # HNSW: Cosine Similarity
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            similarities = np.dot(embeddings_norm, query_norm)
            ground_truth = np.argsort(-similarities)[:retrieve_k]
        elif "FAISS" in store_name:
            # FAISS: L2 Distance
            distances = np.linalg.norm(embeddings - query_embedding, axis=1)
            ground_truth = np.argsort(distances)[:retrieve_k]
        else:
            # Default: Cosine
            query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
            embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            similarities = np.dot(embeddings_norm, query_norm)
            ground_truth = np.argsort(-similarities)[:retrieve_k]
        
        # Calculate recall
        retrieved_set = set(retrieved_indices[:10])
        ground_truth_set = set(ground_truth[:10])
        recall_10 = len(retrieved_set & ground_truth_set) / len(ground_truth_set)
        
        recalled_set = set(retrieved_indices[:100])
        ground_truth_set_100 = set(ground_truth[:100])
        recall_100 = len(recalled_set & ground_truth_set_100) / len(ground_truth_set_100)
        
        recall_scores_10.append(recall_10)
        recall_scores_100.append(recall_100)

    return np.mean(recall_scores_10), np.mean(recall_scores_100)

# Main benchmarking function
def benchmark_vector_store(vector_store_class,
                          store_params: Dict[str, Any],
                          embeddings: np.ndarray,
                          documents: List[Dict[str, Any]],
                          query_embeddings: np.ndarray,
                          top_k: int = 10,
                          num_queries: Optional[int] = None) -> Optional[BenchmarkMetrics]:
    """
    Complete benchmarking of a single vector store implementation.

    This function:
    1. Creates the vector store
    2. Measures BUILD TIME while building the index
    3. Measures MEMORY USAGE after building
    4. Measures QUERY SPEED for multiple queries
    5. Calculates SEARCH ACCURACY (Recall@k)

    Args:
        vector_store_class: VectorStore class
        store_params: Parameters for the vector store (e.g., {'n_trees': 10})
        embeddings: Document embeddings
        documents: Document metadata
        query_embeddings: Query embeddings for testing
        top_k: Number of results to retrieve
        num_queries: Number of queries to run
    
    Returns:
        BenchmarkMetrics object with all 4 metrics, or None if error
    """
    store_name = store_params.get('name', vector_store_class.__name__)
    params = {k: v for k, v in store_params.items() if k != 'name'}

    print(f"\n{'='*80}")
    print(f"BENCHMARKING {store_name}")
    print(f"{'='*80}\n")

    try:
        # 1. Initialize vector store
        print(f"1. Initializing {store_name}...")
        vector_store = vector_store_class(**params)
        print(f"  Initialized with parameters: {params}\n")

        # 2. Measure BUILD TIME
        print(f"2. Measuring BUILD TIME...")
        mem_before = measure_memory_usage()
        
        build_start = time.time()
        vector_store.build(embeddings, documents)
        build_time = time.time() - build_start
        
        print(f"  Build completed in {build_time:.4f} seconds\n")

        # 3. Measure MEMORY USAGE
        print(f"3. Measuring MEMORY USAGE...")
        mem_after = measure_memory_usage()
        memory_usage = mem_after - mem_before
        
        if memory_usage <= 0:
            memory_usage = mem_after
        
        print(f"  Memory usage: {memory_usage:.2f} MB\n")

        # 4. Measure QUERY SPEED (QPS)
        print(f"4. Measuring QUERY SPEED (QPS)...")
        avg_query_time_ms, qps, query_results = measure_query_speed(
            vector_store,
            query_embeddings,
            top_k=top_k,
            num_queries=num_queries
        )
        print(f"  Query speed: {qps:.2f} QPS (avg {avg_query_time_ms:.4f}ms per query)\n")

        # 5. Calculate SEARCH ACCURACY (Recall@k)
        print(f"5. Calculating SEARCH ACCURACY (Recall@k)...")
        recall_at_10, recall_at_100 = calculate_search_accuracy(
            vector_store,
            embeddings,
            query_embeddings,
            store_name,
            top_k=top_k,
            num_samples=5
        )
        print(f"  Recall@10: {recall_at_10:.4f}")
        print(f"  Recall@100: {recall_at_100:.4f}\n")

        metrics = BenchmarkMetrics(
            store_name=store_name,
            build_time_seconds=build_time,
            memory_usage_mb=memory_usage,
            queries_per_second=qps,
            avg_query_time_ms=avg_query_time_ms,
            recall_at_10=recall_at_10,
            recall_at_100=recall_at_100,
            dataset_size=len(embeddings),
            embedding_dim=embeddings.shape[1],
            num_queries_tested=num_queries or len(query_embeddings)
        )
        
        return metrics
    except Exception as e:
        print(f"Error benchmarking {store_name}: {e}")
        import traceback
        traceback.print_exc()
        return None
    
if __name__ == "__main__":
    print("This module provides benchmarking utilities for vector stores. " 
          "Import its functions to use them.")