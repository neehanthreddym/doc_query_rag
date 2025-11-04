import argparse
import sys
from typing import List
from benchmark import BenchmarkMetrics, benchmark_vector_store
from ..core.data_loader import load_pdf_documents, split_documents
from ..core.embedding_manager import EmbeddingManager
from vector_stores import ANNOYVectorStore, HNSWVectorStore, FAISSVectorStore

TEST_QUERIES = [
    "What is the core architectural innovation of the Transformer model introduced in 'Attention Is All You Need'?",
    "Describe the process that Hierarchical NSW (HNSW) uses to build its multi-layer structure for approximate nearest neighbor search.",
    "According to the abstract, what two key challenges does the RA-RAG framework aim to solve compared to standard RAG?",
    "Explain the two-stage training process used in Dyna-Think Dyna Training (DDT).",
    "Why is compositional multi-tasking a significant challenge for on-device Large Language Models?",
    "Compare the approach of RA-RAG to the conventional RAG described in 'Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.' What is the main difference in their retrieval process?",
]

def print_metric_analysis(results: List[BenchmarkMetrics]):
    """Analyze and print best performer for each metric."""
    
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        return
    
    print(f"\n{'='*80}")
    print("BEST PERFORMERS BY METRIC")
    print(f"{'='*80}\n")
    
    # Build Time (lower is better)
    fastest_build = min(valid_results, key=lambda x: x.build_time_seconds)
    print(f" FASTEST BUILD TIME")
    print(f"    Winner: {fastest_build.store_name}")
    print(f"    Time: {fastest_build.build_time_seconds:.4f} seconds")
    if len(valid_results) > 1:
        for r in valid_results:
            if r != fastest_build:
                speedup = r.build_time_seconds / fastest_build.build_time_seconds
                print(f"   ({r.store_name} is {speedup:.2f}x slower)")
    print()
    
    # Memory Usage (lower is better)
    lowest_memory = min(valid_results, key=lambda x: x.memory_usage_mb)
    print(f" LOWEST MEMORY USAGE")
    print(f"    Winner: {lowest_memory.store_name}")
    print(f"    Memory: {lowest_memory.memory_usage_mb:.2f} MB")
    if len(valid_results) > 1:
        for r in valid_results:
            if r != lowest_memory:
                ratio = r.memory_usage_mb / lowest_memory.memory_usage_mb
                print(f"   ({r.store_name} uses {ratio:.2f}x more memory)")
    print()
    
    # Query Speed (higher is better)
    fastest_query = max(valid_results, key=lambda x: x.queries_per_second)
    print(f" FASTEST QUERY SPEED (QPS)")
    print(f"    Winner: {fastest_query.store_name}")
    print(f"    Speed: {fastest_query.queries_per_second:.2f} QPS")
    print(f"    (Avg time: {fastest_query.avg_query_time_ms:.4f}ms per query)")
    if len(valid_results) > 1:
        for r in valid_results:
            if r != fastest_query:
                speedup = fastest_query.queries_per_second / r.queries_per_second
                print(f"   ({r.store_name} is {speedup:.2f}x slower)")
    print()
    
    #Search Accuracy - Recall@10 (higher is better)
    best_recall_10 = max(valid_results, key=lambda x: x.recall_at_10)
    print(f" BEST SEARCH ACCURACY (Recall@10)")
    print(f"    Winner: {best_recall_10.store_name}")
    print(f"    Recall@10: {best_recall_10.recall_at_10:.4f} ({best_recall_10.recall_at_10*100:.2f}%)")
    if len(valid_results) > 1:
        for r in valid_results:
            if r != best_recall_10:
                diff = (best_recall_10.recall_at_10 - r.recall_at_10) * 100
                print(f"   ({r.store_name}: {r.recall_at_10:.4f}, {diff:.2f}% lower)")
    print()
    
    print(f"{'='*80}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Benchmark vector stores with 4 key metrics',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
        Examples:
          python benchmark_comparison.py --data-dir data/pdf_files
          python benchmark_comparison.py --data-dir data/pdf_files --model all-mpnet-base-v2 --num-queries 100
        '''
    )
    
    parser.add_argument('--data-dir', type=str, default='data/pdf_files', 
                        help='Directory containing PDF files')
    parser.add_argument('--model', type=str, default='all-MiniLM-L6-v2',
                       help='SentenceTransformer model name')
    parser.add_argument('--chunk-size', type=int, default=1000,
                       help='Document chunk size')
    parser.add_argument('--chunk-overlap', type=int, default=50,
                       help='Chunk overlap')
    parser.add_argument('--num-queries', type=int, default=None,
                       help='Number of queries to benchmark')
    parser.add_argument('--top-k', type=int, default=10,
                       help='Number of results to retrieve')
    
    args = parser.parse_args()
    # Load data
    print("LOADING DATA...")
    print("-" * 80)
    
    pdf_documents = load_pdf_documents(args.data_dir)
    if not pdf_documents:
        print("  No PDF documents found")
        sys.exit(1)

    chunked_documents = split_documents(pdf_documents, args.chunk_size, args.chunk_overlap)

    if not chunked_documents:
        print("  No chunks generated")
        sys.exit(1)
    
    # Generate embeddings
    print("\nGENERATING EMBEDDINGS...")
    print("-" * 80)
    embedding_manager = EmbeddingManager(args.model)

    texts = [doc.page_content for doc in chunked_documents]
    embeddings = embedding_manager.generate_embeddings(texts)

    documents = [
        {
            'id': idx,
            'content': doc.page_content,
            'source': doc.metadata.get('source', 'unknown'),
        }
        for idx, doc in enumerate(chunked_documents)
    ]

    # Generate query embeddings
    print("\nGENERATING QUERY EMBEDDINGS...")
    print("-" * 80)
    query_embeddings = embedding_manager.generate_embeddings(TEST_QUERIES)

    # Define vector stores to benchmark
    stores_to_benchmark = [
        {
            'class': ANNOYVectorStore,
            'params': {'n_trees': 10, 'name': 'ANNOY'}
        },
        {
            'class': HNSWVectorStore,
            'params': {'max_connections': 16, 'ef_construction': 200, 'name': 'HNSW'}
        },
        {
            'class': FAISSVectorStore,
            'params': {'n_clusters': 100, 'pq_m': 8, 'name': 'FAISS'}
        },
    ]

    # Run benchmarks
    print("\n" + "="*80)
    print("RUNNING BENCHMARKS FOR EACH METRIC...")
    print("="*80)

    results = []
    for config in stores_to_benchmark:
        result = benchmark_vector_store(
            vector_store_class=config['class'],
            store_params=config['params'],
            embeddings=embeddings,
            documents=documents,
            query_embeddings=query_embeddings,
            top_k=args.top_k,
            num_queries=args.num_queries
        )
        results.append(result)

    # Print detailed results
    print("\n" + "="*80)
    print("DETAILED RESULTS FOR EACH VECTOR STORE")
    print("="*80)

    for result in results:
        if result is not None:
            result.BenchmarkMetrics.print_summary()

    print_metric_analysis(results)

    print("✅ Benchmarking complete!\n")