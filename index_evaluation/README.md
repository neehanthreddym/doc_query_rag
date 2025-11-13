# ANN Index Evaluation Harness for RAG

This project is a Python-based benchmarking harness designed to evaluate and compare the performance of different Approximate Nearest Neighbor (ANN) algorithm implementations. It provides a common interface for multiple vector search libraries and runs a systematic evaluation to measure their trade-offs in terms of speed, memory, and accuracy.

The goal is to provide a data-driven basis for selecting the best vector store "engine" for a Retrieval-Augmented Generation (RAG) application.

## 🚀 Features
- **Strategy Pattern**: Uses an abstract `VectorStoreInterface` to easily add and test new vector search implementations.
- **Factory Function**: Includes a `get_vector_store` factory for easy instantiation of different stores.
- **Comprehensive Benchmarking**: Measures four critical performance metrics for each algorithm.
- **Easy to Run**: A single `evaluation.py` script loads data, generates embeddings, and runs the full benchmark.
- **Detailed Reporting**: Prints a full summary for each algorithm and a final "Best Performers" analysis.

## 🔬 Algorithms Benchmarked
This harness is configured to evaluate three popular and distinct ANN algorithm implementations:
1. `ANNOY` (Spotify): A tree-based algorithm that builds a forest of random projection trees. Known for its simplicity and memory efficiency.
2. `HNSW` (using hnswlib): A graph-based algorithm (Hierarchical Navigable Small Worlds) that is currently one of the most popular choices, known for its excellent speed and accuracy.
3. `FAISS` (IVF+PQ): A library from Meta AI. This implementation uses an Inverted File (IVF) index with Product Quantization (PQ) for a balance of high speed and extreme memory efficiency through vector compression.

## 📊 Evaluation Metrics
The harness evaluates each algorithm on four key metrics:
- **Index Build Time (s)**: The wall-clock time required to construct the search index from the raw document embeddings.
- **Index Memory Usage (MB)**: The amount of additional RAM consumed by the Python process after loading the index into memory.
- **Query Speed (QPS)**: The number of Queries Per Second the index can handle. This is derived from the average query latency over a set of test queries.
- **Search Accuracy (Recall@K)**: The most important metric for an approximate algorithm. It measures what percentage of the true nearest neighbors were found by the algorithm.

## 🏗️ Evaluation files
The project is designed using the Strategy and Factory design patterns to be modular and extensible.
- `vector_store_interface.py`: The Abstract Base Class (ABC) that defines the common "contract" (the build and query methods) that all vector stores must implement. This is the core of the Strategy pattern.
- `vector_stores.py`: Contains the concrete Strategy implementations:
    1. `ANNOYVectorStore`
    2. `HNSWVectorStore`
    3. `FAISSVectorStore`
    
    It also contains the get_vector_store Factory function, which provides a simple way to create instances of these classes by name.
- `benchmark.py`: The core benchmarking logic. It contains the `BenchmarkMetrics` dataclass and the main `benchmark_vector_store` function that systematically runs an instance of the `VectorStoreInterface` through all four performance tests.
- `evaluation.py`: The main entry point to run the entire evaluation. This script:
    - Parses command-line arguments.
    - Loads and processes the source PDF documents.
    - Generates embeddings for all documents and for the test queries.
    - Calls benchmark_vector_store for each configured algorithm.
    - Prints the final comparative report.

## How to run benchmarking
Clone the repo (anns_benchmarking branch):
```bash
git clone https://github.com/neehanthreddym/doc_query_rag.git
cd doc_query_rag
git checkout anns_benchmarking
```

Create virtual environment managed by uv:
```bash
# pip install uv <-- If uv is not installed
uv venv
source .venv/bin/activate # macOS / Linux
# or
.\.venv\Scripts\activate    # Windows
```

Install the dependencies from project.toml+uv.lock:
```bash
uv sync
```
No pip install or requirements files needed.

Run evaluation script:
```bash
uv run python -m index_evaluation.evaluation
```