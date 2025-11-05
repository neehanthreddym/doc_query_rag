# DocQuery: Research Q&A Bot

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.x-FF4B4B.svg)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co/transformers)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20DB-green.svg)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/Groq-API-red.svg)](https://groq.com/)
[![uv](https://img.shields.io/badge/uv-Package%20Manager-orange.svg)](https://docs.astral.sh/uv/)

This project implements a Retrieval-Augmented Generation (RAG) pipeline for querying unstructured PDF documents (Research Papers from arXiv).

This bot will summarize the Research papers related to AI/ML in response to the user query about a Research Paper.

It combines embeddings, vector search, and a large language model to return context-aware answers in real time.

`Note`: Limited Data

## 📊 Application Workflow
<p align="center">
  <img src="assets/RAG-pipeline.svg" alt="RAG Workflow" width="600">
</p>

## 🚀 Features
- **Document Ingestion** (`core/data_loader.py`): Load and chunk PDF documents.
- **Embeddings** (`core/embedding_manager.py`): Generate 384-dim sentence embeddings with `all-MiniLM-L6-v2`.
- **Vector Store** (`core/vector_store.py`): Store and search embeddings using ChromaDB (HNSW indexing).
- **Retriever** (`core/retriever.py`): Fetch relevant context for queries.
- **Pipeline** (`pipelines/rag_pipeline.py`): Combine retriever + LLM (Google’s `gemma2-9b-it`) for RAG responses.
- **Streamlit UI** (`main.py`): Simple and interactive interface for querying documents.
- **Configurable** (`config.py`): Centralized settings for model, database, and pipeline options.
- **Experiments** (`notebooks/rag_pipeline.ipynb`).

## ⚙️ Setup
This project uses [uv](https://docs.astral.sh/uv/) for Python package management.  
Make sure you have `uv` installed first:
```bash
pip install uv
```

Clone the repo and install dependencies:
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
uv sync
```

## ▶️ Usage
**Build the databse** (this is a onetime setup):
- Upload PDFs to the `data/pdf_files path`
- Then run this command
```bash
python main.py --build
```

**API Setup**:
- Get your API key to the gemma2-9b-it model from here [groq-api-keys](https://console.groq.com/keys).
- Create a `.env` file in your project root path and assign your API key to `GROQ_API_KEY`.

**Start the Streamlit app in local**:
```bash
streamlit run app.py
```

Type your query about a research paper published, and get context-aware answers.

## 📂 Project Structure
```
.
├── index_evaluation/        # Similarity search techniques Benchmarking
│   ├── vector_store_interface.py       # Common interface for benchmarking different ANN techniques
├── core/                    # Core components
│   ├── data_loader.py       # PDF loading + chunking
│   ├── embedding_manager.py # Embedding generation
│   ├── retriever.py         # Context retrieval
│   └── vector_store.py      # ChromaDB integration
│
├── data/                    # Input and storage
│   ├── pdf_files/           # Source documents
│   └── vector_store/        # Persisted ChromaDB index
│
├── index_evaluation/              # Benchmarking
│   ├── vector_store_interface.py  # Vector store interface (ABC / Strategy)
│   ├── vector_stores.py           # Wrapers for Indexing algorithms (Concrete Strategies: ANNOY, HNSW, FAISS)
│   ├── benchmark.py               # Benchmarking logic & dataclass
│   └── evaluation.py              # Main script to run the benchmark
│
├── notebooks/
│   └── rag_pipeline.ipynb   # Experiments & benchmarks
│
├── pipelines/
│   └── rag_pipeline.py      # Full RAG pipeline logic
│
├── config.py                # Global configs
├── main.py                  # Streamlit entry point
├── pyproject.toml           # uv dependencies
├── requirements.txt         # pip fallback
├── uv.lock                  # uv lock file
├── .gitignore
└── README.md
```

## To-Do
- Benchmark the retrieval strategies and integrate the best in the Q&A Bot.

## Reference
- https://www.youtube.com/watch?v=fZM3oX4xEyg&list=PLZoTAELRMXVM8Pf4U67L4UuDRgV4TNX9D
- https://www.singlestore.com/blog/a-guide-to-retrieval-augmented-generation-rag/
- https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- https://python.langchain.com/docs/introduction/
- https://console.groq.com/docs/