import os
from dotenv import load_dotenv

load_dotenv()

# --- PATHS ---
DATA_DIR = "./data"
PDF_DIR = os.path.join(DATA_DIR, "pdf_files")
VECTOR_STORE_DIR = os.path.join(DATA_DIR, "vector_store")

# --- DOCUMENT PROCESSING ---
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 150

# --- EMBEDDING MODEL ---
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# --- VECTOR STORE ---
COLLECTION_NAME = "pdf_documents"

# --- LLM and API KEYS ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL_NAME = "openai/gpt-oss-20b"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 65000