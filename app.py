from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import List, Dict, Any, Optional
import uvicorn

from core.data_loader import load_pdf_documents, split_documents
from core.embedding_manager import EmbeddingManager
from core.vector_store import VectorStore
from core.retriever import RAGRetriever
from pipelines.rag_pipeline import RAGPipeline
from langchain_groq import ChatGroq
import config

# Global state
pipeline: Optional[RAGPipeline] = None

def initialize_pipeline():
    """Initializes the RAG pipeline components."""
    global pipeline
    print("--- 🔄 Initializing RAG Pipeline ---")
    try:
        embedding_manager = EmbeddingManager(config.EMBEDDING_MODEL_NAME)
        vector_store = VectorStore(
            collection_name=config.COLLECTION_NAME,
            persist_directory=config.VECTOR_STORE_DIR
        )
        retriever = RAGRetriever(vector_store, embedding_manager)
        
        llm = ChatGroq(
            groq_api_key=config.GROQ_API_KEY,
            model=config.GROQ_MODEL_NAME,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            reasoning_effort="medium"
        )
        
        pipeline = RAGPipeline(retriever, llm)
        print("--- ✅ RAG Pipeline Initialized ---")
    except Exception as e:
        print(f"--- ❌ Error initializing pipeline: {e} ---")
        raise e

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the pipeline on startup
    initialize_pipeline()
    yield
    # Clean up if necessary

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="Doc Query RAG API", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse('static/index.html')

class QueryRequest(BaseModel):
    query: str
    top_k: int = 10

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]

class BuildResponse(BaseModel):
    status: str
    message: str

def run_build_database():
    """Loads, splits, and embeds documents into the vector store."""
    print("--- 📚 Starting Database Build Process ---")
    try:
        documents = load_pdf_documents(config.PDF_DIR)
        if not documents:
            print("No documents found. Exiting build process.")
            return

        chunked_docs = split_documents(
            documents,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        
        embedding_manager = EmbeddingManager(config.EMBEDDING_MODEL_NAME)
        embeddings = embedding_manager.generate_embeddings(
            [doc.page_content for doc in chunked_docs]
        )
        
        vector_store = VectorStore(
            collection_name=config.COLLECTION_NAME,
            persist_directory=config.VECTOR_STORE_DIR
        )
        vector_store.add_documents(chunked_docs, embeddings)
        print("--- ✅ Database Build Process Complete ---")
        
        # Re-initialize pipeline to pick up new data
        initialize_pipeline()
        
    except Exception as e:
        print(f"Error during build: {e}")

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        result = pipeline.ask(query=request.query, top_k=request.top_k)
        return QueryResponse(
            answer=result['answer'],
            confidence=result['confidence'],
            sources=result['sources']
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/build", response_model=BuildResponse)
async def build_database(background_tasks: BackgroundTasks):
    """Triggers the database build process in the background."""
    background_tasks.add_task(run_build_database)
    return BuildResponse(status="accepted", message="Database build started in background")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "pipeline_loaded": pipeline is not None}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
