from core.data_loader import load_pdf_documents, split_documents
from core.embedding_manager import EmbeddingManager
from core.vector_store import VectorStore
from core.retriever import RAGRetriever
from pipelines.rag_pipeline import RAGPipeline
from langchain_groq import ChatGroq
import config
import argparse
import streamlit as st

def build_database():
    """Loads, splits, and embeds documents into the vector store."""
    print("--- 📚 Starting Database Build Process ---")
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

def query_rag(query: str):
    """Initializes components and runs a query through the RAG pipeline."""
    print(f"\n--- ❓ Querying RAG System with: '{query}' ---")
    
    # Initialize components
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
    
    # Get answer
    result = pipeline.ask(query=query, top_k=10)
    
    # Print results
    # print("\n--- 📝 RAG System Response ---")
    # print(f"\nAnswer:\n{result['answer']}")
    # print(f"\nConfidence: {result['confidence']:.4f}")
    # print("\nSources:")
    # for source in result['sources']:
    #     print(f"  - File: {source['source']}, Page: {source['page']}, Score: {source['score']:.4f}")
    print("--- ✅ Query Complete ---")
    return {
        'answer': result['answer'],
        'confidence': result['confidence'],
        'sources': result['sources']
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A RAG Q&A System.")
    parser.add_argument("--build", action="store_true", help="Build the vector database from PDF files.")
    args = parser.parse_args()

    if args.build:
        build_database()
    else:
        def main():
            st.title("Q&A Bot")
            st.write("Research Assistant for AI/ML Papers using RAG System")

            query = st.text_input("Enter your query:")
            if st.button("Submit Query") and query:
                with st.spinner("Querying the corpus of available documents..."):
                    try:
                        result = query_rag(query)
                        if result:
                            st.write("**Answer**")
                            st.write(result['answer'])
                            st.write(f"**Confidence:** {result['confidence']:.4f}")
                            st.write("**Sources**")
                            for source in result['sources']:
                                st.write(f"- File: {source['source']}, Page: {source['page']}, Score: {source['score']:.4f}")
                    except Exception as e:
                        st.error(f"Error: {e}")

        main()
