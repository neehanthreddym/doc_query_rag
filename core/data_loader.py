import os
from typing import List
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_pdf_documents(directory_path: str) -> List[Document]:
    """Loads all PDF files from a specified directory."""
    print(f"Loading PDF documents from: {directory_path}")
    if not os.path.exists(directory_path):
        print(f"Directory not found: {directory_path}")
        return []
        
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.pdf",
        loader_cls=PyMuPDFLoader,
        show_progress=True,
    )
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} document pages.")
    return documents

def split_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    """Splits documents into smaller chunks for processing."""
    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunked_docs = splitter.split_documents(documents)
    print(f"✅ Split {len(documents)} pages into {len(chunked_docs)} chunks.")
    return chunked_docs