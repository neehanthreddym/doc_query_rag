import os
import re
import fitz  # PyMuPDF
from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import glob

# PDF TEXT CLEANING UTILITIES
def clean_text(text: str) -> str:
    """Cleans raw PDF text for better chunking."""
    
    # Fix hyphenated line breaks: "informa-\ntion" → "information"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Remove line breaks inside paragraphs (but keep paragraph breaks)
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Normalize multiple newlines
    text = re.sub(r"\n{2,}", "\n\n", text)

    # Remove double spaces
    text = re.sub(r" +", " ", text)

    return text.strip()

# MULTI-COLUMN AWARE PDF EXTRACTION (PyMuPDF)
def extract_multicolumn_text(pdf_path: str) -> str:
    """
    Extracts PDF text while respecting multi-column layouts.
    Uses PyMuPDF block coordinates to reconstruct reading order.
    """

    doc = fitz.open(pdf_path)
    full_text = ""

    for page in doc:
        blocks = page.get_text("blocks")

        # Sort by approximate vertical position, then horizontal
        blocks = sorted(blocks, key=lambda b: (round(b[1], -1), b[0]))

        page_text = "\n".join([b[4] for b in blocks if b[4].strip()])
        full_text += page_text + "\n\n"

    return clean_text(full_text)

# LOAD ALL PDFS + APPLY MULTI-COLUMN EXTRACTION
def load_pdf_documents(directory_path: str) -> List[Document]:
    """Loads PDFs and extracts text using multi-column aware parsing."""

    print(f"Loading PDF documents from: {directory_path}")

    if not os.path.exists(directory_path):
        print(f"❌ Directory not found: {directory_path}")
        return []

    pdf_files = glob.glob(f"{directory_path}/**/*.pdf", recursive=True)

    if not pdf_files:
        print("No PDF files found.")
        return []

    docs = []
    print(f"Extracting the text from PDF files...")
    for pdf_path in pdf_files:
        # print(f"Extracting: {os.path.basename(pdf_path)}")

        text = extract_multicolumn_text(pdf_path)
        docs.append(Document(page_content=text, metadata={"source": pdf_path}))

    print(f"✅ Extracted {len(docs)} documents.")
    return docs

# RESEARCH-PAPER-OPTIMIZED CHUNKING
def split_documents(
    documents: List[Document],
    chunk_size: int = 1200,
    chunk_overlap: int = 150
) -> List[Document]:
    """Splits documents into high-quality chunks for RAG."""

    print("Splitting documents into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=[
            "\n\n",  # paragraph
            "\n",    # line
            ". ",    # sentence
            " ",     # word
            ""       # fallback
        ],
    )

    chunks = splitter.split_documents(documents)
    print(f"✅ Split {len(documents)} documents into {len(chunks)} chunks.")

    return chunks

# FULL IMPLEMENTATION
def load_and_chunk_documents(path: str) -> List[Document]:
    """Load PDFs -> clean multi-column text -> chunk."""
    docs = load_pdf_documents(path)
    chunks = split_documents(docs)
    return chunks



# def load_pdf_documents(directory_path: str) -> List[Document]:
#     """Loads all PDF files from a specified directory."""
#     print(f"Loading PDF documents from: {directory_path}")
#     if not os.path.exists(directory_path):
#         print(f"Directory not found: {directory_path}")
#         return []
        
#     loader = DirectoryLoader(
#         directory_path,
#         glob="**/*.pdf",
#         loader_cls=PyMuPDFLoader,
#         show_progress=True,
#     )
#     documents = loader.load()
#     print(f"✅ Loaded {len(documents)} document pages.")
#     return documents

# def split_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
#     """Splits documents into smaller chunks for processing."""
#     print("Splitting documents into chunks...")
#     splitter = RecursiveCharacterTextSplitter(
#         chunk_size=chunk_size,
#         chunk_overlap=chunk_overlap,
#         length_function=len,
#         separators=["\n\n", "\n", " ", ""]
#     )
#     chunked_docs = splitter.split_documents(documents)
#     print(f"✅ Split {len(documents)} pages into {len(chunked_docs)} chunks.")
#     return chunked_docs