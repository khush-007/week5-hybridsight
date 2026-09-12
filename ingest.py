"""
Handles PDF loading, chunking, metadata processing,
and indexing documents into ChromaDB.
"""

# ======================================================
# Standard Library
# ======================================================

import os

# ======================================================
# Third-Party Imports
# ======================================================

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

# ======================================================
# Local Imports
# ======================================================

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    get_embedding_model,
)


# ======================================================
# Load PDF Documents
# ======================================================

def load_documents(pdf_paths: list) -> list:
    """
    Load one or more PDF files.

    Args:
        pdf_paths (list):
            List of PDF file paths.

    Returns:
        list:
            Flat list of LangChain Document objects.
    """

    documents: list = []

    for pdf_path in pdf_paths:

        try:
            loader = PyPDFLoader(pdf_path)

            pages = loader.load()

            documents.extend(pages)

            print(f"✓ Loaded {pdf_path} ({len(pages)} pages)")

        except Exception as e:

            print(f"✗ Failed to load {pdf_path}")
            print(f"Reason: {e}")

    return documents


# ======================================================
# Split Documents into Chunks
# ======================================================

def split_documents(documents: list) -> list:
    """
    Split documents into smaller overlapping chunks.

    Args:
        documents (list):
            List of LangChain Document objects.

    Returns:
        list:
            Chunked Document objects.
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = splitter.split_documents(documents)

    print(f"✓ Created {len(chunks)} chunks.")

    return chunks


# ======================================================
# Update Metadata
# ======================================================

def add_metadata(chunks: list) -> list:
    """
    Clean and update metadata for every chunk.

    Changes:
    - Keeps only the PDF filename.
    - Converts page numbers from 0-based to 1-based.

    Args:
        chunks (list):
            List of chunked Document objects.

    Returns:
        list:
            Updated chunks.
    """

    for chunk in chunks:

        # Get full file path
        source = chunk.metadata.get("source", "")

        # Keep only the filename
        chunk.metadata["source"] = os.path.basename(source)

        # Convert page number to 1-based indexing
        chunk.metadata["page"] = chunk.metadata.get("page", 0) + 1

    print(f"✓ Metadata updated for {len(chunks)} chunks.")

    return chunks


# ======================================================
# Create Chroma Vector Database
# ======================================================

def create_vectorstore(chunks: list):
    embedding_model = get_embedding_model()

    # Connect to the existing collection
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embedding_model,
        collection_name=COLLECTION_NAME,
    )

    # Remove old documents
    existing_ids = vectorstore.get()["ids"]

    if existing_ids:
        vectorstore.delete(ids=existing_ids)
        print(f"✓ Removed {len(existing_ids)} old chunks.")

    # Add the new PDF chunks
    vectorstore.add_documents(chunks)

    print(f"✓ Indexed {len(chunks)} new chunks into ChromaDB.")

    return vectorstore
# ======================================================
# Main Indexing Function
# ======================================================

def index_documents(pdf_paths: list) -> int:
    """
    Complete indexing pipeline.

    Steps:
        1. Load PDFs
        2. Split into chunks
        3. Update metadata
        4. Store in ChromaDB

    Args:
        pdf_paths (list):
            List of PDF file paths.

    Returns:
        int:
            Total number of indexed chunks.
    """

    print("\n===================================")
    print("      INDEXING STARTED")
    print("===================================\n")

    # Step 1: Load documents
    documents = load_documents(pdf_paths)

    if not documents:
        print("❌ No documents were loaded.")
        return 0

    # Step 2: Split documents
    chunks = split_documents(documents)

    # Step 3: Update metadata
    chunks = add_metadata(chunks)

    # Step 4: Store in ChromaDB
    create_vectorstore(chunks)

    print("\n===================================")
    print("      INDEXING COMPLETED")
    print("===================================")
    print(f"✓ Total Chunks Indexed : {len(chunks)}\n")

    return len(chunks)