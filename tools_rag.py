"""
RAG tool for HybridSight.

Allows the LangGraph agent to search
the user's uploaded PDF documents.
"""

from langchain_core.tools import tool
from langchain_chroma import Chroma

from config import (
    get_embedding_model,
    CHROMA_DB_PATH,
    COLLECTION_NAME,
    TOP_K,
)


# ======================================================
# Load Existing ChromaDB
# ======================================================

embedding_model = get_embedding_model()

vectorstore = Chroma(
    persist_directory=CHROMA_DB_PATH,
    embedding_function=embedding_model,
    collection_name=COLLECTION_NAME,
)


# ======================================================
# Check Whether PDF Documents Exist
# ======================================================

def has_documents():
    """
    Return True if at least one PDF chunk
    is currently stored in ChromaDB.
    """

    return vectorstore._collection.count() > 0


# ======================================================
# RAG Tool
# ======================================================

@tool
def search_documents(query: str) -> str:
    """
    Search the user's uploaded PDF documents for information
    relevant to the query.

    Use this tool when the user asks about:
    - an uploaded PDF
    - their notes
    - their document
    - information contained in their uploaded files

    Do NOT use this tool for:
    - general knowledge
    - current events
    - recent news
    - information not related to the user's documents
    """

    # --------------------------------------------------
    # Check whether documents exist
    # --------------------------------------------------

    if not has_documents():
        return "No documents uploaded yet."

    # --------------------------------------------------
    # Retrieve relevant chunks
    # --------------------------------------------------

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": TOP_K
        }
    )

    chunks = retriever.invoke(query)

    # --------------------------------------------------
    # No relevant content
    # --------------------------------------------------

    if not chunks:
        return "No relevant content found in the uploaded documents."

    # --------------------------------------------------
    # Return chunks with source and page information
    # --------------------------------------------------

    results = []

    for chunk in chunks:

        page = chunk.metadata.get(
            "page",
            "?"
        )

        source = chunk.metadata.get(
            "source",
            "unknown"
        )

        results.append(
            f"[Source: {source}, p.{page}]\n"
            f"{chunk.page_content}"
        )

    return "\n\n".join(results)