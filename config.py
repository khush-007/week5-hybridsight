"""
Configuration settings for DocBuddy Pro.

This file contains:
- Environment variables
- Project constants
- LLM configuration
- Embedding model configuration
"""

import os

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# ======================================================
# Load Environment Variables
# ======================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env")

# ======================================================
# Project Constants
# ======================================================

CHUNK_SIZE = 700
CHUNK_OVERLAP = 50

TOP_K = 3

COLLECTION_NAME = "docbuddy"

CHROMA_DB_PATH = "./chroma_store"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

LLM_MODEL = "openai/gpt-oss-120b"

TEMPERATURE = 0

# ======================================================
# Helper Functions
# ======================================================

def get_embedding_model():
    """
    Create and return the embedding model.
    """

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
    )

    return embedding_model


def get_llm():
    """
    Create and return the Groq LLM.
    """

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model=LLM_MODEL,
        temperature=TEMPERATURE,
        max_tokens=1000,
    )

    return llm