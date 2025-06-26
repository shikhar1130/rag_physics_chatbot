# src/__init__.py

"""
rag_physics_chatbot

A modular Retrieval-Augmented Generation (RAG) system for physics Q&A.
Provides:
- ingestion: PDF → text chunks
- embedding_index: chunk → FAISS index
- retriever: similarity search over FAISS
- qa_pipeline: LangChain RetrievalQA chain
- evaluate: benchmark evaluation on custom QA set
"""

# Ingestion
from .ingestion import load_paper, chunk_text, ingest_folder

# Embedding & Indexing
from .embedding_index import load_chunks, create_embeddings, build_and_save_index

# Retrieval
from .retriever import setup_retriever

# QA Pipeline
from .qa_pipeline import build_qa_chain

# Evaluation
from .evaluate import load_benchmark, evaluate

# Define what gets imported with `from rag_physics_chatbot import *`
__all__ = [
    # ingestion
    "load_paper", "chunk_text", "ingest_folder",
    # embedding & indexing
    "load_chunks", "create_embeddings", "build_and_save_index",
    # qa pipeline
    "build_qa_chain",
    # evaluation
    "load_benchmark", "evaluate",
    "setup_retriever",
]
