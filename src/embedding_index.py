import os
import json
from pathlib import Path
import faiss
import numpy as np
# from langchain_openai import OpenAIEmbeddings  # pip install langchain-openai
from .utils import get_api_key, setup_logger
from langchain_community.embeddings import HuggingFaceEmbeddings

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

logger = setup_logger(__name__)


def load_chunks(path: Path) -> list[dict]:
    """
    Load JSON list of {'doc_id', 'text'} entries.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def create_embeddings(chunks: list[dict]) -> np.ndarray:
    """
    Embed each chunk into a vector using a local HuggingFace model.
    """
    logger.info("Creating embeddings for chunks with HuggingFace model")
    # all-MiniLM-L6-v2 is small and fast; feel free to swap models
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    texts = [c['text'] for c in chunks]
    # embed_documents batches under the hood
    vectors = embedder.embed_documents(texts)
    return np.array(vectors, dtype='float32')


def build_and_save_index(
    chunks_path: Path,
    index_path: Path,
    meta_path: Path
):
    """
    1. Load chunks from JSON
    2. Create embeddings
    3. Build FAISS L2 index and save it
    4. Persist metadata for lookup
    """
    chunks = load_chunks(chunks_path)
    vectors = create_embeddings(chunks)

    # Build FAISS index
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    logger.info(f"FAISS index saved to {index_path} with {vectors.shape[0]} vectors")

    # Save metadata mapping row -> chunk dict
    meta = {i: chunks[i] for i in range(len(chunks))}
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata saved to {meta_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build FAISS index over text chunks using HuggingFace embeddings")
    parser.add_argument("--chunks", type=Path, default=Path("data/chunks.json"), help="Path to chunks.json")
    parser.add_argument("--index", type=Path, default=Path("data/index.faiss"), help="Output FAISS index file path")
    parser.add_argument("--meta", type=Path, default=Path("data/meta.json"), help="Output metadata JSON path")
    args = parser.parse_args()

    build_and_save_index(args.chunks, args.index, args.meta)