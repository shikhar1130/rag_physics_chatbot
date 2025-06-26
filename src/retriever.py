import warnings
import numpy as np
import faiss
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from .utils import load_json

# Suppress deprecation warnings
warnings.simplefilter("ignore", DeprecationWarning)

def setup_retriever(
    index_path: Path,
    meta_path: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> "FAISSRetriever":
    """
    Initialize a FAISSRetriever with a local HuggingFace embedding model.
    """
    embedder = HuggingFaceEmbeddings(model_name=model_name)
    return FAISSRetriever(
        index_path=index_path,
        meta_path=meta_path,
        embedding_model=embedder
    )

class FAISSRetriever:
    """
    FAISS-based retriever: loads index & metadata, embeds queries, returns Document objects.
    """
    def __init__(
        self,
        index_path: Path,
        meta_path: Path,
        embedding_model
    ):
        # Load the FAISS index
        self.index = faiss.read_index(str(index_path))
        # Load metadata mapping row indices to chunk data
        self.metadata = load_json(meta_path)
        # HuggingFaceEmbeddings instance
        self.embedder = embedding_model

    def get_relevant(self, query: str, k: int = 5) -> list[Document]:
        """
        Embed the query, search the FAISS index, and return top-k Document objects.
        """
        # Embed query
        q_vec = self.embedder.embed_query(query)
        # Search the index
        D, I = self.index.search(np.array([q_vec], dtype="float32"), k)
        # Build Document list
        docs: list[Document] = []
        for idx in I[0]:
            item = self.metadata[str(idx)]
            docs.append(
                Document(
                    page_content=item["text"],
                    metadata={"doc_id": item["doc_id"]}
                )
            )
        return docs