# retriever.py
import warnings
import numpy as np
import faiss
from pathlib import Path
from typing import Any, Dict, List
from langchain.schema import BaseRetriever, Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from .utils import load_json

# Suppress deprecation warnings
warnings.simplefilter("ignore", DeprecationWarning)

def setup_retriever(
    index_path: Path,
    meta_path: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> "FAISSRetriever":
    """
    Load data and initialize the FAISSRetriever.
    """
    index = faiss.read_index(str(index_path))
    metadata = load_json(meta_path)
    embedder = HuggingFaceEmbeddings(model_name=model_name)
    
    return FAISSRetriever(
        index=index,
        metadata=metadata,
        embedder=embedder
    )

class FAISSRetriever(BaseRetriever):
    """
    FAISS-based retriever that conforms to Pydantic model validation
    by declaring fields at the class level.
    """
    index: Any
    metadata: Dict[str, Any]
    embedder: HuggingFaceEmbeddings
    # Add a field for search_kwargs, making the retriever configurable
    search_kwargs: Dict[str, Any] = {"k": 5}

    def _get_relevant_documents(self, query: str, *, run_manager=None, **kwargs) -> list[Document]:
        """
        LangChain calls this method. Embed query and perform FAISS search.
        """
        # Use the 'k' value from the configured search_kwargs
        k = self.search_kwargs.get('k', 5)
        
        q_vec = self.embedder.embed_query(query)
        distances, indices = self.index.search(np.array([q_vec], dtype="float32"), k)
        docs: list[Document] = []
        for idx in indices[0]:
            item = self.metadata[str(idx)]
            docs.append(
                Document(
                    page_content=item["text"],
                    metadata={"doc_id": item["doc_id"]}
                )
            )
        return docs

    async def _aget_relevant_documents(self, query: str, *, run_manager=None, **kwargs) -> list[Document]:
        """
        Asynchronous version of _get_relevant_documents.
        """
        return self._get_relevant_documents(query, run_manager=run_manager, **kwargs)