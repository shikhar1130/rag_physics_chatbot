# import warnings
# import numpy as np
# import faiss
# from pathlib import Path
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.schema import Document
# from .utils import load_json

# # Suppress deprecation warnings
# warnings.simplefilter("ignore", DeprecationWarning)

# def setup_retriever(
#     index_path: Path,
#     meta_path: Path,
#     model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
# ) -> "FAISSRetriever":
#     """
#     Initialize a FAISSRetriever with a local HuggingFace embedding model.
#     """
#     embedder = HuggingFaceEmbeddings(model_name=model_name)
#     return FAISSRetriever(
#         index_path=index_path,
#         meta_path=meta_path,
#         embedding_model=embedder
#     )

# class FAISSRetriever:
#     """
#     FAISS-based retriever: loads index & metadata, embeds queries, returns Document objects.
#     """
#     def __init__(
#         self,
#         index_path: Path,
#         meta_path: Path,
#         embedding_model
#     ):
#         # Load the FAISS index
#         self.index = faiss.read_index(str(index_path))
#         # Load metadata mapping row indices to chunk data
#         self.metadata = load_json(meta_path)
#         # HuggingFaceEmbeddings instance
#         self.embedder = embedding_model

#     def get_relevant(self, query: str, k: int = 5) -> list[Document]:
#         """
#         Embed the query, search the FAISS index, and return top-k Document objects.
#         """
#         # Embed query
#         q_vec = self.embedder.embed_query(query)
#         # Search the index
#         D, I = self.index.search(np.array([q_vec], dtype="float32"), k)
#         # Build Document list
#         docs: list[Document] = []
#         for idx in I[0]:
#             item = self.metadata[str(idx)]
#             docs.append(
#                 Document(
#                     page_content=item["text"],
#                     metadata={"doc_id": item["doc_id"]}
#                 )
#             )
#         return docs



from pathlib import Path
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
import json

def setup_retriever(
    index_path: Path = Path("data/index.faiss"),
    meta_path: Path = Path("data/meta.json"),
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> FAISS:
    """
    Load a FAISS index and associated metadata from disk and return the LangChain FAISS vectorstore.
    """
    # 1. Initialize the embedding model
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    # 2. Load the FAISS vectorstore from the specified folder
    vectorstore = FAISS.load_local(
        folder_path=str(index_path.parent),
        embeddings=embeddings,
        index_name=index_path.stem,
        allow_dangerous_deserialization=True
    )

    # 3. (Optional) Load metadata and merge with the vectorstore's docstore
    # Uncomment and adjust if you have metadata to associate
    with open(meta_path, "r") as f:
        metadata = json.load(f)
        for doc_id, meta in metadata.items():
            if doc_id in vectorstore.docstore._dict:
                vectorstore.docstore._dict[doc_id].metadata.update(meta)

    return vectorstore
