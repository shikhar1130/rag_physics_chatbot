# src/qa_pipeline.py

from pathlib import Path
from langchain.chains import RetrievalQA
from .retriever import setup_retriever
from langchain_community.llms import HuggingFacePipeline

def build_qa_chain(
    index_path: Path = Path("data/index.faiss"),
    meta_path: Path = Path("data/meta.json"),
    hf_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    hf_gen_model: str = "google/flan-t5-small",
    temperature: float = 1.0,
    k: int = 2
) -> RetrievalQA:
    """Build a RetrievalQA chain using:
       • sentence-transformers/all-MiniLM-L6-v2 for embeddings
       • google/flan-t5-small for generation"""
       
    # 1. The setup_retriever function now returns the final retriever object.
    #    Let's name the variable appropriately.
    retriever = setup_retriever(
        index_path=index_path,
        meta_path=meta_path,
        model_name=hf_embed_model
    )
    # Set the search_kwargs directly on our retriever instance.
    retriever.search_kwargs = {"k": k}
    
    # The erroneous call to .as_retriever() has been removed.

    # 2. Build the HuggingFace generation pipeline
    from transformers import pipeline as hf_pipeline
    gen_pipe = hf_pipeline(
        "text2text-generation",
        model=hf_gen_model,
        tokenizer=hf_gen_model,
        device=-1,
        max_length=512,
        do_sample=False,
    )
    llm = HuggingFacePipeline(pipeline=gen_pipe)


    # 3. Wire up the RetrievalQA chain with the corrected retriever
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="map_reduce",
        retriever=retriever,
        return_source_documents=True
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run RAG Physics Chatbot QA pipeline with local HF models"
    )
    parser.add_argument(
        "--question", type=str, required=True,
        help="Physics question to ask"
    )
    parser.add_argument(
        "--index", type=Path, default=Path("data/index.faiss"),
        help="Path to FAISS index"
    )
    parser.add_argument(
        "--meta", type=Path, default=Path("data/meta.json"),
        help="Path to metadata JSON"
    )
    parser.add_argument(
        "--hf_embed_model", type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace embedder model"
    )
    parser.add_argument(
        "--hf_gen_model", type=str,
        default="google/flan-t5-small",
        help="HuggingFace generation model for the pipeline"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Unused for deterministic generation; adjust pipeline kwargs manually if needed"
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Number of documents to retrieve"
    )
    args = parser.parse_args()

    qa = build_qa_chain(
        index_path=args.index,
        meta_path=args.meta,
        hf_embed_model=args.hf_embed_model,
        hf_gen_model=args.hf_gen_model,
        temperature=args.temperature,
        k=args.k
    )

    # --- Start of Changed Block ---

    # Run the chain once by calling it as a function and store the full result.
    # The input must be a dictionary with the key 'query'.
    result = qa({"query": args.question})

    # Extract the answer and source documents from the result dictionary.
    answer = result.get('result', 'No answer was generated.')
    sources = result.get('source_documents', [])

    print("Answer:", answer)

    print("\nSource Documents:")
    if sources:
        for doc in sources:
            # Clean up page_content display to avoid excessive newlines
            page_content_preview = ' '.join(doc.page_content.split())[:200]
            print(f"- {doc.metadata['doc_id']}: {page_content_preview}...")
    else:
        print("No source documents were retrieved.")
    
    # --- End of Changed Block ---


if __name__ == "__main__":
    main()