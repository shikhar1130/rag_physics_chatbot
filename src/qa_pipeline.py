from pathlib import Path
from langchain.chains import RetrievalQA
from .retriever import setup_retriever
from langchain_community.llms import HuggingFacePipeline

def build_qa_chain(
    index_path: Path = Path("data/index.faiss"),
    meta_path: Path = Path("data/meta.json"),
    hf_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    hf_gen_model: str = "tiiuae/falcon-7b-instruct",
    temperature: float = 0.0,
    k: int = 5
) -> RetrievalQA:
    """
    Build a RetrievalQA chain using local HuggingFace embeddings and a local HF generation model.
    - Retrieval via sentence-transformers/all-MiniLM-L6-v2
    - Generation via HuggingFace pipeline model (e.g., Falcon-7B)
    """
    # Setup retriever with HF embeddings
    retriever = setup_retriever(
        index_path=index_path,
        meta_path=meta_path,
        model_name=hf_embed_model
    )

        # Use a local HuggingFace model/inference pipeline for generation
    # Instantiate via from_model_id: specify task and generation params
    llm = HuggingFacePipeline.from_model_id(
        model_id=hf_gen_model,
        task="text-generation",
        pipeline_kwargs={"temperature": temperature}
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever.get_relevant,
        return_source_documents=True
    )


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run RAG Physics Chatbot QA pipeline with local HF models")
    parser.add_argument("--question", type=str, required=True, help="Physics question to ask")
    parser.add_argument("--index", type=Path, default=Path("data/index.faiss"), help="FAISS index path")
    parser.add_argument("--meta", type=Path, default=Path("data/meta.json"), help="Metadata JSON path")
    parser.add_argument(
        "--hf_embed_model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace embedder model"
    )
    parser.add_argument(
        "--hf_gen_model", type=str, default="tiiuae/falcon-7b-instruct",
        help="HuggingFace generation model for the pipeline"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        help="Generation temperature for the HF pipeline"
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

    answer = qa.run(args.question)
    print("Answer:", answer)
    res = qa({"query": args.question})
    print("Source Documents:")
    for doc in res["source_documents"]:
        print(f"- {doc.metadata['doc_id']}{doc.page_content[:200]}...")


if __name__ == "__main__":
    main()
