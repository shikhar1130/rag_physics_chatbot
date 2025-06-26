# src/evaluate.py

import os
from pathlib import Path
import pandas as pd
from .qa_pipeline import build_qa_chain
from .utils import get_api_key

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

def load_benchmark(path: Path) -> pd.DataFrame:
    """
    Load a CSV with exactly two columns: 
      - question
      - reference_answer
    """
    return pd.read_csv(path)

def evaluate(path: Path, k: int = 5) -> None:
    """
    Run the RAG QA chain and compute exact-match accuracy.
    """
    api_key = get_api_key()
    qa_chain = build_qa_chain(openai_api_key=api_key, k=k)

    df = load_benchmark(path)
    total = len(df)
    correct = 0

    for _, row in df.iterrows():
        question = row["question"]
        reference = row["reference_answer"]

        # Get the model's answer
        result = qa_chain.run(question)

        # Normalize both strings
        norm_res = result.strip().lower()
        norm_ref = reference.strip().lower()

        if norm_res == norm_ref:
            correct += 1

    accuracy = correct / total * 100
    print(f"Evaluated {total} questions → Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Evaluate RAG Physics Chatbot with exact-match metric"
    )
    parser.add_argument(
        "--benchmark", type=Path, default=Path("data/qa_benchmark.csv"),
        help="CSV file with question, reference_answer"
    )
    parser.add_argument(
        "--k", type=int, default=5,
        help="Top-k passages to retrieve"
    )
    args = parser.parse_args()
    evaluate(args.benchmark, k=args.k)
