# ingestion.py

import os
import json
from pathlib import Path
from typing import List, Tuple

import PyPDF2  # pip install PyPDF2

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

# 1. Load raw text from a PDF file
def load_paper(path: Path) -> str:
    """
    Extracts and concatenates all text from the given PDF.
    """
    text = []
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text.append(page.extract_text() or "")
    return "\n".join(text)

# 2. Break a long string into overlapping chunks
def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Splits text into pieces of ~chunk_size characters, overlapping by overlap.
    """
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# 3. Walk a folder of PDFs, chunk each, and save as JSON
def ingest_folder(input_dir: Path, output_path: Path):
    """
    Processes every PDF in input_dir, chunks it, and writes a JSON list of
    {"doc_id": str, "text": str} entries to output_path.
    """
    all_chunks: List[Tuple[str, str]] = []
    for pdf in input_dir.glob("*.pdf"):
        doc_id = pdf.stem
        raw = load_paper(pdf)
        for i, chunk in enumerate(chunk_text(raw)):
            all_chunks.append((f"{doc_id}_{i}", chunk))

    # write out metadata for later embedding/indexing
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([{"doc_id": cid, "text": txt} for cid, txt in all_chunks], f, indent=2)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest physics papers into text chunks")
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=Path("data/raw_papers"),
        help="Folder containing PDF files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/chunks.json"),
        help="Where to write chunk metadata",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ingest_folder(args.input_dir, args.output)
    print(f"Ingested papers from {args.input_dir} into {args.output} ({os.path.getsize(args.output)} bytes)")
