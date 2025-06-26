# RAG Physics Chatbot

This repository contains a small Retrieval-Augmented Generation (RAG) system built around a few research papers on theoretical physics.  PDFs are converted to text, embedded with a sentence-transformer model and stored in a FAISS index.  A lightweight question answering pipeline then retrieves relevant passages and uses a local HuggingFace model to generate an answer.

## Setup

1. Create a Python environment (Python 3.10+ recommended) and install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   The scripts rely on `faiss`, `langchain` and the HuggingFace ecosystem.  A GPU is optional but recommended when using larger generation models.

2. (Optional) Place additional PDF files in `data/raw_papers/` if you want to extend the corpus.

## Building the FAISS index

The repository already contains `data/index.faiss` and `data/meta.json`, but they can be rebuilt from scratch:

1. Convert PDFs into text chunks:
   ```bash
   python src/ingestion.py --input_dir data/raw_papers --output data/chunks.json
   ```

2. Embed the chunks and create the FAISS index with accompanying metadata:
   ```bash
   python src/embedding_index.py \
       --chunks data/chunks.json \
       --index data/index.faiss \
       --meta data/meta.json
   ```

## Running the QA pipeline

Once the index is available you can ask physics questions using a local HuggingFace generation model (Falcon‑7B is the default):

```bash
python src/qa_pipeline.py --question "What is axion production through cosmic strings?"
```

The script prints the answer followed by the source document identifiers and excerpts that were retrieved from the FAISS index.

## Usage example

```
$ python src/qa_pipeline.py --question "How does the surface code detect bit-flip errors?"
Answer: The surface code uses a lattice of qubits and measurements of stabilizers to locate bit-flip errors by looking for changes in the parity checks.

Source Documents:
- 9706014v1_2: ...
```

This demonstrates how the retrieval step provides context for the generated answer.
