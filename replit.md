# EDA Decision Agent

A Streamlit-based RAG application that automates data preprocessing decisions based on EDA (Exploratory Data Analysis) reports. It uses Gemini for reasoning and FAISS + sentence-transformers for knowledge retrieval over a curated scikit-learn knowledge base.

## Project Layout

- `ui/app.py` — Streamlit frontend (entry point of the deployed app)
- `cli.py` — Optional CLI runner for the same pipeline
- `rag/` — RAG pipeline: parser, query builder, retriever, reasoning, critic, knowledge base
- `extractor/` — PDF parsing, sentence rebuilding, text cleaning, chunking
- `core/` — Logger, config, exceptions, tracking
- `context/`, `total_context/` — Project self-context (docstrings, structure)
- `tests/` — Test scaffolding
- `data/pdfs/` — Source knowledge files (e.g., `grounded_brain.md`) used by the RAG knowledge base
- `storage/` — FAISS index, doc chunks, and source fingerprint (created at runtime)

## Running on Replit

- The frontend is served by Streamlit on port 5000 (host `0.0.0.0`) via the `Start application` workflow.
- `.streamlit/config.toml` disables CORS/XSRF and runs in headless mode so the Replit preview iframe works correctly.
- The pipeline (clicking "Run Pipeline") requires a `GEMINI_API_KEY` secret. Without it the UI loads and shows a warning.

## Deployment

Deployment is configured for autoscale, running:

```
streamlit run ui/app.py --server.port 5000 --server.address 0.0.0.0
```

## Notes

- `requirements.txt` reflects a Windows development snapshot (it pins versions like `pywin32` and uses a local `packaging` wheel path) and is not directly installable on Linux. On Replit, dependencies are managed via the package manager — `streamlit` is already installed. Heavy ML dependencies (`torch`, `sentence-transformers`, `faiss-cpu`, `google-generativeai`, etc.) can be added with the package manager when the full pipeline is exercised.
