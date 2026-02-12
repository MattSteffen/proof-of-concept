## `rag` package overview

This folder contains a minimal Retrieval-Augmented Generation (RAG) example that runs locally with Ollama.

### Files

- `main.py`: End-to-end demo of a naive RAG pipeline with:
  - Ollama embedding calls through the Python client
  - A simple in-memory vector store using cosine similarity
  - Retrieval of top matching chunks
  - Prompt construction and answer generation with an Ollama chat model
- `sample_document.txt`: Tiny sample corpus split by paragraphs and indexed by `main.py`.

### Design notes

- The vector store is intentionally simple and in-memory only.
- Document chunking is done by splitting `sample_document.txt` into paragraphs.
- Base URL uses Ollama host format (`http://localhost:11434`) and is configurable via environment variables:
  - `OLLAMA_BASE_URL`
  - `OLLAMA_EMBED_MODEL`
  - `OLLAMA_LLM_MODEL`


