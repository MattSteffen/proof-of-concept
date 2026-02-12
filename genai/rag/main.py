"""Minimal local RAG pipeline using Ollama and in-memory vectors."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path

from ollama import Client


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    denominator = _norm(a) * _norm(b)
    if denominator == 0:
        return 0.0
    return _dot(a, b) / denominator


class EmbeddingModel:
    def __init__(self, model_name: str, base_url: str) -> None:
        self.model_name = model_name
        # Ollama's Python client expects host, not /api/* endpoints.
        self.client = Client(host=base_url)

    def embed(self, text: str) -> list[float]:
        response = self.client.embed(model=self.model_name, input=text)
        embeddings = response.get("embeddings", [])
        if not embeddings:
            raise ValueError("Embedding response did not include vectors.")
        return embeddings[0]


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    score: float


class VectorDatabase:
    def __init__(self, model: EmbeddingModel) -> None:
        self.model = model
        self._rows: list[tuple[str, list[float]]] = []

    def add(self, text: str) -> None:
        embedding = self.model.embed(text)
        self._rows.append((text, embedding))

    def search(self, query: str, top_k: int = 2) -> list[RetrievedChunk]:
        if not self._rows:
            return []

        query_embedding = self.model.embed(query)
        scored_rows = [
            RetrievedChunk(text=row_text, score=_cosine_similarity(query_embedding, row_embedding))
            for row_text, row_embedding in self._rows
        ]
        scored_rows.sort(key=lambda chunk: chunk.score, reverse=True)
        return scored_rows[:top_k]


class LLMModel:
    def __init__(self, model_name: str, base_url: str) -> None:
        self.model_name = model_name
        self.client = Client(host=base_url)

    def generate(self, prompt: str) -> str:
        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        message = response.get("message", {})
        content = message.get("content")
        if not content:
            raise ValueError("Chat response did not include message content.")
        return content


def _build_prompt(question: str, context_chunks: list[RetrievedChunk]) -> str:
    context = "\n".join(f"- {chunk.text}" for chunk in context_chunks)
    return (
        "You are a helpful assistant. Answer the question using only the context below. "
        "If the answer is not in the context, say you do not know.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )


def _load_sample_documents() -> list[str]:
    sample_document_path = Path(__file__).parent / "sample_document.txt"
    if not sample_document_path.exists():
        raise FileNotFoundError(f"Missing sample document: {sample_document_path}")

    raw_text = sample_document_path.read_text(encoding="utf-8")
    paragraphs = [paragraph.strip() for paragraph in raw_text.split("\n\n") if paragraph.strip()]
    if not paragraphs:
        raise ValueError(f"No usable paragraphs in sample document: {sample_document_path}")
    return paragraphs


def main() -> None:
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    embedding_model_name = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")
    llm_model_name = os.getenv("OLLAMA_LLM_MODEL", "gpt-oss:20b")

    embedding_model = EmbeddingModel(model_name=embedding_model_name, base_url=base_url)
    vector_database = VectorDatabase(model=embedding_model)
    llm_model = LLMModel(model_name=llm_model_name, base_url=base_url)

    sample_documents = _load_sample_documents()
    for document in sample_documents:
        vector_database.add(document)

    question = "What does RAG stand for and what is it used for?"
    retrieved = vector_database.search(question, top_k=2)
    prompt = _build_prompt(question=question, context_chunks=retrieved)
    answer = llm_model.generate(prompt)

    print(f"Question: {question}")
    print("\nRetrieved context:")
    for chunk in retrieved:
        print(f"- score={chunk.score:.4f} | {chunk.text}")
    print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    main()