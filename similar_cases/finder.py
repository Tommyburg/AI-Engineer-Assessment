"""Embedding-based similar-case retrieval."""

from __future__ import annotations

import math
from typing import Protocol

from similar_cases.models import Case, SearchResult


class EmbeddingClient(Protocol):
    """Minimum interface required from an embedding provider."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed each input text in the same order."""
        ...


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity for two vectors."""
    if len(left) != len(right):
        raise ValueError("Embedding dimensions must match")

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0

    dot_product = sum(a * b for a, b in zip(left, right, strict=True))
    return dot_product / (left_norm * right_norm)


class SimilarCaseFinder:
    """Index case summaries and retrieve the closest matches."""

    def __init__(self, embedding_client: EmbeddingClient) -> None:
        self._embedding_client = embedding_client
        self._cases: list[Case] = []
        self._case_embeddings: list[list[float]] = []

    def index_cases(self, cases: list[Case]) -> None:
        """Replace the current index with the supplied cases."""
        if not cases:
            raise ValueError("At least one case is required")

        embeddings = self._embedding_client.embed([case.summary for case in cases])
        if len(embeddings) != len(cases):
            raise RuntimeError("Embedding provider returned an unexpected result count")

        self._cases = list(cases)
        self._case_embeddings = embeddings

    def search(self, query: str, *, top_k: int = 3) -> list[SearchResult]:
        """Return at most ``top_k`` cases in descending similarity order."""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query must not be empty")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        if not self._cases:
            raise RuntimeError("No cases indexed")

        query_embeddings = self._embedding_client.embed([clean_query])
        if len(query_embeddings) != 1:
            raise RuntimeError("Embedding provider did not return one query embedding")

        results = [
            SearchResult(
                case=case,
                similarity_score=cosine_similarity(query_embeddings[0], embedding),
            )
            for case, embedding in zip(
                self._cases,
                self._case_embeddings,
                strict=True,
            )
        ]
        return sorted(results, key=lambda result: result.similarity_score, reverse=True)[:top_k]