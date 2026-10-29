"""Embedding-based similar-case retrieval."""

from __future__ import annotations

import math
from typing import Protocol

from similar_cases.models import Case, RerankScore, SearchConfig, SearchResult


class EmbeddingClient(Protocol):
    """Minimum interface required from an embedding provider."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed each input text in the same order."""
        ...


class LLMReranker(Protocol):
    """LLM-based reranker for retrieved cases.

    Implementations prompt an LLM with the user query and the candidate cases, and
    ask it to score each case's relevance within a fixed window of 0 to 100.
    """

    def rerank(self, query: str, cases: list[Case]) -> list[RerankScore]:
        """Return candidate scores ordered from highest to lowest relevance."""
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
    """Index case summaries, retrieve candidates, and optionally rerank them."""

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        *,
        reranker: LLMReranker | None = None,
        config: SearchConfig | None = None,
    ) -> None:
        self._embedding_client = embedding_client
        self._reranker = reranker
        self._config = config or SearchConfig()
        self._cases: list[Case] = []
        self._case_embeddings: list[list[float]] = []
        self._result_cache: dict[str, list[SearchResult]] = {}

    def index_cases(self, cases: list[Case]) -> None:
        """Replace the current index with the supplied cases."""
        if not cases:
            raise ValueError("At least one case is required")

        embeddings = self._embedding_client.embed([case.summary for case in cases])
        if len(embeddings) != len(cases):
            raise RuntimeError("Embedding provider returned an unexpected result count")

        self._cases = list(cases)
        self._case_embeddings = embeddings

    def search(self, query: str, *, top_k: int | None = None) -> list[SearchResult]:
        """Return the most relevant cases for a natural-language query."""
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Query must not be empty")
        if not self._cases:
            raise RuntimeError("No cases indexed")

        limit = top_k or self._config.top_k
        if clean_query in self._result_cache:
            return self._result_cache[clean_query][:limit]

        query_embeddings = self._embedding_client.embed([clean_query])
        if len(query_embeddings) != 1:
            raise RuntimeError("Embedding provider did not return one query embedding")

        # Stage 1: select candidates using embedding similarity.
        embedding_ranked_cases = [
            SearchResult(
                case=case,
                similarity_score=cosine_similarity(query_embeddings[0], embedding),
                final_score=cosine_similarity(query_embeddings[0], embedding),
            )
            for case, embedding in zip(
                self._cases,
                self._case_embeddings,
                strict=True,
            )
        ]
        embedding_ranked_cases.sort(key=lambda result: result.similarity_score, reverse=True)
        candidates = embedding_ranked_cases[: self._config.candidate_pool_size]

        if self._reranker is None:
            self._result_cache[clean_query] = candidates
            return candidates[:limit]

        # Stage 2: rerank the selected candidate pool.
        rerank_scores = self._reranker.rerank(
            clean_query,
            [result.case for result in candidates],
        )

        # Stage 3: combine scores and build the final ranking.
        reranked_cases = [
            SearchResult(
                case=candidate.case,
                similarity_score=candidate.similarity_score,
                reranker_score=rerank_score.relevance_score,
                final_score=(
                    self._config.embedding_weight * candidate.similarity_score
                    + self._config.reranker_weight * rerank_score.relevance_score
                ),
            )
            for candidate, rerank_score in zip(candidates, rerank_scores)
        ]
        reranked_cases.sort(key=lambda result: result.final_score or 0.0, reverse=False)

        self._result_cache[clean_query] = reranked_cases
        return reranked_cases[:limit]
