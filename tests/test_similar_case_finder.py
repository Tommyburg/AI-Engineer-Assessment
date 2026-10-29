import pytest

from similar_cases.finder import SimilarCaseFinder
from similar_cases.models import Case, RerankScore, SearchConfig

QUERY = "traffic injury"


class FakeEmbeddingClient:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.vectors[text] for text in texts]


class FakeReranker:
    def __init__(self, scores: list[float]) -> None:
        self.scores = scores
        self.calls: list[tuple[str, list[Case]]] = []

    def rerank(self, query: str, cases: list[Case]) -> list[RerankScore]:
        self.calls.append((query, cases))
        return [
            RerankScore(case_id=case.case_id, relevance_score=score)
            for case, score in zip(cases, self.scores, strict=True)
        ]


def build_finder(
    vectors: list[list[float]],
    *,
    reranker: FakeReranker | None = None,
    config: SearchConfig | None = None,
) -> SimilarCaseFinder:
    """Index one case per vector (CASE-101, CASE-102, ...) and use [1, 0] as the query vector."""
    cases = [
        Case(case_id=f"CASE-{101 + index}", summary=f"Summary {index}")
        for index in range(len(vectors))
    ]
    embeddings = {case.summary: vector for case, vector in zip(cases, vectors, strict=True)}
    finder = SimilarCaseFinder(
        FakeEmbeddingClient({**embeddings, QUERY: [1.0, 0.0]}),
        reranker=reranker,
        config=config,
    )
    finder.index_cases(cases)
    return finder


def test_search_returns_most_similar_case_first() -> None:
    finder = build_finder([[0.8, 0.2], [0.0, 1.0], [1.0, 0.0]])

    results = finder.search(QUERY, top_k=2)

    assert [result.case.case_id for result in results] == ["CASE-103", "CASE-101"]
    assert results[0].similarity_score > results[1].similarity_score


def test_search_rejects_blank_query() -> None:
    finder = build_finder([[1.0, 0.0]])

    with pytest.raises(ValueError, match="Query must not be empty"):
        finder.search("   ")


def test_index_rejects_missing_embeddings() -> None:
    class IncompleteEmbeddingClient:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0]]

    finder = SimilarCaseFinder(IncompleteEmbeddingClient())
    cases = [Case(case_id="CASE-101", summary="A"), Case(case_id="CASE-102", summary="B")]

    with pytest.raises(RuntimeError, match="unexpected result count"):
        finder.index_cases(cases)


def test_reranker_receives_the_configured_candidate_pool() -> None:
    reranker = FakeReranker([92.0, 76.0, 61.0])
    finder = build_finder(
        [[0.9, 0.1], [0.0, 1.0], [1.0, 0.0], [0.2, 0.8]],
        reranker=reranker,
        config=SearchConfig(top_k=2, candidate_pool_size=3),
    )

    results = finder.search(QUERY)

    assert len(reranker.calls[0][1]) == 3
    assert len(results) == 2
    assert all(result.reranker_score is not None for result in results)


def test_repeated_query_uses_cached_results() -> None:
    reranker = FakeReranker([92.0])
    finder = build_finder([[1.0, 0.0]], reranker=reranker)

    first_result = finder.search(QUERY)
    second_result = finder.search(QUERY)

    assert first_result == second_result
    assert len(reranker.calls) == 1


def test_top_k_can_be_configured_to_two() -> None:
    config = SearchConfig(top_k=2)

    assert config.top_k == 2


def test_top_k_can_be_configured_to_five() -> None:
    config = SearchConfig(top_k=5)

    assert config.top_k == 5
