from similar_cases.finder import SimilarCaseFinder
from similar_cases.models import Case


class FakeEmbeddingClient:
    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self.vectors = vectors
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [self.vectors[text] for text in texts]


def test_search_returns_most_similar_case_first() -> None:
    cases = [
        Case(case_id="CASE-101", summary="Rear-end collision with neck injury"),
        Case(case_id="CASE-102", summary="Water damage caused by a leaking roof"),
        Case(case_id="CASE-103", summary="Cyclist injured at a roundabout"),
    ]
    client = FakeEmbeddingClient(
        {
            cases[0].summary: [0.8, 0.2],
            cases[1].summary: [0.0, 1.0],
            cases[2].summary: [1.0, 0.0],
            "traffic injury": [1.0, 0.0],
        }
    )
    finder = SimilarCaseFinder(client)
    finder.index_cases(cases)

    results = finder.search("traffic injury", top_k=2)

    assert [result.case.case_id for result in results] == ["CASE-103", "CASE-101"]
    assert results[0].similarity_score > results[1].similarity_score


def test_search_rejects_blank_query() -> None:
    case = Case(case_id="CASE-101", summary="Rear-end collision")
    finder = SimilarCaseFinder(FakeEmbeddingClient({case.summary: [1.0, 0.0]}))
    finder.index_cases([case])

    try:
        finder.search("   ")
    except ValueError as error:
        assert str(error) == "Query must not be empty"
    else:
        raise AssertionError("Expected a blank query to be rejected")


def test_index_rejects_missing_embeddings() -> None:
    cases = [
        Case(case_id="CASE-101", summary="Rear-end collision"),
        Case(case_id="CASE-102", summary="Water damage"),
    ]

    class IncompleteEmbeddingClient:
        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, 0.0]]

    finder = SimilarCaseFinder(IncompleteEmbeddingClient())

    try:
        finder.index_cases(cases)
    except RuntimeError as error:
        assert "unexpected result count" in str(error)
    else:
        raise AssertionError("Expected incomplete embeddings to be rejected")