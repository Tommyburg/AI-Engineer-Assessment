"""Search for comparable insurance cases."""

from similar_cases.finder import EmbeddingClient, SimilarCaseFinder
from similar_cases.models import Case, SearchResult

__all__ = ["Case", "EmbeddingClient", "SearchResult", "SimilarCaseFinder"]