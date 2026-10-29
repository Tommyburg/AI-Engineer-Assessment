"""Domain models for similar-case search."""

from pydantic import BaseModel, ConfigDict, Field


class Case(BaseModel):
    """A historic insurance case available for retrieval."""

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)


class SearchResult(BaseModel):
    """A case and the scores produced by the search stages."""

    model_config = ConfigDict(frozen=True)

    case: Case
    similarity_score: float = Field(ge=-1.0, le=1.0)
    reranker_score: float | None = Field(default=None, ge=0.0, le=100.0)
    final_score: float | None = None


class RerankScore(BaseModel):
    """Relevance assigned to a case by a reranking provider."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    relevance_score: float = Field(ge=0.0, le=100.0)


class SearchConfig(BaseModel):
    """Configuration for retrieval and reranking."""

    top_k: int = 3
    candidate_pool_size: int = 8
    embedding_weight: float = 0.4
    reranker_weight: float = 0.6
