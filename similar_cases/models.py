"""Domain models for similar-case search."""

from pydantic import BaseModel, ConfigDict, Field


class Case(BaseModel):
    """A historic insurance case available for retrieval."""

    model_config = ConfigDict(frozen=True)

    case_id: str = Field(min_length=1)
    summary: str = Field(min_length=1)


class SearchResult(BaseModel):
    """A case and its cosine similarity to the search query."""

    model_config = ConfigDict(frozen=True)

    case: Case
    similarity_score: float = Field(ge=-1.0, le=1.0)