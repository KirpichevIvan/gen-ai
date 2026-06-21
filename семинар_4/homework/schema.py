"""Response schema for the homework RAG pipeline."""

from pydantic import BaseModel, Field


class RAGAnswer(BaseModel):
    answer: str = Field(description="Final answer based only on retrieved context")
    quotes: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Short exact quotes from retrieved chunks",
    )
    confidence: float = Field(
        ge=0,
        le=1,
        description="Answer confidence: low when retrieved context is weak",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Chunk ids used for the answer, e.g. 'habr_01...__recursive__0'",
    )
