from __future__ import annotations

from pydantic import BaseModel, Field


class DocChunk(BaseModel):
    """A searchable documentation unit with symbolic metadata."""

    id: str
    repo: str
    path: str
    title: str
    heading_path: list[str] = Field(default_factory=list)
    text: str
    symbols: list[str] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)
    start_line: int = 1
    end_line: int = 1


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str
    weight: float = 1.0


class SearchHit(BaseModel):
    chunk_id: str
    repo: str
    path: str
    title: str
    heading_path: list[str] = Field(default_factory=list)
    snippet: str
    score: float
    neural_score: float = 0.0
    lexical_score: float = 0.0
    symbolic_score: float = 0.0
    matched_symbols: list[str] = Field(default_factory=list)
    graph_hops: list[str] = Field(default_factory=list)


class IndexMeta(BaseModel):
    repo: str
    source: str
    chunk_count: int
    symbol_count: int
    edge_count: int
    embed_model: str
    created_at: str
