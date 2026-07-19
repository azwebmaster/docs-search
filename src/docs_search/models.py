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


class RagSource(BaseModel):
    """A retrieved documentation chunk cited by a RAG answer."""

    chunk_id: str
    repo: str
    path: str
    title: str
    heading_path: list[str] = Field(default_factory=list)
    snippet: str
    text: str = ""
    score: float = 0.0


class RagAnswer(BaseModel):
    """Answer produced by retrieval-augmented generation over an index."""

    question: str
    answer: str
    sources: list[RagSource] = Field(default_factory=list)
    model: str = ""
    index_name: str | None = None
    index_version: str | None = None
    repo: str | None = None


class IndexMeta(BaseModel):
    """Metadata for a saved documentation index."""

    name: str
    version: str
    repo: str
    source: str
    chunk_count: int
    symbol_count: int
    edge_count: int
    embed_model: str
    created_at: str

    def key(self) -> str:
        return f"{self.name}@{self.version}"


class RegistryEntry(BaseModel):
    """One published index listed in an S3 registry manifest."""

    name: str
    version: str
    repo: str
    source: str = ""
    chunk_count: int = 0
    symbol_count: int = 0
    edge_count: int = 0
    embed_model: str = ""
    created_at: str = ""
    s3_key: str = ""
    size_bytes: int = 0
    published_at: str = ""

    def key(self) -> str:
        return f"{self.name}@{self.version}"


class RegistryManifest(BaseModel):
    """Root manifest stored at the registry prefix."""

    updated_at: str = ""
    indices: list[RegistryEntry] = Field(default_factory=list)
