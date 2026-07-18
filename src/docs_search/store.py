from __future__ import annotations

import json
from pathlib import Path

import networkx as nx
import numpy as np

from docs_search.config import DEFAULT_INDEX_DIR
from docs_search.models import DocChunk, GraphEdge, IndexMeta


class IndexStore:
    """Filesystem-backed local index for one documentation repository."""

    def __init__(self, repo_slug: str, index_dir: Path | None = None) -> None:
        self.repo_slug = repo_slug
        self.root = (index_dir or DEFAULT_INDEX_DIR) / repo_slug.replace("/", "__")
        self.root.mkdir(parents=True, exist_ok=True)

    @property
    def chunks_path(self) -> Path:
        return self.root / "chunks.jsonl"

    @property
    def embeddings_path(self) -> Path:
        return self.root / "embeddings.npy"

    @property
    def edges_path(self) -> Path:
        return self.root / "edges.jsonl"

    @property
    def graph_path(self) -> Path:
        return self.root / "graph.json"

    @property
    def meta_path(self) -> Path:
        return self.root / "meta.json"

    def save(
        self,
        chunks: list[DocChunk],
        embeddings: np.ndarray,
        edges: list[GraphEdge],
        graph: nx.MultiDiGraph,
        meta: IndexMeta,
    ) -> Path:
        with self.chunks_path.open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(chunk.model_dump_json() + "\n")

        np.save(self.embeddings_path, embeddings)

        with self.edges_path.open("w", encoding="utf-8") as fh:
            for edge in edges:
                fh.write(edge.model_dump_json() + "\n")

        # Node-link data is portable and enough to rebuild the MultiDiGraph.
        data = nx.node_link_data(graph, edges="links")
        self.graph_path.write_text(json.dumps(data), encoding="utf-8")
        self.meta_path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
        return self.root

    def exists(self) -> bool:
        return self.chunks_path.exists() and self.embeddings_path.exists() and self.meta_path.exists()

    def load_chunks(self) -> list[DocChunk]:
        chunks: list[DocChunk] = []
        with self.chunks_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    chunks.append(DocChunk.model_validate_json(line))
        return chunks

    def load_embeddings(self) -> np.ndarray:
        return np.load(self.embeddings_path)

    def load_graph(self) -> nx.MultiDiGraph:
        data = json.loads(self.graph_path.read_text(encoding="utf-8"))
        return nx.node_link_graph(data, edges="links", multigraph=True, directed=True)

    def load_meta(self) -> IndexMeta:
        return IndexMeta.model_validate_json(self.meta_path.read_text(encoding="utf-8"))


def list_indexes(index_dir: Path | None = None) -> list[IndexMeta]:
    root = index_dir or DEFAULT_INDEX_DIR
    if not root.exists():
        return []
    metas: list[IndexMeta] = []
    for meta_path in sorted(root.glob("*/meta.json")):
        metas.append(IndexMeta.model_validate_json(meta_path.read_text(encoding="utf-8")))
    return metas
