from __future__ import annotations

import json
import shutil
import tarfile
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np

from docs_search.config import (
    DEFAULT_INDEX_DIR,
    DEFAULT_INDEX_VERSION,
    default_name_from_repo,
    sanitize_index_name,
    sanitize_index_version,
)
from docs_search.models import DocChunk, GraphEdge, IndexMeta


class IndexStore:
    """Filesystem-backed local index for one named/versioned documentation set."""

    def __init__(
        self,
        name: str,
        version: str = DEFAULT_INDEX_VERSION,
        index_dir: Path | None = None,
    ) -> None:
        self.name = sanitize_index_name(name)
        self.version = sanitize_index_version(version)
        self.index_dir = index_dir or DEFAULT_INDEX_DIR
        self.root = self.index_dir / self.name / self.version
        self.root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_meta(cls, meta: IndexMeta, index_dir: Path | None = None) -> IndexStore:
        return cls(meta.name, meta.version, index_dir=index_dir)

    @classmethod
    def for_repo(
        cls,
        repo_slug: str,
        *,
        name: str | None = None,
        version: str = DEFAULT_INDEX_VERSION,
        index_dir: Path | None = None,
    ) -> IndexStore:
        resolved = name or default_name_from_repo(repo_slug)
        return cls(resolved, version, index_dir=index_dir)

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
        if meta.name != self.name or meta.version != self.version:
            raise ValueError("IndexMeta name/version must match IndexStore")

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
        return _load_meta_file(self.meta_path)

    def package_tarball(self, dest: Path | None = None) -> Path:
        """Create a .tar.gz of this index directory."""
        if not self.exists():
            raise FileNotFoundError(f"No complete index at {self.root}")
        if dest is None:
            dest = self.root.parent / f"{self.name}-{self.version}.tar.gz"
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(dest, "w:gz") as tar:
            for path in (
                self.meta_path,
                self.chunks_path,
                self.embeddings_path,
                self.edges_path,
                self.graph_path,
            ):
                if path.exists():
                    tar.add(path, arcname=path.name)
        return dest

    @classmethod
    def extract_tarball(
        cls,
        archive: Path,
        *,
        name: str | None = None,
        version: str | None = None,
        index_dir: Path | None = None,
    ) -> IndexStore:
        """Extract a packaged index into the local index directory."""
        with tempfile.TemporaryDirectory(prefix="docs-search-extract-") as tmp:
            tmp_path = Path(tmp)
            with tarfile.open(archive, "r:gz") as tar:
                tar.extractall(tmp_path, filter="data")
            meta_path = tmp_path / "meta.json"
            if not meta_path.exists():
                raise ValueError(f"Archive {archive} is missing meta.json")
            meta = _load_meta_file(meta_path)
            store = cls(
                name or meta.name,
                version or meta.version,
                index_dir=index_dir,
            )
            # Replace any existing version directory atomically-ish.
            if store.root.exists():
                shutil.rmtree(store.root)
            store.root.mkdir(parents=True, exist_ok=True)
            for item in tmp_path.iterdir():
                shutil.move(str(item), str(store.root / item.name))
            # Ensure meta name/version match destination.
            if store.name != meta.name or store.version != meta.version:
                meta = meta.model_copy(update={"name": store.name, "version": store.version})
                store.meta_path.write_text(meta.model_dump_json(indent=2), encoding="utf-8")
            return store


def _load_meta_file(path: Path) -> IndexMeta:
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Backward-compat for indexes written before name/version existed.
    if "name" not in raw:
        repo = raw.get("repo", "unknown")
        raw["name"] = default_name_from_repo(str(repo))
    if "version" not in raw:
        raw["version"] = DEFAULT_INDEX_VERSION
    return IndexMeta.model_validate(raw)


def list_indexes(index_dir: Path | None = None) -> list[IndexMeta]:
    """List local indexes (name/version layout and legacy flat layout)."""
    root = index_dir or DEFAULT_INDEX_DIR
    if not root.exists():
        return []

    metas: list[IndexMeta] = []
    seen: set[Path] = set()

    # Preferred layout: indexes/<name>/<version>/meta.json
    for meta_path in sorted(root.glob("*/*/meta.json")):
        metas.append(_load_meta_file(meta_path))
        seen.add(meta_path.resolve())

    # Legacy layout: indexes/<repo_slug>/meta.json
    for meta_path in sorted(root.glob("*/meta.json")):
        if meta_path.resolve() in seen:
            continue
        metas.append(_load_meta_file(meta_path))

    metas.sort(key=lambda m: (m.name, m.version, m.created_at))
    return metas


def find_index(
    *,
    name: str | None = None,
    version: str | None = None,
    repo: str | None = None,
    index_dir: Path | None = None,
) -> IndexMeta:
    """Resolve a single local index by name/version and/or repo."""
    metas = list_indexes(index_dir)
    if not metas:
        raise FileNotFoundError("No local indexes found")

    candidates = metas
    if name:
        name = sanitize_index_name(name)
        candidates = [m for m in candidates if m.name == name]
    if repo:
        candidates = [m for m in candidates if m.repo == repo]
    if version:
        version = sanitize_index_version(version)
        candidates = [m for m in candidates if m.version == version]
        if not candidates:
            raise FileNotFoundError(
                f"No local index matching name={name!r} version={version!r} repo={repo!r}"
            )
        return candidates[-1]

    if not candidates:
        raise FileNotFoundError(
            f"No local index matching name={name!r} version={version!r} repo={repo!r}"
        )

    # Prefer newest created_at when version omitted.
    candidates = sorted(candidates, key=lambda m: m.created_at)
    return candidates[-1]


def resolve_store(
    *,
    name: str | None = None,
    version: str | None = None,
    repo: str | None = None,
    index_dir: Path | None = None,
) -> IndexStore:
    meta = find_index(name=name, version=version, repo=repo, index_dir=index_dir)
    # Support legacy flat directories that are not under name/version.
    root = index_dir or DEFAULT_INDEX_DIR
    versioned = root / meta.name / meta.version
    if (versioned / "meta.json").exists():
        return IndexStore(meta.name, meta.version, index_dir=index_dir)

    legacy = root / meta.repo.replace("/", "__")
    if (legacy / "meta.json").exists():
        store = IndexStore.__new__(IndexStore)
        store.name = meta.name
        store.version = meta.version
        store.index_dir = root
        store.root = legacy
        return store

    return IndexStore(meta.name, meta.version, index_dir=index_dir)
