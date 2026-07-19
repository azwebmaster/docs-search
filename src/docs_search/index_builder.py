from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from docs_search.config import DEFAULT_EMBED_MODEL, DEFAULT_INDEX_DIR, DEFAULT_REPOS_DIR
from docs_search.embed import embed_texts
from docs_search.extract import extract_repo_chunks
from docs_search.graph import build_knowledge_graph
from docs_search.ingest import clone_or_update, iter_doc_files, normalize_github_source
from docs_search.models import IndexMeta
from docs_search.store import IndexStore

ProgressFn = Callable[[str], None]


def index_github_repo(
    source: str,
    *,
    branch: str | None = None,
    repos_dir: Path | None = None,
    index_dir: Path | None = None,
    include_dirs: list[str] | None = None,
    embed_model: str = DEFAULT_EMBED_MODEL,
    force_clone: bool = False,
    progress: ProgressFn | None = None,
) -> IndexMeta:
    """Clone/update a GitHub repo and build a local neurosymbolic index."""

    def log(msg: str) -> None:
        if progress:
            progress(msg)

    repos_dir = repos_dir or DEFAULT_REPOS_DIR
    index_dir = index_dir or DEFAULT_INDEX_DIR

    log(f"Fetching {source}…")
    repo_root, repo_slug = clone_or_update(
        source,
        repos_dir=repos_dir,
        branch=branch,
        force=force_clone,
    )

    log("Extracting documentation (symbolic layer)…")
    files = iter_doc_files(repo_root, include_dirs=include_dirs)
    if not files:
        raise RuntimeError(f"No documentation files found in {repo_slug}")

    chunks = extract_repo_chunks(repo_root, repo_slug, files)
    log(f"Parsed {len(files)} files → {len(chunks)} chunks")

    log("Building knowledge graph…")
    graph, edges = build_knowledge_graph(chunks)
    symbol_count = sum(1 for n, d in graph.nodes(data=True) if d.get("kind") == "symbol")

    log(f"Embedding chunks with {embed_model} (neural layer)…")
    texts = [f"{c.title}\n{' > '.join(c.heading_path)}\n{c.text}" for c in chunks]
    embeddings = embed_texts(texts, model_name=embed_model)

    meta = IndexMeta(
        repo=repo_slug,
        source=normalize_github_source(source)[0],
        chunk_count=len(chunks),
        symbol_count=symbol_count,
        edge_count=len(edges),
        embed_model=embed_model,
        created_at=datetime.now(timezone.utc).isoformat(),
    )

    store = IndexStore(repo_slug, index_dir=index_dir)
    out = store.save(chunks, embeddings, edges, graph, meta)
    log(f"Index written to {out}")
    return meta


def index_local_path(
    path: Path,
    *,
    repo_slug: str | None = None,
    index_dir: Path | None = None,
    include_dirs: list[str] | None = None,
    embed_model: str = DEFAULT_EMBED_MODEL,
    progress: ProgressFn | None = None,
) -> IndexMeta:
    """Index a local documentation directory (useful for tests / offline mirrors)."""

    def log(msg: str) -> None:
        if progress:
            progress(msg)

    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(path)

    repo_slug = repo_slug or f"local/{path.name}"
    index_dir = index_dir or DEFAULT_INDEX_DIR

    log("Extracting documentation (symbolic layer)…")
    files = iter_doc_files(path, include_dirs=include_dirs or ["."])
    chunks = extract_repo_chunks(path, repo_slug, files)
    if not chunks:
        raise RuntimeError(f"No documentation chunks extracted from {path}")

    log("Building knowledge graph…")
    graph, edges = build_knowledge_graph(chunks)
    symbol_count = sum(1 for n, d in graph.nodes(data=True) if d.get("kind") == "symbol")

    log(f"Embedding chunks with {embed_model} (neural layer)…")
    texts = [f"{c.title}\n{' > '.join(c.heading_path)}\n{c.text}" for c in chunks]
    embeddings = embed_texts(texts, model_name=embed_model)

    meta = IndexMeta(
        repo=repo_slug,
        source=str(path),
        chunk_count=len(chunks),
        symbol_count=symbol_count,
        edge_count=len(edges),
        embed_model=embed_model,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    store = IndexStore(repo_slug, index_dir=index_dir)
    store.save(chunks, embeddings, edges, graph, meta)
    log(f"Index written to {store.root}")
    return meta
