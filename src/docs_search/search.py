from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

from docs_search.config import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_LEXICAL_WEIGHT,
    DEFAULT_NEURAL_WEIGHT,
    DEFAULT_SYMBOLIC_WEIGHT,
)
from docs_search.embed import embed_query
from docs_search.extract import extract_symbols
from docs_search.graph import expand_via_graph
from docs_search.models import DocChunk, SearchHit
from docs_search.store import resolve_store

_TOKEN_RE = re.compile(r"[a-z0-9_]+(?:\.[a-z0-9_]+)*", re.I)


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


@dataclass
class NeurosymbolicIndex:
    """In-memory neurosymbolic index: neural + lexical + symbolic graph."""

    repo: str
    chunks: list[DocChunk]
    embeddings: np.ndarray
    graph: object
    embed_model: str = DEFAULT_EMBED_MODEL
    _bm25: BM25Okapi | None = None
    _id_to_idx: dict[str, int] | None = None

    def __post_init__(self) -> None:
        corpus = [tokenize(c.text) for c in self.chunks]
        self._bm25 = BM25Okapi(corpus) if corpus else None
        self._id_to_idx = {c.id: i for i, c in enumerate(self.chunks)}

    @classmethod
    def load(
        cls,
        repo_slug: str | None = None,
        index_dir=None,
        *,
        name: str | None = None,
        version: str | None = None,
    ) -> NeurosymbolicIndex:
        store = resolve_store(
            name=name,
            version=version,
            repo=repo_slug,
            index_dir=index_dir,
        )
        if not store.exists():
            label = name or repo_slug or "index"
            raise FileNotFoundError(
                f"No local index for {label!r}. Run: docs-search index <source> --name …"
            )
        meta = store.load_meta()
        return cls(
            repo=meta.repo,
            chunks=store.load_chunks(),
            embeddings=store.load_embeddings(),
            graph=store.load_graph(),
            embed_model=meta.embed_model,
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        neural_weight: float = DEFAULT_NEURAL_WEIGHT,
        lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
        symbolic_weight: float = DEFAULT_SYMBOLIC_WEIGHT,
    ) -> list[SearchHit]:
        if not self.chunks or not query.strip():
            return []

        query_symbols = extract_symbols(query)
        # Also treat bare query tokens that match known symbols.
        known_symbols = {
            data.get("label", node.removeprefix("symbol:"))
            for node, data in self.graph.nodes(data=True)
            if str(node).startswith("symbol:")
        }
        for token in tokenize(query):
            for sym in known_symbols:
                if token == sym.lower() or token in sym.lower():
                    if sym not in query_symbols:
                        query_symbols.append(sym)

        neural_scores = self._neural_scores(query)
        lexical_scores = self._lexical_scores(query)

        # Seed graph expansion from top neural + lexical candidates and query symbols.
        seed_ids = self._seed_chunk_ids(neural_scores, lexical_scores, limit=12)
        symbolic_map = expand_via_graph(
            self.graph,
            seed_ids,
            query_symbols,
            max_hops=2,
            limit=max(40, top_k * 5),
        )

        # Normalize score channels to [0, 1].
        n_norm = _minmax(neural_scores)
        l_norm = _minmax(lexical_scores)
        s_raw = np.zeros(len(self.chunks), dtype=np.float32)
        hop_paths: dict[int, list[str]] = {}
        for chunk_id, (score, path) in symbolic_map.items():
            idx = self._id_to_idx.get(chunk_id)
            if idx is not None:
                s_raw[idx] = score
                hop_paths[idx] = path
        s_norm = _minmax(s_raw)

        combined = (
            neural_weight * n_norm
            + lexical_weight * l_norm
            + symbolic_weight * s_norm
        )

        # Soft boost when query symbols are directly mentioned.
        if query_symbols:
            qset = {s.lower() for s in query_symbols}
            for i, chunk in enumerate(self.chunks):
                overlap = qset.intersection({s.lower() for s in chunk.symbols})
                if overlap:
                    combined[i] += 0.05 * len(overlap)

        ranking = np.argsort(-combined)[:top_k]
        hits: list[SearchHit] = []
        for idx in ranking:
            chunk = self.chunks[int(idx)]
            matched = [
                s
                for s in chunk.symbols
                if s.lower() in {q.lower() for q in query_symbols} or any(
                    t in s.lower() for t in tokenize(query)
                )
            ]
            hits.append(
                SearchHit(
                    chunk_id=chunk.id,
                    repo=chunk.repo,
                    path=chunk.path,
                    title=chunk.title,
                    heading_path=chunk.heading_path,
                    snippet=_snippet(chunk.text, query),
                    score=float(combined[idx]),
                    neural_score=float(n_norm[idx]),
                    lexical_score=float(l_norm[idx]),
                    symbolic_score=float(s_norm[idx]),
                    matched_symbols=matched[:12],
                    graph_hops=hop_paths.get(int(idx), [])[:8],
                )
            )
        return hits

    def _neural_scores(self, query: str) -> np.ndarray:
        if self.embeddings.size == 0:
            return np.zeros(len(self.chunks), dtype=np.float32)
        q = embed_query(query, model_name=self.embed_model)
        return self.embeddings @ q

    def _lexical_scores(self, query: str) -> np.ndarray:
        if self._bm25 is None:
            return np.zeros(len(self.chunks), dtype=np.float32)
        tokens = tokenize(query)
        if not tokens:
            return np.zeros(len(self.chunks), dtype=np.float32)
        return np.asarray(self._bm25.get_scores(tokens), dtype=np.float32)

    def _seed_chunk_ids(
        self,
        neural_scores: np.ndarray,
        lexical_scores: np.ndarray,
        *,
        limit: int,
    ) -> list[str]:
        blend = 0.55 * _minmax(neural_scores) + 0.45 * _minmax(lexical_scores)
        top = np.argsort(-blend)[:limit]
        return [self.chunks[int(i)].id for i in top]


def _minmax(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    lo = float(values.min())
    hi = float(values.max())
    if hi - lo < 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - lo) / (hi - lo)).astype(np.float32)


def _snippet(text: str, query: str, width: int = 240) -> str:
    tokens = tokenize(query)
    lower = text.lower()
    pos = -1
    for token in tokens:
        pos = lower.find(token)
        if pos >= 0:
            break
    if pos < 0:
        snippet = text[:width].strip()
        return snippet + ("…" if len(text) > width else "")
    start = max(0, pos - width // 3)
    end = min(len(text), start + width)
    snippet = text[start:end].strip()
    if start > 0:
        snippet = "…" + snippet
    if end < len(text):
        snippet = snippet + "…"
    return snippet
