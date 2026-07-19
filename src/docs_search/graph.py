from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import networkx as nx

from docs_search.models import DocChunk, GraphEdge


def build_knowledge_graph(chunks: list[DocChunk]) -> tuple[nx.MultiDiGraph, list[GraphEdge]]:
    """Build a symbolic knowledge graph over documentation chunks.

    Node types (encoded in node id prefixes):
      chunk:<id>   — documentation chunk
      doc:<path>   — source document
      symbol:<name>— extracted API / code symbol
      heading:<...> — heading path node
    """
    g = nx.MultiDiGraph()
    edges: list[GraphEdge] = []

    def add_edge(source: str, target: str, relation: str, weight: float = 1.0) -> None:
        g.add_edge(source, target, relation=relation, weight=weight)
        edges.append(GraphEdge(source=source, target=target, relation=relation, weight=weight))

    path_to_chunks: dict[str, list[str]] = defaultdict(list)

    for chunk in chunks:
        chunk_node = f"chunk:{chunk.id}"
        doc_node = f"doc:{chunk.path}"
        g.add_node(chunk_node, kind="chunk", title=chunk.title, path=chunk.path)
        g.add_node(doc_node, kind="doc", path=chunk.path)
        add_edge(doc_node, chunk_node, "CONTAINS")
        path_to_chunks[chunk.path].append(chunk.id)

        if chunk.heading_path:
            heading_node = "heading:" + " > ".join(chunk.heading_path)
            g.add_node(heading_node, kind="heading", label=" > ".join(chunk.heading_path))
            add_edge(chunk_node, heading_node, "UNDER_HEADING")
            # Nest successive heading levels.
            for i in range(1, len(chunk.heading_path)):
                parent = "heading:" + " > ".join(chunk.heading_path[:i])
                child = "heading:" + " > ".join(chunk.heading_path[: i + 1])
                g.add_node(parent, kind="heading")
                g.add_node(child, kind="heading")
                if not g.has_edge(parent, child):
                    add_edge(parent, child, "PARENT_OF")

        for symbol in chunk.symbols:
            sym_node = f"symbol:{symbol}"
            g.add_node(sym_node, kind="symbol", label=symbol)
            add_edge(chunk_node, sym_node, "MENTIONS", weight=1.2)
            add_edge(sym_node, chunk_node, "MENTIONED_IN", weight=1.2)

        for link in chunk.links:
            target_path = _resolve_link(chunk.path, link)
            if target_path:
                target_doc = f"doc:{target_path}"
                g.add_node(target_doc, kind="doc", path=target_path)
                add_edge(doc_node, target_doc, "LINKS_TO")

    # Adjacent chunks within the same document are related.
    for _path, chunk_ids in path_to_chunks.items():
        for left, right in zip(chunk_ids, chunk_ids[1:], strict=False):
            add_edge(f"chunk:{left}", f"chunk:{right}", "NEXT", weight=0.5)
            add_edge(f"chunk:{right}", f"chunk:{left}", "PREV", weight=0.5)

    return g, edges


def _resolve_link(source_path: str, link: str) -> str | None:
    if link.startswith(("http://", "https://", "mailto:")):
        return None
    clean = link.split("?", 1)[0].strip()
    if not clean:
        return None
    # Normalize relative markdown links to repo-relative paths.
    base = Path(source_path).parent
    resolved = (base / clean).as_posix()
    # Collapse ./ and ../
    parts: list[str] = []
    for part in resolved.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    return "/".join(parts)


def expand_via_graph(
    graph: nx.MultiDiGraph,
    seed_chunk_ids: list[str],
    query_symbols: list[str],
    *,
    max_hops: int = 2,
    limit: int = 40,
) -> dict[str, tuple[float, list[str]]]:
    """Symbolically expand candidate chunks through the knowledge graph.

    Returns mapping of chunk_id -> (symbolic_score, hop_path labels).
    """
    scores: dict[str, tuple[float, list[str]]] = {}

    seeds: list[tuple[str, float, list[str]]] = []
    for chunk_id in seed_chunk_ids:
        seeds.append((f"chunk:{chunk_id}", 1.0, [chunk_id]))
    for symbol in query_symbols:
        node = f"symbol:{symbol}"
        if graph.has_node(node):
            seeds.append((node, 1.1, [symbol]))

    visited: set[str] = set()
    frontier = list(seeds)

    while frontier:
        node, score, path = frontier.pop(0)
        if node in visited:
            continue
        visited.add(node)

        if node.startswith("chunk:"):
            chunk_id = node.removeprefix("chunk:")
            prev = scores.get(chunk_id)
            if prev is None or score > prev[0]:
                scores[chunk_id] = (score, path)
            if len(scores) >= limit and score < 0.4:
                continue

        if len(path) > max_hops:
            continue

        if not graph.has_node(node):
            continue

        for _, neighbor, data in graph.out_edges(node, data=True):
            relation = data.get("relation", "")
            weight = float(data.get("weight", 1.0))
            # Prefer symbol and link relations for expansion.
            decay = 0.65 if relation in {"MENTIONS", "MENTIONED_IN", "LINKS_TO"} else 0.45
            next_score = score * decay * weight
            if next_score < 0.15:
                continue
            label = neighbor.split(":", 1)[-1]
            frontier.append((neighbor, next_score, path + [label]))

    return scores
