# docs-search

Convert a GitHub documentation repository into a **local neurosymbolic search** index.

Neurosymbolic here means retrieval that combines:

- **Neural** — dense semantic embeddings (`fastembed` / ONNX, no PyTorch)
- **Symbolic** — a knowledge graph of docs, headings, links, and code symbols
- **Lexical** — BM25 keyword matching

Those signals are fused at query time so symbol-aware questions (APIs, config keys) and natural-language questions both work offline after indexing.

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)

## Install

```bash
uv sync
```

This creates `.venv` and installs the `docs-search` CLI.

## Quick start

```bash
# Build a local index from a GitHub repo (docs/, README, etc.)
uv run docs-search index astral-sh/uv

# Search it
uv run docs-search search "how do I add a dependency?" --repo astral-sh/uv

# List indexes
uv run docs-search list
```

Indexes live under `~/.docs-search/indexes/` by default. Cloned repos live under `~/.docs-search/repos/`.

### Local directory

```bash
uv run docs-search index ./path/to/docs --local
```

### Useful options

```bash
uv run docs-search index owner/repo --include docs --include README.md
uv run docs-search search "WidgetFactory.create" -r owner/repo -k 8
uv run docs-search search "timeout retries" -r owner/repo --json
```

Fusion weights can be tuned with `--neural`, `--lexical`, and `--symbolic`.

## How it works

1. **Ingest** — clone/update the GitHub repo (`git` via GitPython).
2. **Symbolic extract** — split markdown into heading-aware chunks; pull inline/fenced code symbols and markdown links.
3. **Knowledge graph** — connect `doc → chunk → symbol`, heading hierarchy, and `LINKS_TO` relations (`networkx`).
4. **Neural embed** — embed each chunk with `BAAI/bge-small-en-v1.5` via `fastembed`.
5. **Search** — run dense + BM25 retrieval, expand candidates through the graph, fuse scores, return snippets with matched symbols and graph hops.

```text
Query ─┬─► neural similarity
       ├─► BM25 lexical scores
       └─► symbol/graph expansion  ─► fused ranking ─► local hits
```

## Development

```bash
uv sync --group dev
uv run pytest
uv run ruff check src tests
```

## CLI

| Command | Purpose |
|---------|---------|
| `docs-search ingest <repo>` | Clone/update only |
| `docs-search index <repo>` | Clone + build neurosymbolic index |
| `docs-search search <query>` | Hybrid local search |
| `docs-search list` | Show indexed repos |
| `docs-search version` | Print version |
