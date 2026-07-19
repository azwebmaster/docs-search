# docs-search documentation

Guides for installing, indexing, searching, answering questions with RAG, sharing indexes via S3, and using the web API.

## Guides

| Guide | What it covers |
|-------|----------------|
| [Getting started](getting-started.md) | Install, first index, first search and ask |
| [Indexing](indexing.md) | Building named/versioned indexes from GitHub or local docs |
| [Search](search.md) | Hybrid neurosymbolic search, fusion weights, auto-pull |
| [Ask (RAG)](ask-rag.md) | Retrieval-augmented answers with OpenAI-compatible LLMs |
| [S3 registry](registry.md) | Publish, list, and pull shared indexes |
| [Web server & API](web-api.md) | Browse indexes and call search/ask over HTTP |
| [Configuration](configuration.md) | Paths, config file, env vars, naming rules |
| [CLI reference](cli-reference.md) | Every command and option |

## Feature overview

`docs-search` turns documentation (usually a GitHub repo) into a **local neurosymbolic index**:

1. **Neural** — dense embeddings (`fastembed` / ONNX, default `BAAI/bge-small-en-v1.5`)
2. **Lexical** — BM25 keyword matching
3. **Symbolic** — a knowledge graph of docs, headings, links, and code symbols

At query time those signals are fused. You can search for hits, ask an LLM for a grounded answer with citations, publish indexes to an S3 registry, and browse everything in a small web UI.

```text
GitHub / local docs
        │
        ▼
   docs-search index   ──►  ~/.docs-search/indexes/<name>/<version>/
        │
        ├── search / ask  (local; auto-pull from S3 if missing)
        ├── publish / pull (S3 registry)
        └── serve         (web UI + JSON API)
```
