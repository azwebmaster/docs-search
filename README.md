# docs-search

Convert a GitHub documentation repository into a **local neurosymbolic search** index.

Neurosymbolic here means retrieval that combines:

- **Neural** — dense semantic embeddings (`fastembed` / ONNX, no PyTorch)
- **Symbolic** — a knowledge graph of docs, headings, links, and code symbols
- **Lexical** — BM25 keyword matching

Those signals are fused at query time so symbol-aware questions (APIs, config keys) and natural-language questions both work offline after indexing.

Indexes are saved with a **name** and **version**, can be published to an **S3 registry**, and browsed through a small **web server**.

## Documentation

Thorough how-to guides live in [`docs/`](docs/README.md):

| Guide | Topic |
|-------|-------|
| [Getting started](docs/getting-started.md) | Install, first index, search, and ask |
| [Indexing](docs/indexing.md) | Named/versioned indexes from GitHub or local paths |
| [Search](docs/search.md) | Hybrid retrieval, fusion weights, auto-pull |
| [Ask (RAG)](docs/ask-rag.md) | Grounded answers with OpenAI-compatible LLMs |
| [S3 registry](docs/registry.md) | Publish, list, and pull shared indexes |
| [Web server & API](docs/web-api.md) | Browser UI and JSON endpoints |
| [Configuration](docs/configuration.md) | Paths, `config.json`, and environment variables |
| [CLI reference](docs/cli-reference.md) | Every command and option |

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- AWS credentials (only required for S3 registry publish/pull)

## Install

```bash
uv sync
```

This creates `.venv` and installs the `docs-search` CLI.

## Quick start

```bash
# Build a named/versioned local index from a GitHub repo
uv run docs-search index astral-sh/uv --name uv-docs --index-version 1.0.0

# Search it
uv run docs-search search "how do I add a dependency?" --name uv-docs --index-version 1.0.0

# Ask a question with retrieval-augmented generation (OpenAI-compatible LLM)
uv run docs-search ask "how do I add a dependency?" --name uv-docs --index-version 1.0.0

# Search a named index that is not local yet — downloads from the S3 registry first
uv run docs-search search "how do I add a dependency?" --name uv-docs

# List local indexes
uv run docs-search list
```

Indexes live under `~/.docs-search/indexes/<name>/<version>/` by default. Cloned repos live under `~/.docs-search/repos/`.

When you search with `--name` and that index is not on disk, `docs-search` downloads it from the configured S3 registry (newest version if `--index-version` is omitted). Pass `--no-pull` to disable that.

### Local directory

```bash
uv run docs-search index ./path/to/docs --local --name my-docs --index-version 0.1.0
```

### Useful options

```bash
uv run docs-search index owner/repo --name owner-docs --index-version 1.2.0 --include docs
uv run docs-search search "WidgetFactory.create" -n owner-docs -V 1.2.0 -k 8
uv run docs-search search "timeout retries" -n owner-docs --json
```

Fusion weights can be tuned with `--neural`, `--lexical`, and `--symbolic`.

## RAG answers

`docs-search ask` retrieves hybrid search hits, then asks an **OpenAI-compatible**
chat model to answer using only those excerpts (with citations).

```bash
export OPENAI_API_KEY=sk-...
uv run docs-search ask "how do I add a dependency?" --name uv-docs

# Or use a local server such as Ollama
export DOCS_SEARCH_LLM_BASE_URL=http://127.0.0.1:11434/v1
export DOCS_SEARCH_LLM_MODEL=llama3.2
uv run docs-search ask "how do I add a dependency?" --name uv-docs --no-pull
```

Configuration (first match wins):

| Setting | Env / config |
|---------|----------------|
| API key | `DOCS_SEARCH_LLM_API_KEY` → `OPENAI_API_KEY` → `llm_api_key` in `~/.docs-search/config.json` |
| Base URL | `DOCS_SEARCH_LLM_BASE_URL` → `OPENAI_BASE_URL` → `llm_base_url` (default `https://api.openai.com/v1`) |
| Model | `DOCS_SEARCH_LLM_MODEL` → `OPENAI_MODEL` → `llm_model` (default `gpt-4o-mini`) |

CLI overrides: `--model`, `--base-url`, `--api-key`. Use `--json` for machine-readable output.

## S3 registry

Link a shared S3 prefix where published indexes are stored:

```bash
uv run docs-search registry set-url s3://my-bucket/docs-search
uv run docs-search registry url

# Publish a local index
uv run docs-search publish --name uv-docs --index-version 1.0.0

# List remote indexes
uv run docs-search registry list

# Pull into the local index directory
uv run docs-search pull uv-docs 1.0.0

# Or pull the newest published version of a name
uv run docs-search pull uv-docs
```

Registry layout:

```text
s3://bucket/prefix/
  registry.json
  indices/<name>/<version>/meta.json
  indices/<name>/<version>/index.tar.gz
```

The registry URL is saved in `~/.docs-search/config.json`. Override per command with `--registry s3://…`.

## Web server

Browse local indexes and the linked S3 registry:

```bash
uv run docs-search serve
# open http://127.0.0.1:8787
```

JSON endpoints:

- `GET /api/local` — local indexes
- `GET /api/registry` — registry indexes
- `GET /api/health` — health check
- `POST /api/search` — hybrid search (`{"query": "…", "name": "…"}`)
- `POST /api/ask` — RAG answer (`{"query": "…", "name": "…"}` plus optional `model` / `base_url` / `api_key`)

## How it works

1. **Ingest** — clone/update the GitHub repo (`git` via GitPython).
2. **Symbolic extract** — split markdown into heading-aware chunks; pull inline/fenced code symbols and markdown links.
3. **Knowledge graph** — connect `doc → chunk → symbol`, heading hierarchy, and `LINKS_TO` relations (`networkx`).
4. **Neural embed** — embed each chunk with `BAAI/bge-small-en-v1.5` via `fastembed`.
5. **Save** — persist the index under a name + version.
6. **Search / ask / publish / serve** — hybrid search locally, optionally answer with RAG, publish to S3, and browse via the web UI.

```text
Query ─┬─► neural similarity
       ├─► BM25 lexical scores
       └─► symbol/graph expansion  ─► fused ranking ─► local hits
                                              │
                                              └─► (ask) LLM grounded answer + citations
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
| `docs-search index <repo> --name … --index-version …` | Clone + build named/versioned index |
| `docs-search search <query> --name …` | Hybrid search (auto-downloads missing named indexes) |
| `docs-search ask <question> --name …` | RAG answer over retrieved docs (OpenAI-compatible LLM) |
| `docs-search list` | Show local name/version indexes |
| `docs-search publish` | Upload a local index to the S3 registry |
| `docs-search pull <name> [version]` | Download an index from the registry |
| `docs-search registry set-url s3://…` | Link this machine to an S3 registry |
| `docs-search registry list` | List published registry indexes |
| `docs-search serve` | Web UI for local + registry indexes |
| `docs-search version` | Print package version |
