# Getting started

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv)
- `git` on your `PATH` (for cloning GitHub repos)
- AWS credentials only if you use the [S3 registry](registry.md)
- An OpenAI-compatible API key (or a local LLM server) only if you use [ask / RAG](ask-rag.md)

## Install

From the project root:

```bash
uv sync
```

This creates `.venv` and installs the `docs-search` CLI entry point.

Confirm:

```bash
uv run docs-search version
```

For development extras (pytest, ruff):

```bash
uv sync --group dev
```

## First index

Build a named, versioned index from a public GitHub documentation repo:

```bash
uv run docs-search index astral-sh/uv --name uv-docs --index-version 1.0.0
```

What happens:

1. The repo is cloned under `~/.docs-search/repos/`
2. Markdown (and related) docs are chunked with heading awareness
3. Symbols and links are extracted into a knowledge graph
4. Chunks are embedded with the default ONNX model
5. The index is saved at `~/.docs-search/indexes/uv-docs/1.0.0/`

You should see a summary table with chunk, symbol, and edge counts.

## First search

```bash
uv run docs-search search "how do I add a dependency?" --name uv-docs --index-version 1.0.0
```

If you omit `--index-version`, the newest local version of that name is used. If the named index is not on disk and an S3 registry is configured, `search` (and `ask`) will download it automatically — see [Search](search.md#auto-download-from-the-registry).

List what you have locally:

```bash
uv run docs-search list
```

## First RAG answer

```bash
export OPENAI_API_KEY=sk-...
uv run docs-search ask "how do I add a dependency?" --name uv-docs --index-version 1.0.0
```

This retrieves hybrid search hits, then asks an OpenAI-compatible chat model to answer using only those excerpts, with `[1]`, `[2]` citations. Details and local-LLM setup: [Ask (RAG)](ask-rag.md).

## Typical workflows

### Solo / offline

```bash
uv run docs-search index owner/repo --name owner-docs --index-version 1.0.0
uv run docs-search search "WidgetFactory.create" -n owner-docs -V 1.0.0
uv run docs-search ask "How do I configure retries?" -n owner-docs
```

### Team with a shared registry

```bash
# One machine publishes
uv run docs-search registry set-url s3://my-bucket/docs-search
uv run docs-search publish --name owner-docs --index-version 1.0.0

# Another machine searches (downloads on first use)
uv run docs-search registry set-url s3://my-bucket/docs-search
uv run docs-search search "timeout retries" --name owner-docs
```

### Browse in the browser

```bash
uv run docs-search serve
# open http://127.0.0.1:8787
```

## Next steps

- [Indexing](indexing.md) — include paths, local dirs, versions
- [Search](search.md) — fusion weights, JSON output, `--no-pull`
- [Ask (RAG)](ask-rag.md) — models, Ollama, API response shape
- [S3 registry](registry.md) — publish / pull layout
- [Web server & API](web-api.md) — HTTP endpoints
- [Configuration](configuration.md) — env vars and `config.json`
- [CLI reference](cli-reference.md) — full flag list
