# Indexing

Build a local neurosymbolic index with `docs-search index`. Every index is stored under a **name** and **version**.

## Commands

### Index a GitHub repository

```bash
uv run docs-search index OWNER/REPO --name NAME --index-version VERSION
```

Accepted source forms:

- `owner/repo` (e.g. `astral-sh/uv`)
- `https://github.com/owner/repo`
- `git@github.com:owner/repo.git`

Examples:

```bash
uv run docs-search index astral-sh/uv --name uv-docs --index-version 1.0.0
uv run docs-search index https://github.com/fastapi/fastapi --name fastapi --index-version 0.2.0 -b main
```

Options:

| Option | Description |
|--------|-------------|
| `--name` / `-n` | Index name. Default: repo slug with `/` → `__` (e.g. `astral-sh__uv`) |
| `--index-version` / `-V` | Version string (default `0.1.0`) |
| `--branch` / `-b` | Git branch to clone |
| `--include` / `-i` | Doc subdirectory to include (repeatable). Default: `docs`, `doc`, `documentation`, `wiki`, `.` |
| `--model` | Embedding model (default `BAAI/bge-small-en-v1.5`) |
| `--repos-dir` | Where clones live (default `~/.docs-search/repos`) |
| `--index-dir` | Where indexes live (default `~/.docs-search/indexes`) |
| `--force` | Force re-clone before indexing |

### Index a local directory

```bash
uv run docs-search index ./path/to/docs --local --name my-docs --index-version 0.1.0
```

If `SOURCE` already exists as a path, `--local` is optional — the CLI treats it as a local directory automatically.

### Clone without indexing

```bash
uv run docs-search ingest owner/repo
uv run docs-search ingest owner/repo --branch main --force
```

Useful when you want the repo on disk first. Clones land under `~/.docs-search/repos/<owner>__<repo>/`.

## What gets indexed

Document files with these extensions:

- `.md`, `.mdx`, `.rst`, `.txt`

By default the indexer looks under `docs/`, `doc/`, `documentation/`, `wiki/`, and the repo root (`.`). Narrow that with `--include`:

```bash
uv run docs-search index owner/repo --name owner-docs -V 1.0.0 --include docs --include guide
```

### Chunking and symbols

For each file:

1. Content is split into **heading-aware chunks** (max ~1200 characters, ~120 overlap).
2. Inline and fenced code symbols (API-like identifiers) are extracted.
3. Markdown links become graph edges (`LINKS_TO`).
4. A knowledge graph connects `doc → chunk → symbol` plus heading hierarchy.

Then each chunk is embedded for neural retrieval.

## Names and versions

Names and versions must match:

```text
^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$   # name
^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$    # version
```

Spaces become `-`; `/` in names becomes `__`. Prefer explicit names when publishing to a shared registry:

```bash
# Good for sharing
--name uv-docs --index-version 1.2.0

# Auto-derived (fine for solo use)
# astral-sh/uv → name astral-sh__uv, version 0.1.0
```

Re-indexing the same name + version **overwrites** that local slot. Bump `--index-version` to keep older builds.

## On-disk layout

```text
~/.docs-search/
  repos/
    owner__repo/          # cloned git working tree
  indexes/
    <name>/
      <version>/
        meta.json         # IndexMeta
        chunks.jsonl      # DocChunk per line
        embeddings.npy    # float matrix
        edges.jsonl       # GraphEdge per line
        graph.json        # networkx node-link graph
```

## List local indexes

```bash
uv run docs-search list
uv run docs-search list --index-dir /custom/indexes
```

Columns: Name, Version, Repo, Chunks, Symbols, Edges, Model.

## Tips

- Prefer a stable `--name` that does not include the GitHub org if you rebrand or fork.
- Use semantic versions (`1.0.0`, `1.1.0`) so teammates can pin with `--index-version`.
- After indexing, [publish](registry.md) so others can [search without building](search.md#auto-download-from-the-registry).
- Large monorepos: restrict with `--include` to keep indexes small and relevant.
