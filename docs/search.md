# Search

`docs-search search` runs **hybrid neurosymbolic search** over a local index: neural similarity, BM25 lexical scores, and symbol/graph expansion are fused into a single ranking.

## Basic usage

```bash
uv run docs-search search "how do I add a dependency?" --name uv-docs --index-version 1.0.0
```

### Selecting an index

| How | Behavior |
|-----|----------|
| `--name` / `-n` | Use that named index (newest version if `-V` omitted) |
| `--index-version` / `-V` | Pin a specific version |
| `--repo` / `-r` | Find by `owner/repo` when you did not pass `--name` |
| Neither name nor repo | If exactly one local index (or one unique name) exists, use it; otherwise error |

Examples:

```bash
# Newest local version of uv-docs
uv run docs-search search "lockfile" -n uv-docs

# Pin version
uv run docs-search search "lockfile" -n uv-docs -V 1.0.0

# By source repo slug
uv run docs-search search "WidgetFactory.create" --repo owner/repo

# Only one index on disk — selectors optional
uv run docs-search search "timeout"
```

## Result format

Human-readable output shows ranked panels with:

- File path and heading path
- Snippet
- Fused `score` plus `neural` / `lexical` / `symbolic` components
- Matched symbols (when any)
- Graph hops (when graph expansion contributed)

JSON mode:

```bash
uv run docs-search search "timeout retries" -n owner-docs --json
```

Each hit matches the `SearchHit` model:

```json
{
  "chunk_id": "...",
  "repo": "owner/repo",
  "path": "docs/config.md",
  "title": "Retries",
  "heading_path": ["Configuration", "Retries"],
  "snippet": "...",
  "score": 0.812,
  "neural_score": 0.71,
  "lexical_score": 0.55,
  "symbolic_score": 0.40,
  "matched_symbols": ["Client.timeout"],
  "graph_hops": ["chunk:…", "symbol:Client.timeout"]
}
```

## Fusion weights

Defaults (must sum conceptually to a balanced blend; they are used as relative weights):

| Signal | Default | Flag | Good for |
|--------|---------|------|----------|
| Neural | `0.45` | `--neural` | Paraphrases, conceptual questions |
| Lexical | `0.35` | `--lexical` | Exact phrases, error strings |
| Symbolic | `0.20` | `--symbolic` | API names, config keys, code identifiers |

```bash
# Emphasize symbols for API lookup
uv run docs-search search "WidgetFactory.create" -n owner-docs --symbolic 0.5 --neural 0.3 --lexical 0.2

# Emphasize keywords
uv run docs-search search "ECONNRESET retry" -n owner-docs --lexical 0.55 --neural 0.3 --symbolic 0.15
```

Also useful:

| Option | Default | Description |
|--------|---------|-------------|
| `--top` / `-k` | `5` | Number of hits (1–50) |
| `--index-dir` | `~/.docs-search/indexes` | Alternate index root |
| `--registry` | configured URL | Override S3 registry for auto-pull |
| `--no-pull` | off | Never download missing named indexes |
| `--json` | off | Machine-readable hits |

## Auto-download from the registry

When you pass `--name` and that index is **not** on disk:

1. `docs-search` looks up the configured S3 registry (`registry set-url` or `--registry`).
2. If `--index-version` is omitted, it pulls the **newest published** version of that name.
3. The archive is unpacked into `~/.docs-search/indexes/<name>/<version>/`.
4. Search proceeds as if the index had always been local.

```bash
# Configure once
uv run docs-search registry set-url s3://my-bucket/docs-search

# First search downloads uv-docs (newest remote version)
uv run docs-search search "add a dependency" --name uv-docs

# Pin remote version
uv run docs-search search "add a dependency" --name uv-docs -V 1.0.0

# Fail if not already local
uv run docs-search search "add a dependency" --name uv-docs --no-pull
```

Notes:

- Auto-pull requires `--name`. Searching by `--repo` alone does not trigger a registry download.
- AWS credentials must allow `s3:GetObject` (and listing) on the registry prefix.
- Explicit `docs-search pull` is still available when you want to prefetch; see [S3 registry](registry.md).

## Query tips

- **Natural language:** `"how do I configure authentication?"` — neural + lexical shine.
- **Symbols:** `"WidgetFactory.create"`, `"max_retries"` — symbolic boost helps.
- **Mixed:** `"timeout retries Client"` — all three signals contribute.
- Raise `-k` when using results as RAG context or when exploring unfamiliar docs.

## Related

- Build indexes: [Indexing](indexing.md)
- Answer with an LLM: [Ask (RAG)](ask-rag.md)
- Share indexes: [S3 registry](registry.md)
- HTTP search: [Web server & API](web-api.md)
