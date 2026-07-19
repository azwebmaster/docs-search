# CLI reference

Entry point: `docs-search` (via `uv run docs-search …`).

```bash
uv run docs-search --help
uv run docs-search <command> --help
```

## `version`

Print the package version.

```bash
uv run docs-search version
```

## `ingest`

Clone or update a GitHub repository without building an index.

| Argument / option | Description |
|-------------------|-------------|
| `SOURCE` | GitHub URL or `owner/repo` |
| `--branch` / `-b` | Branch to clone |
| `--repos-dir` | Clone root (default `~/.docs-search/repos`) |
| `--force` | Re-clone even if present |

## `index`

Build a named/versioned neurosymbolic index from GitHub or a local path.

| Argument / option | Description |
|-------------------|-------------|
| `SOURCE` | GitHub URL, `owner/repo`, or local path |
| `--local` | Treat `SOURCE` as a local directory |
| `--name` / `-n` | Index name (default from repo slug) |
| `--index-version` / `-V` | Version to save (default `0.1.0`) |
| `--branch` / `-b` | Git branch |
| `--include` / `-i` | Include subdirectory (repeatable) |
| `--model` | Embedding model |
| `--repos-dir` | Clone root |
| `--index-dir` | Index root |
| `--force` | Force re-clone before indexing |

See [Indexing](indexing.md).

## `search`

Hybrid neurosymbolic search.

| Argument / option | Description |
|-------------------|-------------|
| `QUERY` | Natural-language or symbol-aware query |
| `--name` / `-n` | Saved index name |
| `--index-version` / `-V` | Version (default: newest matching) |
| `--repo` / `-r` | `owner/repo` when name omitted |
| `--top` / `-k` | Hit count (1–50, default 5) |
| `--neural` / `--lexical` / `--symbolic` | Fusion weights |
| `--index-dir` | Index root |
| `--registry` | `s3://…` for auto-pull |
| `--no-pull` | Do not download missing named indexes |
| `--json` | Emit JSON hits |

See [Search](search.md).

## `ask`

RAG answer over retrieved documentation.

| Argument / option | Description |
|-------------------|-------------|
| `QUESTION` | Question to answer from the docs |
| `--name` / `-n` | Saved index name |
| `--index-version` / `-V` | Version |
| `--repo` / `-r` | `owner/repo` when name omitted |
| `--top` / `-k` | Chunks for context (default 5) |
| `--neural` / `--lexical` / `--symbolic` | Fusion weights |
| `--model` / `-m` | Chat model |
| `--base-url` | OpenAI-compatible API base |
| `--api-key` | API key (also reads `DOCS_SEARCH_LLM_API_KEY` / `OPENAI_API_KEY`) |
| `--index-dir` | Index root |
| `--registry` | `s3://…` for auto-pull |
| `--no-pull` | Do not download missing named indexes |
| `--json` | Emit full `RagAnswer` JSON |

See [Ask (RAG)](ask-rag.md).

## `list`

Show local name/version indexes.

| Option | Description |
|--------|-------------|
| `--index-dir` | Index root |

## `publish`

Upload a local index to the S3 registry.

| Option | Description |
|--------|-------------|
| `--name` / `-n` | Local index name |
| `--index-version` / `-V` | Local index version |
| `--repo` / `-r` | Filter by `owner/repo` |
| `--registry` | `s3://bucket/prefix` (default: configured) |
| `--index-dir` | Index root |

## `pull`

Download an index from the registry.

| Argument / option | Description |
|-------------------|-------------|
| `NAME` | Registry index name |
| `VERSION` | Optional; default newest published |
| `--registry` | Override registry URL |
| `--index-dir` | Destination index root |

## `registry set-url`

Persist the S3 registry URL in `~/.docs-search/config.json`.

| Argument | Description |
|----------|-------------|
| `URL` | `s3://bucket/prefix` |

## `registry url`

Print the configured registry URL (exit code 1 if unset).

## `registry list`

List published registry indexes.

| Option | Description |
|--------|-------------|
| `--registry` | Override registry URL |

See [S3 registry](registry.md).

## `serve`

Start the web UI and JSON API.

| Option | Description |
|--------|-------------|
| `--host` | Bind host (default `127.0.0.1`) |
| `--port` / `-p` | Bind port (default `8787`) |
| `--index-dir` | Local indexes root |
| `--registry` | Override registry URL for this process |

See [Web server & API](web-api.md).
