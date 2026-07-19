# Web server & API

`docs-search serve` starts a small FastAPI app to browse local indexes and the linked S3 registry, and to run search / ask over HTTP.

## Start the server

```bash
uv run docs-search serve
# http://127.0.0.1:8787
```

Options:

| Option | Default | Description |
|--------|---------|-------------|
| `--host` | `127.0.0.1` | Bind address |
| `--port` / `-p` | `8787` | Bind port |
| `--index-dir` | `~/.docs-search/indexes` | Local indexes root |
| `--registry` | configured URL | Override S3 registry for this process |

```bash
uv run docs-search serve --host 0.0.0.0 --port 9000
```

Open `http://127.0.0.1:8787` for the HTML UI: local table, registry table, and a filter box (`?q=`).

## JSON endpoints

### `GET /api/health`

```bash
curl -s http://127.0.0.1:8787/api/health
# {"status":"ok"}
```

### `GET /api/local`

List all local `IndexMeta` objects.

```bash
curl -s http://127.0.0.1:8787/api/local
```

### `GET /api/local/{name}/{version}`

Single local index metadata, or `404`.

### `GET /api/registry`

```json
{
  "registry_url": "s3://my-bucket/docs-search",
  "error": null,
  "indices": [ /* RegistryEntry … */ ]
}
```

If no registry is configured, `registry_url` is null and `error` explains why. Registry access failures set `error` and may return an empty `indices` list.

### `GET /api/registry/{name}/{version}`

One registry entry plus `registry_url`, or `404` / `502`.

### `POST /api/search`

Hybrid search. Body (`SearchRequest`):

| Field | Type | Default | Notes |
|-------|------|---------|-------|
| `query` | string | required | Min length 1 |
| `name` | string | null | Preferred selector |
| `version` | string | null | Pin version |
| `repo` | string | null | Alternate selector |
| `top_k` | int | `5` | 1–50 |
| `neural_weight` | float | `0.45` | |
| `lexical_weight` | float | `0.35` | |
| `symbolic_weight` | float | `0.20` | |
| `pull_missing` | bool | `true` | Auto-download named indexes |

```bash
curl -s http://127.0.0.1:8787/api/search \
  -H 'content-type: application/json' \
  -d '{
    "query": "how do I add a dependency?",
    "name": "uv-docs",
    "version": "1.0.0",
    "top_k": 8
  }'
```

Response:

```json
{
  "query": "…",
  "index_name": "uv-docs",
  "index_version": "1.0.0",
  "repo": "astral-sh/uv",
  "hits": [ /* SearchHit … */ ]
}
```

### `POST /api/ask`

Same retrieval fields as search, plus optional LLM overrides:

| Field | Type | Notes |
|-------|------|-------|
| `model` | string | Chat model name |
| `base_url` | string | OpenAI-compatible base URL |
| `api_key` | string | Provider key (prefer env on the server host) |

```bash
curl -s http://127.0.0.1:8787/api/ask \
  -H 'content-type: application/json' \
  -d '{
    "query": "how do I add a dependency?",
    "name": "uv-docs",
    "top_k": 5
  }'
```

Returns a full `RagAnswer` (`question`, `answer`, `sources`, `model`, index fields). Errors: `400` (bad request / validation), `404` (index missing), `502` (LLM or registry failure).

## Index resolution

If neither `name` nor `repo` is set:

- One unique local name → use it
- Exactly one local index → use its name/repo
- Otherwise → `400` asking for `name` or `repo`

When `name` is set and `pull_missing` is true, missing indexes are downloaded from the process registry URL (same behavior as the CLI).

## Security notes

- Default bind is localhost only. Exposing `--host 0.0.0.0` publishes search/ask on the network.
- Prefer configuring LLM keys via environment on the server, not via `api_key` in client requests, if the API is reachable beyond your machine.
- Registry credentials use the server process AWS identity.

## Related

- [Ask (RAG)](ask-rag.md) — generation behavior
- [Search](search.md) — hit semantics
- [S3 registry](registry.md) — what `/api/registry` lists
