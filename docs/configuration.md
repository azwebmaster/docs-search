# Configuration

## Data directories

| Path | Purpose |
|------|---------|
| `~/.docs-search/` | Default data root |
| `~/.docs-search/repos/` | Cloned GitHub repositories |
| `~/.docs-search/indexes/` | Named/versioned indexes |
| `~/.docs-search/config.json` | User config (registry URL, LLM defaults) |

Override per command with `--repos-dir` and `--index-dir` where supported.

## `config.json`

Created or updated by `docs-search registry set-url` and editable by hand:

```json
{
  "registry_url": "s3://my-bucket/docs-search",
  "llm_base_url": "http://127.0.0.1:11434/v1",
  "llm_model": "llama3.2",
  "llm_api_key": "optional-prefer-env-instead"
}
```

| Key | Used by |
|-----|---------|
| `registry_url` | `publish`, `pull`, `registry list`, auto-pull on `search`/`ask`, `serve` |
| `llm_base_url` | `ask` (and `/api/ask`) when env/flags unset |
| `llm_model` | `ask` default model |
| `llm_api_key` | `ask` when env unset (prefer environment variables) |

Invalid JSON causes a clear error on load.

## Environment variables

### LLM / RAG

| Variable | Purpose |
|----------|---------|
| `DOCS_SEARCH_LLM_API_KEY` | Preferred API key for `ask` |
| `OPENAI_API_KEY` | Fallback API key |
| `DOCS_SEARCH_LLM_BASE_URL` | Preferred OpenAI-compatible base URL |
| `OPENAI_BASE_URL` | Fallback base URL |
| `DOCS_SEARCH_LLM_MODEL` | Preferred chat model |
| `OPENAI_MODEL` | Fallback chat model |

CLI flags (`--api-key`, `--base-url`, `--model`) always win over env and config.

### AWS (registry)

Uses the standard boto3 chain, for example:

- `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN`
- `AWS_PROFILE` / `AWS_DEFAULT_REGION`
- Instance / container roles

No docs-search-specific AWS env vars.

## Built-in defaults

| Setting | Default |
|---------|---------|
| Embed model | `BAAI/bge-small-en-v1.5` |
| Index version (when creating) | `0.1.0` |
| Neural / lexical / symbolic weights | `0.45` / `0.35` / `0.20` |
| Search/ask `top_k` | `5` |
| LLM base URL | `https://api.openai.com/v1` |
| LLM model | `gpt-4o-mini` |
| RAG temperature | `0.2` |
| RAG max tokens | `1024` |
| Serve host/port | `127.0.0.1:8787` |
| Doc extensions | `.md`, `.mdx`, `.rst`, `.txt` |
| Default include dirs | `docs`, `doc`, `documentation`, `wiki`, `.` |
| Chunk size / overlap | ~1200 / ~120 characters |

## Naming rules

Index **names** and **versions** must be filesystem-safe:

- Start with a letter or digit
- Then letters, digits, `.`, `_`, or `-`
- Max length 128 (name) / 64 (version)

`sanitize_index_name` also turns `/` into `__` and spaces into `-`.

Default name from a repo slug: `owner/repo` → `owner__repo`.

## Related

- [Getting started](getting-started.md)
- [Ask (RAG)](ask-rag.md) — LLM resolution examples
- [S3 registry](registry.md) — `registry_url` usage
- [CLI reference](cli-reference.md)
