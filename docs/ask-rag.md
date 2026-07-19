# Ask (RAG)

`docs-search ask` answers a question with **retrieval-augmented generation**:

1. Hybrid neurosymbolic search retrieves the top chunks from an index.
2. Those excerpts are sent to an **OpenAI-compatible** chat API.
3. The model must answer from the excerpts only, citing them as `[1]`, `[2]`, …

## Basic usage

```bash
export OPENAI_API_KEY=sk-...
uv run docs-search ask "how do I add a dependency?" --name uv-docs --index-version 1.0.0
```

Human output:

- A green **Answer** panel with inline citations
- A **Sources** list mapping `[n]` → path, heading, and retrieval score

JSON:

```bash
uv run docs-search ask "how do I add a dependency?" -n uv-docs --json
```

Shape (`RagAnswer`):

```json
{
  "question": "how do I add a dependency?",
  "answer": "Use `uv add` … [1]",
  "sources": [
    {
      "chunk_id": "...",
      "repo": "astral-sh/uv",
      "path": "docs/concepts/projects.md",
      "title": "...",
      "heading_path": ["…"],
      "snippet": "…",
      "text": "full chunk text…",
      "score": 0.81
    }
  ],
  "model": "gpt-4o-mini",
  "index_name": "uv-docs",
  "index_version": "1.0.0",
  "repo": "astral-sh/uv"
}
```

## Index selection and auto-pull

Same rules as [search](search.md):

- `--name` / `-n`, `--index-version` / `-V`, `--repo` / `-r`
- Missing named indexes download from the S3 registry unless `--no-pull`
- Override registry with `--registry s3://…`

```bash
uv run docs-search ask "timeout behavior" --name uv-docs --no-pull
```

## LLM configuration

### Precedence

| Setting | Resolution order (first match wins) |
|---------|-------------------------------------|
| API key | `--api-key` → `DOCS_SEARCH_LLM_API_KEY` → `OPENAI_API_KEY` → `llm_api_key` in `~/.docs-search/config.json` |
| Base URL | `--base-url` → `DOCS_SEARCH_LLM_BASE_URL` → `OPENAI_BASE_URL` → `llm_base_url` → `https://api.openai.com/v1` |
| Model | `--model` / `-m` → `DOCS_SEARCH_LLM_MODEL` → `OPENAI_MODEL` → `llm_model` → `gpt-4o-mini` |

Local servers (Ollama, LM Studio, etc.) often do not need an API key.

### OpenAI

```bash
export OPENAI_API_KEY=sk-...
uv run docs-search ask "…" -n uv-docs
# optional
uv run docs-search ask "…" -n uv-docs --model gpt-4o
```

### Ollama

```bash
# Start Ollama and pull a model first, e.g. `ollama pull llama3.2`
export DOCS_SEARCH_LLM_BASE_URL=http://127.0.0.1:11434/v1
export DOCS_SEARCH_LLM_MODEL=llama3.2
uv run docs-search ask "how do I add a dependency?" --name uv-docs --no-pull
```

Or one-shot flags:

```bash
uv run docs-search ask "…" -n uv-docs \
  --base-url http://127.0.0.1:11434/v1 \
  --model llama3.2
```

### Persist defaults in config

`~/.docs-search/config.json`:

```json
{
  "registry_url": "s3://my-bucket/docs-search",
  "llm_base_url": "http://127.0.0.1:11434/v1",
  "llm_model": "llama3.2"
}
```

Do not commit API keys. Prefer env vars for secrets.

## Retrieval options

Shared with search:

| Option | Default | Description |
|--------|---------|-------------|
| `--top` / `-k` | `5` | Chunks passed to the LLM |
| `--neural` / `--lexical` / `--symbolic` | `0.45` / `0.35` / `0.20` | Fusion weights |
| `--index-dir` | `~/.docs-search/indexes` | Index root |
| `--registry` | configured | S3 URL for auto-pull |
| `--no-pull` | off | Disable auto-download |
| `--json` | off | Full `RagAnswer` JSON |

Increase `-k` for harder questions that need more context; decrease it to reduce cost and noise.

## How grounding works

The system prompt instructs the model to:

- Use **only** the provided documentation excerpts
- Cite with bracketed numbers matching excerpt order
- Say clearly when the excerpts are insufficient (no invented APIs)

Generation defaults (library-level): temperature `0.2`, max tokens `1024`, HTTP timeout 60s.

## HTTP API

The web server exposes the same flow:

```bash
curl -s http://127.0.0.1:8787/api/ask \
  -H 'content-type: application/json' \
  -d '{"query":"how do I add a dependency?","name":"uv-docs"}'
```

Optional body fields: `version`, `repo`, `top_k`, fusion weights, `pull_missing`, `model`, `base_url`, `api_key`. See [Web server & API](web-api.md).

## Troubleshooting

| Symptom | What to check |
|---------|----------------|
| `Error: … API key` / auth failure | Set `OPENAI_API_KEY` or use a local `--base-url` that does not need a key |
| Connection refused to base URL | Is Ollama/LM Studio running? Is the path `/v1`? |
| Empty or vague answers | Raise `-k`, retune fusion weights, or confirm the index contains the topic (`search` first) |
| Index not found | Pass `--name`, publish/pull from registry, or drop `--no-pull` |
| Slow first ask | Embedding model may download on first search; LLM cold start on local servers |

## Related

- [Search](search.md) — retrieval layer only
- [Configuration](configuration.md) — env and config keys
- [CLI reference](cli-reference.md) — full `ask` flag list
