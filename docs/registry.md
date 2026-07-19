# S3 registry

Share named/versioned indexes through an S3 prefix. One machine builds and publishes; others pull (or let `search` / `ask` auto-download).

## Prerequisites

- AWS credentials with access to the bucket/prefix (`boto3` default credential chain)
- A writable S3 location, e.g. `s3://my-bucket/docs-search`

Typical IAM needs:

- Publish: `s3:PutObject`, `s3:GetObject` on the prefix (manifest read-modify-write)
- Pull / auto-download: `s3:GetObject` (and ability to read `registry.json`)

## Configure the registry URL

```bash
uv run docs-search registry set-url s3://my-bucket/docs-search
uv run docs-search registry url
```

The URL is stored in `~/.docs-search/config.json` as `registry_url`. Any command that talks to S3 can override with `--registry s3://…`.

## Publish a local index

```bash
uv run docs-search publish --name uv-docs --index-version 1.0.0
```

If you omit `--name` / `--index-version` / `--repo`, the CLI resolves a single matching local index the same way search does — with multiple indexes you must be explicit.

```bash
uv run docs-search publish -n uv-docs -V 1.0.0 --registry s3://other-bucket/prefix
```

Publish packs the index directory into a tarball, uploads it, and updates the registry manifest.

## List remote indexes

```bash
uv run docs-search registry list
uv run docs-search registry list --registry s3://my-bucket/docs-search
```

## Pull into the local index directory

```bash
# Specific version
uv run docs-search pull uv-docs 1.0.0

# Newest published version for that name
uv run docs-search pull uv-docs
```

After a pull:

```bash
uv run docs-search list
uv run docs-search search "…" --name uv-docs --no-pull
```

## Auto-download on search / ask

You usually do **not** need an explicit `pull`. With a configured registry:

```bash
uv run docs-search search "add a dependency" --name uv-docs
uv run docs-search ask "add a dependency" --name uv-docs
```

If `uv-docs` is missing locally, it is downloaded first. Disable with `--no-pull`. Details: [Search → Auto-download](search.md#auto-download-from-the-registry).

## Object layout

```text
s3://bucket/prefix/
  registry.json
  indices/<name>/<version>/meta.json
  indices/<name>/<version>/index.tar.gz
```

- `registry.json` — manifest of all published entries (`RegistryManifest`)
- `meta.json` — copy of index metadata for that name/version
- `index.tar.gz` — archive of the local index directory (`chunks.jsonl`, `embeddings.npy`, graph files, etc.)

## Workflow example

```bash
# Builder
uv run docs-search index astral-sh/uv --name uv-docs --index-version 1.0.0
uv run docs-search registry set-url s3://acme-docs/docs-search
uv run docs-search publish --name uv-docs --index-version 1.0.0

# Consumer (laptop / CI)
uv run docs-search registry set-url s3://acme-docs/docs-search
uv run docs-search search "workspace members" --name uv-docs
# or prefetch
uv run docs-search pull uv-docs 1.0.0
```

## Tips

- Bump `--index-version` when the source docs change so consumers can pin.
- Keep registry URLs consistent across the team (document the `s3://` URI in your internal runbook).
- Use `--no-pull` in air-gapped or deterministic CI once indexes are vendored locally.
- The web UI lists registry entries when configured — see [Web server & API](web-api.md).

## Related

- [Indexing](indexing.md) — create what you publish
- [Search](search.md) — auto-pull behavior
- [Configuration](configuration.md) — where `registry_url` lives
