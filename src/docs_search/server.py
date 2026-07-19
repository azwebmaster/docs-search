from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from docs_search.config import (
    DEFAULT_INDEX_DIR,
    DEFAULT_LEXICAL_WEIGHT,
    DEFAULT_NEURAL_WEIGHT,
    DEFAULT_RAG_TOP_K,
    DEFAULT_SYMBOLIC_WEIGHT,
    get_registry_url,
)
from docs_search.models import IndexMeta, RegistryEntry
from docs_search.store import list_indexes

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.responses import HTMLResponse
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "fastapi is required for the web server. Install with: uv sync"
    ) from exc


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    name: Optional[str] = None
    version: Optional[str] = None
    repo: Optional[str] = None
    top_k: int = Field(default=DEFAULT_RAG_TOP_K, ge=1, le=50)
    neural_weight: float = DEFAULT_NEURAL_WEIGHT
    lexical_weight: float = DEFAULT_LEXICAL_WEIGHT
    symbolic_weight: float = DEFAULT_SYMBOLIC_WEIGHT
    pull_missing: bool = True


class AskRequest(SearchRequest):
    model: Optional[str] = None
    base_url: Optional[str] = None
    api_key: Optional[str] = None


def create_app(
    *,
    index_dir: Path | None = None,
    registry_url: str | None = None,
    registry_client: Any | None = None,
) -> FastAPI:
    """Create the docs-search registry browser app."""
    resolved_index_dir = index_dir or DEFAULT_INDEX_DIR
    configured_registry = (registry_url or get_registry_url() or "").strip() or None

    app = FastAPI(
        title="docs-search",
        description="Browse local documentation indexes and the linked S3 registry.",
        version="0.1.0",
    )

    def _local() -> list[IndexMeta]:
        return list_indexes(resolved_index_dir)

    def _registry() -> tuple[str | None, list[RegistryEntry], str | None]:
        if not configured_registry:
            return None, [], "No registry configured"
        try:
            from docs_search.registry import S3Registry

            reg = S3Registry(configured_registry, client=registry_client)
            return reg.url, reg.list_entries(), None
        except Exception as exc:  # noqa: BLE001 - surface registry errors in UI/API
            return configured_registry, [], str(exc)

    def _load_index(
        *,
        name: str | None,
        version: str | None,
        repo: str | None,
        pull_missing: bool,
    ):
        from docs_search.ingest import normalize_github_source
        from docs_search.search import NeurosymbolicIndex

        resolved_repo = repo
        if resolved_repo is not None:
            try:
                resolved_repo = normalize_github_source(resolved_repo)[1]
            except ValueError:
                pass

        metas = _local()
        resolved_name = name
        if resolved_name is None and resolved_repo is None:
            names = sorted({m.name for m in metas})
            if len(names) == 1:
                resolved_name = names[0]
            elif len(metas) == 1:
                resolved_name = metas[0].name
                resolved_repo = metas[0].repo
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Pass name or repo when multiple local indexes exist",
                )

        try:
            return NeurosymbolicIndex.load(
                resolved_repo,
                index_dir=resolved_index_dir,
                name=resolved_name,
                version=version,
                pull_missing=pull_missing and bool(resolved_name),
                registry_url=configured_registry,
                registry_client=registry_client,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/local")
    def api_local() -> list[dict[str, Any]]:
        return [m.model_dump() for m in _local()]

    @app.get("/api/local/{name}/{version}")
    def api_local_one(name: str, version: str) -> dict[str, Any]:
        for meta in _local():
            if meta.name == name and meta.version == version:
                return meta.model_dump()
        raise HTTPException(status_code=404, detail=f"Local index {name}@{version} not found")

    @app.get("/api/registry")
    def api_registry() -> dict[str, Any]:
        url, entries, error = _registry()
        return {
            "registry_url": url,
            "error": error,
            "indices": [e.model_dump() for e in entries],
        }

    @app.get("/api/registry/{name}/{version}")
    def api_registry_one(name: str, version: str) -> dict[str, Any]:
        url, entries, error = _registry()
        if error and not entries:
            raise HTTPException(status_code=502, detail=error)
        for entry in entries:
            if entry.name == name and entry.version == version:
                payload = entry.model_dump()
                payload["registry_url"] = url
                return payload
        raise HTTPException(status_code=404, detail=f"Registry index {name}@{version} not found")

    @app.post("/api/search")
    def api_search(body: SearchRequest) -> dict[str, Any]:
        index = _load_index(
            name=body.name,
            version=body.version,
            repo=body.repo,
            pull_missing=body.pull_missing,
        )
        hits = index.search(
            body.query,
            top_k=body.top_k,
            neural_weight=body.neural_weight,
            lexical_weight=body.lexical_weight,
            symbolic_weight=body.symbolic_weight,
        )
        return {
            "query": body.query,
            "index_name": index.name,
            "index_version": index.version,
            "repo": index.repo,
            "hits": [h.model_dump() for h in hits],
        }

    @app.post("/api/ask")
    def api_ask(body: AskRequest) -> dict[str, Any]:
        from docs_search.rag import RagError, ask

        index = _load_index(
            name=body.name,
            version=body.version,
            repo=body.repo,
            pull_missing=body.pull_missing,
        )
        try:
            result = ask(
                index,
                body.query,
                top_k=body.top_k,
                neural_weight=body.neural_weight,
                lexical_weight=body.lexical_weight,
                symbolic_weight=body.symbolic_weight,
                model=body.model,
                base_url=body.base_url,
                api_key=body.api_key,
            )
        except RagError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return result.model_dump()

    @app.get("/", response_class=HTMLResponse)
    def home(q: str | None = Query(default=None)) -> HTMLResponse:
        local = _local()
        reg_url, remote, reg_error = _registry()
        query = (q or "").strip().lower()
        if query:
            local = [
                m
                for m in local
                if query in m.name.lower()
                or query in m.version.lower()
                or query in m.repo.lower()
            ]
            remote = [
                e
                for e in remote
                if query in e.name.lower()
                or query in e.version.lower()
                or query in e.repo.lower()
            ]
        body = _render_page(
            local=local,
            remote=remote,
            registry_url=reg_url,
            registry_error=reg_error,
            index_dir=resolved_index_dir,
            query=q or "",
        )
        return HTMLResponse(body)

    return app


def _render_page(
    *,
    local: list[IndexMeta],
    remote: list[RegistryEntry],
    registry_url: str | None,
    registry_error: str | None,
    index_dir: Path,
    query: str,
) -> str:
    local_rows = "".join(_local_row(m) for m in local) or (
        '<tr><td colspan="6" class="empty">No local indexes yet.</td></tr>'
    )
    remote_rows = "".join(_remote_row(e) for e in remote) or (
        '<tr><td colspan="6" class="empty">No registry indexes found.</td></tr>'
    )
    reg_label = html.escape(registry_url) if registry_url else "not configured"
    error_block = (
        f'<p class="error">Registry error: {html.escape(registry_error)}</p>'
        if registry_error
        else ""
    )
    q_value = html.escape(query)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>docs-search · indexes</title>
  <style>
    :root {{
      --ink: #1c1917;
      --muted: #57534e;
      --line: #d6d3d1;
      --panel: rgba(255, 255, 255, 0.72);
      --accent: #0f766e;
      --accent-ink: #f0fdfa;
      --warn: #9a3412;
      --bg0: #ecfdf5;
      --bg1: #fafaf9;
      --bg2: #e7e5e4;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: "IBM Plex Sans", "Source Sans 3", "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(1200px 600px at 10% -10%, #99f6e4 0%, transparent 55%),
        radial-gradient(900px 500px at 100% 0%, #fde68a 0%, transparent 45%),
        linear-gradient(180deg, var(--bg0), var(--bg1) 40%, var(--bg2));
    }}
    main {{
      width: min(1100px, calc(100% - 2rem));
      margin: 0 auto;
      padding: 2.5rem 0 4rem;
    }}
    header {{
      display: grid;
      gap: 0.75rem;
      margin-bottom: 2rem;
    }}
    .brand {{
      font-family: "IBM Plex Mono", "JetBrains Mono", ui-monospace, monospace;
      font-size: clamp(2rem, 4vw, 2.75rem);
      letter-spacing: -0.04em;
      margin: 0;
    }}
    .lede {{
      margin: 0;
      max-width: 42rem;
      color: var(--muted);
      font-size: 1.05rem;
      line-height: 1.5;
    }}
    .meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem 1.25rem;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    .meta code {{
      font-family: "IBM Plex Mono", "JetBrains Mono", ui-monospace, monospace;
      color: var(--ink);
      background: rgba(255,255,255,0.55);
      padding: 0.1rem 0.35rem;
    }}
    form.search {{
      display: flex;
      gap: 0.5rem;
      margin: 0.5rem 0 0;
    }}
    form.search input {{
      flex: 1;
      min-width: 0;
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 0.4rem;
      padding: 0.65rem 0.8rem;
      font: inherit;
    }}
    form.search button {{
      border: 0;
      border-radius: 0.4rem;
      background: var(--accent);
      color: var(--accent-ink);
      padding: 0.65rem 1rem;
      font: inherit;
      font-weight: 600;
      cursor: pointer;
    }}
    section {{
      margin-top: 1.75rem;
    }}
    section h2 {{
      margin: 0 0 0.35rem;
      font-size: 1.25rem;
      letter-spacing: -0.02em;
    }}
    section p.hint {{
      margin: 0 0 0.85rem;
      color: var(--muted);
      font-size: 0.95rem;
    }}
    .table-wrap {{
      overflow-x: auto;
      border: 1px solid var(--line);
      background: var(--panel);
      backdrop-filter: blur(6px);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
    }}
    th, td {{
      text-align: left;
      padding: 0.7rem 0.85rem;
      border-bottom: 1px solid var(--line);
      vertical-align: top;
    }}
    th {{
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      font-weight: 600;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    .empty {{ color: var(--muted); }}
    .error {{
      color: var(--warn);
      background: rgba(254, 243, 199, 0.7);
      border: 1px solid #f59e0b;
      padding: 0.65rem 0.8rem;
      margin: 0 0 0.85rem;
    }}
    .pill {{
      display: inline-block;
      font-family: "IBM Plex Mono", ui-monospace, monospace;
      font-size: 0.85rem;
    }}
    footer {{
      margin-top: 2rem;
      color: var(--muted);
      font-size: 0.85rem;
    }}
    a {{ color: var(--accent); }}
  </style>
</head>
<body>
  <main>
    <header>
      <h1 class="brand">docs-search</h1>
      <p class="lede">Local neurosymbolic documentation indexes and the linked S3 registry.</p>
      <div class="meta">
        <span>Local dir <code>{html.escape(str(index_dir))}</code></span>
        <span>Registry <code>{reg_label}</code></span>
      </div>
      <form class="search" method="get" action="/">
        <input type="search" name="q" value="{q_value}" placeholder="Filter by name, version, or repo" />
        <button type="submit">Filter</button>
      </form>
    </header>

    <section>
      <h2>Local indexes</h2>
      <p class="hint">Saved on this machine with a name and version.</p>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Version</th>
              <th>Repo</th>
              <th>Chunks</th>
              <th>Symbols</th>
              <th>Created</th>
            </tr>
          </thead>
          <tbody>{local_rows}</tbody>
        </table>
      </div>
    </section>

    <section>
      <h2>S3 registry</h2>
      <p class="hint">Published indexes available from the configured registry.</p>
      {error_block}
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Version</th>
              <th>Repo</th>
              <th>Chunks</th>
              <th>Size</th>
              <th>Published</th>
            </tr>
          </thead>
          <tbody>{remote_rows}</tbody>
        </table>
      </div>
    </section>

    <footer>
      JSON API:
      <a href="/api/local">/api/local</a> ·
      <a href="/api/registry">/api/registry</a> ·
      <a href="/api/health">/api/health</a> ·
      POST /api/search ·
      POST /api/ask
    </footer>
  </main>
</body>
</html>"""


def _local_row(meta: IndexMeta) -> str:
    return (
        "<tr>"
        f'<td class="pill">{html.escape(meta.name)}</td>'
        f'<td class="pill">{html.escape(meta.version)}</td>'
        f"<td>{html.escape(meta.repo)}</td>"
        f"<td>{meta.chunk_count}</td>"
        f"<td>{meta.symbol_count}</td>"
        f"<td>{html.escape(meta.created_at)}</td>"
        "</tr>"
    )


def _remote_row(entry: RegistryEntry) -> str:
    size = _format_bytes(entry.size_bytes)
    return (
        "<tr>"
        f'<td class="pill">{html.escape(entry.name)}</td>'
        f'<td class="pill">{html.escape(entry.version)}</td>'
        f"<td>{html.escape(entry.repo)}</td>"
        f"<td>{entry.chunk_count}</td>"
        f"<td>{html.escape(size)}</td>"
        f"<td>{html.escape(entry.published_at or entry.created_at)}</td>"
        "</tr>"
    )


def _format_bytes(size: int) -> str:
    if size < 1024:
        return f"{size} B"
    if size < 1024**2:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024**2):.1f} MB"


def run_server(
    *,
    host: str = "127.0.0.1",
    port: int = 8787,
    index_dir: Path | None = None,
    registry_url: str | None = None,
) -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "uvicorn is required for the web server. Install with: uv sync"
        ) from exc

    app = create_app(index_dir=index_dir, registry_url=registry_url)
    uvicorn.run(app, host=host, port=port, log_level="info")
