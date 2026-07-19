from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from docs_search import __version__
from docs_search.config import (
    DEFAULT_EMBED_MODEL,
    DEFAULT_INDEX_DIR,
    DEFAULT_INDEX_VERSION,
    DEFAULT_LEXICAL_WEIGHT,
    DEFAULT_NEURAL_WEIGHT,
    DEFAULT_REPOS_DIR,
    DEFAULT_SYMBOLIC_WEIGHT,
    get_registry_url,
    set_registry_url,
)
from docs_search.index_builder import index_github_repo, index_local_path
from docs_search.ingest import normalize_github_source
from docs_search.search import NeurosymbolicIndex
from docs_search.store import list_indexes

app = typer.Typer(
    name="docs-search",
    help="Neurosymbolic local search over GitHub documentation repositories.",
    no_args_is_help=True,
    add_completion=False,
)
registry_app = typer.Typer(
    name="registry",
    help="Configure and browse the S3 index registry.",
    no_args_is_help=True,
)
app.add_typer(registry_app, name="registry")
console = Console()


@app.callback()
def _root() -> None:
    """Convert GitHub docs repos into local neurosymbolic search indexes."""


@app.command("version")
def version_cmd() -> None:
    """Print the package version."""
    console.print(__version__)


@app.command("ingest")
def ingest_cmd(
    source: str = typer.Argument(..., help="GitHub URL or owner/repo"),
    branch: Optional[str] = typer.Option(None, "--branch", "-b", help="Branch to clone"),
    repos_dir: Path = typer.Option(DEFAULT_REPOS_DIR, "--repos-dir"),
    force: bool = typer.Option(False, "--force", help="Re-clone even if present"),
) -> None:
    """Clone or update a GitHub repository locally (without indexing)."""
    from docs_search.ingest import clone_or_update

    path, slug = clone_or_update(source, repos_dir=repos_dir, branch=branch, force=force)
    console.print(f"[green]Ready[/green] {slug} → {path}")


@app.command("index")
def index_cmd(
    source: str = typer.Argument(..., help="GitHub URL, owner/repo, or local path"),
    branch: Optional[str] = typer.Option(None, "--branch", "-b"),
    local: bool = typer.Option(False, "--local", help="Treat SOURCE as a local directory"),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Index name to save as (default: derived from repo slug)",
    ),
    index_version: str = typer.Option(
        DEFAULT_INDEX_VERSION,
        "--index-version",
        "-V",
        help="Index version to save as (e.g. 1.0.0)",
    ),
    include: Optional[list[str]] = typer.Option(
        None,
        "--include",
        "-i",
        help="Doc subdirectories to include (repeatable). Default: docs,doc,documentation,.",
    ),
    embed_model: str = typer.Option(DEFAULT_EMBED_MODEL, "--model"),
    repos_dir: Path = typer.Option(DEFAULT_REPOS_DIR, "--repos-dir"),
    index_dir: Path = typer.Option(DEFAULT_INDEX_DIR, "--index-dir"),
    force: bool = typer.Option(False, "--force", help="Force re-clone before indexing"),
) -> None:
    """Build a neurosymbolic index and save it with a name and version."""

    def progress(msg: str) -> None:
        console.print(f"[dim]•[/dim] {msg}")

    try:
        if local or Path(source).exists():
            meta = index_local_path(
                Path(source),
                name=name,
                version=index_version,
                index_dir=index_dir,
                include_dirs=include,
                embed_model=embed_model,
                progress=progress,
            )
        else:
            meta = index_github_repo(
                source,
                branch=branch,
                name=name,
                version=index_version,
                repos_dir=repos_dir,
                index_dir=index_dir,
                include_dirs=include,
                embed_model=embed_model,
                force_clone=force,
                progress=progress,
            )
    except Exception as exc:  # noqa: BLE001 - surface CLI errors cleanly
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    table = Table(title="Index ready", show_header=False)
    table.add_row("Name", meta.name)
    table.add_row("Version", meta.version)
    table.add_row("Repo", meta.repo)
    table.add_row("Chunks", str(meta.chunk_count))
    table.add_row("Symbols", str(meta.symbol_count))
    table.add_row("Graph edges", str(meta.edge_count))
    table.add_row("Embed model", meta.embed_model)
    table.add_row("Created", meta.created_at)
    console.print(table)


@app.command("search")
def search_cmd(
    query: str = typer.Argument(..., help="Natural-language or symbol-aware query"),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Saved index name",
    ),
    index_version: Optional[str] = typer.Option(
        None,
        "--index-version",
        "-V",
        help="Saved index version (default: newest matching name/repo)",
    ),
    repo: Optional[str] = typer.Option(
        None,
        "--repo",
        "-r",
        help="owner/repo to search (used when --name is omitted)",
    ),
    top_k: int = typer.Option(5, "--top", "-k", min=1, max=50),
    index_dir: Path = typer.Option(DEFAULT_INDEX_DIR, "--index-dir"),
    neural: float = typer.Option(DEFAULT_NEURAL_WEIGHT, "--neural"),
    lexical: float = typer.Option(DEFAULT_LEXICAL_WEIGHT, "--lexical"),
    symbolic: float = typer.Option(DEFAULT_SYMBOLIC_WEIGHT, "--symbolic"),
    registry: Optional[str] = typer.Option(
        None,
        "--registry",
        help="s3://bucket/prefix used when downloading a missing named index",
    ),
    no_pull: bool = typer.Option(
        False,
        "--no-pull",
        help="Do not download a missing named index from the registry",
    ),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON results"),
) -> None:
    """Run neurosymbolic hybrid search over a local index.

    If ``--name`` is given and that index is not local, it is downloaded from the
    S3 registry automatically (unless ``--no-pull``).
    """
    metas = list_indexes(index_dir)
    if not metas and name is None:
        console.print(
            "[red]No indexes found.[/red] Pass --name to download from the registry, "
            "or run: docs-search index owner/repo --name …"
        )
        raise typer.Exit(code=1)

    if repo is not None:
        try:
            repo = normalize_github_source(repo)[1]
        except ValueError:
            pass

    if name is None and repo is None:
        names = sorted({m.name for m in metas})
        if len(names) == 1:
            name = names[0]
        elif len(metas) == 1:
            name = metas[0].name
            repo = metas[0].repo
        else:
            labels = ", ".join(f"{m.name}@{m.version}" for m in metas)
            console.print(f"[red]Multiple indexes:[/red] {labels}. Pass --name or --repo.")
            raise typer.Exit(code=1)

    def _on_pull(pulled_name: str, pulled_version: str) -> None:
        console.print(
            f"[dim]Index not found locally; downloading {pulled_name}@{pulled_version} "
            f"from registry…[/dim]"
        )

    try:
        index = NeurosymbolicIndex.load(
            repo,
            index_dir=index_dir,
            name=name,
            version=index_version,
            pull_missing=not no_pull,
            registry_url=registry,
            on_pull=_on_pull,
        )
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc

    hits = index.search(
        query,
        top_k=top_k,
        neural_weight=neural,
        lexical_weight=lexical,
        symbolic_weight=symbolic,
    )

    if json_out:
        import json

        console.print_json(json.dumps([h.model_dump() for h in hits]))
        return

    if not hits:
        console.print("No results.")
        return

    for rank, hit in enumerate(hits, start=1):
        heading = " › ".join(hit.heading_path) if hit.heading_path else hit.title
        body = (
            f"[bold]{hit.path}[/bold]  {heading}\n"
            f"{hit.snippet}\n"
            f"[dim]score={hit.score:.3f}  "
            f"neural={hit.neural_score:.2f}  "
            f"lexical={hit.lexical_score:.2f}  "
            f"symbolic={hit.symbolic_score:.2f}[/dim]"
        )
        if hit.matched_symbols:
            body += f"\n[cyan]symbols:[/cyan] {', '.join(hit.matched_symbols[:8])}"
        if hit.graph_hops:
            body += f"\n[magenta]graph:[/magenta] {' → '.join(hit.graph_hops[:6])}"
        console.print(Panel(body, title=f"{rank}. {hit.title}", border_style="blue"))


@app.command("list")
def list_cmd(
    index_dir: Path = typer.Option(DEFAULT_INDEX_DIR, "--index-dir"),
) -> None:
    """List locally saved named/versioned indexes."""
    metas = list_indexes(index_dir)
    if not metas:
        console.print("No indexes yet.")
        return
    table = Table("Name", "Version", "Repo", "Chunks", "Symbols", "Edges", "Model")
    for meta in metas:
        table.add_row(
            meta.name,
            meta.version,
            meta.repo,
            str(meta.chunk_count),
            str(meta.symbol_count),
            str(meta.edge_count),
            meta.embed_model,
        )
    console.print(table)


@app.command("publish")
def publish_cmd(
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Local index name"),
    index_version: Optional[str] = typer.Option(
        None,
        "--index-version",
        "-V",
        help="Local index version",
    ),
    repo: Optional[str] = typer.Option(None, "--repo", "-r", help="owner/repo filter"),
    registry: Optional[str] = typer.Option(
        None,
        "--registry",
        help="s3://bucket/prefix (default: configured registry URL)",
    ),
    index_dir: Path = typer.Option(DEFAULT_INDEX_DIR, "--index-dir"),
) -> None:
    """Publish a local named/versioned index to the S3 registry."""
    from docs_search.registry import publish_local_index

    if repo is not None:
        try:
            repo = normalize_github_source(repo)[1]
        except ValueError:
            pass

    try:
        entry = publish_local_index(
            name=name,
            version=index_version,
            repo=repo,
            registry_url=registry,
            index_dir=index_dir,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[green]Published[/green] {entry.name}@{entry.version} → s3://…/{entry.s3_key} "
        f"({entry.size_bytes} bytes)"
    )


@app.command("pull")
def pull_cmd(
    name: str = typer.Argument(..., help="Index name in the registry"),
    index_version: Optional[str] = typer.Argument(
        None,
        help="Index version in the registry (default: newest published)",
    ),
    registry: Optional[str] = typer.Option(
        None,
        "--registry",
        help="s3://bucket/prefix (default: configured registry URL)",
    ),
    index_dir: Path = typer.Option(DEFAULT_INDEX_DIR, "--index-dir"),
) -> None:
    """Download an index from the S3 registry into the local index directory."""
    from docs_search.registry import pull_registry_index

    try:
        meta = pull_registry_index(
            name,
            index_version,
            registry_url=registry,
            index_dir=index_dir,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(
        f"[green]Pulled[/green] {meta.name}@{meta.version} "
        f"({meta.chunk_count} chunks) → {index_dir / meta.name / meta.version}"
    )


@registry_app.command("set-url")
def registry_set_url_cmd(
    url: str = typer.Argument(..., help="s3://bucket/prefix for the shared index registry"),
) -> None:
    """Link this machine to an S3 registry URL."""
    from docs_search.registry import parse_registry_url

    try:
        loc = parse_registry_url(url)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    path = set_registry_url(loc.uri)
    console.print(f"[green]Registry set[/green] {loc.uri}")
    console.print(f"[dim]Saved to {path}[/dim]")


@registry_app.command("url")
def registry_url_cmd() -> None:
    """Show the configured S3 registry URL."""
    url = get_registry_url()
    if not url:
        console.print("No registry configured.")
        raise typer.Exit(code=1)
    console.print(url)


@registry_app.command("list")
def registry_list_cmd(
    registry: Optional[str] = typer.Option(
        None,
        "--registry",
        help="s3://bucket/prefix (default: configured registry URL)",
    ),
) -> None:
    """List indexes published to the S3 registry."""
    from docs_search.registry import list_registry_entries

    try:
        entries = list_registry_entries(registry_url=registry)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    if not entries:
        console.print("Registry is empty.")
        return

    table = Table("Name", "Version", "Repo", "Chunks", "Size", "Published")
    for entry in entries:
        table.add_row(
            entry.name,
            entry.version,
            entry.repo,
            str(entry.chunk_count),
            str(entry.size_bytes),
            entry.published_at or entry.created_at,
        )
    console.print(table)


@app.command("serve")
def serve_cmd(
    host: str = typer.Option("127.0.0.1", "--host", help="Bind host"),
    port: int = typer.Option(8787, "--port", "-p", help="Bind port"),
    index_dir: Path = typer.Option(DEFAULT_INDEX_DIR, "--index-dir"),
    registry: Optional[str] = typer.Option(
        None,
        "--registry",
        help="s3://bucket/prefix (default: configured registry URL)",
    ),
) -> None:
    """Start a web server that surfaces local indexes and the S3 registry."""
    from docs_search.server import run_server

    url = registry or get_registry_url()
    console.print(f"[green]Serving[/green] http://{host}:{port}")
    console.print(f"[dim]Local indexes:[/dim] {index_dir}")
    console.print(f"[dim]Registry:[/dim] {url or 'not configured'}")
    try:
        run_server(host=host, port=port, index_dir=index_dir, registry_url=registry)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
