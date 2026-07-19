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
    DEFAULT_LEXICAL_WEIGHT,
    DEFAULT_NEURAL_WEIGHT,
    DEFAULT_REPOS_DIR,
    DEFAULT_SYMBOLIC_WEIGHT,
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
    """Build a neurosymbolic index (neural embeddings + symbolic knowledge graph)."""

    def progress(msg: str) -> None:
        console.print(f"[dim]•[/dim] {msg}")

    try:
        if local or Path(source).exists():
            meta = index_local_path(
                Path(source),
                index_dir=index_dir,
                include_dirs=include,
                embed_model=embed_model,
                progress=progress,
            )
        else:
            meta = index_github_repo(
                source,
                branch=branch,
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
    repo: Optional[str] = typer.Option(
        None,
        "--repo",
        "-r",
        help="owner/repo to search (required if multiple indexes exist)",
    ),
    top_k: int = typer.Option(5, "--top", "-k", min=1, max=50),
    index_dir: Path = typer.Option(DEFAULT_INDEX_DIR, "--index-dir"),
    neural: float = typer.Option(DEFAULT_NEURAL_WEIGHT, "--neural"),
    lexical: float = typer.Option(DEFAULT_LEXICAL_WEIGHT, "--lexical"),
    symbolic: float = typer.Option(DEFAULT_SYMBOLIC_WEIGHT, "--symbolic"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON results"),
) -> None:
    """Run neurosymbolic hybrid search over a local index."""
    metas = list_indexes(index_dir)
    if not metas:
        console.print("[red]No indexes found.[/red] Run: docs-search index owner/repo")
        raise typer.Exit(code=1)

    if repo is None:
        if len(metas) == 1:
            repo = metas[0].repo
        else:
            names = ", ".join(m.repo for m in metas)
            console.print(f"[red]Multiple indexes:[/red] {names}. Pass --repo.")
            raise typer.Exit(code=1)
    else:
        # Accept URL or slug.
        try:
            repo = normalize_github_source(repo)[1]
        except ValueError:
            pass

    try:
        index = NeurosymbolicIndex.load(repo, index_dir=index_dir)
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
    """List locally indexed repositories."""
    metas = list_indexes(index_dir)
    if not metas:
        console.print("No indexes yet.")
        return
    table = Table("Repo", "Chunks", "Symbols", "Edges", "Model")
    for meta in metas:
        table.add_row(
            meta.repo,
            str(meta.chunk_count),
            str(meta.symbol_count),
            str(meta.edge_count),
            meta.embed_model,
        )
    console.print(table)


if __name__ == "__main__":
    app()
