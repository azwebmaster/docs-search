"""Neurosymbolic local search over GitHub documentation repositories."""

__version__ = "0.1.0"


def main() -> None:
    """CLI entrypoint used by the console script fallback."""
    from docs_search.cli import app

    app()
