from __future__ import annotations

from pathlib import Path

DEFAULT_DATA_DIR = Path.home() / ".docs-search"
DEFAULT_REPOS_DIR = DEFAULT_DATA_DIR / "repos"
DEFAULT_INDEX_DIR = DEFAULT_DATA_DIR / "indexes"

DOC_EXTENSIONS = {".md", ".mdx", ".rst", ".txt"}
DEFAULT_DOC_GLOBS = ("docs", "doc", "documentation", "wiki", ".")

# Lightweight local ONNX embedding model (no PyTorch required).
DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Hybrid fusion weights (neural / lexical / symbolic).
DEFAULT_NEURAL_WEIGHT = 0.45
DEFAULT_LEXICAL_WEIGHT = 0.35
DEFAULT_SYMBOLIC_WEIGHT = 0.20

CHUNK_MAX_CHARS = 1200
CHUNK_OVERLAP_CHARS = 120
