from __future__ import annotations

import json
import os
import re
from pathlib import Path

DEFAULT_DATA_DIR = Path.home() / ".docs-search"
DEFAULT_REPOS_DIR = DEFAULT_DATA_DIR / "repos"
DEFAULT_INDEX_DIR = DEFAULT_DATA_DIR / "indexes"
DEFAULT_CONFIG_PATH = DEFAULT_DATA_DIR / "config.json"

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

DEFAULT_INDEX_VERSION = "0.1.0"

# OpenAI-compatible chat API used by RAG (`docs-search ask`).
DEFAULT_LLM_BASE_URL = "https://api.openai.com/v1"
DEFAULT_LLM_MODEL = "gpt-4o-mini"
DEFAULT_RAG_TOP_K = 5
DEFAULT_RAG_TEMPERATURE = 0.2
DEFAULT_RAG_MAX_TOKENS = 1024

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def sanitize_index_name(name: str) -> str:
    """Normalize a user-facing index name into a filesystem-safe slug."""
    cleaned = name.strip().replace("/", "__").replace(" ", "-")
    if not _NAME_RE.fullmatch(cleaned):
        raise ValueError(
            f"Invalid index name {name!r}. Use letters, digits, '.', '_', or '-'."
        )
    return cleaned


def sanitize_index_version(version: str) -> str:
    cleaned = version.strip()
    if not _VERSION_RE.fullmatch(cleaned):
        raise ValueError(
            f"Invalid index version {version!r}. Use letters, digits, '.', '_', or '-'."
        )
    return cleaned


def default_name_from_repo(repo_slug: str) -> str:
    return sanitize_index_name(repo_slug.replace("/", "__"))


def load_user_config(path: Path | None = None) -> dict:
    cfg_path = path or DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        return {}
    try:
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid config JSON at {cfg_path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Config at {cfg_path} must be a JSON object")
    return data


def save_user_config(data: dict, path: Path | None = None) -> Path:
    cfg_path = path or DEFAULT_CONFIG_PATH
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return cfg_path


def get_registry_url(path: Path | None = None) -> str | None:
    return load_user_config(path).get("registry_url")


def set_registry_url(url: str, path: Path | None = None) -> Path:
    data = load_user_config(path)
    data["registry_url"] = url.strip()
    return save_user_config(data, path)


def get_llm_base_url(path: Path | None = None) -> str:
    """Resolve the OpenAI-compatible API base URL.

    Precedence: ``DOCS_SEARCH_LLM_BASE_URL`` → ``OPENAI_BASE_URL`` →
    config ``llm_base_url`` → default OpenAI URL.
    """
    for key in ("DOCS_SEARCH_LLM_BASE_URL", "OPENAI_BASE_URL"):
        value = os.environ.get(key, "").strip()
        if value:
            return value.rstrip("/")
    cfg = load_user_config(path).get("llm_base_url")
    if isinstance(cfg, str) and cfg.strip():
        return cfg.strip().rstrip("/")
    return DEFAULT_LLM_BASE_URL


def get_llm_model(path: Path | None = None) -> str:
    """Resolve the chat model name for RAG generation."""
    for key in ("DOCS_SEARCH_LLM_MODEL", "OPENAI_MODEL"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    cfg = load_user_config(path).get("llm_model")
    if isinstance(cfg, str) and cfg.strip():
        return cfg.strip()
    return DEFAULT_LLM_MODEL


def get_llm_api_key(path: Path | None = None) -> str | None:
    """Resolve the API key for the chat provider.

    Precedence: ``DOCS_SEARCH_LLM_API_KEY`` → ``OPENAI_API_KEY`` →
    config ``llm_api_key``. Local servers (e.g. Ollama) may omit a key.
    """
    for key in ("DOCS_SEARCH_LLM_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    cfg = load_user_config(path).get("llm_api_key")
    if isinstance(cfg, str) and cfg.strip():
        return cfg.strip()
    return None
