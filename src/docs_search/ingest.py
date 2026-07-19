from __future__ import annotations

import re
import shutil
from pathlib import Path
from urllib.parse import urlparse

from git import Repo

from docs_search.config import DEFAULT_REPOS_DIR, DOC_EXTENSIONS


_GITHUB_SSH = re.compile(r"^git@github\.com:(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?$")
_GITHUB_PATH = re.compile(r"^/(?P<owner>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?/?$")


def normalize_github_source(source: str) -> tuple[str, str, str]:
    """Return (clone_url, owner/repo, local_slug) for a GitHub source string."""
    source = source.strip()

    if source.startswith("git@"):
        match = _GITHUB_SSH.match(source)
        if not match:
            raise ValueError(f"Unsupported SSH GitHub URL: {source}")
        owner, repo = match.group("owner"), match.group("repo")
        slug = f"{owner}/{repo}"
        return f"https://github.com/{slug}.git", slug, f"{owner}__{repo}"

    if source.startswith(("http://", "https://")):
        parsed = urlparse(source)
        if parsed.netloc not in {"github.com", "www.github.com"}:
            raise ValueError("Only GitHub repositories are supported")
        match = _GITHUB_PATH.match(parsed.path)
        if not match:
            raise ValueError(f"Could not parse GitHub URL: {source}")
        owner, repo = match.group("owner"), match.group("repo")
        slug = f"{owner}/{repo}"
        return f"https://github.com/{slug}.git", slug, f"{owner}__{repo}"

    # owner/repo shorthand
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", source):
        owner, repo = source.split("/", 1)
        slug = f"{owner}/{repo}"
        return f"https://github.com/{slug}.git", slug, f"{owner}__{repo}"

    raise ValueError(
        "Expected a GitHub URL or owner/repo shorthand, "
        f"got: {source!r}"
    )


def clone_or_update(
    source: str,
    repos_dir: Path | None = None,
    *,
    branch: str | None = None,
    force: bool = False,
) -> tuple[Path, str]:
    """Clone or update a GitHub repo locally. Returns (local_path, repo_slug)."""
    repos_dir = repos_dir or DEFAULT_REPOS_DIR
    repos_dir.mkdir(parents=True, exist_ok=True)

    clone_url, slug, local_name = normalize_github_source(source)
    dest = repos_dir / local_name

    if force and dest.exists():
        shutil.rmtree(dest)

    if dest.exists() and (dest / ".git").exists():
        repo = Repo(dest)
        repo.remotes.origin.fetch()
        if branch:
            repo.git.checkout(branch)
            repo.git.pull("origin", branch)
        else:
            repo.git.pull()
    else:
        if dest.exists():
            shutil.rmtree(dest)
        kwargs: dict = {"url": clone_url, "to_path": str(dest), "depth": 1}
        if branch:
            kwargs["branch"] = branch
        Repo.clone_from(**kwargs)

    return dest, slug


def iter_doc_files(
    root: Path,
    *,
    include_dirs: list[str] | None = None,
) -> list[Path]:
    """Find documentation files under a repository checkout."""
    include_dirs = include_dirs or ["docs", "doc", "documentation", "."]
    found: dict[Path, Path] = {}

    for dirname in include_dirs:
        base = root if dirname in {".", ""} else root / dirname
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in DOC_EXTENSIONS:
                continue
            # Skip common non-doc noise.
            parts = {p.lower() for p in path.parts}
            if any(skip in parts for skip in {".git", "node_modules", ".venv", "venv", "dist"}):
                continue
            found[path.resolve()] = path

    return sorted(found.values(), key=lambda p: str(p).lower())
