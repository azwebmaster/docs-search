from __future__ import annotations

import hashlib
import re
from pathlib import Path

from docs_search.config import CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS
from docs_search.models import DocChunk

# Identifiers that look like APIs / code symbols in prose and fenced code.
_SYMBOL_RE = re.compile(
    r"\b(?:"
    r"[A-Z][a-z]+(?:[A-Z][a-zA-Z0-9]+)+"  # PascalCase
    r"|[a-z_][a-z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+"  # dotted.path
    r"|[a-z_][a-z0-9_]{2,}(?=\()"  # function(
    r")\b"
)
_CODE_FENCE_RE = re.compile(r"```[\w+-]*\n(.*?)```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


def _chunk_id(repo: str, path: str, title: str, start_line: int) -> str:
    raw = f"{repo}:{path}:{title}:{start_line}"
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


_DOC_SUFFIXES = (".md", ".mdx", ".rst", ".txt", ".html")


def _is_symbol(token: str) -> bool:
    token = token.strip()
    if not (1 < len(token) < 80):
        return False
    if token.startswith("http") or "/" in token or "\\" in token:
        return False
    lower = token.lower()
    if lower.endswith(_DOC_SUFFIXES):
        return False
    return True


def extract_symbols(text: str) -> list[str]:
    symbols: set[str] = set()
    for match in _INLINE_CODE_RE.finditer(text):
        token = match.group(1).strip()
        if _is_symbol(token):
            symbols.add(token)
    for match in _CODE_FENCE_RE.finditer(text):
        for sym in _SYMBOL_RE.findall(match.group(1)):
            if _is_symbol(sym):
                symbols.add(sym)
    for sym in _SYMBOL_RE.findall(text):
        if _is_symbol(sym):
            symbols.add(sym)
    return sorted(symbols)


def extract_links(text: str) -> list[str]:
    links: list[str] = []
    for _label, target in _LINK_RE.findall(text):
        target = target.strip()
        if target and not target.startswith("#"):
            links.append(target.split("#", 1)[0])
    return links


def _split_sections(text: str) -> list[tuple[list[str], str, int, int]]:
    """Split markdown into (heading_path, body, start_line, end_line)."""
    lines = text.splitlines()
    if not lines:
        return []

    sections: list[tuple[list[str], list[str], int, int]] = []
    stack: list[tuple[int, str]] = []
    current_lines: list[str] = []
    start_line = 1

    def flush(end_line: int) -> None:
        nonlocal current_lines, start_line
        body = "\n".join(current_lines).strip()
        if body:
            heading_path = [title for _, title in stack]
            sections.append((heading_path, current_lines[:], start_line, end_line))
        current_lines = []

    for idx, line in enumerate(lines, start=1):
        heading = _HEADING_RE.match(line)
        if heading:
            if current_lines:
                flush(idx - 1)
            level = len(heading.group(1))
            title = heading.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            start_line = idx
            current_lines = [line]
        else:
            if not current_lines and not stack:
                # Preamble before first heading.
                start_line = idx
            current_lines.append(line)

    if current_lines:
        flush(len(lines))

    # Convert accumulated line lists into joined text.
    result: list[tuple[list[str], str, int, int]] = []
    for heading_path, body_lines, start, end in sections:
        result.append((heading_path, "\n".join(body_lines).strip(), start, end))
    return result


def _window_chunks(text: str, max_chars: int, overlap: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        # Prefer breaking on paragraph boundaries.
        if end < len(text):
            break_at = text.rfind("\n\n", start, end)
            if break_at > start + max_chars // 3:
                end = break_at
        parts.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [p for p in parts if p]


def extract_chunks_from_markdown(
    text: str,
    *,
    repo: str,
    path: str,
    max_chars: int = CHUNK_MAX_CHARS,
    overlap: int = CHUNK_OVERLAP_CHARS,
) -> list[DocChunk]:
    chunks: list[DocChunk] = []
    sections = _split_sections(text)
    if not sections:
        sections = [([], text.strip(), 1, max(1, text.count("\n") + 1))]

    for heading_path, body, start_line, end_line in sections:
        title = heading_path[-1] if heading_path else Path(path).stem
        for i, window in enumerate(_window_chunks(body, max_chars, overlap)):
            symbols = extract_symbols(window)
            links = extract_links(window)
            chunk_title = title if i == 0 else f"{title} (part {i + 1})"
            chunks.append(
                DocChunk(
                    id=_chunk_id(repo, path, chunk_title, start_line + i),
                    repo=repo,
                    path=path,
                    title=chunk_title,
                    heading_path=heading_path,
                    text=window,
                    symbols=symbols,
                    links=links,
                    start_line=start_line,
                    end_line=end_line,
                )
            )
    return chunks


def extract_repo_chunks(repo_root: Path, repo_slug: str, files: list[Path]) -> list[DocChunk]:
    chunks: list[DocChunk] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rel = str(path.relative_to(repo_root)).replace("\\", "/")
        chunks.extend(extract_chunks_from_markdown(text, repo=repo_slug, path=rel))
    return chunks
