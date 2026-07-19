from __future__ import annotations

from typing import Any, Callable

import httpx

from docs_search.config import (
    DEFAULT_LEXICAL_WEIGHT,
    DEFAULT_LLM_BASE_URL,
    DEFAULT_NEURAL_WEIGHT,
    DEFAULT_RAG_MAX_TOKENS,
    DEFAULT_RAG_TEMPERATURE,
    DEFAULT_RAG_TOP_K,
    DEFAULT_SYMBOLIC_WEIGHT,
    get_llm_api_key,
    get_llm_base_url,
    get_llm_model,
)
from docs_search.models import RagAnswer, RagSource, SearchHit
from docs_search.search import NeurosymbolicIndex

SYSTEM_PROMPT = """\
You are a documentation assistant. Answer the user's question using only the \
provided documentation excerpts. Cite sources inline with bracketed numbers \
like [1], [2] that match the excerpt numbers. If the excerpts do not contain \
enough information, say so clearly and suggest what is missing. Be concise \
and accurate; do not invent APIs, options, or behavior that are not present \
in the excerpts."""


class RagError(RuntimeError):
    """Raised when retrieval-augmented generation fails."""


def build_sources(index: NeurosymbolicIndex, hits: list[SearchHit]) -> list[RagSource]:
    """Map search hits to RAG sources including full chunk text when available."""
    sources: list[RagSource] = []
    for hit in hits:
        chunk = index.chunk_by_id(hit.chunk_id)
        sources.append(
            RagSource(
                chunk_id=hit.chunk_id,
                repo=hit.repo,
                path=hit.path,
                title=hit.title,
                heading_path=hit.heading_path,
                snippet=hit.snippet,
                text=chunk.text if chunk is not None else hit.snippet,
                score=hit.score,
            )
        )
    return sources


def format_context(sources: list[RagSource]) -> str:
    """Render retrieved sources as numbered excerpts for the LLM prompt."""
    blocks: list[str] = []
    for i, source in enumerate(sources, start=1):
        heading = " › ".join(source.heading_path) if source.heading_path else source.title
        body = (source.text or source.snippet).strip()
        blocks.append(
            f"[{i}] {source.path} — {heading}\n{body}"
        )
    return "\n\n".join(blocks)


def build_messages(question: str, sources: list[RagSource]) -> list[dict[str, str]]:
    context = format_context(sources)
    user = (
        f"Documentation excerpts:\n\n{context}\n\n"
        f"Question: {question.strip()}\n\n"
        "Answer with citations like [1] when you rely on an excerpt."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def chat_complete(
    messages: list[dict[str, str]],
    *,
    model: str,
    base_url: str,
    api_key: str | None = None,
    temperature: float = DEFAULT_RAG_TEMPERATURE,
    max_tokens: int = DEFAULT_RAG_MAX_TOKENS,
    timeout: float = 60.0,
    client: httpx.Client | None = None,
) -> str:
    """Call an OpenAI-compatible chat completions endpoint."""
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    owns_client = client is None
    http = client or httpx.Client(timeout=timeout)
    try:
        response = http.post(url, headers=headers, json=payload)
    except httpx.HTTPError as exc:
        raise RagError(f"LLM request failed: {exc}") from exc
    finally:
        if owns_client:
            http.close()

    if response.status_code >= 400:
        detail = response.text.strip()[:500] or response.reason_phrase
        raise RagError(f"LLM request failed ({response.status_code}): {detail}")

    try:
        data = response.json()
        content = data["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RagError("LLM response was missing chat completion content") from exc

    if not isinstance(content, str) or not content.strip():
        raise RagError("LLM returned an empty answer")
    return content.strip()


def ask(
    index: NeurosymbolicIndex,
    question: str,
    *,
    top_k: int = DEFAULT_RAG_TOP_K,
    neural_weight: float = DEFAULT_NEURAL_WEIGHT,
    lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
    symbolic_weight: float = DEFAULT_SYMBOLIC_WEIGHT,
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    temperature: float = DEFAULT_RAG_TEMPERATURE,
    max_tokens: int = DEFAULT_RAG_MAX_TOKENS,
    timeout: float = 60.0,
    client: httpx.Client | None = None,
    complete: Callable[..., str] | None = None,
) -> RagAnswer:
    """Retrieve relevant docs and generate a grounded answer.

    Uses an OpenAI-compatible chat API (OpenAI, Ollama, LM Studio, etc.).
    Pass ``complete`` to inject a custom generator (useful in tests).
    """
    cleaned = question.strip()
    if not cleaned:
        raise ValueError("Question must not be empty")

    hits = index.search(
        cleaned,
        top_k=top_k,
        neural_weight=neural_weight,
        lexical_weight=lexical_weight,
        symbolic_weight=symbolic_weight,
    )
    sources = build_sources(index, hits)

    resolved_model = model or get_llm_model()
    resolved_base = (base_url or get_llm_base_url()).rstrip("/")
    resolved_key = api_key if api_key is not None else get_llm_api_key()

    if not sources:
        return RagAnswer(
            question=cleaned,
            answer=(
                "I could not find relevant documentation for that question "
                "in the selected index."
            ),
            sources=[],
            model=resolved_model,
            index_name=index.name,
            index_version=index.version,
            repo=index.repo,
        )

    # Local OpenAI-compatible servers often accept any/empty key; cloud APIs need one.
    if (
        complete is None
        and not resolved_key
        and resolved_base.rstrip("/") == DEFAULT_LLM_BASE_URL.rstrip("/")
    ):
        raise RagError(
            "No LLM API key configured. Set DOCS_SEARCH_LLM_API_KEY or OPENAI_API_KEY, "
            "or point DOCS_SEARCH_LLM_BASE_URL at a local OpenAI-compatible server "
            "(for example http://127.0.0.1:11434/v1 for Ollama)."
        )

    messages = build_messages(cleaned, sources)
    generator = complete or chat_complete
    answer = generator(
        messages,
        model=resolved_model,
        base_url=resolved_base,
        api_key=resolved_key,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        client=client,
    )

    return RagAnswer(
        question=cleaned,
        answer=answer,
        sources=sources,
        model=resolved_model,
        index_name=index.name,
        index_version=index.version,
        repo=index.repo,
    )
