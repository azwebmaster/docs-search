from pathlib import Path

import httpx
import numpy as np
import pytest

from docs_search.config import (
    DEFAULT_LLM_BASE_URL,
    get_llm_api_key,
    get_llm_base_url,
    get_llm_model,
)
from docs_search.index_builder import index_local_path
from docs_search.models import RagSource
from docs_search.rag import RagError, ask, build_messages, build_sources, chat_complete, format_context
from docs_search.search import NeurosymbolicIndex
from docs_search.server import create_app


FIXTURE = Path(__file__).parent / "fixtures" / "sample_docs"


def _fake_embed(texts, model_name="x", batch_size=32):
    dim = 32
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for ch in text.lower():
            out[i, ord(ch) % dim] += 1.0
        norm = np.linalg.norm(out[i]) or 1.0
        out[i] /= norm
    return out


def _build_index(tmp_path, monkeypatch):
    monkeypatch.setattr("docs_search.index_builder.embed_texts", _fake_embed)
    monkeypatch.setattr(
        "docs_search.search.embed_query",
        lambda q, model_name="x": _fake_embed([q])[0],
    )
    meta = index_local_path(
        FIXTURE,
        repo_slug="local/sample",
        name="sample",
        version="1.0.0",
        index_dir=tmp_path / "indexes",
        include_dirs=["."],
        embed_model="fake-model",
    )
    index = NeurosymbolicIndex.load(
        "local/sample",
        index_dir=tmp_path / "indexes",
        name="sample",
        version="1.0.0",
    )
    return meta, index


def test_llm_config_helpers(monkeypatch, tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text(
        '{"llm_base_url": "http://cfg.example/v1", "llm_model": "cfg-model", '
        '"llm_api_key": "cfg-key"}\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("DOCS_SEARCH_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("DOCS_SEARCH_LLM_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("DOCS_SEARCH_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    assert get_llm_base_url(cfg) == "http://cfg.example/v1"
    assert get_llm_model(cfg) == "cfg-model"
    assert get_llm_api_key(cfg) == "cfg-key"

    monkeypatch.setenv("OPENAI_BASE_URL", "http://env.example/v1/")
    monkeypatch.setenv("OPENAI_MODEL", "env-model")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    assert get_llm_base_url(cfg) == "http://env.example/v1"
    assert get_llm_model(cfg) == "env-model"
    assert get_llm_api_key(cfg) == "env-key"

    monkeypatch.setenv("DOCS_SEARCH_LLM_BASE_URL", "http://docs.example/v1")
    monkeypatch.setenv("DOCS_SEARCH_LLM_MODEL", "docs-model")
    monkeypatch.setenv("DOCS_SEARCH_LLM_API_KEY", "docs-key")
    assert get_llm_base_url(cfg) == "http://docs.example/v1"
    assert get_llm_model(cfg) == "docs-model"
    assert get_llm_api_key(cfg) == "docs-key"


def test_format_context_numbers_sources():
    sources = [
        RagSource(
            chunk_id="c1",
            repo="local/sample",
            path="widgets.md",
            title="Widgets",
            heading_path=["Widgets", "Creating widgets"],
            snippet="Use WidgetFactory",
            text="Use `WidgetFactory` to construct widgets.",
            score=0.9,
        )
    ]
    rendered = format_context(sources)
    assert rendered.startswith("[1] widgets.md — Widgets › Creating widgets")
    assert "WidgetFactory" in rendered
    messages = build_messages("How do I create a widget?", sources)
    assert messages[0]["role"] == "system"
    assert "Question: How do I create a widget?" in messages[1]["content"]


def test_ask_uses_retrieved_context(tmp_path, monkeypatch):
    _, index = _build_index(tmp_path, monkeypatch)

    def fake_complete(messages, **kwargs):
        assert "WidgetFactory" in messages[1]["content"]
        return "Create widgets with WidgetFactory.create(). [1]"

    result = ask(
        index,
        "How do I create a widget with WidgetFactory?",
        top_k=3,
        complete=fake_complete,
        model="fake-model",
        base_url="http://example.test/v1",
        api_key="unused",
    )
    assert "WidgetFactory" in result.answer
    assert result.sources
    assert result.model == "fake-model"
    assert result.index_name == "sample"
    assert result.index_version == "1.0.0"
    assert result.sources[0].text


def test_ask_empty_question_raises(tmp_path, monkeypatch):
    _, index = _build_index(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="empty"):
        ask(index, "   ", complete=lambda *a, **k: "nope")


def test_ask_requires_api_key_for_default_openai(tmp_path, monkeypatch):
    _, index = _build_index(tmp_path, monkeypatch)
    monkeypatch.delenv("DOCS_SEARCH_LLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("docs_search.rag.get_llm_api_key", lambda path=None: None)
    monkeypatch.setattr("docs_search.rag.get_llm_base_url", lambda path=None: DEFAULT_LLM_BASE_URL)
    with pytest.raises(RagError, match="API key"):
        ask(index, "How do I create a widget?", model="gpt-4o-mini")


def test_chat_complete_parses_response():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/chat/completions")
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": "  Hello from docs.  "}}
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        text = chat_complete(
            [{"role": "user", "content": "hi"}],
            model="gpt-test",
            base_url="https://api.example.com/v1",
            api_key="test-key",
            client=client,
        )
    assert text == "Hello from docs."


def test_chat_complete_error_status():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text="unauthorized")

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        with pytest.raises(RagError, match="401"):
            chat_complete(
                [{"role": "user", "content": "hi"}],
                model="gpt-test",
                base_url="https://api.example.com/v1",
                api_key="bad",
                client=client,
            )


def test_api_search(tmp_path, monkeypatch):
    _build_index(tmp_path, monkeypatch)
    app = create_app(index_dir=tmp_path / "indexes")

    from fastapi.testclient import TestClient

    client = TestClient(app)
    search = client.post(
        "/api/search",
        json={"query": "WidgetFactory create", "name": "sample", "version": "1.0.0"},
    )
    assert search.status_code == 200
    payload = search.json()
    assert payload["hits"]
    assert payload["index_name"] == "sample"


def test_api_ask_with_mocked_llm(tmp_path, monkeypatch):
    _build_index(tmp_path, monkeypatch)

    def fake_ask(index, question, **kwargs):
        from docs_search.models import RagAnswer

        hits = index.search(question, top_k=kwargs.get("top_k", 3))
        sources = build_sources(index, hits)
        return RagAnswer(
            question=question,
            answer="Use WidgetFactory. [1]",
            sources=sources,
            model=kwargs.get("model") or "fake",
            index_name=index.name,
            index_version=index.version,
            repo=index.repo,
        )

    monkeypatch.setattr("docs_search.rag.ask", fake_ask)
    app = create_app(index_dir=tmp_path / "indexes")
    from fastapi.testclient import TestClient

    client = TestClient(app)
    resp = client.post(
        "/api/ask",
        json={
            "query": "How do I create a widget?",
            "name": "sample",
            "version": "1.0.0",
            "model": "fake-model",
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "WidgetFactory" in data["answer"]
    assert data["sources"]
