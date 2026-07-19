from pathlib import Path

import numpy as np

from docs_search.extract import extract_repo_chunks
from docs_search.graph import build_knowledge_graph
from docs_search.index_builder import index_local_path
from docs_search.search import NeurosymbolicIndex


FIXTURE = Path(__file__).parent / "fixtures" / "sample_docs"


def test_neurosymbolic_search_ranks_relevant_chunk(tmp_path, monkeypatch):
    def fake_embed(texts, model_name="x", batch_size=32):
        # Deterministic bag-of-chars projection for offline tests.
        dim = 32
        out = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for ch in text.lower():
                out[i, ord(ch) % dim] += 1.0
            norm = np.linalg.norm(out[i]) or 1.0
            out[i] /= norm
        return out

    monkeypatch.setattr("docs_search.index_builder.embed_texts", fake_embed)
    monkeypatch.setattr("docs_search.search.embed_query", lambda q, model_name="x": fake_embed([q])[0])

    meta = index_local_path(
        FIXTURE,
        repo_slug="local/sample",
        index_dir=tmp_path / "indexes",
        include_dirs=["."],
        embed_model="fake-model",
    )
    assert meta.chunk_count >= 2
    assert meta.symbol_count >= 1

    index = NeurosymbolicIndex.load("local/sample", index_dir=tmp_path / "indexes")
    hits = index.search("How do I create a widget with WidgetFactory?", top_k=3)
    assert hits
    top = hits[0]
    assert "widget" in top.path.lower() or "widget" in top.snippet.lower()
    assert top.score > 0


def test_graph_built_from_fixture():
    files = sorted(FIXTURE.rglob("*.md"))
    chunks = extract_repo_chunks(FIXTURE, "local/sample", files)
    graph, edges = build_knowledge_graph(chunks)
    assert any(e.relation == "LINKS_TO" for e in edges)
    assert graph.number_of_nodes() >= len(chunks)
