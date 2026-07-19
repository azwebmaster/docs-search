from docs_search.extract import extract_chunks_from_markdown, extract_symbols
from docs_search.graph import build_knowledge_graph, expand_via_graph


SAMPLE = """# Widgets

Overview of the Widget API.

## Creating widgets

Call `WidgetFactory.create()` to build a widget.

```python
factory = WidgetFactory()
widget = factory.create(name="demo")
```

See [configuration](./config.md) for options.

## Deleting widgets

Use `Widget.delete()` when finished.
"""


def test_extract_symbols_from_inline_and_code():
    symbols = extract_symbols(SAMPLE)
    assert "WidgetFactory.create()" in symbols or "WidgetFactory" in symbols
    assert any("Widget" in s for s in symbols)


def test_extract_chunks_preserves_heading_path_and_links():
    chunks = extract_chunks_from_markdown(SAMPLE, repo="acme/widgets", path="docs/widgets.md")
    assert len(chunks) >= 2
    create = next(c for c in chunks if "Creating" in c.title or "Creating" in " ".join(c.heading_path))
    assert create.heading_path[:1] == ["Widgets"]
    assert any(link.endswith("config.md") for link in create.links)


def test_knowledge_graph_links_symbols_and_docs():
    chunks = extract_chunks_from_markdown(SAMPLE, repo="acme/widgets", path="docs/widgets.md")
    graph, edges = build_knowledge_graph(chunks)
    assert any(e.relation == "MENTIONS" for e in edges)
    assert any(e.relation == "LINKS_TO" for e in edges)
    assert graph.number_of_nodes() > 0

    seed = [chunks[0].id]
    expanded = expand_via_graph(graph, seed, ["WidgetFactory"], max_hops=2, limit=20)
    assert expanded
