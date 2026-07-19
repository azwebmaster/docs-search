import pytest

from docs_search.ingest import normalize_github_source


@pytest.mark.parametrize(
    ("source", "slug"),
    [
        ("https://github.com/astral-sh/uv", "astral-sh/uv"),
        ("https://github.com/astral-sh/uv.git", "astral-sh/uv"),
        ("astral-sh/uv", "astral-sh/uv"),
        ("git@github.com:astral-sh/uv.git", "astral-sh/uv"),
    ],
)
def test_normalize_github_source(source: str, slug: str):
    clone_url, got_slug, local_name = normalize_github_source(source)
    assert got_slug == slug
    assert clone_url.endswith(".git")
    assert "__" in local_name


def test_normalize_rejects_non_github():
    with pytest.raises(ValueError):
        normalize_github_source("https://gitlab.com/foo/bar")
