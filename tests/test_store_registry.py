from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from docs_search.config import default_name_from_repo, set_registry_url
from docs_search.index_builder import index_local_path
from docs_search.models import RegistryManifest
from docs_search.registry import (
    S3Registry,
    ensure_local_index,
    parse_registry_url,
    publish_local_index,
    pull_registry_index,
)
from docs_search.search import NeurosymbolicIndex
from docs_search.server import create_app
from docs_search.store import IndexStore, find_index, list_indexes


FIXTURE = Path(__file__).parent / "fixtures" / "sample_docs"


class FakeS3:
    """Minimal in-memory S3 stub for registry tests."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], bytes] = {}

    def put_object(self, *, Bucket: str, Key: str, Body: bytes, ContentType: str | None = None):
        if isinstance(Body, str):
            Body = Body.encode("utf-8")
        self.objects[(Bucket, Key)] = Body
        return {}

    def get_object(self, *, Bucket: str, Key: str):
        try:
            body = self.objects[(Bucket, Key)]
        except KeyError as exc:
            err = type("ClientError", (Exception,), {})("NoSuchKey")
            err.response = {"Error": {"Code": "NoSuchKey"}}  # type: ignore[attr-defined]
            raise err from exc

        class _Body:
            def __init__(self, data: bytes) -> None:
                self._data = data

            def read(self) -> bytes:
                return self._data

        return {"Body": _Body(body)}

    def upload_file(self, filename: str, Bucket: str, Key: str) -> None:
        self.objects[(Bucket, Key)] = Path(filename).read_bytes()

    def download_file(self, Bucket: str, Key: str, Filename: str) -> None:
        try:
            data = self.objects[(Bucket, Key)]
        except KeyError as exc:
            raise FileNotFoundError(Key) from exc
        Path(Filename).write_bytes(data)


def _fake_embed(texts, model_name="x", batch_size=32):
    dim = 16
    out = np.zeros((len(texts), dim), dtype=np.float32)
    for i, text in enumerate(texts):
        for ch in text.lower():
            out[i, ord(ch) % dim] += 1.0
        norm = np.linalg.norm(out[i]) or 1.0
        out[i] /= norm
    return out


@pytest.fixture
def indexed(tmp_path, monkeypatch):
    monkeypatch.setattr("docs_search.index_builder.embed_texts", _fake_embed)
    monkeypatch.setattr(
        "docs_search.search.embed_query",
        lambda q, model_name="x": _fake_embed([q])[0],
    )
    index_dir = tmp_path / "indexes"
    meta = index_local_path(
        FIXTURE,
        repo_slug="local/sample",
        name="sample-docs",
        version="1.2.0",
        index_dir=index_dir,
        include_dirs=["."],
        embed_model="fake-model",
    )
    return meta, index_dir


def test_parse_registry_url():
    loc = parse_registry_url("s3://my-bucket/docs/registry")
    assert loc.bucket == "my-bucket"
    assert loc.prefix == "docs/registry"
    assert loc.key("indices", "a", "1", "index.tar.gz") == "docs/registry/indices/a/1/index.tar.gz"


def test_index_saved_with_name_and_version(indexed):
    meta, index_dir = indexed
    assert meta.name == "sample-docs"
    assert meta.version == "1.2.0"
    assert (index_dir / "sample-docs" / "1.2.0" / "meta.json").exists()

    found = find_index(name="sample-docs", version="1.2.0", index_dir=index_dir)
    assert found.key() == "sample-docs@1.2.0"

    listed = list_indexes(index_dir)
    assert len(listed) == 1
    assert listed[0].name == "sample-docs"

    index = NeurosymbolicIndex.load(name="sample-docs", version="1.2.0", index_dir=index_dir)
    assert index.repo == "local/sample"
    assert index.search("widget")


def test_package_and_extract_tarball(indexed, tmp_path):
    meta, index_dir = indexed
    store = IndexStore(meta.name, meta.version, index_dir=index_dir)
    archive = store.package_tarball(tmp_path / "pack.tar.gz")
    assert archive.exists()

    dest = tmp_path / "restored"
    restored = IndexStore.extract_tarball(archive, index_dir=dest)
    assert restored.exists()
    assert restored.load_meta().name == "sample-docs"


def test_s3_registry_publish_and_pull(indexed, tmp_path):
    meta, index_dir = indexed
    fake = FakeS3()
    registry_url = "s3://test-bucket/docs-search"
    reg = S3Registry(registry_url, client=fake)

    entry = reg.publish(meta, index_dir=index_dir)
    assert entry.name == "sample-docs"
    assert entry.version == "1.2.0"
    assert entry.size_bytes > 0
    assert (fake.objects.get(("test-bucket", "docs-search/registry.json")))

    manifest = RegistryManifest.model_validate_json(
        fake.objects[("test-bucket", "docs-search/registry.json")]
    )
    assert len(manifest.indices) == 1

    pull_dir = tmp_path / "pulled"
    pulled = reg.pull("sample-docs", "1.2.0", index_dir=pull_dir)
    assert pulled.chunk_count == meta.chunk_count
    assert (pull_dir / "sample-docs" / "1.2.0" / "embeddings.npy").exists()


def test_publish_helper_uses_config(indexed, tmp_path, monkeypatch):
    meta, index_dir = indexed
    cfg = tmp_path / "config.json"
    set_registry_url("s3://cfg-bucket/prefix", path=cfg)
    monkeypatch.setattr("docs_search.registry.get_registry_url", lambda: "s3://cfg-bucket/prefix")

    fake = FakeS3()
    entry = publish_local_index(
        name=meta.name,
        version=meta.version,
        registry_url=None,
        index_dir=index_dir,
        client=fake,
    )
    assert entry.s3_key.endswith("index.tar.gz")

    pulled = pull_registry_index(
        meta.name,
        meta.version,
        registry_url="s3://cfg-bucket/prefix",
        index_dir=tmp_path / "out",
        client=fake,
    )
    assert pulled.name == meta.name


def test_search_auto_pulls_missing_named_index(indexed, tmp_path, monkeypatch):
    meta, index_dir = indexed
    fake = FakeS3()
    registry_url = "s3://auto-pull-bucket/reg"
    S3Registry(registry_url, client=fake).publish(meta, index_dir=index_dir)
    monkeypatch.setattr(
        "docs_search.registry.get_registry_url",
        lambda: registry_url,
    )

    empty_dir = tmp_path / "empty-indexes"
    empty_dir.mkdir()
    pulled_versions: list[tuple[str, str]] = []

    index = NeurosymbolicIndex.load(
        name="sample-docs",
        index_dir=empty_dir,
        registry_url=registry_url,
        registry_client=fake,
        on_pull=lambda n, v: pulled_versions.append((n, v)),
    )
    assert pulled_versions == [("sample-docs", "1.2.0")]
    assert (empty_dir / "sample-docs" / "1.2.0" / "meta.json").exists()
    hits = index.search("widget")
    assert hits


def test_ensure_local_index_pulls_latest_when_version_omitted(indexed, tmp_path, monkeypatch):
    meta, index_dir = indexed
    fake = FakeS3()
    registry_url = "s3://latest-bucket/reg"
    reg = S3Registry(registry_url, client=fake)
    reg.publish(meta, index_dir=index_dir)

    # Publish a second version with newer published_at by re-indexing.
    newer = index_local_path(
        FIXTURE,
        repo_slug="local/sample",
        name="sample-docs",
        version="2.0.0",
        index_dir=index_dir,
        include_dirs=["."],
        embed_model="fake-model",
    )
    reg.publish(newer, index_dir=index_dir)

    monkeypatch.setattr(
        "docs_search.registry.get_registry_url",
        lambda: registry_url,
    )
    empty_dir = tmp_path / "pulled-latest"
    ensured = ensure_local_index(
        name="sample-docs",
        index_dir=empty_dir,
        registry_url=registry_url,
        client=fake,
    )
    assert ensured.version == "2.0.0"
    assert (empty_dir / "sample-docs" / "2.0.0" / "embeddings.npy").exists()


def test_ensure_local_index_no_pull_without_name(indexed, tmp_path):
    _, index_dir = indexed
    with pytest.raises(FileNotFoundError):
        ensure_local_index(
            name=None,
            repo="missing/repo",
            index_dir=index_dir,
            pull_missing=True,
        )


def test_load_respects_no_pull(indexed, tmp_path, monkeypatch):
    meta, index_dir = indexed
    fake = FakeS3()
    registry_url = "s3://nopull-bucket/reg"
    S3Registry(registry_url, client=fake).publish(meta, index_dir=index_dir)
    monkeypatch.setattr(
        "docs_search.registry.get_registry_url",
        lambda: registry_url,
    )
    empty_dir = tmp_path / "still-empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        NeurosymbolicIndex.load(
            name="sample-docs",
            index_dir=empty_dir,
            pull_missing=False,
            registry_url=registry_url,
            registry_client=fake,
        )


def test_find_entry_returns_exact_and_latest(indexed, tmp_path):
    meta, index_dir = indexed
    fake = FakeS3()
    reg = S3Registry("s3://find-bucket/reg", client=fake)
    reg.publish(meta, index_dir=index_dir)
    newer = index_local_path(
        FIXTURE,
        repo_slug="local/sample",
        name="sample-docs",
        version="9.9.9",
        index_dir=index_dir,
        include_dirs=["."],
        embed_model="fake-model",
    )
    reg.publish(newer, index_dir=index_dir)

    exact = reg.find_entry("sample-docs", "1.2.0")
    assert exact is not None
    assert exact.version == "1.2.0"
    latest = reg.find_entry("sample-docs")
    assert latest is not None
    assert latest.version == "9.9.9"
    assert reg.find_entry("sample-docs", "0.0.1") is None
    assert reg.find_entry("missing-name") is None


def test_web_server_lists_local_and_registry(indexed, tmp_path):
    meta, index_dir = indexed
    fake = FakeS3()
    registry_url = "s3://web-bucket/reg"
    S3Registry(registry_url, client=fake).publish(meta, index_dir=index_dir)

    app = create_app(
        index_dir=index_dir,
        registry_url=registry_url,
        registry_client=fake,
    )
    client = TestClient(app)

    local = client.get("/api/local").json()
    assert local[0]["name"] == "sample-docs"
    assert local[0]["version"] == "1.2.0"

    remote = client.get("/api/registry").json()
    assert remote["registry_url"] == "s3://web-bucket/reg"
    assert remote["error"] is None
    assert remote["indices"][0]["name"] == "sample-docs"

    page = client.get("/")
    assert page.status_code == 200
    assert "sample-docs" in page.text
    assert "docs-search" in page.text
    assert "Local indexes" in page.text
    assert "S3 registry" in page.text


def test_default_name_from_repo():
    assert default_name_from_repo("astral-sh/uv") == "astral-sh__uv"


def test_legacy_meta_load(tmp_path):
    root = tmp_path / "indexes" / "astral-sh__uv"
    root.mkdir(parents=True)
    (root / "meta.json").write_text(
        json.dumps(
            {
                "repo": "astral-sh/uv",
                "source": "https://github.com/astral-sh/uv.git",
                "chunk_count": 1,
                "symbol_count": 0,
                "edge_count": 0,
                "embed_model": "x",
                "created_at": "2020-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    # Incomplete legacy index still lists via meta.
    metas = list_indexes(tmp_path / "indexes")
    assert len(metas) == 1
    assert metas[0].name == "astral-sh__uv"
    assert metas[0].version == "0.1.0"
