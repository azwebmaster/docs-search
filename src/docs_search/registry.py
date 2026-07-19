from __future__ import annotations

import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from docs_search.config import (
    DEFAULT_INDEX_DIR,
    get_registry_url,
    sanitize_index_name,
    sanitize_index_version,
)
from docs_search.models import IndexMeta, RegistryEntry, RegistryManifest
from docs_search.store import IndexStore, find_index


@dataclass(frozen=True)
class S3Location:
    """Parsed s3://bucket/prefix location."""

    bucket: str
    prefix: str

    @property
    def uri(self) -> str:
        if self.prefix:
            return f"s3://{self.bucket}/{self.prefix}"
        return f"s3://{self.bucket}"

    def key(self, *parts: str) -> str:
        chunks = [p.strip("/") for p in (self.prefix, *parts) if p and p.strip("/")]
        return "/".join(chunks)


def parse_registry_url(url: str) -> S3Location:
    """Parse an s3://bucket[/prefix] registry URL."""
    raw = url.strip()
    if not raw:
        raise ValueError("Registry URL is empty")
    parsed = urlparse(raw)
    if parsed.scheme != "s3":
        raise ValueError(f"Registry URL must use s3:// scheme, got {raw!r}")
    bucket = parsed.netloc.strip()
    if not bucket:
        raise ValueError(f"Registry URL missing bucket: {raw!r}")
    prefix = parsed.path.lstrip("/").rstrip("/")
    return S3Location(bucket=bucket, prefix=prefix)


def resolve_registry_url(url: str | None = None) -> str:
    resolved = (url or get_registry_url() or "").strip()
    if not resolved:
        raise ValueError(
            "No registry configured. Pass --registry s3://bucket/prefix "
            "or run: docs-search registry set-url s3://bucket/prefix"
        )
    return resolved


class S3Registry:
    """Client for a shared S3 index registry."""

    MANIFEST_NAME = "registry.json"

    def __init__(self, url: str, *, client: Any | None = None) -> None:
        self.location = parse_registry_url(url)
        self._client = client

    @property
    def url(self) -> str:
        return self.location.uri

    @property
    def client(self) -> Any:
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - dependency guard
                raise RuntimeError(
                    "boto3 is required for S3 registry operations. Install with: uv sync"
                ) from exc
            self._client = boto3.client("s3")
        return self._client

    @property
    def manifest_key(self) -> str:
        return self.location.key(self.MANIFEST_NAME)

    def load_manifest(self) -> RegistryManifest:
        try:
            response = self.client.get_object(
                Bucket=self.location.bucket,
                Key=self.manifest_key,
            )
        except Exception as exc:
            # boto3 ClientError code NoSuchKey / 404 → empty manifest.
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if code in {"NoSuchKey", "404", "NotFound"}:
                return RegistryManifest()
            message = str(exc)
            if "NoSuchKey" in message or "Not Found" in message:
                return RegistryManifest()
            raise
        body = response["Body"].read().decode("utf-8")
        if not body.strip():
            return RegistryManifest()
        return RegistryManifest.model_validate_json(body)

    def save_manifest(self, manifest: RegistryManifest) -> None:
        manifest.updated_at = datetime.now(timezone.utc).isoformat()
        payload = manifest.model_dump_json(indent=2)
        self.client.put_object(
            Bucket=self.location.bucket,
            Key=self.manifest_key,
            Body=payload.encode("utf-8"),
            ContentType="application/json",
        )

    def list_entries(self) -> list[RegistryEntry]:
        manifest = self.load_manifest()
        return sorted(manifest.indices, key=lambda e: (e.name, e.version, e.published_at))

    def publish(
        self,
        meta: IndexMeta,
        *,
        index_dir: Path | None = None,
    ) -> RegistryEntry:
        """Upload a local named/versioned index and update the registry manifest."""
        store = IndexStore(meta.name, meta.version, index_dir=index_dir or DEFAULT_INDEX_DIR)
        if not store.exists():
            raise FileNotFoundError(f"Local index not found: {meta.key()} at {store.root}")

        s3_key = self.location.key("indices", meta.name, meta.version, "index.tar.gz")
        with tempfile.TemporaryDirectory(prefix="docs-search-publish-") as tmp:
            archive = Path(tmp) / "index.tar.gz"
            store.package_tarball(archive)
            size_bytes = archive.stat().st_size
            self.client.upload_file(str(archive), self.location.bucket, s3_key)

        # Also publish meta.json beside the archive for cheap listing/inspection.
        meta_key = self.location.key("indices", meta.name, meta.version, "meta.json")
        self.client.put_object(
            Bucket=self.location.bucket,
            Key=meta_key,
            Body=meta.model_dump_json(indent=2).encode("utf-8"),
            ContentType="application/json",
        )

        entry = RegistryEntry(
            name=meta.name,
            version=meta.version,
            repo=meta.repo,
            source=meta.source,
            chunk_count=meta.chunk_count,
            symbol_count=meta.symbol_count,
            edge_count=meta.edge_count,
            embed_model=meta.embed_model,
            created_at=meta.created_at,
            s3_key=s3_key,
            size_bytes=size_bytes,
            published_at=datetime.now(timezone.utc).isoformat(),
        )

        manifest = self.load_manifest()
        manifest.indices = [
            e for e in manifest.indices if not (e.name == entry.name and e.version == entry.version)
        ]
        manifest.indices.append(entry)
        self.save_manifest(manifest)
        return entry

    def pull(
        self,
        name: str,
        version: str | None = None,
        *,
        index_dir: Path | None = None,
    ) -> IndexMeta:
        """Download a registry index into the local index directory.

        When ``version`` is omitted, pulls the newest published entry for ``name``.
        """
        entry = self.find_entry(name, version)
        if entry is None:
            label = f"{name}@{version}" if version else name
            raise FileNotFoundError(f"Registry has no index {label}")

        with tempfile.TemporaryDirectory(prefix="docs-search-pull-") as tmp:
            archive = Path(tmp) / "index.tar.gz"
            self.client.download_file(self.location.bucket, entry.s3_key, str(archive))
            store = IndexStore.extract_tarball(
                archive,
                name=entry.name,
                version=entry.version,
                index_dir=index_dir or DEFAULT_INDEX_DIR,
            )
            return store.load_meta()

    def find_entry(self, name: str, version: str | None = None) -> RegistryEntry | None:
        """Find a registry entry by name, optionally pinned to a version.

        When ``version`` is omitted, returns the newest entry for ``name``
        (by ``published_at``, then ``created_at``).
        """
        name = sanitize_index_name(name)
        entries = [e for e in self.list_entries() if e.name == name]
        if version is not None:
            version = sanitize_index_version(version)
            for entry in entries:
                if entry.version == version:
                    return entry
            return None
        if not entries:
            return None
        return sorted(
            entries,
            key=lambda e: (e.published_at or "", e.created_at or ""),
        )[-1]

    def get_entry(self, name: str, version: str) -> RegistryEntry | None:
        return self.find_entry(name, version)


def publish_local_index(
    *,
    name: str | None = None,
    version: str | None = None,
    repo: str | None = None,
    registry_url: str | None = None,
    index_dir: Path | None = None,
    client: Any | None = None,
) -> RegistryEntry:
    meta = find_index(name=name, version=version, repo=repo, index_dir=index_dir)
    registry = S3Registry(resolve_registry_url(registry_url), client=client)
    return registry.publish(meta, index_dir=index_dir)


def pull_registry_index(
    name: str,
    version: str | None = None,
    *,
    registry_url: str | None = None,
    index_dir: Path | None = None,
    client: Any | None = None,
) -> IndexMeta:
    registry = S3Registry(resolve_registry_url(registry_url), client=client)
    return registry.pull(name, version, index_dir=index_dir)


def ensure_local_index(
    *,
    name: str | None = None,
    version: str | None = None,
    repo: str | None = None,
    index_dir: Path | None = None,
    registry_url: str | None = None,
    client: Any | None = None,
    pull_missing: bool = True,
    on_pull: Any | None = None,
) -> IndexMeta:
    """Return a local index, downloading from the registry when missing.

    When ``pull_missing`` is true and a ``name`` is given, a missing local index
    is fetched from the configured S3 registry (latest version if ``version`` is
    omitted). Searching by repo alone does not trigger a download.
    """
    try:
        return find_index(name=name, version=version, repo=repo, index_dir=index_dir)
    except FileNotFoundError as local_exc:
        if not pull_missing or not name:
            raise

        try:
            resolved_url = resolve_registry_url(registry_url)
        except ValueError as exc:
            raise FileNotFoundError(
                f"No local index for {name!r}"
                + (f" version={version!r}" if version else "")
                + f". {exc}"
            ) from local_exc

        registry = S3Registry(resolved_url, client=client)
        entry = registry.find_entry(name, version)
        if entry is None:
            label = f"{name}@{version}" if version else name
            raise FileNotFoundError(
                f"No local index for {label!r} and registry has no matching entry"
            ) from local_exc

        if on_pull is not None:
            on_pull(entry.name, entry.version)

        return registry.pull(entry.name, entry.version, index_dir=index_dir)


def list_registry_entries(
    *,
    registry_url: str | None = None,
    client: Any | None = None,
) -> list[RegistryEntry]:
    registry = S3Registry(resolve_registry_url(registry_url), client=client)
    return registry.list_entries()
