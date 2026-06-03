"""Pinecone serverless wrapper for the Medium RAG pipeline.

Thin facade over the raw `pinecone` SDK so that ingest, retrieval, and the
experiment runner share one client construction, one index lifecycle, and one
namespace-scoped surface (`upsert`, `query`, `delete_namespace`,
`namespace_stats`). No embedding, no chunking.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from src.config import Config, load_config


UPSERT_BATCH_SIZE = 100
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"
INDEX_READY_TIMEOUT_S = 60
INDEX_POLL_INTERVAL_S = 1
WRITE_CONSISTENCY_TIMEOUT_S = 10
WRITE_CONSISTENCY_POLL_S = 0.5


@dataclass(frozen=True)
class Match:
    """One retrieval result.

    `frozen=True` freezes the field bindings, not the contents of `metadata`.
    Downstream code can still mutate the dict — treat it as read-only by
    convention.
    """

    id: str
    score: float
    metadata: dict


@lru_cache(maxsize=1)
def _client(api_key: str) -> Pinecone:
    """Cached Pinecone client. Keyed on api_key so tests that swap keys via
    monkeypatch get a fresh client without manual cache invalidation."""
    return Pinecone(api_key=api_key)


def _index(cfg: Config):
    """Return a handle to cfg.pinecone_index. Assumes ensure_index() has run."""
    return _client(cfg.pinecone_api_key).Index(cfg.pinecone_index)


def ensure_index(cfg: Config | None = None) -> None:
    """Create the serverless cosine index named cfg.pinecone_index with
    dimension cfg.embed_dim if it does not exist; wait until status.ready.

    No-op if already present AND already the right dimension and cosine
    metric. If the index exists but has the wrong dimension or metric, raise
    RuntimeError — never auto-delete.
    """
    if cfg is None:
        cfg = load_config()
    pc = _client(cfg.pinecone_api_key)
    name = cfg.pinecone_index

    existing = pc.list_indexes().names()
    if name in existing:
        info = pc.describe_index(name)
        if info.dimension != cfg.embed_dim:
            raise RuntimeError(
                f"Pinecone index {name!r} exists with dimension "
                f"{info.dimension}, expected {cfg.embed_dim} (1536). Refusing "
                f"to auto-delete. Use `pc.delete_index({name!r})` from a one-off "
                f"script if you really mean to recreate it."
            )
        if info.metric != "cosine":
            raise RuntimeError(
                f"Pinecone index {name!r} exists with metric {info.metric!r}, "
                f"expected 'cosine'. Refusing to auto-delete."
            )
        _wait_until_ready(pc, name)
        return

    try:
        pc.create_index(
            name=name,
            dimension=cfg.embed_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
    except Exception as exc:
        msg = str(exc).lower()
        if "quota" in msg or "limit" in msg:
            raise RuntimeError(
                "Pinecone free-tier index quota reached. Delete an unused index "
                "via the dashboard or with `pc.delete_index(name)` from a one-off "
                "script, then retry. We deliberately do not auto-delete here."
            ) from exc
        raise

    _wait_until_ready(pc, name)


def _wait_until_ready(pc: Pinecone, name: str) -> None:
    deadline = time.monotonic() + INDEX_READY_TIMEOUT_S
    polls = 0
    while True:
        info = pc.describe_index(name)
        polls += 1
        if info.status.ready:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Pinecone index {name!r} did not become ready within "
                f"{INDEX_READY_TIMEOUT_S}s ({polls} polls)."
            )
        time.sleep(INDEX_POLL_INTERVAL_S)


def upsert(
    namespace: str,
    ids: list[str],
    vectors: list[list[float]],
    metadatas: list[dict],
    cfg: Config | None = None,
) -> int:
    """Upsert `ids[i] -> (vectors[i], metadatas[i])` into `namespace` in
    batches of <= UPSERT_BATCH_SIZE. Returns total vectors written.

    Duplicate IDs within a single call are forwarded to Pinecone unchanged
    (last-write-wins). Metadata is passed through without validation — the
    caller (Component 7) owns the schema.

    Use deterministic IDs (e.g. f"{article_id}-{chunk_idx}") so re-ingest
    overwrites cleanly.
    """
    if cfg is None:
        cfg = load_config()
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace must be a non-empty string")
    if not (len(ids) == len(vectors) == len(metadatas)):
        raise ValueError(
            f"ids, vectors, metadatas must have equal length "
            f"(got {len(ids)}, {len(vectors)}, {len(metadatas)})"
        )
    for i, vid in enumerate(ids):
        if not isinstance(vid, str) or not vid:
            raise ValueError(f"id at position {i} must be a non-empty string")
    for i, vec in enumerate(vectors):
        if len(vec) != cfg.embed_dim:
            raise ValueError(
                f"vector at position {i} has dimension {len(vec)}, "
                f"expected {cfg.embed_dim}"
            )

    if not ids:
        return 0

    index = _index(cfg)
    written = 0
    for start in range(0, len(ids), UPSERT_BATCH_SIZE):
        end = start + UPSERT_BATCH_SIZE
        batch = [
            {"id": vid, "values": vec, "metadata": meta}
            for vid, vec, meta in zip(
                ids[start:end], vectors[start:end], metadatas[start:end]
            )
        ]
        index.upsert(vectors=batch, namespace=namespace)
        written += len(batch)
    return written


def query(
    namespace: str,
    vector: list[float],
    top_k: int,
    cfg: Config | None = None,
) -> list[Match]:
    """Query `namespace` for the top_k nearest neighbours of `vector`.

    Always requests metadata, never the stored vector body. Returns Match
    list ordered by descending score. Missing or empty namespace returns [].
    If top_k exceeds the namespace's vector count, returns whatever's there.
    """
    if cfg is None:
        cfg = load_config()
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace must be a non-empty string")
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1 (got {top_k})")

    index = _index(cfg)
    response = index.query(
        vector=vector,
        top_k=top_k,
        namespace=namespace,
        include_metadata=True,
        include_values=False,
    )
    # The real Pinecone SDK models (QueryResponse / ScoredVector) raise
    # PineconeApiAttributeError on subscript for an attribute that was never
    # set — unlike a plain dict that returns None. A vector upserted without
    # metadata therefore has no "metadata" attribute at all, and a response
    # with no hits may omit "matches" entirely. Guard both with `in` (which
    # the SDK models support) so we degrade to empty rather than crashing.
    matches = (response["matches"] if "matches" in response else None) or []
    return [
        Match(
            id=m["id"],
            score=float(m["score"]),
            metadata=dict(m["metadata"]) if ("metadata" in m and m["metadata"]) else {},
        )
        for m in matches
    ]


def delete_namespace(namespace: str, cfg: Config | None = None) -> None:
    """Delete every vector in `namespace`. Safe on non-existent namespaces."""
    if cfg is None:
        cfg = load_config()
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace must be a non-empty string")
    index = _index(cfg)
    try:
        index.delete(delete_all=True, namespace=namespace)
    except Exception as exc:
        msg = str(exc).lower()
        if "not found" in msg or "404" in msg:
            return
        raise


def namespace_stats(namespace: str, cfg: Config | None = None) -> dict[str, Any]:
    """Return {'vector_count': int} for `namespace`. Returns
    {'vector_count': 0} if the namespace doesn't exist."""
    if cfg is None:
        cfg = load_config()
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("namespace must be a non-empty string")
    index = _index(cfg)
    stats = index.describe_index_stats()
    namespaces = stats["namespaces"] or {}
    ns = namespaces.get(namespace)
    if ns is None:
        return {"vector_count": 0}
    return {"vector_count": int(ns["vector_count"])}
