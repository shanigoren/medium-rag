"""Live Pinecone API smoke tests.

Run with `pytest --smoke -v -m smoke -k vectorstore`. Skipped by default.
All tests scope writes to the `_smoke_vectorstore` namespace and clean up
in fixture teardown, even on failure. They must NOT touch `prod` or `exp_*`.
"""

from __future__ import annotations

import time
from typing import Callable

import pytest

from src.rag import vectorstore
from src.rag.vectorstore import (
    WRITE_CONSISTENCY_POLL_S,
    WRITE_CONSISTENCY_TIMEOUT_S,
    delete_namespace,
    ensure_index,
    namespace_stats,
    query,
    upsert,
)


NS = "_smoke_vectorstore"
EMBED_DIM = 1536


def _one_hot(i: int, dim: int = EMBED_DIM) -> list[float]:
    v = [0.0] * dim
    v[i] = 1.0
    return v


def _poll_until(
    predicate: Callable[[], bool],
    timeout_s: float = WRITE_CONSISTENCY_TIMEOUT_S,
    interval_s: float = WRITE_CONSISTENCY_POLL_S,
) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return False


@pytest.fixture
def smoke_namespace():
    ensure_index()
    delete_namespace(NS)  # leftover from a prior failed run
    try:
        yield NS
    finally:
        delete_namespace(NS)


@pytest.mark.smoke
def test_ensure_index_idempotent():
    ensure_index()
    t0 = time.monotonic()
    ensure_index()
    # Second call: index already exists, no creation. Should be fast.
    assert (time.monotonic() - t0) < vectorstore.INDEX_READY_TIMEOUT_S


@pytest.mark.smoke
def test_upsert_query_delete_roundtrip(smoke_namespace):
    ns = smoke_namespace
    n = 5
    ids = [f"vec-{i}" for i in range(n)]
    vectors = [_one_hot(i) for i in range(n)]
    metadatas = [{"label": f"item-{i}"} for i in range(n)]

    written = upsert(ns, ids, vectors, metadatas)
    assert written == n

    became_visible = _poll_until(
        lambda: namespace_stats(ns)["vector_count"] == n
    )
    assert became_visible, (
        f"Upserted {n} vectors but namespace_stats never reported {n}; "
        "either the wrapper is broken or Pinecone is unusually slow."
    )

    results = query(ns, _one_hot(2), top_k=3)
    assert len(results) >= 1
    assert results[0].id == "vec-2"
    assert results[0].metadata.get("label") == "item-2"


@pytest.mark.smoke
def test_delete_namespace_eventually_consistent(smoke_namespace):
    ns = smoke_namespace
    n = 3
    ids = [f"vec-{i}" for i in range(n)]
    vectors = [_one_hot(i) for i in range(n)]
    metadatas = [{"i": i} for i in range(n)]

    upsert(ns, ids, vectors, metadatas)
    assert _poll_until(lambda: namespace_stats(ns)["vector_count"] == n)

    delete_namespace(ns)
    assert _poll_until(lambda: namespace_stats(ns)["vector_count"] == 0), (
        "namespace_stats never reported 0 after delete_namespace; "
        "delete propagation took longer than expected."
    )


@pytest.mark.smoke
def test_query_missing_namespace_returns_empty():
    ensure_index()
    assert query("_does_not_exist_xyz", _one_hot(0), top_k=5) == []


@pytest.mark.smoke
def test_upsert_batch_boundary_real(smoke_namespace):
    ns = smoke_namespace
    n = 150  # forces 2 batches: 100 + 50
    ids = [f"vec-{i}" for i in range(n)]
    vectors = [_one_hot(i) for i in range(n)]
    metadatas = [{"i": i} for i in range(n)]

    upsert(ns, ids, vectors, metadatas)
    became_visible = _poll_until(
        lambda: namespace_stats(ns)["vector_count"] == n,
        timeout_s=WRITE_CONSISTENCY_TIMEOUT_S * 2,  # 2 batches → more headroom
    )
    assert became_visible, (
        f"Upserted {n} vectors across 2 batches but stats never reported {n}."
    )

    # Verify a vector from the second batch is queryable.
    results = query(ns, _one_hot(73), top_k=1)
    assert len(results) == 1
    assert results[0].id == "vec-73"
