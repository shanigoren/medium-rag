"""Live API smoke tests for Component 7 (ingestion pipeline).

Run with `pytest --smoke -v -m smoke -k ingest`. Skipped by default. Hits real
LLMod.AI embeddings + real Pinecone (cost: a few cents). Each test scopes writes
to a throwaway namespace and deletes it in teardown, even on failure -- never
touches `prod` or `exp_*`.
"""

from __future__ import annotations

import time
from typing import Callable

import pytest

from scripts.ingest import _load_with_overrides, run_ingest
from src.config import load_config
from src.llm.clients import get_embeddings
from src.rag.embed import EMBED_BATCH_SIZE
from src.rag.vectorstore import (
    WRITE_CONSISTENCY_POLL_S,
    WRITE_CONSISTENCY_TIMEOUT_S,
    delete_namespace,
    namespace_stats,
    query,
)

NS_ROUNDTRIP = "_smoke_ingest"
NS_1024 = "_smoke_ingest_1024"
NS_IDEMPOTENT = "_smoke_ingest_idempotent"


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
def throwaway_roundtrip():
    try:
        yield NS_ROUNDTRIP
    finally:
        delete_namespace(NS_ROUNDTRIP)


@pytest.fixture
def throwaway_1024():
    try:
        yield NS_1024
    finally:
        delete_namespace(NS_1024)


@pytest.fixture
def throwaway_idempotent():
    try:
        yield NS_IDEMPOTENT
    finally:
        delete_namespace(NS_IDEMPOTENT)


@pytest.mark.smoke
def test_smoke_ingest_10_articles_roundtrips(throwaway_roundtrip):
    ns = throwaway_roundtrip
    cfg = load_config()
    stats = run_ingest(ns, cfg, limit=10, clean=True)
    assert 20 <= stats.vectors_upserted <= 120, stats

    visible = _poll_until(
        lambda: namespace_stats(ns)["vector_count"] == stats.vectors_upserted
    )
    assert visible, (
        f"upserted {stats.vectors_upserted} but stats never reported it"
    )

    qv = get_embeddings(cfg).embed_query("the main idea of the first article")
    matches = query(ns, qv, top_k=1, cfg=cfg)
    assert matches
    m = matches[0].metadata

    # The full schema is present, with a non-empty RAW chunk (no embed prefix).
    for key in ("article_id", "title", "url", "timestamp", "chunk", "chunk_idx"):
        assert key in m, f"missing metadata key {key!r}"
    assert isinstance(m["article_id"], str)
    assert m["chunk"]
    assert not m["chunk"].startswith("Title:")

    # omit-empty held on real data: no metadata value is [] or None.
    for k, v in m.items():
        assert v is not None, f"{k} is None"
        assert v != [], f"{k} is an empty list"


@pytest.mark.smoke
def test_smoke_chunk_size_1024_fills_a_full_embed_batch_without_rejection(throwaway_1024):
    """C6 token-ceiling handoff. The ceiling only bites when ONE
    embed_documents call carries a FULL EMBED_BATCH_SIZE-text batch at ~1090
    tokens each. A small slice would pass vacuously, so ingest enough articles
    to produce >= EMBED_BATCH_SIZE chunks at chunk_size=1024, then assert the
    ingest completed without the proxy rejecting an over-large request."""
    ns = throwaway_1024
    cfg = _load_with_overrides({"chunk_size": "1024"})
    assert cfg.chunk_size == 1024

    stats = run_ingest(ns, cfg, limit=300, clean=True)
    if stats.chunks_total < EMBED_BATCH_SIZE:
        pytest.skip(
            f"inconclusive: only {stats.chunks_total} chunks (< {EMBED_BATCH_SIZE}); "
            "raise --limit to fill a full embed batch"
        )

    # Reaching here means at least one internal embed_documents call was a full
    # EMBED_BATCH_SIZE-text batch and the proxy accepted it (no exception above).
    assert stats.vectors_upserted == stats.chunks_total
    visible = _poll_until(
        lambda: namespace_stats(ns)["vector_count"] == stats.vectors_upserted,
        timeout_s=WRITE_CONSISTENCY_TIMEOUT_S * 3,
    )
    assert visible, "1024-chunk ingest never became fully visible"


@pytest.mark.smoke
def test_smoke_ingest_is_idempotent_on_rerun(throwaway_idempotent):
    """Checkpoint A gate: re-ingesting the SAME config into the SAME namespace
    (no --clean) is a clean deterministic-ID overwrite, NOT a duplicate. The
    vector count after the second run must equal the count after the first --
    deterministic IDs f"{row_idx}-{chunk_idx}" map every chunk to the same slot
    (last-write-wins), so a re-run overwrites in place rather than doubling.
    """
    ns = throwaway_idempotent
    cfg = load_config()

    # First ingest into a guaranteed-empty namespace.
    first = run_ingest(ns, cfg, limit=10, clean=True)
    assert first.vectors_upserted > 0, first
    assert _poll_until(
        lambda: namespace_stats(ns)["vector_count"] == first.vectors_upserted
    ), "first ingest never became fully visible"
    count_after_first = namespace_stats(ns)["vector_count"]

    # Re-ingest the SAME 10 articles at the SAME config, WITHOUT --clean.
    second = run_ingest(ns, cfg, limit=10, clean=False)
    assert second.vectors_upserted == first.vectors_upserted, (first, second)

    # Pinecone serverless is eventually consistent: describe_index_stats can
    # TRANSIENTLY report an inflated count right after an in-place overwrite
    # (replica/data-freshness lag counts the old and new copy briefly) before it
    # converges. Upserting the SAME deterministic ID updates rather than
    # duplicates, so idempotency means the count CONVERGES BACK to the original
    # -- not that it never momentarily rises. Poll for convergence over a
    # generous window; a genuine non-deterministic-ID bug would instead stay
    # doubled and never converge, so this still fails on a real regression.
    converged = _poll_until(
        lambda: namespace_stats(ns)["vector_count"] == count_after_first,
        timeout_s=120.0,
        interval_s=3.0,
    )
    final = namespace_stats(ns)["vector_count"]
    assert converged, (
        f"re-ingest left the vector count at {final} and it never converged "
        f"back to {count_after_first} within 120s; deterministic-ID overwrite "
        "must not permanently duplicate vectors"
    )
