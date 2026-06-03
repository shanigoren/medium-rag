"""Live API smoke tests for Component 8 (the retriever).

Run with `pytest --smoke -v -m smoke -k retriever`. Skipped by default. Hits real
LLMod.AI embeddings + the live `smoke` namespace produced by C7/CP-A. Cost: a
couple of cents (one embed_query per test).

READ-ONLY: these tests CONSUME the shared `smoke` namespace and must NOT delete
it (CP-B/C/D still need it). Each guards with a stats check and SKIPs (never
fails) if `smoke` is empty -- a missing fixture is an environment gap, not a code
bug; the remedy is to ingest first.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.config import load_config
from src.llm.clients import get_embeddings
from src.rag.retriever import retrieve
from src.rag.vectorstore import namespace_stats, query

SMOKE_NS = "smoke"
_INGEST_HINT = (
    "namespace 'smoke' is empty; run "
    "`python scripts/ingest.py --limit 10 --namespace smoke` first"
)


def _require_smoke(cfg):
    if namespace_stats(SMOKE_NS, cfg)["vector_count"] == 0:
        pytest.skip(_INGEST_HINT)


# --------------------------------------------------------------------------
# Zero-extra-cost live coverage.
#
# A query embedding is the ONLY paid step on the read path (Pinecone reads are
# free on the serverless free tier). So `smoke_ctx` embeds ONE query a single
# time (module scope) and caches the real vector. The tests below REPLAY that
# cached vector into retrieve() by patching `src.rag.retriever.get_embeddings`,
# so retrieve()'s real dedup/ranking/metadata logic runs against the live
# namespace with NO further embedding or LLM calls -- only free Pinecone reads.
# This lets us assert strong, airtight properties (computed from the same real
# matches via an independent oracle) instead of the loose checks live
# non-determinism would otherwise force.
# --------------------------------------------------------------------------


class _ReplayEmbedder:
    """Stand-in whose embed_query returns a pre-computed real vector. No I/O."""

    def __init__(self, vec: list[float]) -> None:
        self._vec = vec

    def embed_query(self, text: str) -> list[float]:
        return self._vec


@pytest.fixture(scope="module")
def smoke_ctx():
    """Embed ONE real query against the live API (the only cost), pick a usable
    anchor from the live namespace, and cache the vector for replay. Skips the
    whole live-replay suite if `smoke` is empty."""
    cfg = load_config()
    if namespace_stats(SMOKE_NS, cfg)["vector_count"] == 0:
        pytest.skip(_INGEST_HINT)
    qvec = get_embeddings(cfg).embed_query("ideas and lessons worth sharing")  # the one paid call
    raw = query(SMOKE_NS, qvec, top_k=max(cfg.retrieval_fetch_k, cfg.top_k), cfg=cfg)
    usable = [m for m in raw if m.metadata.get("article_id") and ("chunk" in m.metadata)]
    if not usable:
        pytest.skip("no well-formed vectors in 'smoke'; re-ingest")
    return SimpleNamespace(cfg=cfg, qvec=qvec)


def _replay(monkeypatch, qvec):
    monkeypatch.setattr(
        "src.rag.retriever.get_embeddings", lambda cfg=None: _ReplayEmbedder(qvec)
    )


def _first_seen_distinct(matches, k):
    """Independent oracle: collapse `matches` (score-desc) to the first-seen
    chunk per article_id, skipping unusable, up to k. Mirrors retrieve(dedup=True)
    so we can cross-check it against REAL Pinecone candidates."""
    seen: set[str] = set()
    out: list[str] = []
    for m in matches:
        md = m.metadata
        if "article_id" not in md or "chunk" not in md:
            continue
        aid = str(md["article_id"])
        if aid in seen:
            continue
        seen.add(aid)
        out.append(aid)
        if len(out) == k:
            break
    return out


@pytest.mark.smoke
def test_smoke_retrieve_finds_expected_article():
    """Recall@5 on a known item: pick one ingested article, build a query from
    its title words, and assert its article_id is among the returned ids.
    Validates the embed-asymmetry seam -- a RAW query still hits a title-prefixed
    document."""
    cfg = load_config()
    _require_smoke(cfg)

    # Grab a real, well-formed match from the namespace to use as the known item.
    probe_vec = get_embeddings(cfg).embed_query("the main idea of an article")
    probe = query(SMOKE_NS, probe_vec, top_k=5, cfg=cfg)
    probe = [m for m in probe if m.metadata.get("article_id") and m.metadata.get("title")]
    assert probe, "could not pull any well-formed match from the 'smoke' namespace"

    target = probe[0]
    expected_id = str(target.metadata["article_id"])
    title_words = " ".join(str(target.metadata["title"]).split()[:8])

    out = retrieve(title_words, SMOKE_NS, cfg, top_k=5)
    assert expected_id in {r.article_id for r in out}, (
        f"expected article {expected_id!r} not in top-5 for query {title_words!r}"
    )


@pytest.mark.smoke
def test_smoke_list_query_returns_distinct_articles():
    """A broad topical query with top_k=3, dedup=True returns 3 records whose
    article_ids are all DISTINCT. Early, isolated check of the type-2 dedupe."""
    cfg = load_config()
    _require_smoke(cfg)

    out = retrieve("lessons and ideas worth sharing", SMOKE_NS, cfg, top_k=3, dedup=True)
    ids = [r.article_id for r in out]
    assert len(ids) == len(set(ids)), f"dedup=True returned repeats: {ids}"


@pytest.mark.smoke
def test_smoke_no_dedup_can_return_repeat_article():
    """dedup=False with a query that strongly matches one article (its own title
    words) and top_k=5: assert the no-dedup path does NOT silently dedupe --
    either an article_id repeats, OR all 5 chunks are returned without
    collapsing. Confirms the flag changes live behaviour."""
    cfg = load_config()
    _require_smoke(cfg)

    # Anchor on a real article's title words so retrieval concentrates on it.
    probe_vec = get_embeddings(cfg).embed_query("an article")
    probe = query(SMOKE_NS, probe_vec, top_k=1, cfg=cfg)
    probe = [m for m in probe if m.metadata.get("title")]
    assert probe, "could not pull a titled match from the 'smoke' namespace"
    title_words = " ".join(str(probe[0].metadata["title"]).split()[:8])

    out = retrieve(title_words, SMOKE_NS, cfg, top_k=5, dedup=False)
    ids = [r.article_id for r in out]
    has_repeat = len(ids) != len(set(ids))
    returned_full_topk = len(ids) == 5
    assert has_repeat or returned_full_topk, (
        f"dedup=False unexpectedly collapsed to {len(ids)} distinct ids: {ids}"
    )


# --- Replayed-vector live tests (no extra embedding/LLM calls; free Pinecone reads) ---


@pytest.mark.smoke
def test_smoke_dedup_true_matches_hand_computed_first_seen(smoke_ctx, monkeypatch):
    """Airtight dedup logic on REAL data: retrieve(dedup=True, top_k=K) returns
    exactly the first-K-distinct article_ids (first-seen == best) of the real
    over-fetched candidates, as computed by an independent oracle over the same
    Pinecone results."""
    cfg = smoke_ctx.cfg
    fk = max(cfg.retrieval_fetch_k, 3)
    raw = query(SMOKE_NS, smoke_ctx.qvec, top_k=fk, cfg=cfg)
    expected = _first_seen_distinct(raw, 3)

    _replay(monkeypatch, smoke_ctx.qvec)
    out = retrieve("replayed", SMOKE_NS, cfg, top_k=3, dedup=True)
    assert [r.article_id for r in out] == expected


@pytest.mark.smoke
def test_smoke_dedup_false_matches_raw_top_k(smoke_ctx, monkeypatch):
    """Airtight no-dedup on REAL data: retrieve(dedup=False, top_k=K) returns
    exactly the real top-K usable chunks, in order, with no collapsing."""
    cfg = smoke_ctx.cfg
    raw = query(SMOKE_NS, smoke_ctx.qvec, top_k=5, cfg=cfg)
    expected = [
        str(m.metadata["article_id"])
        for m in raw
        if m.metadata.get("article_id") and ("chunk" in m.metadata)
    ]

    _replay(monkeypatch, smoke_ctx.qvec)
    out = retrieve("replayed", SMOKE_NS, cfg, top_k=5, dedup=False)
    assert [r.article_id for r in out] == expected


@pytest.mark.smoke
def test_smoke_metadata_schema_roundtrips(smoke_ctx, monkeypatch):
    """C7 -> C8 metadata contract on REAL stored data: every returned chunk has a
    non-empty string article_id, non-empty string chunk, int chunk_idx >= 0,
    list authors, and string title."""
    _replay(monkeypatch, smoke_ctx.qvec)
    out = retrieve("replayed", SMOKE_NS, smoke_ctx.cfg, top_k=5, dedup=False)
    assert out, "expected at least one result from a populated namespace"
    for r in out:
        assert isinstance(r.article_id, str) and r.article_id
        assert isinstance(r.chunk, str) and r.chunk
        assert isinstance(r.chunk_idx, int) and r.chunk_idx >= 0
        assert isinstance(r.authors, list)
        assert isinstance(r.tags, list)
        assert isinstance(r.title, str)


@pytest.mark.smoke
def test_smoke_chunk_is_clean_prose_not_prefixed(smoke_ctx, monkeypatch):
    """The raw-chunk contract survives the real write->read round-trip: no
    returned chunk carries the embed-time 'Title:'/'Tags:' prefix (C7 stores the
    raw chunk; only the embedded string was prefixed)."""
    _replay(monkeypatch, smoke_ctx.qvec)
    out = retrieve("replayed", SMOKE_NS, smoke_ctx.cfg, top_k=5, dedup=False)
    for r in out:
        assert not r.chunk.lstrip().startswith("Title:"), r.chunk[:60]


@pytest.mark.smoke
def test_smoke_article_ids_are_numeric_strings_in_range(smoke_ctx, monkeypatch):
    """smoke == first 10 CSV rows, so every article_id is a stringified int in
    [0, 10). Reinforces the string invariant AND proves we read the expected
    slice back."""
    _replay(monkeypatch, smoke_ctx.qvec)
    out = retrieve("replayed", SMOKE_NS, smoke_ctx.cfg, top_k=5, dedup=True)
    for r in out:
        assert r.article_id.isdigit() and 0 <= int(r.article_id) < 10, r.article_id


@pytest.mark.smoke
def test_smoke_results_ranked_capped_and_contiguous(smoke_ctx, monkeypatch):
    """On REAL data: scores are non-increasing, ranks are exactly [1..n]
    contiguous, len <= top_k, and (dedup=True) article_ids are all distinct."""
    _replay(monkeypatch, smoke_ctx.qvec)
    out = retrieve("replayed", SMOKE_NS, smoke_ctx.cfg, top_k=5, dedup=True)
    assert len(out) <= 5
    assert [r.rank for r in out] == list(range(1, len(out) + 1))
    scores = [r.score for r in out]
    assert scores == sorted(scores, reverse=True)
    ids = [r.article_id for r in out]
    assert len(ids) == len(set(ids))


@pytest.mark.smoke
def test_smoke_both_modes_agree_on_rank1_article(smoke_ctx, monkeypatch):
    """For the SAME query vector, dedup=True and dedup=False return the same
    rank-1 article (both see the single top chunk; dedup only changes what
    follows). A cheap cross-mode consistency check on real data."""
    _replay(monkeypatch, smoke_ctx.qvec)
    cfg = smoke_ctx.cfg
    top_true = retrieve("replayed", SMOKE_NS, cfg, top_k=5, dedup=True)
    top_false = retrieve("replayed", SMOKE_NS, cfg, top_k=5, dedup=False)
    assert top_true and top_false
    assert top_true[0].article_id == top_false[0].article_id
