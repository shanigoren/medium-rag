"""Offline unit tests for `src.rag.vectorstore`.

The `cfg` and `fake_pc` fixtures live in `tests/conftest.py`; the in-memory
Pinecone stand-in lives in `tests/_fake_pinecone.py`.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.rag import vectorstore
from src.rag.vectorstore import (
    Match,
    delete_namespace,
    ensure_index,
    namespace_stats,
    query,
    upsert,
)


def _vec(dim: int = 1536) -> list[float]:
    return [0.0] * dim


# ---------- _client caching ----------------------------------------------


def test_client_cached_for_same_api_key(fake_pc, cfg):
    ensure_index(cfg)
    ensure_index(cfg)
    assert len(fake_pc) == 1


def test_client_rebuilt_for_different_api_key(fake_pc, cfg, monkeypatch):
    ensure_index(cfg)
    monkeypatch.setenv("PINECONE_API_KEY", "pc-other")
    from src.config import load_config

    cfg2 = load_config()
    ensure_index(cfg2)
    assert len(fake_pc) == 2
    assert fake_pc[1].api_key == "pc-other"


# ---------- ensure_index --------------------------------------------------


def test_ensure_index_creates_when_missing(fake_pc, cfg):
    ensure_index(cfg)
    assert len(fake_pc[0].created) == 1
    entry = fake_pc[0].created[0]
    assert entry["name"] == cfg.pinecone_index
    assert entry["dimension"] == 1536
    assert entry["metric"] == "cosine"
    assert entry["cloud"] == "aws"
    assert entry["region"] == "us-east-1"


def test_ensure_index_noop_when_present(fake_pc, cfg):
    from tests._fake_pinecone import _FakeIndexInfo

    ensure_index(cfg)  # creates
    fake = fake_pc[0]
    fake.created.clear()
    fake.index_infos[cfg.pinecone_index] = _FakeIndexInfo()
    ensure_index(cfg)
    assert fake.created == []


def test_ensure_index_rejects_wrong_dimension(fake_pc, cfg):
    from tests._fake_pinecone import _FakeIndexInfo

    # Seed by calling once so the cached _FakePinecone exists.
    ensure_index(cfg)
    fake = fake_pc[0]
    fake.index_infos[cfg.pinecone_index] = _FakeIndexInfo(dimension=768)
    with pytest.raises(RuntimeError, match=r"dimension.*1536|1536.*dimension"):
        ensure_index(cfg)


def test_ensure_index_rejects_wrong_metric(fake_pc, cfg):
    from tests._fake_pinecone import _FakeIndexInfo

    ensure_index(cfg)
    fake = fake_pc[0]
    fake.index_infos[cfg.pinecone_index] = _FakeIndexInfo(metric="euclidean")
    with pytest.raises(RuntimeError, match=r"metric.*cosine|cosine.*metric"):
        ensure_index(cfg)


def test_ensure_index_waits_for_ready(fake_pc, cfg, monkeypatch):
    from tests._fake_pinecone import _FakeIndexInfo

    monkeypatch.setattr(vectorstore, "INDEX_POLL_INTERVAL_S", 0)
    monkeypatch.setattr(vectorstore, "INDEX_READY_TIMEOUT_S", 5)

    # Pre-populate the index as not-ready so create_index isn't called.
    # We need a fake first — build one by triggering _client.
    pc = vectorstore._client(cfg.pinecone_api_key)
    pc.index_infos[cfg.pinecone_index] = _FakeIndexInfo(ready=False)

    state = {"calls": 0}

    def flip_after_two(_name: str) -> None:
        state["calls"] += 1
        if state["calls"] >= 3:
            pc.index_infos[cfg.pinecone_index].status.ready = True

    pc.describe_hook = flip_after_two
    ensure_index(cfg)
    assert state["calls"] >= 2


def test_ensure_index_times_out_when_never_ready(fake_pc, cfg, monkeypatch):
    from tests._fake_pinecone import _FakeIndexInfo

    monkeypatch.setattr(vectorstore, "INDEX_READY_TIMEOUT_S", 0.1)
    monkeypatch.setattr(vectorstore, "INDEX_POLL_INTERVAL_S", 0.01)

    pc = vectorstore._client(cfg.pinecone_api_key)
    pc.index_infos[cfg.pinecone_index] = _FakeIndexInfo(ready=False)

    with pytest.raises(TimeoutError, match=r"ready"):
        ensure_index(cfg)


def test_ensure_index_surfaces_quota_error(fake_pc, cfg):
    pc = vectorstore._client(cfg.pinecone_api_key)
    pc.quota_reached = True
    with pytest.raises(RuntimeError, match=r"(?i)quota"):
        ensure_index(cfg)


# ---------- upsert: batching + ordering -----------------------------------


def test_upsert_empty_makes_zero_calls(fake_pc, cfg):
    assert upsert("ns", [], [], [], cfg=cfg) == 0
    # Either no client was constructed at all, or it was but no upsert calls
    # were issued — both satisfy "zero calls."
    assert not fake_pc or fake_pc[0]._index.upserts == []


def test_upsert_returns_total_count(fake_pc, cfg):
    ids = [f"id-{i}" for i in range(7)]
    vecs = [_vec() for _ in range(7)]
    metas = [{"i": i} for i in range(7)]
    assert upsert("ns", ids, vecs, metas, cfg=cfg) == 7


def test_upsert_at_exactly_batch_size(fake_pc, cfg):
    n = 100
    ids = [f"id-{i:03d}" for i in range(n)]
    vecs = [_vec() for _ in range(n)]
    metas = [{"i": i} for i in range(n)]
    upsert("ns", ids, vecs, metas, cfg=cfg)
    assert len(fake_pc[0]._index.upserts) == 1
    assert len(fake_pc[0]._index.upserts[0]["vectors"]) == 100


def test_upsert_at_batch_size_plus_one(fake_pc, cfg):
    n = 101
    ids = [f"id-{i:03d}" for i in range(n)]
    vecs = [_vec() for _ in range(n)]
    metas = [{"i": i} for i in range(n)]
    upsert("ns", ids, vecs, metas, cfg=cfg)
    counts = [len(b["vectors"]) for b in fake_pc[0]._index.upserts]
    assert counts == [100, 1]


def test_upsert_batches_at_100(fake_pc, cfg):
    n = 250
    ids = [f"id-{i:03d}" for i in range(n)]
    vecs = [_vec() for _ in range(n)]
    metas = [{"i": i} for i in range(n)]
    upsert("ns", ids, vecs, metas, cfg=cfg)
    batches = fake_pc[0]._index.upserts
    counts = [len(b["vectors"]) for b in batches]
    assert counts == [100, 100, 50]
    assert all(b["namespace"] == "ns" for b in batches)


def test_upsert_aligns_ids_and_metadatas_across_batches(fake_pc, cfg):
    n = 150
    ids = [f"id-{i:03d}" for i in range(n)]
    vecs = [_vec() for _ in range(n)]
    metas = [{"i": i} for i in range(n)]
    upsert("ns", ids, vecs, metas, cfg=cfg)
    seen = []
    for batch in fake_pc[0]._index.upserts:
        for entry in batch["vectors"]:
            seen.append((entry["id"], entry["metadata"]["i"]))
    assert len(seen) == n
    for vid, idx in seen:
        assert int(vid.split("-")[1]) == idx


def test_upsert_passes_duplicate_ids_through(fake_pc, cfg):
    v = _vec()
    upsert("ns", ["a", "a", "b"], [v, v, v], [{}, {}, {}], cfg=cfg)
    flat = [
        entry["id"] for batch in fake_pc[0]._index.upserts for entry in batch["vectors"]
    ]
    assert flat == ["a", "a", "b"]


# ---------- upsert: validation --------------------------------------------


def test_upsert_validates_lengths(fake_pc, cfg):
    with pytest.raises(ValueError, match=r"length"):
        upsert("ns", ["a", "b", "c"], [_vec(), _vec()], [{}, {}, {}], cfg=cfg)
    assert not fake_pc or fake_pc[0]._index.upserts == []


def test_upsert_validates_vector_dimension(fake_pc, cfg):
    with pytest.raises(ValueError, match=r"dimension.*1536|1536"):
        upsert("ns", ["a"], [[0.0] * 512], [{}], cfg=cfg)
    assert not fake_pc or fake_pc[0]._index.upserts == []


def test_upsert_rejects_empty_namespace(fake_pc, cfg):
    with pytest.raises(ValueError, match=r"namespace"):
        upsert("", ["a"], [_vec()], [{}], cfg=cfg)


def test_upsert_rejects_empty_id(fake_pc, cfg):
    with pytest.raises(ValueError, match=r"id"):
        upsert("ns", ["", "b"], [_vec(), _vec()], [{}, {}], cfg=cfg)


# ---------- query ---------------------------------------------------------


def test_query_returns_match_dataclasses(fake_pc, cfg):
    # Trigger client construction
    ensure_index(cfg)
    fake_pc[0]._index.query_response = {
        "matches": [
            {"id": "a", "score": 0.9, "metadata": {"title": "A"}},
            {"id": "b", "score": 0.7, "metadata": {"title": "B"}},
        ]
    }
    results = query("ns", _vec(), top_k=2, cfg=cfg)
    assert results == [
        Match(id="a", score=0.9, metadata={"title": "A"}),
        Match(id="b", score=0.7, metadata={"title": "B"}),
    ]
    assert all(isinstance(r, Match) for r in results)


def test_query_empty_returns_empty_list(fake_pc, cfg):
    assert query("ns", _vec(), top_k=5, cfg=cfg) == []


def test_query_handles_match_without_metadata(fake_pc, cfg):
    """A vector upserted without metadata yields a match that has no
    'metadata' attribute at all on the real SDK (subscript raises). We must
    degrade to an empty dict, not crash."""
    ensure_index(cfg)
    fake_pc[0]._index.query_response = {
        "matches": [
            {"id": "a", "score": 0.9},  # no 'metadata' key
            {"id": "b", "score": 0.8, "metadata": {"title": "B"}},
        ]
    }
    results = query("ns", _vec(), top_k=2, cfg=cfg)
    assert results == [
        Match(id="a", score=0.9, metadata={}),
        Match(id="b", score=0.8, metadata={"title": "B"}),
    ]


def test_query_handles_response_without_matches_key(fake_pc, cfg):
    """A response that omits 'matches' entirely (no hits) returns []."""
    ensure_index(cfg)
    fake_pc[0]._index.query_response = {}  # no 'matches' key at all
    assert query("ns", _vec(), top_k=5, cfg=cfg) == []


def test_query_handles_null_metadata(fake_pc, cfg):
    """A match whose 'metadata' is explicitly None degrades to {}."""
    ensure_index(cfg)
    fake_pc[0]._index.query_response = {
        "matches": [{"id": "a", "score": 0.5, "metadata": None}]
    }
    assert query("ns", _vec(), top_k=1, cfg=cfg) == [
        Match(id="a", score=0.5, metadata={})
    ]


def test_query_top_k_one(fake_pc, cfg):
    ensure_index(cfg)
    fake_pc[0]._index.query_response = {
        "matches": [{"id": "a", "score": 0.9, "metadata": {}}]
    }
    results = query("ns", _vec(), top_k=1, cfg=cfg)
    assert fake_pc[0]._index.queries[0]["top_k"] == 1
    assert len(results) == 1


def test_query_top_k_larger_than_population(fake_pc, cfg):
    ensure_index(cfg)
    fake_pc[0]._index.query_response = {
        "matches": [
            {"id": "a", "score": 0.9, "metadata": {}},
            {"id": "b", "score": 0.8, "metadata": {}},
            {"id": "c", "score": 0.7, "metadata": {}},
        ]
    }
    results = query("ns", _vec(), top_k=100, cfg=cfg)
    assert fake_pc[0]._index.queries[0]["top_k"] == 100
    assert len(results) == 3


def test_query_passes_include_metadata_true_and_values_false(fake_pc, cfg):
    query("ns", _vec(), top_k=5, cfg=cfg)
    recorded = fake_pc[0]._index.queries[0]
    assert recorded["include_metadata"] is True
    assert recorded["include_values"] is False


def test_query_passes_namespace_and_top_k(fake_pc, cfg):
    query("foo", _vec(), top_k=7, cfg=cfg)
    recorded = fake_pc[0]._index.queries[0]
    assert recorded["namespace"] == "foo"
    assert recorded["top_k"] == 7


def test_query_rejects_empty_namespace(fake_pc, cfg):
    with pytest.raises(ValueError, match=r"namespace"):
        query("", _vec(), top_k=5, cfg=cfg)


def test_query_rejects_zero_top_k(fake_pc, cfg):
    with pytest.raises(ValueError, match=r"top_k"):
        query("ns", _vec(), top_k=0, cfg=cfg)


# ---------- delete_namespace + namespace_stats ----------------------------


def test_delete_namespace_calls_delete_all(fake_pc, cfg):
    delete_namespace("ns", cfg=cfg)
    assert fake_pc[0]._index.deletes == [{"delete_all": True, "namespace": "ns"}]


def test_delete_namespace_rejects_empty(fake_pc, cfg):
    with pytest.raises(ValueError, match=r"namespace"):
        delete_namespace("", cfg=cfg)


def test_namespace_stats_missing_returns_zero(fake_pc, cfg):
    assert namespace_stats("absent", cfg=cfg) == {"vector_count": 0}


def test_namespace_stats_existing_returns_count(fake_pc, cfg):
    ensure_index(cfg)
    fake_pc[0]._index.stats_response = {
        "namespaces": {"foo": {"vector_count": 42}},
        "dimension": 1536,
    }
    assert namespace_stats("foo", cfg=cfg) == {"vector_count": 42}


def test_namespace_stats_rejects_empty(fake_pc, cfg):
    with pytest.raises(ValueError, match=r"namespace"):
        namespace_stats("", cfg=cfg)


# ---------- Match dataclass -----------------------------------------------


def test_match_is_frozen():
    m = Match(id="a", score=0.5, metadata={"k": "v"})
    with pytest.raises(dataclasses.FrozenInstanceError):
        m.id = "b"  # type: ignore[misc]
