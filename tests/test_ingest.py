"""Offline unit tests for Component 7 (ingestion pipeline).

Zero network. Strategy: run `chunk_text` for real (pure CPU, deterministic), but
stub the two network seams -- `embed_batch` (a stateful, order-revealing stub)
and the C5 Pinecone functions (the existing `fake_pc` fixture). `run_ingest` /
`main` tests source articles from `tests/fixtures/tiny_articles.csv`, never the
50 MB corpus.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from scripts.ingest import (
    FLUSH_CHUNKS,
    IngestStats,
    _records_for_article,
    build_records,
    ingest_articles,
    main,
    run_ingest,
)
from src.data.csv_loader import Article
from src.rag.chunking import chunk_text
from src.rag.embed import build_embed_text

TINY_CSV = str(Path(__file__).resolve().parent / "fixtures" / "tiny_articles.csv")

# Long enough to produce multiple chunks at the final default chunk_size=768.
LONG_BODY = "The quick brown fox jumps over the lazy dog. " * 120


def make_article(
    row_idx: int,
    text: str = LONG_BODY,
    *,
    title: str = "Some Title",
    url: str = "http://example.com",
    authors=("Author One",),
    timestamp: str = "2021-01-01",
    tags=("alpha",),
) -> Article:
    return Article(
        row_idx=row_idx,
        title=title,
        text=text,
        url=url,
        authors=list(authors),
        timestamp=timestamp,
        tags=list(tags),
    )


# --------------------------------------------------------------------------- #
# Stateful, order-revealing embed stub (vector[0] == GLOBAL record index).     #
# --------------------------------------------------------------------------- #
class _IngestStubEmbed:
    """embed_batch(texts, cfg) -> one 1536-d vector per text whose [0] element is
    the GLOBAL input index across all calls. STATEFUL: `next_idx` advances by
    len(texts) per call so alignment is provable across flush boundaries."""

    def __init__(self) -> None:
        self.next_idx = 0
        self.calls: list[list[str]] = []

    def __call__(self, texts, cfg=None):
        self.calls.append(list(texts))
        out: list[list[float]] = []
        for _ in texts:
            v = [0.0] * 1536
            v[0] = float(self.next_idx)
            self.next_idx += 1
            out.append(v)
        return out


@pytest.fixture
def fake_embed(monkeypatch):
    stub = _IngestStubEmbed()
    monkeypatch.setattr("scripts.ingest.embed_batch", stub)
    return stub


@pytest.fixture
def env(monkeypatch):
    """Minimum env so main()'s own load_config() succeeds."""
    monkeypatch.setenv("LLMOD_API_KEY", "sk-test")
    monkeypatch.setenv("LLMOD_BASE_URL", "https://api.llmod.ai/v1")
    monkeypatch.setenv("PINECONE_API_KEY", "pc-test")


def _flatten_upserts(fake) -> list[dict]:
    """All upserted vector dicts ({'id','values','metadata'}) in recorded order."""
    out: list[dict] = []
    for call in fake._index.upserts:
        out.extend(call["vectors"])
    return out


# --------------------------------------------------------------------------- #
# build_records -- schema, IDs, alignment (pure, no stubs)                     #
# --------------------------------------------------------------------------- #
def test_build_records_ids_are_articleid_dash_chunkidx(cfg):
    art = make_article(42)
    ids, _texts, _metas = build_records([art], cfg)
    n = len(chunk_text(art.text, cfg.chunk_size, cfg.overlap_ratio))
    assert n >= 1
    assert ids == [f"42-{i}" for i in range(n)]


def test_build_records_metadata_has_exact_schema_keys(cfg):
    art = make_article(1, authors=("Alice",), tags=("food",))
    _ids, _texts, metas = build_records([art], cfg)
    assert metas
    for m in metas:
        assert set(m.keys()) == {
            "article_id", "title", "authors", "url", "timestamp",
            "tags", "chunk", "chunk_idx",
        }
        assert m["article_id"] == "1"


def test_build_records_stores_raw_chunk_not_embed_string(cfg):
    cfg = replace(cfg, embed_content="title_chunk")
    art = make_article(3, title="My Title")
    _ids, texts, metas = build_records([art], cfg)
    assert texts[0].startswith("Title: My Title")
    assert not metas[0]["chunk"].startswith("Title:")
    assert metas[0]["chunk"] in texts[0]


def test_build_records_embed_text_matches_build_embed_text(cfg):
    art = make_article(5)
    chunks = chunk_text(art.text, cfg.chunk_size, cfg.overlap_ratio)
    expected = [build_embed_text(art, c, cfg.embed_content) for c in chunks]
    _ids, texts, _metas = build_records([art], cfg)
    assert texts == expected


def test_build_records_three_lists_index_aligned_and_equal_length(cfg):
    art = make_article(9)
    ids, texts, metas = build_records([art], cfg)
    assert len(ids) == len(texts) == len(metas)
    for i, (vid, m) in enumerate(zip(ids, metas)):
        assert vid == f"{m['article_id']}-{m['chunk_idx']}"
        assert m["chunk_idx"] == i


def test_build_records_skips_zero_chunk_articles(cfg):
    empty = make_article(0, text="   ")
    populated = make_article(1)
    ids, _texts, _metas = build_records([empty, populated], cfg)
    assert ids
    assert all(vid.startswith("1-") for vid in ids)


def test_build_records_chunk_idx_is_int_article_id_is_str(cfg):
    _ids, _texts, metas = build_records([make_article(7)], cfg)
    assert isinstance(metas[0]["article_id"], str)
    assert isinstance(metas[0]["chunk_idx"], int)


def test_build_records_empty_authors_tags_keys_omitted(cfg):
    art = make_article(2, authors=(), tags=())
    _ids, _texts, metas = build_records([art], cfg)
    assert metas
    for m in metas:
        assert "authors" not in m
        assert "tags" not in m
        assert set(m.keys()) == {
            "article_id", "title", "url", "timestamp", "chunk", "chunk_idx",
        }


def test_build_records_respects_override_mode(cfg):
    cfg = replace(cfg, embed_content="chunk_only")
    art = make_article(4)
    _ids, texts, metas = build_records([art], cfg)
    assert texts == [m["chunk"] for m in metas]


def test_build_records_concatenates_articles_in_order(cfg):
    a = make_article(7)
    b = make_article(8)
    na = len(chunk_text(a.text, cfg.chunk_size, cfg.overlap_ratio))
    nb = len(chunk_text(b.text, cfg.chunk_size, cfg.overlap_ratio))
    expected = [f"7-{i}" for i in range(na)] + [f"8-{i}" for i in range(nb)]
    ids, _texts, _metas = build_records([a, b], cfg)
    assert ids == expected


def test_build_records_empty_input_returns_three_empty_lists(cfg):
    assert build_records([], cfg) == ([], [], [])


def test_records_is_deterministic_across_two_calls(cfg):
    art = make_article(11)
    first = _records_for_article(art, cfg)
    second = _records_for_article(art, cfg)
    assert first == second


def test_build_records_empty_scalar_fields_kept_as_empty_string(cfg):
    art = make_article(6, text="A short body that yields one chunk.", title="")
    _ids, _texts, metas = build_records([art], cfg)
    assert metas[0]["title"] == ""


def test_build_records_title_tags_chunk_mode(cfg):
    """(A) title_tags_chunk end-to-end: the embed string carries BOTH the Title:
    and Tags: prefixes, while metadata stays clean -- raw un-prefixed chunk and
    tags as the raw list (not the joined string). The third embed mode, and the
    one where the raw-vs-embedded split is most likely to leak (tags appear in
    both the embed prefix and the metadata)."""
    cfg = replace(cfg, embed_content="title_tags_chunk")
    art = make_article(3, title="My Title", tags=("alpha", "beta"))
    _ids, texts, metas = build_records([art], cfg)
    assert texts[0].startswith("Title: My Title\nTags: alpha, beta\n\n")
    m = metas[0]
    assert not m["chunk"].startswith("Title:")
    assert m["chunk"] in texts[0]
    assert m["tags"] == ["alpha", "beta"]  # raw list, never the joined string


def test_build_records_chunk_size_affects_chunk_count(cfg):
    """(B) Config EFFECT, not just plumbing: a smaller chunk_size produces MORE
    chunks for the same body, proving the value actually reaches chunk_text (the
    C1->C4 wiring). The --override tests stub run_ingest and only check the cfg
    value; this runs the real chunker through build_records."""
    art = make_article(0)  # LONG_BODY
    big = build_records([art], replace(cfg, chunk_size=1024))[0]
    small = build_records([art], replace(cfg, chunk_size=128))[0]
    assert len(big) >= 1
    assert len(small) > len(big)


def test_real_csv_empty_lists_omitted_and_empty_title_kept(cfg):
    """(C) Through the REAL load_articles -> build_records chain (NOT hand-built
    Articles): the empty-authors/tags fixture row omits those keys, and the
    empty-title fixture row keeps title=''. Exercises C2 parsing
    (ast.literal_eval of '[]', pandas fillna -> '') feeding C7's metadata rules,
    which the hand-built-Article tests bypass entirely. This is the reason the
    fixture CSV carries those two special rows."""
    from src.data.csv_loader import load_articles

    articles = load_articles(TINY_CSV)
    _ids, _texts, metas = build_records(articles, cfg)
    by_article: dict[str, list[dict]] = {}
    for m in metas:
        by_article.setdefault(m["article_id"], []).append(m)

    # Fixture row 1 (A Quiet Walk): empty authors + tags -> keys omitted.
    assert by_article["1"]
    for m in by_article["1"]:
        assert "authors" not in m
        assert "tags" not in m

    # Fixture row 2: empty title -> kept as "" (empty scalar, not omitted).
    assert by_article["2"]
    for m in by_article["2"]:
        assert "title" in m
        assert m["title"] == ""


# --------------------------------------------------------------------------- #
# ingest_articles -- flush loop, alignment to Pinecone, stats                  #
# --------------------------------------------------------------------------- #
def test_ingest_upserts_aligned_vectors_and_metadata(fake_pc, fake_embed, cfg, monkeypatch):
    import src.rag.vectorstore as vectorstore

    monkeypatch.setattr("scripts.ingest.FLUSH_CHUNKS", 3)
    vectorstore.ensure_index(cfg)
    articles = [make_article(1), make_article(2)]
    expected_ids, _texts, _metas = build_records(articles, cfg)
    assert len(expected_ids) > 3  # force multiple flushes

    ingest_articles(articles, "ns", cfg)
    flat = _flatten_upserts(fake_pc[0])
    assert [v["id"] for v in flat] == expected_ids
    for i, v in enumerate(flat):
        assert v["values"][0] == float(i)
        assert v["id"] == f"{v['metadata']['article_id']}-{v['metadata']['chunk_idx']}"


def test_ingest_flushes_in_chunks_of_flush_chunks(fake_pc, fake_embed, cfg, monkeypatch):
    import src.rag.vectorstore as vectorstore

    monkeypatch.setattr("scripts.ingest.FLUSH_CHUNKS", 2)
    vectorstore.ensure_index(cfg)
    articles = [make_article(1), make_article(2)]
    expected_ids, _texts, _metas = build_records(articles, cfg)
    assert len(expected_ids) > 2

    ingest_articles(articles, "ns", cfg)
    assert len(fake_embed.calls) > 1
    upserted = [v["id"] for v in _flatten_upserts(fake_pc[0])]
    assert set(upserted) == set(expected_ids)
    assert len(upserted) == len(expected_ids)  # no dupes


def test_ingest_stats_counts(fake_pc, fake_embed, cfg):
    import src.rag.vectorstore as vectorstore

    vectorstore.ensure_index(cfg)
    articles = [make_article(0), make_article(1, text="   "), make_article(2)]
    stats = ingest_articles(articles, "ns", cfg)
    assert stats.articles_total == 3
    assert stats.articles_chunked == 2
    assert stats.articles_skipped == 1
    assert stats.chunks_total == stats.vectors_upserted
    assert stats.chunks_total > 0


def test_ingest_writes_to_given_namespace_not_cfg_default(fake_pc, fake_embed, cfg):
    import src.rag.vectorstore as vectorstore

    assert cfg.pinecone_namespace != "exp_x"
    vectorstore.ensure_index(cfg)
    ingest_articles([make_article(0)], "exp_x", cfg)
    assert fake_pc[0]._index.upserts
    for call in fake_pc[0]._index.upserts:
        assert call["namespace"] == "exp_x"


def test_ingest_empty_article_list_no_calls(fake_pc, fake_embed, cfg):
    import src.rag.vectorstore as vectorstore

    vectorstore.ensure_index(cfg)
    stats = ingest_articles([], "ns", cfg)
    assert fake_embed.calls == []
    assert fake_pc[0]._index.upserts == []
    assert stats == IngestStats(0, 0, 0, 0, 0, "ns")


def test_ingest_exact_flush_boundary_no_spurious_empty_flush(fake_pc, fake_embed, cfg, monkeypatch):
    import src.rag.vectorstore as vectorstore

    vectorstore.ensure_index(cfg)
    articles = [make_article(1), make_article(2)]
    expected_ids, _t, _m = build_records(articles, cfg)
    total = len(expected_ids)
    monkeypatch.setattr("scripts.ingest.FLUSH_CHUNKS", total)

    ingest_articles(articles, "ns", cfg)
    assert len(fake_embed.calls) == 1
    assert all(len(c) > 0 for c in fake_embed.calls)


def test_ingest_single_article_overshoots_buffer(fake_pc, fake_embed, cfg, monkeypatch):
    import src.rag.vectorstore as vectorstore

    monkeypatch.setattr("scripts.ingest.FLUSH_CHUNKS", 1)
    vectorstore.ensure_index(cfg)
    art = make_article(1)
    expected_ids, _t, _m = build_records([art], cfg)
    assert len(expected_ids) > 1  # one article overshoots the buffer

    ingest_articles([art], "ns", cfg)
    flat = _flatten_upserts(fake_pc[0])
    assert [v["id"] for v in flat] == expected_ids
    for i, v in enumerate(flat):
        assert v["values"][0] == float(i)


def test_ingest_propagates_embed_error(fake_pc, cfg, monkeypatch):
    import src.rag.vectorstore as vectorstore

    monkeypatch.setattr("scripts.ingest.FLUSH_CHUNKS", 2)
    vectorstore.ensure_index(cfg)

    state = {"n": 0}

    def exploding(texts, cfg=None):
        state["n"] += 1
        if state["n"] >= 2:
            raise RuntimeError("embed boom on 2nd flush")
        return [[0.0] * 1536 for _ in texts]

    monkeypatch.setattr("scripts.ingest.embed_batch", exploding)
    articles = [make_article(1), make_article(2)]
    assert len(build_records(articles, cfg)[0]) > 2

    with pytest.raises(RuntimeError, match="embed boom"):
        ingest_articles(articles, "ns", cfg)


def test_ingest_upsert_metadata_omits_empty_lists(fake_pc, fake_embed, cfg):
    import src.rag.vectorstore as vectorstore

    vectorstore.ensure_index(cfg)
    ingest_articles([make_article(0, authors=(), tags=())], "ns", cfg)
    flat = _flatten_upserts(fake_pc[0])
    assert flat
    for v in flat:
        assert "authors" not in v["metadata"]
        assert "tags" not in v["metadata"]


# --------------------------------------------------------------------------- #
# run_ingest / --clean wiring (articles from the fixture CSV)                  #
# --------------------------------------------------------------------------- #
def test_run_ingest_calls_ensure_index_then_upserts(fake_pc, fake_embed, cfg):
    stats = run_ingest("ns", cfg, csv_path=TINY_CSV)
    fake = fake_pc[0]
    assert cfg.pinecone_index in fake.index_infos  # ensure_index created it
    assert fake._index.upserts
    for call in fake._index.upserts:
        assert call["namespace"] == "ns"
    assert stats.vectors_upserted == stats.chunks_total > 0


def test_run_ingest_forwards_limit_to_load_articles(fake_pc, fake_embed, cfg):
    run_ingest("ns", cfg, limit=2, csv_path=TINY_CSV)
    ids = {v["metadata"]["article_id"] for v in _flatten_upserts(fake_pc[0])}
    assert ids <= {"0", "1"}
    assert ids  # something was ingested


def test_clean_false_records_no_delete(fake_pc, fake_embed, cfg):
    run_ingest("ns", cfg, csv_path=TINY_CSV, clean=False)
    assert fake_pc[0]._index.deletes == []


def test_clean_deletes_before_upsert(fake_pc, fake_embed, cfg):
    fake_seen: list[int] = []

    def configure():
        fake = fake_pc[0]
        orig = fake._index.upsert

        def wrapped(vectors, namespace):
            fake_seen.append(len(fake._index.deletes))
            return orig(vectors, namespace)

        fake._index.upsert = wrapped

    import src.rag.vectorstore as vectorstore

    vectorstore.ensure_index(cfg)  # force fake creation so we can wrap upsert
    configure()
    run_ingest("ns", cfg, csv_path=TINY_CSV, clean=True)
    assert fake_pc[0]._index.deletes  # a delete was recorded
    assert fake_seen and all(d >= 1 for d in fake_seen)  # delete preceded upsert


def test_clean_polls_namespace_until_zero_before_upsert(fake_pc, fake_embed, cfg, monkeypatch):
    import src.rag.vectorstore as vectorstore

    monkeypatch.setattr("scripts.ingest.WRITE_CONSISTENCY_POLL_S", 0.0)
    vectorstore.ensure_index(cfg)
    fake = fake_pc[0]

    calls = {"n": 0}

    def fake_stats():
        calls["n"] += 1
        count = 7 if calls["n"] == 1 else 0  # non-zero first, then settled
        return {"namespaces": {"ns": {"vector_count": count}}, "dimension": 1536}

    fake._index.describe_index_stats = fake_stats
    orig_upsert = fake._index.upsert
    seen: list[int] = []

    def wrapped(vectors, namespace):
        seen.append(calls["n"])
        return orig_upsert(vectors, namespace)

    fake._index.upsert = wrapped

    run_ingest("ns", cfg, csv_path=TINY_CSV, clean=True)
    assert calls["n"] >= 2     # it looped, not a single read
    assert seen                # ingest proceeded
    assert min(seen) >= 2      # every upsert happened after the poll saw zero


# --------------------------------------------------------------------------- #
# CLI / main contract (exit codes, flag handling)                             #
# --------------------------------------------------------------------------- #
def test_main_success_returns_zero(capsys, env, fake_pc, fake_embed):
    rc = main(["--namespace", "demo", "--csv", TINY_CSV])
    assert rc == 0
    out = capsys.readouterr().out
    assert "OK:" in out
    assert "namespace=demo" in out
    assert "vectors=" in out


def test_main_missing_namespace_exits_nonzero(env):
    with pytest.raises(SystemExit) as exc:
        main(["--csv", TINY_CSV])
    assert exc.value.code != 0


def test_main_failure_returns_nonzero(capsys, env, monkeypatch):
    def boom(cfg=None):
        raise RuntimeError("index wrong dimension")

    monkeypatch.setattr("scripts.ingest.ensure_index", boom)
    rc = main(["--namespace", "demo", "--csv", TINY_CSV])
    assert rc != 0
    assert "ERROR" in capsys.readouterr().out


def test_main_missing_csv_returns_nonzero(capsys, env):
    """(D) A bad --csv path -> load_articles raises FileNotFoundError -> main
    catches it into a readable non-zero exit, not a raw traceback. load_articles
    runs before ensure_index in run_ingest, so no Pinecone fake is needed."""
    rc = main(["--namespace", "demo", "--csv", "does_not_exist_xyz.csv"])
    assert rc != 0
    assert "ERROR" in capsys.readouterr().out


def test_override_unknown_key_exits_nonzero_listing_allowed(capsys, env):
    rc = main(["--namespace", "demo", "--override", "reasoning_effort=high"])
    assert rc != 0
    out = capsys.readouterr().out
    assert "reasoning_effort" in out
    assert "chunk_size" in out  # the allowed set is listed


@pytest.mark.parametrize(
    "bad",
    ["chunk_size=5000", "overlap_ratio=0.9", "top_k=99"],
)
def test_override_out_of_range_value_rejected(capsys, env, bad):
    rc = main(["--namespace", "demo", "--csv", TINY_CSV, "--override", bad])
    assert rc != 0
    assert "ERROR" in capsys.readouterr().out


def test_override_malformed_no_equals_errors(capsys, env):
    rc = main(["--namespace", "demo", "--override", "chunk_size768"])
    assert rc != 0
    assert "ERROR" in capsys.readouterr().out


def test_override_multiple_flags_all_apply(env, monkeypatch):
    captured = {}

    def capture(namespace, cfg=None, *, limit=None, clean=False, csv_path=None):
        captured["cfg"] = cfg
        return IngestStats(0, 0, 0, 0, 0, namespace)

    monkeypatch.setattr("scripts.ingest.run_ingest", capture)
    rc = main([
        "--namespace", "demo",
        "--override", "chunk_size=768",
        "--override", "embed_content=title_tags_chunk",
    ])
    assert rc == 0
    assert captured["cfg"].chunk_size == 768
    assert captured["cfg"].embed_content == "title_tags_chunk"


def test_override_chunk_size_flows_into_config(env, monkeypatch):
    captured = {}

    def capture(namespace, cfg=None, *, limit=None, clean=False, csv_path=None):
        captured["cfg"] = cfg
        return IngestStats(0, 0, 0, 0, 0, namespace)

    monkeypatch.setattr("scripts.ingest.run_ingest", capture)
    rc = main(["--namespace", "demo", "--override", "chunk_size=256"])
    assert rc == 0
    assert captured["cfg"].chunk_size == 256


def test_override_restores_env_after_load(env, monkeypatch):
    import os

    before = os.environ.get("CHUNK_SIZE")

    def capture(namespace, cfg=None, *, limit=None, clean=False, csv_path=None):
        return IngestStats(0, 0, 0, 0, 0, namespace)

    monkeypatch.setattr("scripts.ingest.run_ingest", capture)
    main(["--namespace", "demo", "--override", "chunk_size=256"])
    assert os.environ.get("CHUNK_SIZE") == before


def test_stdout_is_cp1252_encodable(capsys, env, fake_pc, fake_embed, monkeypatch):
    monkeypatch.setattr("scripts.ingest.FLUSH_CHUNKS", 1)  # force per-flush prints
    rc = main(["--namespace", "demo", "--csv", TINY_CSV])
    assert rc == 0
    out = capsys.readouterr().out
    assert "flushed" in out  # progress lines printed
    out.encode("cp1252")  # must not raise
