"""Offline unit tests for Component 8 (`src.rag.retriever`).

Run on every `pytest`; ZERO network. Strategy: drive C5's REAL `query` against
the in-memory `fake_pc` (so the Match-building / include_metadata contract is
exercised for real), and stub only the query embedding via `fake_query_embeddings`.

Results are driven by setting `fakes[0]._index.query_response = {"matches": [...]}`
with fabricated matches whose metadata carries the C7 schema. The matches list
MUST be pre-sorted by score desc to mirror Pinecone (C5 does not re-sort).

The `cfg`, `fake_pc`, and `fake_query_embeddings` fixtures live in conftest.py.
"""

from __future__ import annotations

import pytest

from src.rag import vectorstore
from src.rag.retriever import RetrievedChunk, retrieve


def _m(article_id, score, chunk_idx, chunk="some text", title="Some Title",
       authors=("X",), tags=("Topic",)):
    """Build one fake Pinecone hit (a dict) shaped like an SDK match, carrying
    the C7 metadata schema."""
    return {
        "id": f"{article_id}-{chunk_idx}",
        "score": score,
        "metadata": {
            "article_id": article_id,
            "title": title,
            "authors": list(authors),
            "tags": list(tags),
            "chunk": chunk,
            "chunk_idx": chunk_idx,
        },
    }


def _set_matches(fake_pc, cfg, matches):
    """Trigger fake-client construction and set the query_response. The retriever
    calls the real C5 `query`, which goes through the cached `_client` -- so the
    instance we configure here is the same one `retrieve` will hit."""
    vectorstore._client(cfg.pinecone_api_key)  # construct + cache the fake
    fake_pc[0]._index.query_response = {"matches": list(matches)}
    return fake_pc[0]._index


# ---------- dedup=True (default) -----------------------------------------


def test_dedupes_to_one_chunk_per_article(fake_pc, fake_query_embeddings, cfg):
    """Matches span articles [7,7,8,7,9] -> result article_ids == ['7','8','9']
    (each distinct, no repeats)."""
    _set_matches(fake_pc, cfg, [
        _m("7", 0.95, 0),
        _m("7", 0.90, 1),
        _m("8", 0.80, 0),
        _m("7", 0.70, 2),
        _m("9", 0.60, 0),
    ])
    out = retrieve("q", "ns", cfg)
    assert [r.article_id for r in out] == ["7", "8", "9"]


def test_keeps_highest_score_chunk_per_article(fake_pc, fake_query_embeddings, cfg):
    """Article '7' appears twice: (score 0.91, chunk_idx 3) and (0.62, chunk_idx 0).
    The kept '7' record has score 0.91 and chunk_idx 3 (first-seen == best)."""
    _set_matches(fake_pc, cfg, [
        _m("7", 0.91, 3),
        _m("8", 0.80, 0),
        _m("7", 0.62, 0),
    ])
    out = retrieve("q", "ns", cfg)
    rec7 = next(r for r in out if r.article_id == "7")
    assert rec7.score == 0.91
    assert rec7.chunk_idx == 3


def test_returns_at_most_top_k_distinct_articles(fake_pc, fake_query_embeddings, cfg):
    """6 distinct articles in matches, top_k=3 -> exactly 3 records returned."""
    _set_matches(fake_pc, cfg, [
        _m(str(i), 0.9 - i * 0.1, 0) for i in range(6)
    ])
    out = retrieve("q", "ns", cfg, top_k=3)
    assert len(out) == 3


def test_fewer_distinct_than_top_k_returns_all(fake_pc, fake_query_embeddings, cfg):
    """Matches span only 2 distinct articles, top_k=5 -> 2 records, no padding."""
    _set_matches(fake_pc, cfg, [
        _m("7", 0.9, 0),
        _m("7", 0.8, 1),
        _m("8", 0.7, 0),
    ])
    out = retrieve("q", "ns", cfg, top_k=5)
    assert len(out) == 2
    assert {r.article_id for r in out} == {"7", "8"}


def test_results_ordered_by_score_descending(fake_pc, fake_query_embeddings, cfg):
    """Returned scores are non-increasing across ranks 1..n."""
    _set_matches(fake_pc, cfg, [
        _m("7", 0.9, 0),
        _m("8", 0.7, 0),
        _m("9", 0.5, 0),
    ])
    out = retrieve("q", "ns", cfg)
    scores = [r.score for r in out]
    assert scores == sorted(scores, reverse=True)


def test_rank_is_1_based_and_sequential(fake_pc, fake_query_embeddings, cfg):
    """ranks == [1,2,3,...] in returned order, assigned AFTER dedupe."""
    _set_matches(fake_pc, cfg, [
        _m("7", 0.9, 0),
        _m("7", 0.85, 1),  # dropped duplicate -> must not consume a rank
        _m("8", 0.7, 0),
        _m("9", 0.5, 0),
    ])
    out = retrieve("q", "ns", cfg)
    assert [r.rank for r in out] == [1, 2, 3]


@pytest.mark.parametrize("dedup", [True, False])
def test_empty_namespace_returns_empty_list(dedup, fake_pc, fake_query_embeddings, cfg):
    """query_response = {'matches': []} -> retrieve(..., dedup=dedup) == [] in
    BOTH modes."""
    _set_matches(fake_pc, cfg, [])
    assert retrieve("q", "ns", cfg, dedup=dedup) == []


def test_over_fetches_fetch_k_not_top_k(fake_pc, fake_query_embeddings, cfg):
    """retrieve(..., top_k=3, fetch_k=30): the recorded C5 query call used
    top_k=30 -- proves over-fetch."""
    _set_matches(fake_pc, cfg, [_m("7", 0.9, 0)])
    retrieve("q", "ns", cfg, top_k=3, fetch_k=30)
    assert fake_pc[0]._index.queries[0]["top_k"] == 30


def test_fetch_k_defaults_to_cfg_retrieval_fetch_k(fake_pc, fake_query_embeddings, cfg):
    """No explicit fetch_k -> recorded query top_k == cfg.retrieval_fetch_k."""
    _set_matches(fake_pc, cfg, [_m("7", 0.9, 0)])
    retrieve("q", "ns", cfg)
    assert fake_pc[0]._index.queries[0]["top_k"] == cfg.retrieval_fetch_k


def test_top_k_defaults_to_cfg_top_k(fake_pc, fake_query_embeddings, cfg):
    """No explicit top_k, matches span >cfg.top_k articles -> len==cfg.top_k."""
    _set_matches(fake_pc, cfg, [
        _m(str(i), 0.99 - i * 0.01, 0) for i in range(cfg.top_k + 4)
    ])
    out = retrieve("q", "ns", cfg)
    assert len(out) == cfg.top_k


def test_explicit_top_k_overrides_cfg(fake_pc, fake_query_embeddings, cfg):
    """retrieve(..., top_k=2) returns 2 distinct articles even though cfg.top_k==20."""
    assert cfg.top_k == 20
    _set_matches(fake_pc, cfg, [
        _m("7", 0.9, 0), _m("8", 0.8, 0), _m("9", 0.7, 0), _m("10", 0.6, 0),
    ])
    out = retrieve("q", "ns", cfg, top_k=2)
    assert len(out) == 2


def test_fetch_k_clamped_to_at_least_top_k(fake_pc, fake_query_embeddings, cfg):
    """retrieve(..., top_k=8, fetch_k=3) -> recorded query top_k >= 8 (clamp,
    not raise)."""
    _set_matches(fake_pc, cfg, [_m("7", 0.9, 0)])
    retrieve("q", "ns", cfg, top_k=8, fetch_k=3)
    assert fake_pc[0]._index.queries[0]["top_k"] >= 8


def test_query_embedded_raw_no_prefix(fake_query_embeddings, fake_pc, cfg):
    """embed_query was called with the EXACT query string -- no 'Title:' prefix
    added by the retriever."""
    _set_matches(fake_pc, cfg, [_m("7", 0.9, 0)])
    retrieve("marketing as conversation", "ns", cfg)
    assert fake_query_embeddings.queries[-1] == "marketing as conversation"


def test_article_id_is_string_even_if_metadata_has_int(fake_pc, fake_query_embeddings, cfg):
    """A match whose metadata['article_id'] is the int 42 -> result.article_id
    == '42' (str). Guards the recall@k coercion bug (cross-cutting invariant)."""
    _set_matches(fake_pc, cfg, [_m(42, 0.9, 0)])
    out = retrieve("q", "ns", cfg)
    assert out[0].article_id == "42"
    assert isinstance(out[0].article_id, str)


def test_chunk_is_raw_passthrough(fake_pc, fake_query_embeddings, cfg):
    """result.chunk == metadata['chunk'] verbatim; it does NOT start with
    'Title:' (C7 stored the raw chunk; the retriever returns it unmodified)."""
    _set_matches(fake_pc, cfg, [_m("7", 0.9, 0, chunk="The raw body of the chunk.")])
    out = retrieve("q", "ns", cfg)
    assert out[0].chunk == "The raw body of the chunk."
    assert not out[0].chunk.startswith("Title:")


def test_chunk_idx_coerced_to_int(fake_pc, fake_query_embeddings, cfg):
    """metadata['chunk_idx'] given as '3' (str) -> result.chunk_idx == 3 (int)."""
    _set_matches(fake_pc, cfg, [_m("7", 0.9, "3")])
    out = retrieve("q", "ns", cfg)
    assert out[0].chunk_idx == 3
    assert isinstance(out[0].chunk_idx, int)


def test_absent_optional_metadata_uses_fallbacks(fake_pc, fake_query_embeddings, cfg):
    """A match with required article_id+chunk but NO 'authors'/'tags'/'title' ->
    result.authors == [], result.tags == [], result.title == '' (no KeyError)."""
    _set_matches(fake_pc, cfg, [
        {"id": "7-0", "score": 0.9, "metadata": {"article_id": "7", "chunk": "body"}}
    ])
    out = retrieve("q", "ns", cfg)
    assert out[0].authors == []
    assert out[0].tags == []
    assert out[0].title == ""


def test_tags_extracted_from_metadata(fake_pc, fake_query_embeddings, cfg):
    """A match carrying a 'tags' list -> result.tags is that list (used by the C9
    prompt header for topical judgment; not in the API context)."""
    _set_matches(fake_pc, cfg, [
        _m("7", 0.9, 0, tags=("Mental Health", "Science")),
    ])
    out = retrieve("q", "ns", cfg)
    assert out[0].tags == ["Mental Health", "Science"]


def test_skips_matches_missing_required_metadata(fake_pc, fake_query_embeddings, cfg):
    """A match with metadata={} (no article_id/chunk) is skipped; a well-formed
    match in the same response is still returned."""
    _set_matches(fake_pc, cfg, [
        {"id": "bad", "score": 0.95, "metadata": {}},
        _m("8", 0.80, 0),
    ])
    out = retrieve("q", "ns", cfg)
    assert [r.article_id for r in out] == ["8"]


def test_unusable_first_occurrence_does_not_block_same_article(fake_pc, fake_query_embeddings, cfg):
    """dedup=True seen-pollution guard: an UNUSABLE match for article '7' (has
    article_id but NO 'chunk', so _to_chunk returns None) appears BEFORE a valid
    '7' chunk. Result still contains '7' -- the skipped match must not have
    added '7' to `seen`."""
    _set_matches(fake_pc, cfg, [
        {"id": "7-x", "score": 0.99, "metadata": {"article_id": "7"}},  # no chunk
        _m("7", 0.80, 0, chunk="good 7 chunk"),
    ])
    out = retrieve("q", "ns", cfg)
    assert [r.article_id for r in out] == ["7"]
    assert out[0].chunk == "good 7 chunk"


def test_type2_list_query_surfaces_ge_3_distinct(fake_pc, fake_query_embeddings, cfg):
    """8 matches spanning >=3 articles, top_k=3 -> 3 DISTINCT article_ids.
    Unit-level mirror of the CP-B 'list 3 articles' assertion."""
    _set_matches(fake_pc, cfg, [
        _m("7", 0.99, 0),
        _m("7", 0.95, 1),
        _m("8", 0.90, 0),
        _m("7", 0.85, 2),
        _m("8", 0.80, 1),
        _m("9", 0.75, 0),
        _m("10", 0.70, 0),
        _m("9", 0.65, 1),
    ])
    out = retrieve("list 3 articles about X", "ns", cfg, top_k=3)
    ids = [r.article_id for r in out]
    assert len(ids) == 3
    assert len(set(ids)) == 3


@pytest.mark.parametrize("dedup", [True, False])
def test_empty_query_raises_valueerror(dedup, fake_query_embeddings, cfg):
    """retrieve('   ', 'ns', cfg, dedup=dedup) -> ValueError in BOTH modes;
    embed_query is never called (validation precedes any work)."""
    with pytest.raises(ValueError):
        retrieve("   ", "ns", cfg, dedup=dedup)
    assert fake_query_embeddings.queries == []


def test_cfg_none_loads_default_config(fake_pc, fake_query_embeddings, monkeypatch):
    """retrieve('q', 'ns') with cfg OMITTED -> load_config() is used; the call
    succeeds and the recorded query top_k == the default cfg.retrieval_fetch_k.
    Exercises the `cfg = cfg or load_config()` branch."""
    monkeypatch.setenv("LLMOD_API_KEY", "sk-test")
    monkeypatch.setenv("LLMOD_BASE_URL", "https://api.llmod.ai/v1")
    monkeypatch.setenv("PINECONE_API_KEY", "pc-test")
    from src.config import load_config

    loaded = load_config()
    # Construct the fake client under the same api key the retriever will use.
    vectorstore._client(loaded.pinecone_api_key)
    fake_pc[0]._index.query_response = {"matches": [_m("7", 0.9, 0)]}

    out = retrieve("q", "ns")
    assert out[0].article_id == "7"
    assert fake_pc[0]._index.queries[0]["top_k"] == loaded.retrieval_fetch_k


def test_namespace_passed_through_to_query(fake_pc, fake_query_embeddings, cfg):
    """retrieve(..., namespace='exp_x') -> recorded C5 query used namespace
    'exp_x'."""
    _set_matches(fake_pc, cfg, [_m("7", 0.9, 0)])
    retrieve("q", "exp_x", cfg)
    assert fake_pc[0]._index.queries[0]["namespace"] == "exp_x"


# ---------- dedup=False mode ---------------------------------------------


def test_no_dedup_returns_top_k_chunks_with_duplicate_articles(fake_pc, fake_query_embeddings, cfg):
    """Matches span articles [7,7,8], top_k=3, dedup=False -> 3 records whose
    article_ids are ['7','7','8'] in score order (duplicates KEPT)."""
    _set_matches(fake_pc, cfg, [
        _m("7", 0.9, 0),
        _m("7", 0.8, 1),
        _m("8", 0.7, 0),
    ])
    out = retrieve("q", "ns", cfg, top_k=3, dedup=False)
    assert [r.article_id for r in out] == ["7", "7", "8"]


def test_no_dedup_ignores_fetch_k_and_queries_top_k(fake_pc, fake_query_embeddings, cfg):
    """retrieve(..., top_k=3, fetch_k=30, dedup=False): the recorded C5 query
    used top_k=3 -- fetch_k is IGNORED entirely (not just defaulted)."""
    _set_matches(fake_pc, cfg, [_m("7", 0.9, 0)])
    retrieve("q", "ns", cfg, top_k=3, fetch_k=30, dedup=False)
    assert fake_pc[0]._index.queries[0]["top_k"] == 3


def test_no_dedup_preserves_score_order_and_rank(fake_pc, fake_query_embeddings, cfg):
    """dedup=False results are score-descending; ranks are [1,2,3,...] over the
    returned (un-collapsed) list."""
    _set_matches(fake_pc, cfg, [
        _m("7", 0.9, 0),
        _m("7", 0.8, 1),
        _m("8", 0.7, 0),
    ])
    out = retrieve("q", "ns", cfg, top_k=3, dedup=False)
    assert [r.score for r in out] == [0.9, 0.8, 0.7]
    assert [r.rank for r in out] == [1, 2, 3]


def test_no_dedup_fewer_matches_than_top_k_returns_all(fake_pc, fake_query_embeddings, cfg):
    """dedup=False, namespace returns only 2 matches for top_k=5 -> 2 records,
    no padding."""
    _set_matches(fake_pc, cfg, [
        _m("7", 0.9, 0),
        _m("8", 0.8, 0),
    ])
    out = retrieve("q", "ns", cfg, top_k=5, dedup=False)
    assert len(out) == 2


def test_no_dedup_skips_unusable_match_with_contiguous_ranks(fake_pc, fake_query_embeddings, cfg):
    """dedup=False with one metadata={} match among k -> that one is skipped, the
    result has k-1 records (we do NOT over-fetch to refill the dropped slot), AND
    the surviving ranks are contiguous [1..k-1] with no gap where the skip was."""
    _set_matches(fake_pc, cfg, [
        _m("7", 0.9, 0),
        {"id": "bad", "score": 0.8, "metadata": {}},
        _m("8", 0.7, 0),
    ])
    out = retrieve("q", "ns", cfg, top_k=3, dedup=False)
    assert len(out) == 2
    assert [r.rank for r in out] == [1, 2]
    assert [r.article_id for r in out] == ["7", "8"]


def test_dedup_default_is_true(fake_pc, fake_query_embeddings, cfg):
    """Calling retrieve WITHOUT the dedup kwarg on matches [7,7,8] collapses to
    ['7','8'] -- the default is the safe distinct-articles baseline."""
    _set_matches(fake_pc, cfg, [
        _m("7", 0.9, 0),
        _m("7", 0.8, 1),
        _m("8", 0.7, 0),
    ])
    out = retrieve("q", "ns", cfg)
    assert [r.article_id for r in out] == ["7", "8"]


def test_type3_query_with_no_dedup_keeps_multiple_chunks_of_one_article(fake_pc, fake_query_embeddings, cfg):
    """dedup=False, matches all from article '7' (chunk_idx 0,1,2), top_k=3 ->
    3 records all article_id '7' with distinct chunk_idx -- the depth a summary
    needs (the case dedup=True would have collapsed to a single chunk)."""
    _set_matches(fake_pc, cfg, [
        _m("7", 0.9, 0),
        _m("7", 0.8, 1),
        _m("7", 0.7, 2),
    ])
    out = retrieve("summarise article 7", "ns", cfg, top_k=3, dedup=False)
    assert [r.article_id for r in out] == ["7", "7", "7"]
    assert [r.chunk_idx for r in out] == [0, 1, 2]


def test_retrieved_chunk_has_exactly_expected_fields():
    """RetrievedChunk carries exactly {article_id, title, authors, tags, chunk,
    score, chunk_idx, rank}."""
    import dataclasses

    names = {f.name for f in dataclasses.fields(RetrievedChunk)}
    assert names == {
        "article_id", "title", "authors", "tags", "chunk", "score", "chunk_idx", "rank"
    }
