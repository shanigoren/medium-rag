"""Offline unit tests for the embed builder (Component 6).

Zero network calls. `build_embed_text` is pure (no stubbing); `embed_batch` is
exercised against the `fake_embeddings` stub from conftest so batching and
order-preservation are asserted without burning API budget.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.rag.embed import EMBED_BATCH_SIZE, build_embed_text, embed_batch


def _article(title="A Title", tags=("ai", "rag")):
    """Lightweight duck-typed stand-in for csv_loader.Article — only .title and
    .tags are read by build_embed_text. Deliberately NOT a real Article so the
    test never imports pandas."""
    return SimpleNamespace(title=title, tags=list(tags))


# --------------------------------------------------------------------------- #
# build_embed_text — rendering (snapshots)
# --------------------------------------------------------------------------- #


def test_build_chunk_only_returns_chunk_verbatim():
    chunk = "  leading and trailing whitespace kept  "
    assert build_embed_text(_article(), chunk, "chunk_only") == chunk


def test_build_title_chunk_prepends_title():
    art = _article(title="How to Build Habits")
    out = build_embed_text(art, "body text here", "title_chunk")
    assert out == "Title: How to Build Habits\n\nbody text here"


def test_build_title_tags_chunk_prepends_title_and_tags():
    art = _article(title="How to Build Habits", tags=["ai", "rag"])
    out = build_embed_text(art, "body text here", "title_tags_chunk")
    assert out == "Title: How to Build Habits\nTags: ai, rag\n\nbody text here"


def test_build_title_tags_chunk_empty_tags_emits_empty_tags_line():
    art = _article(title="No Tags", tags=[])
    out = build_embed_text(art, "body", "title_tags_chunk")
    # Trailing space after 'Tags:' is intentional and pinned (design choice #5).
    assert out == "Title: No Tags\nTags: \n\nbody"


def test_build_title_tags_chunk_single_tag_no_separator():
    art = _article(title="One", tags=["solo"])
    out = build_embed_text(art, "body", "title_tags_chunk")
    assert out == "Title: One\nTags: solo\n\nbody"


def test_build_uses_duck_typed_article():
    # SimpleNamespace, not csv_loader.Article — proves no Article/pandas dep.
    art = SimpleNamespace(title="Duck", tags=["typed"])
    out = build_embed_text(art, "x", "title_tags_chunk")
    assert out == "Title: Duck\nTags: typed\n\nx"


def test_build_rejects_unknown_mode():
    with pytest.raises(ValueError) as exc:
        build_embed_text(_article(), "x", "nonsense")
    msg = str(exc.value)
    assert "nonsense" in msg
    for mode in ("chunk_only", "title_chunk", "title_tags_chunk"):
        assert mode in msg


def test_build_modes_match_config_enum():
    """The modes build_embed_text accepts are exactly the config enum — catches
    drift if a mode is added to config but not here (or vice versa)."""
    from src.config import _VALID_EMBED_CONTENT
    from src.rag.embed import _VALID_MODES

    assert set(_VALID_MODES) == set(_VALID_EMBED_CONTENT)
    # And every one renders without error.
    for mode in _VALID_MODES:
        build_embed_text(_article(), "chunk", mode)


# --------------------------------------------------------------------------- #
# embed_batch — batching + order (stub client)
# --------------------------------------------------------------------------- #


def test_embed_batch_empty_returns_empty_no_calls(fake_embeddings):
    assert embed_batch([]) == []
    assert fake_embeddings.batches == []


def test_embed_batch_single_batch(fake_embeddings):
    texts = [f"t{i}" for i in range(10)]
    result = embed_batch(texts)
    assert len(result) == 10
    assert all(len(v) == 1536 for v in result)
    assert len(fake_embeddings.batches) == 1


def test_embed_batch_at_exactly_batch_size(fake_embeddings):
    texts = [f"t{i}" for i in range(EMBED_BATCH_SIZE)]
    result = embed_batch(texts)
    assert len(result) == EMBED_BATCH_SIZE
    # Boundary: a `>` vs `>=` slip would emit an empty second batch.
    assert len(fake_embeddings.batches) == 1


def test_embed_batch_at_batch_size_plus_one(fake_embeddings):
    texts = [f"t{i}" for i in range(EMBED_BATCH_SIZE + 1)]
    result = embed_batch(texts)
    assert len(result) == EMBED_BATCH_SIZE + 1
    sizes = [len(b) for b in fake_embeddings.batches]
    assert sizes == [EMBED_BATCH_SIZE, 1]


def test_embed_batch_preserves_order_across_batches(fake_embeddings):
    n = 2 * EMBED_BATCH_SIZE + 7
    texts = [f"t{i}" for i in range(n)]
    result = embed_batch(texts)
    assert len(result) == n
    # The stub encodes each text's global input index in vector[0].
    for i in range(n):
        assert result[i][0] == float(i)


def test_embed_batch_forwards_cfg(fake_embeddings, cfg):
    # Stub ignores cfg; this only proves the cfg path doesn't crash and the
    # signature matches Component 3's convention.
    result = embed_batch(["a", "b"], cfg=cfg)
    assert len(result) == 2
