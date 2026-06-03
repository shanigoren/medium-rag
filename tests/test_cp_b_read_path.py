"""Checkpoint B -- Read path / full RAG loop (integrates C3 + C10 + C8 + C9 + C11).

This is the INTEGRATION GATE, kept as a separate artifact from the C11 component
tests (`test_chain.py` offline, `test_chain_smoke.py` live). Per-component smoke
proves "the unit runs live"; CP-B proves the five components interact correctly
across their seams, calling the real `answer(q, namespace="smoke")` against the
live `smoke` namespace produced by C7/CP-A.

Runs only with `pytest --smoke`. Read-only -- creates/deletes nothing (CP-C/D
reuse the `smoke` namespace). Select as a gate with `-k cp_b`.

Seams covered:
  1. embed asymmetry  -- a RAW query still retrieves a TITLE-prefixed document.
  2. original-question invariant -- augmented_prompt['User'] holds the ORIGINAL
     question (the strict 'not the rewrite' direction is pinned deterministically
     by the offline test_augmented_prompt_uses_original_question_not_rewrite).
  3. type-2 dedupe    -- a 'list 3 articles' question yields >= 3 DISTINCT articles.
  4. wire contract    -- to_api_dict() matches the /api/prompt schema exactly.
"""

from __future__ import annotations

import pytest

from src.config import load_config
from src.data.csv_loader import load_articles
from src.rag.chain import answer
from src.rag.vectorstore import namespace_stats

# The 10-article slice that was ingested into the `smoke` namespace (C7/CP-A).
_SMOKE_LIMIT = 10


def _require_smoke(cfg) -> None:
    if namespace_stats("smoke", cfg)["vector_count"] == 0:
        pytest.skip("run scripts/ingest.py --limit 10 --namespace smoke first")


def _target_article():
    """A deterministic, distinctive ingested article: the one with the longest
    title among the 10-row smoke slice. Its title makes a strong recall query."""
    articles = load_articles(limit=_SMOKE_LIMIT)
    return max(articles, key=lambda a: len(a.title))


@pytest.mark.smoke
def test_cp_b_embed_asymmetry_recall():
    """Seam 1. A question built from one ingested article's title -> that
    article_id is among the context article_ids. Proves a RAW query still
    retrieves the TITLE-prefixed document end-to-end (recall on a known item)."""
    cfg = load_config()
    _require_smoke(cfg)

    target = _target_article()
    res = answer(f"Find the article titled '{target.title}'.", cfg, namespace="smoke")

    ids = [row["article_id"] for row in res.context]
    assert str(target.row_idx) in ids, (
        f"expected article {target.row_idx} ({target.title!r}) in context ids {ids}"
    )


@pytest.mark.smoke
def test_cp_b_augmented_prompt_is_original_question():
    """Seam 2. answer(original_q).augmented_prompt['User'] contains the ORIGINAL
    question verbatim (cross-cutting invariant, on live data)."""
    cfg = load_config()
    _require_smoke(cfg)

    original = "I want practical, beginner-friendly advice on building habits that stick. Which article would you recommend, and why?"
    res = answer(original, cfg, namespace="smoke")

    assert original in res.augmented_prompt["User"]


@pytest.mark.smoke
def test_cp_b_list3_returns_distinct_articles():
    """Seam 3. A 'list 3 articles about <topic>' question -> the rewriter sets
    dedup=True, the retriever collapses to distinct articles, so the context has
    >= 3 DISTINCT article_ids -- end-to-end, not just the unit."""
    cfg = load_config()
    _require_smoke(cfg)

    res = answer("List exactly 3 articles about the human brain. Return only the titles.", cfg, namespace="smoke")

    distinct = {row["article_id"] for row in res.context}
    assert len(distinct) >= 3, f"expected >= 3 distinct articles, got {sorted(distinct)}"


@pytest.mark.smoke
def test_cp_b_output_shape_matches_api_contract():
    """Seam 4. to_api_dict() has exactly {'response','context','Augmented_prompt'}
    (capital-A); every context row has exactly {'article_id','title','chunk',
    'score'}; the inner prompt dict has {'System','User'}. Locks that C12 can be a
    thin pass-through."""
    cfg = load_config()
    _require_smoke(cfg)

    api = answer("What is one article about?", cfg, namespace="smoke").to_api_dict()

    assert set(api) == {"response", "context", "Augmented_prompt"}
    assert set(api["Augmented_prompt"]) == {"System", "User"}
    assert isinstance(api["response"], str)
    assert api["context"], "context should be non-empty for a live query"
    for row in api["context"]:
        assert set(row) == {"article_id", "title", "chunk", "score"}
        assert isinstance(row["article_id"], str)
        assert isinstance(row["score"], float)
