"""Live LLMod.AI embedding smoke tests for the embed builder (Component 6).

Run with `pytest --smoke -v -m smoke -k embed`. Skipped by default. Embeds a
handful of short strings — cost is a few cents, no Pinecone involvement.
"""

from __future__ import annotations

import math

import pytest

from src.rag.embed import embed_batch


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


@pytest.mark.smoke
def test_embed_batch_returns_1536d_vectors_for_three_strings():
    texts = ["hello world", "machine learning", "medium article"]
    vectors = embed_batch(texts)
    assert len(vectors) == 3
    for v in vectors:
        assert len(v) == 1536
        assert all(isinstance(x, float) for x in v)


@pytest.mark.smoke
def test_embed_batch_is_order_stable_and_deterministic():
    texts = ["hello world", "machine learning", "medium article"]
    first = embed_batch(texts)
    second = embed_batch(texts)
    # Same input -> position-wise near-identical vectors (cosine ~1.0). Proves
    # order is preserved across calls and the embedder is effectively
    # deterministic; tiny numerical drift is tolerated.
    for a, b in zip(first, second):
        assert _cosine(a, b) > 0.999
    # And distinct strings are not collapsed to the same vector.
    assert _cosine(first[0], first[1]) < 0.999
