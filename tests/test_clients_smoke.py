"""Live-API smoke tests. Run with `pytest --smoke -v -k smoke`.

Each test hits LLMod.AI directly. Combined cost: << $0.001.
Auth/proxy errors propagate unmodified so they're easy to diagnose.
"""

from __future__ import annotations

import math

import pytest
from langchain_core.messages import HumanMessage

from src.llm.clients import get_chat, get_embeddings


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


@pytest.mark.smoke
def test_embeddings_hello_returns_1536_vector() -> None:
    vec = get_embeddings().embed_query("hello")
    assert isinstance(vec, list)
    assert len(vec) == 1536
    assert all(isinstance(v, float) for v in vec)
    assert any(v != 0.0 for v in vec)


@pytest.mark.smoke
def test_embeddings_are_unit_normalized() -> None:
    """text-embedding-3-small returns L2-normalized vectors. A norm far from
    1.0 means the proxy returned a different model or mangled the payload."""
    vec = get_embeddings().embed_query("hello")
    norm = math.sqrt(sum(v * v for v in vec))
    assert math.isclose(norm, 1.0, abs_tol=1e-3), f"norm={norm}"


@pytest.mark.smoke
def test_embeddings_have_semantic_ordering() -> None:
    """Confirm vectors are *meaningful*, not just shaped right:
    cos(cat, kitten) > cos(cat, spaceship)."""
    emb = get_embeddings()
    vecs = emb.embed_documents(["cat", "kitten", "spaceship"])
    cat, kitten, spaceship = vecs
    sim_close = _cosine(cat, kitten)
    sim_far = _cosine(cat, spaceship)
    assert sim_close > sim_far, (
        f"cos(cat,kitten)={sim_close:.4f} not > cos(cat,spaceship)={sim_far:.4f}"
    )


@pytest.mark.smoke
def test_embeddings_batch_matches_single() -> None:
    """embed_documents([x])[0] must equal embed_query(x) — catches a wiring
    bug where the two paths hit different models / endpoints."""
    emb = get_embeddings()
    text = "the quick brown fox"
    single = emb.embed_query(text)
    batched = emb.embed_documents([text])[0]
    sim = _cosine(single, batched)
    assert sim > 0.999, f"cosine(single, batched)={sim:.6f}"


@pytest.mark.smoke
def test_chat_say_hi_returns_nonempty_string() -> None:
    response = get_chat().invoke([HumanMessage("say hi in one word")])
    text = response.content if hasattr(response, "content") else str(response)
    assert isinstance(text, str)
    assert text.strip() != ""
