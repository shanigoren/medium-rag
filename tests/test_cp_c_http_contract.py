"""Checkpoint C -- HTTP contract (integrates C11 + C12 + C1). FREE, no --smoke.

The CP-C integration gate: the whole RAG loop behind the real ASGI app, validating
the wire format byte-for-byte. Unlike `test_api.py` (which stubs `answer`), this
runs the REAL chain composition (C10 rewrite -> C8 retrieve -> C9 prompt -> C11
answer -> to_api_dict -> C12 HTTP) and fakes only the three money edges (LLM,
embeddings, Pinecone), so it proves the components still interlock through HTTP at
ZERO cost. The contract first meets the real services at CP-F (the live URL).

Fixtures (all from conftest.py): `fake_pc` (in-memory Pinecone), `fake_query_embeddings`
(stub query vector in the retriever), `fake_chat` (the rewriter LLM), `fake_chain_chat`
(the answer LLM), and `cfg` (supplies the dummy env so the chain's internal
load_config() succeeds offline; its pinecone_api_key keys the cached fake client).

Select as a gate with `-k cp_c`.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

import api.index as api_index
from src.rag import vectorstore


def _m(article_id, score, chunk_idx, *, title="Some Title", chunk="some passage text",
       authors=("Author A",), tags=("Topic",)):
    """One fake Pinecone hit shaped like an SDK match, carrying the C7 metadata
    schema C8's `_to_chunk` reads."""
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


def _seed(fake_pc, cfg, matches):
    """Construct + cache the fake client (keyed on cfg.pinecone_api_key, the same
    key the chain's internal load_config produces) and set its query_response, so
    the real retriever reads these matches."""
    vectorstore._client(cfg.pinecone_api_key)
    fake_pc[0]._index.query_response = {"matches": list(matches)}


def test_cp_c_prompt_wire_contract(fake_pc, fake_query_embeddings, fake_chat, fake_chain_chat, cfg):
    """Full loop behind HTTP, real chain, faked edges. Body keys EXACTLY
    {'response','context','Augmented_prompt'} (capital-A); inner {'System','User'};
    every context row EXACTLY {'article_id','title','chunk','score'} with
    article_id:str, score:float; response == the canned answer; context non-empty."""
    _seed(fake_pc, cfg, [_m("3", 0.91, 0), _m("8", 0.77, 1), _m("12", 0.55, 0)])
    fake_chat.set_content('{"query": "some topic", "dedup": false}')
    fake_chain_chat.set_content("Canned grounded answer.")

    client = TestClient(api_index.app)
    resp = client.post("/api/prompt", json={"question": "What is one article about?"})
    assert resp.status_code == 200
    body = resp.json()

    assert set(body) == {"response", "context", "Augmented_prompt"}
    assert "augmented_prompt" not in body
    assert set(body["Augmented_prompt"]) == {"System", "User"}
    assert body["response"] == "Canned grounded answer."
    assert body["context"], "context should be non-empty for a seeded query"
    for row in body["context"]:
        assert set(row) == {"article_id", "title", "chunk", "score"}
        assert isinstance(row["article_id"], str)
        assert isinstance(row["score"], float)


def test_cp_c_augmented_prompt_is_original_question(fake_pc, fake_query_embeddings, fake_chat, fake_chain_chat, cfg):
    """Cross-cutting invariant at the HTTP layer. The fake rewriter returns a query
    that DIFFERS from the question; Augmented_prompt['User'] CONTAINS the original
    question verbatim and does NOT contain the rewritten query (deterministic here
    because the rewrite is fixed by the fake)."""
    _seed(fake_pc, cfg, [_m("3", 0.91, 0)])
    fake_chat.set_content('{"query": "DIFFERENT_REWRITE", "dedup": false}')
    fake_chain_chat.set_content("Canned answer.")

    original = "I want beginner-friendly advice on gardening. Which article do you recommend, and why?"
    client = TestClient(api_index.app)
    body = client.post("/api/prompt", json={"question": original}).json()

    user_prompt = body["Augmented_prompt"]["User"]
    assert original in user_prompt
    assert "DIFFERENT_REWRITE" not in user_prompt


def test_cp_c_list3_returns_distinct_articles(fake_pc, fake_query_embeddings, fake_chat, fake_chain_chat, cfg):
    """Type-2 dedupe end-to-end. The fake rewriter returns dedup=true; the seeded
    matches span 3 distinct article_ids (with a duplicate to prove collapse), so a
    'list 3 articles' POST yields a context with >= 3 DISTINCT article_ids."""
    _seed(fake_pc, cfg, [
        _m("10", 0.95, 0),
        _m("10", 0.90, 1),   # duplicate of 10 -> must collapse
        _m("11", 0.80, 0),
        _m("12", 0.70, 0),
    ])
    fake_chat.set_content('{"query": "education", "dedup": true}')
    fake_chain_chat.set_content("1. A\n2. B\n3. C")

    client = TestClient(api_index.app)
    body = client.post(
        "/api/prompt",
        json={"question": "List exactly 3 articles about education. Return only the titles."},
    ).json()

    distinct = {row["article_id"] for row in body["context"]}
    assert len(distinct) >= 3, f"expected >= 3 distinct articles, got {sorted(distinct)}"


def test_cp_c_stats_reflects_committed_config(cfg):
    """GET /api/stats -> EXACTLY {'chunk_size','overlap_ratio','top_k'} equal to the
    committed config.yaml (768 / 0.10 / 20). No LLM/Pinecone touched. CP-F re-runs
    this identical assertion against the live Vercel URL."""
    client = TestClient(api_index.app)
    body = client.get("/api/stats").json()
    assert body == {"chunk_size": 768, "overlap_ratio": 0.10, "top_k": 20}
