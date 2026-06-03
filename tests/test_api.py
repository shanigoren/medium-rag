"""Component 12 -- API wrapper tests (offline, free).

Tests the FastAPI layer in ISOLATION: `api.index.answer` is stubbed, so the chain
never runs and nothing touches LLMod.AI / Pinecone. We assert the wrapper's four
jobs -- routing, body validation, error mapping, and byte-for-byte serialization
of C11's `to_api_dict()`.

The real chain composition behind the app is exercised (still for free, with the
money boundaries faked) by `tests/test_cp_c_http_contract.py` -- the CP-C gate.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

import api.index as api_index
from src.rag.chain import AnswerResult
from src.rag.query_writer import RewriteResult

# A token that appears ONLY in the stubbed rewrite query, never in the question or
# the augmented prompt -- so a test can prove the rewrite never reaches the wire.
_REWRITE_ONLY = "REWRITE_ONLY_TOKEN"


def _stub_result(question: str = "the original question") -> AnswerResult:
    """A representative AnswerResult. `context` carries a duplicate article_id to
    prove the wrapper neither collapses nor reorders rows, and `rewrite` carries a
    sentinel query to prove it is dropped from the wire dict."""
    return AnswerResult(
        response="A test answer.",
        context=[
            {"article_id": "7", "title": "T-seven", "chunk": "body one", "score": 0.83},
            {"article_id": "7", "title": "T-seven", "chunk": "body two", "score": 0.71},
        ],
        augmented_prompt={"System": "SYS prompt", "User": f"Question: {question}\n\n(context)"},
        rewrite=RewriteResult(query=_REWRITE_ONLY, dedup=False),
    )


@pytest.fixture
def client() -> TestClient:
    """A TestClient over the real app. Server exceptions re-raise by default so a
    genuine bug surfaces loudly; the 500-mapping test builds its own client with
    `raise_server_exceptions=False`."""
    return TestClient(api_index.app)


def _patch_answer(monkeypatch, fn) -> list:
    """Patch `api.index.answer` with `fn` and return a list recording each call's
    positional args (so a test can assert the question passed through)."""
    calls: list = []

    def _recorder(*args, **kwargs):
        calls.append(args)
        return fn(*args, **kwargs)

    monkeypatch.setattr(api_index, "answer", _recorder)
    return calls


# ---------- /api/prompt -- shape & casing (the wire contract) ----------------


def test_prompt_returns_200_and_exact_wire_dict(client, monkeypatch):
    """POST a question -> 200; body == stub.to_api_dict() (deep equality)."""
    _patch_answer(monkeypatch, lambda q, *a, **k: _stub_result(q))
    resp = client.post("/api/prompt", json={"question": "what is X about?"})
    assert resp.status_code == 200
    assert resp.json() == _stub_result("what is X about?").to_api_dict()


def test_prompt_augmented_prompt_capital_a_and_inner_keys(client, monkeypatch):
    """Body has 'Augmented_prompt' (capital A), NOT 'augmented_prompt'; the inner
    dict keys are exactly {'System','User'}."""
    _patch_answer(monkeypatch, lambda q, *a, **k: _stub_result(q))
    body = client.post("/api/prompt", json={"question": "q"}).json()
    assert "Augmented_prompt" in body
    assert "augmented_prompt" not in body
    assert set(body["Augmented_prompt"]) == {"System", "User"}


def test_prompt_context_rows_have_exactly_four_keys(client, monkeypatch):
    """Every context row's key set == {'article_id','title','chunk','score'} -- no
    authors/chunk_idx/rank/dedup leak; article_id is a str, score a float."""
    _patch_answer(monkeypatch, lambda q, *a, **k: _stub_result(q))
    body = client.post("/api/prompt", json={"question": "q"}).json()
    assert body["context"]
    for row in body["context"]:
        assert set(row) == {"article_id", "title", "chunk", "score"}
        assert isinstance(row["article_id"], str)
        assert isinstance(row["score"], float)


def test_prompt_preserves_context_order_and_duplicate_ids(client, monkeypatch):
    """The two stub rows (both article_id '7') survive in order, uncollapsed."""
    _patch_answer(monkeypatch, lambda q, *a, **k: _stub_result(q))
    body = client.post("/api/prompt", json={"question": "q"}).json()
    assert [r["article_id"] for r in body["context"]] == ["7", "7"]
    assert [r["chunk"] for r in body["context"]] == ["body one", "body two"]


def test_prompt_passes_original_question_to_answer(client, monkeypatch):
    """The recording stub received req.question verbatim (the API does not mutate
    the question before handing it to the chain)."""
    calls = _patch_answer(monkeypatch, lambda q, *a, **k: _stub_result(q))
    client.post("/api/prompt", json={"question": "  Keep   me  verbatim  "})
    assert calls and calls[0][0] == "  Keep   me  verbatim  "


def test_prompt_no_rewrite_or_dedup_on_the_wire(client, monkeypatch):
    """The wire body excludes the rewritten query and 'dedup' even though
    AnswerResult.rewrite carries them."""
    _patch_answer(monkeypatch, lambda q, *a, **k: _stub_result(q))
    raw = client.post("/api/prompt", json={"question": "q"}).text
    body = json.loads(raw)
    assert _REWRITE_ONLY not in raw
    assert "dedup" not in body
    assert "rewrite" not in body


# ---------- /api/prompt -- request validation & error mapping ----------------


def test_prompt_missing_question_field_returns_422(client, monkeypatch):
    """POST {} -> 422 (pydantic); answer is never called."""
    calls = _patch_answer(monkeypatch, lambda q, *a, **k: _stub_result(q))
    assert client.post("/api/prompt", json={}).status_code == 422
    assert calls == []


def test_prompt_non_string_question_returns_422(client, monkeypatch):
    """POST {"question": 123} -> 422 (pydantic does not coerce int->str); answer
    is never called."""
    calls = _patch_answer(monkeypatch, lambda q, *a, **k: _stub_result(q))
    assert client.post("/api/prompt", json={"question": 123}).status_code == 422
    assert calls == []


def test_prompt_non_json_body_returns_422(client):
    """A non-JSON body -> 422 (FastAPI cannot build PromptRequest)."""
    resp = client.post(
        "/api/prompt",
        content=b"not json at all",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422


def test_prompt_value_error_maps_to_400(client, monkeypatch):
    """Stub answer raises ValueError -> 400, detail carries the message (the
    empty/whitespace-question path)."""
    def _raise(*a, **k):
        raise ValueError("question must be a non-empty, non-whitespace string")

    monkeypatch.setattr(api_index, "answer", _raise)
    resp = client.post("/api/prompt", json={"question": "   "})
    assert resp.status_code == 400
    assert "non-empty" in resp.json()["detail"]


def test_prompt_unexpected_error_maps_to_500(monkeypatch):
    """Stub answer raises RuntimeError -> 500 (not swallowed, not a 200-with-empty
    answer). Uses raise_server_exceptions=False so the 500 comes back as a
    response rather than re-raising into the test."""
    def _raise(*a, **k):
        raise RuntimeError("proxy down")

    monkeypatch.setattr(api_index, "answer", _raise)
    client = TestClient(api_index.app, raise_server_exceptions=False)
    resp = client.post("/api/prompt", json={"question": "q"})
    assert resp.status_code == 500


# ---------- /api/stats -- exact 3-key schema, live config --------------------


def test_stats_returns_exactly_three_keys(client, cfg):
    """GET /api/stats -> 200; key set == {'chunk_size','overlap_ratio','top_k'} --
    no dataclass field leaks. (`cfg` supplies the env load_config() needs.)"""
    body = client.get("/api/stats").json()
    assert set(body) == {"chunk_size", "overlap_ratio", "top_k"}


def test_stats_values_and_types_match_config(client, cfg):
    """Against the repo config.yaml: chunk_size==768 (int), overlap_ratio==0.10
    (float), top_k==5 (int)."""
    body = client.get("/api/stats").json()
    assert body["chunk_size"] == 768 and isinstance(body["chunk_size"], int)
    assert body["overlap_ratio"] == 0.10 and isinstance(body["overlap_ratio"], float)
    assert body["top_k"] == 5 and isinstance(body["top_k"], int)


def test_stats_reflects_live_config_override(client, cfg, monkeypatch):
    """monkeypatch TOP_K=8 -> GET /api/stats top_k == 8. Proves the endpoint reads
    config fresh each call (assignment 'must always reflect current values'), not a
    cached import-time snapshot."""
    monkeypatch.setenv("TOP_K", "8")
    assert client.get("/api/stats").json()["top_k"] == 8
