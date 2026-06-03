"""Offline unit tests for Component 11 (the Chain).

Zero network. We stub the three boundaries the chain composes -- `rewrite_query`,
`retrieve`, `get_chat` -- by patching the names as imported into `src.rag.chain`,
and test the chain's COMPOSITION only (each underlying component has its own
suite). The chat boundary uses the shared `fake_chain_chat` fixture (conftest).
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from src.prompts import SYSTEM_PROMPT
from src.rag.chain import AnswerResult, answer
from src.rag.query_writer import RewriteResult
from src.rag.retriever import RetrievedChunk


def _chunk(
    article_id="7",
    title="A Title",
    authors=("Jane Doe",),
    chunk="raw chunk body",
    score=0.9,
    chunk_idx=0,
    rank=1,
    tags=("Topic",),
) -> RetrievedChunk:
    return RetrievedChunk(
        article_id=article_id,
        title=title,
        authors=list(authors),
        tags=list(tags),
        chunk=chunk,
        score=score,
        chunk_idx=chunk_idx,
        rank=rank,
    )


def _install(monkeypatch, *, rewrite, chunks=None, retrieve_raises=None):
    """Stub `src.rag.chain.rewrite_query` and `.retrieve`; record their call args.

    Returns a dict that fills with `calls['rewrite']` / `calls['retrieve']` so a
    test can assert exactly what the chain forwarded.
    """
    calls: dict = {}

    def fake_rewrite(question, cfg=None):
        calls["rewrite"] = {"question": question, "cfg": cfg}
        return rewrite

    def fake_retrieve(query, namespace, cfg=None, *, top_k=None, fetch_k=None, dedup=True):
        calls["retrieve"] = {
            "query": query,
            "namespace": namespace,
            "cfg": cfg,
            "dedup": dedup,
        }
        if retrieve_raises is not None:
            raise retrieve_raises
        return list(chunks) if chunks is not None else []

    monkeypatch.setattr("src.rag.chain.rewrite_query", fake_rewrite)
    monkeypatch.setattr("src.rag.chain.retrieve", fake_retrieve)
    return calls


# --------------------------------------------------------------------------- #
# Mainstream wiring & shape
# --------------------------------------------------------------------------- #


def test_answer_returns_response_context_augmented_prompt(monkeypatch, fake_chain_chat, cfg):
    _install(monkeypatch, rewrite=RewriteResult("habits", False), chunks=[_chunk()])
    fake_chain_chat.set_content("the answer")
    res = answer("How do habits form?", cfg)
    assert isinstance(res, AnswerResult)
    assert res.response == "the answer"
    assert isinstance(res.context, list) and len(res.context) == 1
    assert set(res.augmented_prompt) == {"System", "User"}


def test_retrieve_called_with_rewritten_query_and_dedup(monkeypatch, fake_chain_chat, cfg):
    calls = _install(monkeypatch, rewrite=RewriteResult("REWRITE_TOKEN", True), chunks=[])
    fake_chain_chat.set_content("x")
    answer("the long original question", cfg)
    assert calls["retrieve"]["query"] == "REWRITE_TOKEN"  # the rewrite, not the original
    assert calls["retrieve"]["dedup"] is True


def test_retrieve_called_with_default_namespace(monkeypatch, fake_chain_chat, cfg):
    calls = _install(monkeypatch, rewrite=RewriteResult("q", False), chunks=[])
    fake_chain_chat.set_content("x")
    answer("a question", cfg)
    assert calls["retrieve"]["namespace"] == cfg.pinecone_namespace


def test_namespace_override_forwarded_to_retrieve(monkeypatch, fake_chain_chat, cfg):
    calls = _install(monkeypatch, rewrite=RewriteResult("q", False), chunks=[])
    fake_chain_chat.set_content("x")
    answer("a question", cfg, namespace="smoke")
    assert calls["retrieve"]["namespace"] == "smoke"


def test_dedup_true_forwarded(monkeypatch, fake_chain_chat, cfg):
    calls = _install(monkeypatch, rewrite=RewriteResult("q", True), chunks=[])
    fake_chain_chat.set_content("x")
    answer("list 3 articles about education", cfg)
    assert calls["retrieve"]["dedup"] is True


def test_dedup_false_forwarded(monkeypatch, fake_chain_chat, cfg):
    calls = _install(monkeypatch, rewrite=RewriteResult("q", False), chunks=[])
    fake_chain_chat.set_content("x")
    answer("summarise the article on X", cfg)
    assert calls["retrieve"]["dedup"] is False


def test_chat_invoked_with_system_and_human_messages(monkeypatch, fake_chain_chat, cfg):
    _install(monkeypatch, rewrite=RewriteResult("q", False), chunks=[_chunk()])
    fake_chain_chat.set_content("x")
    res = answer("a question", cfg)
    assert len(fake_chain_chat.calls) == 1  # invoked exactly once
    msgs = fake_chain_chat.calls[0]
    assert len(msgs) == 2
    assert isinstance(msgs[0], SystemMessage) and msgs[0].content == SYSTEM_PROMPT
    assert isinstance(msgs[1], HumanMessage)
    assert msgs[1].content == res.augmented_prompt["User"]


def test_to_api_dict_wire_shape_and_casing(monkeypatch, fake_chain_chat, cfg):
    _install(monkeypatch, rewrite=RewriteResult("q", False), chunks=[_chunk()])
    fake_chain_chat.set_content("the answer")
    res = answer("a question", cfg)
    api = res.to_api_dict()
    assert list(api) == ["response", "context", "Augmented_prompt"]
    assert api["response"] == res.response
    assert api["context"] is res.context
    assert set(api["Augmented_prompt"]) == {"System", "User"}


# --------------------------------------------------------------------------- #
# Cross-cutting invariant: original question, both directions
# --------------------------------------------------------------------------- #


def test_augmented_prompt_uses_original_question_not_rewrite(monkeypatch, fake_chain_chat, cfg):
    original = "the long original question about education systems"
    _install(monkeypatch, rewrite=RewriteResult("REWRITE_TOKEN", False), chunks=[_chunk()])
    fake_chain_chat.set_content("x")
    res = answer(original, cfg)
    user = res.augmented_prompt["User"]
    assert original in user
    assert "REWRITE_TOKEN" not in user


def test_dedup_and_rewrite_not_in_api_dict(monkeypatch, fake_chain_chat, cfg):
    """The wire dict (to_api_dict) excludes the rewritten query and dedup even
    though AnswerResult.rewrite carries them (next test)."""
    _install(monkeypatch, rewrite=RewriteResult("REWRITE_TOKEN", True), chunks=[_chunk()])
    fake_chain_chat.set_content("the answer")
    res = answer("a question", cfg)
    serialized = json.dumps(res.to_api_dict())
    assert "REWRITE_TOKEN" not in serialized
    assert "dedup" not in serialized


def test_rewrite_exposed_on_result_for_debug_and_eval(monkeypatch, fake_chain_chat, cfg):
    """AnswerResult.rewrite is the EXACT RewriteResult the answer was built from
    (so the demo/eval see the actually-used query+dedup, not a re-call)."""
    rw = RewriteResult("REWRITE_TOKEN", True)
    _install(monkeypatch, rewrite=rw, chunks=[_chunk()])
    fake_chain_chat.set_content("the answer")
    res = answer("a question", cfg)
    assert res.rewrite is rw
    assert res.rewrite.query == "REWRITE_TOKEN"
    assert res.rewrite.dedup is True


# --------------------------------------------------------------------------- #
# cfg threading (budget-critical: protects the Phase B/C sweeps)
# --------------------------------------------------------------------------- #


def test_cfg_threaded_to_retrieve_and_get_chat_and_rewrite(monkeypatch, fake_chain_chat, cfg):
    calls = _install(monkeypatch, rewrite=RewriteResult("q", False), chunks=[])
    fake_chain_chat.set_content("x")
    answer("a question", cfg)
    assert calls["rewrite"]["cfg"] is cfg
    assert calls["retrieve"]["cfg"] is cfg
    assert fake_chain_chat.cfgs[0] is cfg


def test_chat_uses_cfg_reasoning_effort_not_minimal(monkeypatch, fake_chain_chat, cfg):
    _install(monkeypatch, rewrite=RewriteResult("q", False), chunks=[])
    fake_chain_chat.set_content("x")
    answer("a question", cfg)
    # No reasoning_effort override -> get_chat uses cfg.reasoning_effort, not "minimal".
    assert fake_chain_chat.reasoning_efforts == [None]


def test_cfg_none_loads_default_config(monkeypatch, fake_chain_chat, cfg):
    monkeypatch.setattr("src.rag.chain.load_config", lambda: cfg)
    calls = _install(monkeypatch, rewrite=RewriteResult("q", False), chunks=[])
    fake_chain_chat.set_content("x")
    answer("a real question", cfg=None)
    assert calls["rewrite"]["cfg"] is cfg
    assert calls["retrieve"]["cfg"] is cfg
    assert fake_chain_chat.cfgs[0] is cfg


# --------------------------------------------------------------------------- #
# Context projection
# --------------------------------------------------------------------------- #


def test_context_rows_have_exactly_four_keys(monkeypatch, fake_chain_chat, cfg):
    _install(monkeypatch, rewrite=RewriteResult("q", False), chunks=[_chunk(), _chunk(article_id="8")])
    fake_chain_chat.set_content("x")
    res = answer("a question", cfg)
    for row in res.context:
        assert set(row) == {"article_id", "title", "chunk", "score"}


def test_context_rows_values_and_types(monkeypatch, fake_chain_chat, cfg):
    chunks = [
        _chunk(article_id=7, title="First", chunk="body one", score=1, chunk_idx=3),  # int id/score
        _chunk(article_id="8", title="Second", chunk="body two", score=0.42),
    ]
    _install(monkeypatch, rewrite=RewriteResult("q", False), chunks=chunks)
    fake_chain_chat.set_content("x")
    res = answer("a question", cfg)
    assert [r["article_id"] for r in res.context] == ["7", "8"]  # order preserved, str-coerced
    assert isinstance(res.context[0]["article_id"], str)
    assert isinstance(res.context[0]["score"], float) and res.context[0]["score"] == 1.0
    assert res.context[0]["chunk"] == "body one"  # raw passthrough


def test_context_keeps_duplicate_article_ids_under_dedup_false(monkeypatch, fake_chain_chat, cfg):
    chunks = [
        _chunk(article_id="7", chunk_idx=0),
        _chunk(article_id="7", chunk_idx=1),
        _chunk(article_id="8", chunk_idx=0),
    ]
    _install(monkeypatch, rewrite=RewriteResult("q", False), chunks=chunks)
    fake_chain_chat.set_content("x")
    res = answer("summarise the article", cfg)
    assert [r["article_id"] for r in res.context] == ["7", "7", "8"]  # not collapsed


def test_context_empty_when_no_chunks(monkeypatch, fake_chain_chat, cfg):
    _install(monkeypatch, rewrite=RewriteResult("q", True), chunks=[])
    fake_chain_chat.set_content("I don't know based on the provided Medium articles data.")
    res = answer("a question with no hits", cfg)
    assert res.context == []
    assert "(no relevant context retrieved)" in res.augmented_prompt["User"]
    assert len(fake_chain_chat.calls) == 1  # chat still called once
    assert res.response  # a response is still returned


# --------------------------------------------------------------------------- #
# Response extraction
# --------------------------------------------------------------------------- #


def test_response_text_plain_string_stripped(monkeypatch, fake_chain_chat, cfg):
    _install(monkeypatch, rewrite=RewriteResult("q", False), chunks=[])
    fake_chain_chat.set_content("  the answer\n")
    res = answer("a question", cfg)
    assert res.response == "the answer"


def test_response_text_joins_list_content(monkeypatch, fake_chain_chat, cfg):
    _install(monkeypatch, rewrite=RewriteResult("q", False), chunks=[])
    fake_chain_chat.set_content([{"type": "text", "text": "Part A "}, {"type": "text", "text": "Part B"}])
    res = answer("a question", cfg)
    assert res.response == "Part A Part B"


# --------------------------------------------------------------------------- #
# Edge cases / failure policy
# --------------------------------------------------------------------------- #


def test_empty_question_raises_valueerror(monkeypatch, fake_chain_chat, cfg):
    calls = _install(monkeypatch, rewrite=RewriteResult("q", False), chunks=[])
    fake_chain_chat.set_content("x")
    with pytest.raises(ValueError):
        answer("   ", cfg)
    assert calls == {}  # rewrite / retrieve never called
    assert fake_chain_chat.calls == []  # chat never called


def test_chat_exception_propagates(monkeypatch, fake_chain_chat, cfg):
    _install(monkeypatch, rewrite=RewriteResult("q", False), chunks=[])
    fake_chain_chat.set_error(RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        answer("a question", cfg)


def test_retrieve_valueerror_propagates(monkeypatch, fake_chain_chat, cfg):
    _install(monkeypatch, rewrite=RewriteResult("q", False), retrieve_raises=ValueError("bad query"))
    fake_chain_chat.set_content("x")
    with pytest.raises(ValueError, match="bad query"):
        answer("a valid question", cfg)
