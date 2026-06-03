"""Offline unit tests for Component 10 (`src.rag.query_writer`).

Run on every `pytest`; ZERO network. The `fake_chat` fixture (conftest) patches
`src.rag.query_writer.get_chat` with a recording `_StubChat`; set its `.content`
(or `.set_error(...)`) per test to drive `rewrite_query`.

Two groups:
  - stub tests: happy path, dedup parsing/coercion, fence stripping, the messages
    the model receives, the forced reasoning_effort, and every fallback branch.
  - replay-bank tests: parametrized over the committed real-capture recordings
    (`tests/fixtures/rewriter_recordings.json`) -- the dedup-classification audit
    on realistic model output, still entirely offline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.rag.query_writer import REWRITER_SYSTEM_PROMPT, RewriteResult, rewrite_query


# --- happy path & parsing --------------------------------------------------


def test_happy_path_returns_query_and_dedup(fake_chat, cfg):
    fake_chat.set_content('{"query": "education", "dedup": true}')
    result = rewrite_query("List 3 articles about education", cfg)
    assert isinstance(result, RewriteResult)
    assert result.query == "education"
    assert result.dedup is True


def test_dedup_false_parsed(fake_chat, cfg):
    fake_chat.set_content('{"query": "remote work", "dedup": false}')
    result = rewrite_query("Summarise an article about remote work", cfg)
    assert result.dedup is False
    assert isinstance(result.dedup, bool)


def test_strips_json_code_fence(fake_chat, cfg):
    fake_chat.set_content('```json\n{"query": "education", "dedup": true}\n```')
    result = rewrite_query("List 3 articles about education", cfg)
    assert result.query == "education"
    assert result.dedup is True


def test_strips_bare_code_fence(fake_chat, cfg):
    fake_chat.set_content('```\n{"query": "education", "dedup": true}\n```')
    result = rewrite_query("List 3 articles about education", cfg)
    assert result.query == "education"


def test_dedup_string_true_coerced_to_bool(fake_chat, cfg):
    fake_chat.set_content('{"query": "x ok", "dedup": "true"}')
    result = rewrite_query("question", cfg)
    assert result.dedup is True


def test_dedup_string_false_coerced(fake_chat, cfg):
    fake_chat.set_content('{"query": "x ok", "dedup": "FALSE"}')
    result = rewrite_query("question", cfg)
    assert result.dedup is False


# --- what the model is asked ----------------------------------------------


def test_question_passed_to_model_in_human_turn(fake_chat, cfg):
    question = "List exactly 3 articles about education. Return only the titles."
    fake_chat.set_content('{"query": "education", "dedup": true}')
    rewrite_query(question, cfg)
    messages = fake_chat.calls[0]
    assert messages[-1].content == question  # human turn = the ORIGINAL question


def test_system_prompt_present(fake_chat, cfg):
    fake_chat.set_content('{"query": "education", "dedup": true}')
    rewrite_query("List 3 articles about education", cfg)
    system_text = fake_chat.calls[0][0].content
    assert "vector database of Medium articles" in system_text
    assert '"dedup"' in system_text  # braces un-escaped after templating
    # sanity: the source constant kept its escaped braces
    assert '{{"query"' in REWRITER_SYSTEM_PROMPT


def test_reasoning_effort_minimal_requested(fake_chat, cfg):
    """get_chat is called with reasoning_effort='minimal' (NOT cfg.reasoning_effort,
    which is 'low'), regardless of the config value."""
    assert cfg.reasoning_effort == "low"
    fake_chat.set_content('{"query": "education", "dedup": true}')
    rewrite_query("List 3 articles about education", cfg)
    assert fake_chat.reasoning_efforts == ["minimal"]


# --- fallback branches: all -> RewriteResult(question, dedup=True) ----------


def test_fallback_on_model_exception(fake_chat, cfg):
    fake_chat.set_error(RuntimeError("proxy 500"))
    result = rewrite_query("the original question", cfg)
    assert result.query == "the original question"
    assert result.dedup is True


def test_fallback_on_empty_output(fake_chat, cfg):
    fake_chat.set_content("")
    result = rewrite_query("the original question", cfg)
    assert result == RewriteResult(query="the original question", dedup=True)


def test_fallback_on_non_json(fake_chat, cfg):
    fake_chat.set_content("here is your query: education")
    result = rewrite_query("the original question", cfg)
    assert result == RewriteResult(query="the original question", dedup=True)


def test_fallback_on_non_object_json(fake_chat, cfg):
    fake_chat.set_content('["education", true]')
    result = rewrite_query("the original question", cfg)
    assert result == RewriteResult(query="the original question", dedup=True)


def test_fallback_on_missing_query_key(fake_chat, cfg):
    fake_chat.set_content('{"dedup": true}')
    result = rewrite_query("the original question", cfg)
    assert result == RewriteResult(query="the original question", dedup=True)


def test_fallback_on_missing_dedup_key(fake_chat, cfg):
    fake_chat.set_content('{"query": "education"}')
    result = rewrite_query("the original question", cfg)
    assert result == RewriteResult(query="the original question", dedup=True)


def test_fallback_on_too_short_query(fake_chat, cfg):
    fake_chat.set_content('{"query": "ed", "dedup": false}')
    result = rewrite_query("the original question", cfg)
    assert result.query == "the original question"
    assert result.dedup is True


def test_fallback_on_non_bool_non_string_dedup(fake_chat, cfg):
    fake_chat.set_content('{"query": "education ok", "dedup": 1}')
    result = rewrite_query("the original question", cfg)
    assert result == RewriteResult(query="the original question", dedup=True)


def test_fallback_on_non_string_query(fake_chat, cfg):
    fake_chat.set_content('{"query": 42, "dedup": false}')
    result = rewrite_query("the original question", cfg)
    assert result == RewriteResult(query="the original question", dedup=True)


# --- config resolution -----------------------------------------------------


def test_cfg_none_loads_default_config(fake_chat, cfg):
    """rewrite_query(..., cfg=None) exercises the load_config() branch. The `cfg`
    fixture has already set the required env vars, so load_config() succeeds."""
    fake_chat.set_content('{"query": "education ok", "dedup": false}')
    result = rewrite_query("some question", cfg=None)
    assert result.query == "education ok"
    assert result.dedup is False


# --- replay bank (parametrized over the real-capture recordings) -----------

_RECORDINGS = json.loads(
    (Path(__file__).parent / "fixtures" / "rewriter_recordings.json").read_text(
        encoding="utf-8"
    )
)


@pytest.mark.parametrize(
    "entry",
    _RECORDINGS,
    ids=[f"type{e['type']}-{e['question'][:32]}" for e in _RECORDINGS],
)
def test_recording_parses_and_matches_expected_dedup(entry, fake_chat, cfg):
    """Replay each captured response through the stub: it must parse (not fall
    back), the query must be non-empty, and the parsed dedup must equal the
    human-authored expected_dedup. This is the dedup-classification audit and
    the parser robustness check running offline on realistic model output.

    (We deliberately do NOT assert query length -- such a check is brittle. The
    prompt instructs a focused, un-augmented query; what we assert here is that
    it parsed and classified correctly.)"""
    question = entry["question"]
    fake_chat.set_content(entry["recorded_response"])
    result = rewrite_query(question, cfg)
    assert result.query != question, "recording fell back -> parse failed"
    assert result.query.strip() != ""
    assert result.dedup == entry["expected_dedup"]


def _tokens(text: str) -> set[str]:
    """Lowercase alphanumeric tokens with crude singularization (strip a trailing
    's' on words >3 chars), so 'articles'/'article' and 'habits'/'habit' match."""
    toks = re.findall(r"[a-z0-9]+", text.lower())
    return {t[:-1] if len(t) > 3 and t.endswith("s") else t for t in toks}


@pytest.mark.parametrize(
    "entry",
    _RECORDINGS,
    ids=[f"type{e['type']}-{e['question'][:32]}" for e in _RECORDINGS],
)
def test_recording_query_stays_close_to_question(entry, fake_chat, cfg):
    """No-augmentation guardrail: the rewritten query must use the question's OWN
    words, not invented synonyms/keywords. We require >=80% of the query's tokens
    to appear in the question (lowercased + crude singularization). Catches
    keyword augmentation / topic drift while tolerating minor morphology and the
    odd connective. The 'stay close to the original wording' contract from the
    prompt, asserted on real captured output."""
    fake_chat.set_content(entry["recorded_response"])
    result = rewrite_query(entry["question"], cfg)
    q_tokens = _tokens(result.query)
    question_tokens = _tokens(entry["question"])
    assert q_tokens, "query produced no tokens"
    extra = q_tokens - question_tokens
    ratio = len(q_tokens & question_tokens) / len(q_tokens)
    assert ratio >= 0.8, (
        f"query drifted from the question (only {ratio:.0%} of query words are in "
        f"the question); added words: {sorted(extra)}; query={result.query!r}"
    )


def test_recordings_bank_is_well_formed():
    """Guard the fixture itself: a meaningful number of entries, all four
    assignment question types present, required fields typed correctly."""
    assert len(_RECORDINGS) >= 12
    types = {e["type"] for e in _RECORDINGS}
    assert types == {1, 2, 3, 4}
    for e in _RECORDINGS:
        assert isinstance(e["question"], str) and e["question"]
        assert isinstance(e["expected_dedup"], bool)
        assert isinstance(e["recorded_response"], str) and e["recorded_response"]
