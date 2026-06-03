"""Offline unit / "snapshot" tests for Component 9 (`src.prompts`).

Run on every `pytest`; ZERO network, ZERO fixtures from conftest (no
Pinecone/LLM). Chunk fakes are built inline as a tiny local dataclass exposing
the four duck-typed attributes (.article_id/.title/.authors/.chunk). Direct
string assertions -- the project's "snapshot" convention (no snapshot library).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.prompts import SYSTEM_PROMPT, build_augmented_prompt, render_user_prompt


@dataclass
class _FakeChunk:
    """Minimal stand-in for RetrievedChunk -- only the attributes C9 reads.

    `tags` is last (after `chunk`) so existing positional constructions
    `_FakeChunk(id, title, authors, chunk)` keep working unchanged."""

    article_id: str
    title: str
    authors: list = field(default_factory=list)
    chunk: str = ""
    tags: list = field(default_factory=list)


# The mandatory two-paragraph block, ASCII-normalized, exactly as it must appear
# verbatim inside SYSTEM_PROMPT.
_MANDATORY = (
    "You are a Medium-article assistant that answers questions strictly and only based on "
    "the Medium articles dataset context provided to you (metadata and article passages). "
    "You must not use any external knowledge, the open internet, or information that is not "
    "explicitly contained in the retrieved context. If the answer cannot be determined from "
    'the provided context, respond: "I don\'t know based on the provided Medium articles data."'
    "\n\n"
    "Always explain your answer using the given context, quoting or paraphrasing the relevant "
    "article passage or metadata when helpful."
)


# ---------- system prompt ------------------------------------------------


def test_system_prompt_contains_mandatory_text_verbatim():
    """The full mandatory two-paragraph block (ASCII form) is a substring of
    SYSTEM_PROMPT, character-for-character."""
    assert _MANDATORY in SYSTEM_PROMPT


def test_system_prompt_contains_idk_sentinel():
    """The exact sentence (ASCII apostrophe) is present."""
    assert "I don't know based on the provided Medium articles data." in SYSTEM_PROMPT


def test_system_prompt_has_style_appendix():
    """Style-guidance markers present AFTER the mandatory block -- clarifications
    added, constraints kept."""
    appendix_start = SYSTEM_PROMPT.index("Style guidance")
    mandatory_end = SYSTEM_PROMPT.index(_MANDATORY) + len(_MANDATORY)
    assert appendix_start > mandatory_end
    appendix = SYSTEM_PROMPT[appendix_start:]
    assert "list" in appendix
    assert "summari" in appendix
    assert "recommend" in appendix


def test_system_prompt_is_ascii():
    """SYSTEM_PROMPT.encode('ascii') does not raise -- locks the
    quote-normalization decision and cp1252 safety."""
    SYSTEM_PROMPT.encode("ascii")


def test_system_prompt_explains_tag_usage():
    """The appendix tells the model tags exist and how to use them: tags help judge
    topical relevance, but facts/summaries/justifications come from the passage
    (so tags don't erode grounding)."""
    appendix = SYSTEM_PROMPT[SYSTEM_PROMPT.index("Style guidance"):]
    assert "Tags" in appendix
    assert "topic" in appendix.lower()
    # grounding guard: passage is the source of substance, not tags
    assert "passage" in appendix.lower()


# ---------- user prompt --------------------------------------------------


def test_user_prompt_includes_original_question_verbatim():
    """The question string appears exactly in the rendered user prompt
    (cross-cutting invariant: original question, never a rewrite)."""
    q = "Which article would you recommend for building habits that stick, and why?"
    out = render_user_prompt(q, [_FakeChunk("1", "T", ["A"], "body")])
    assert q in out


def test_user_prompt_numbers_chunks_in_order():
    """Headers [1], [2], [3] appear, in the order the chunks were passed in
    (C9 does not re-sort)."""
    chunks = [
        _FakeChunk("a", "First", ["A"], "b1"),
        _FakeChunk("b", "Second", ["B"], "b2"),
        _FakeChunk("c", "Third", ["C"], "b3"),
    ]
    out = render_user_prompt("q", chunks)
    i1, i2, i3 = out.index("[1]"), out.index("[2]"), out.index("[3]")
    assert i1 < i2 < i3
    # numbering follows input order, not title order
    assert out.index("First") < out.index("Second") < out.index("Third")


def test_user_prompt_header_format():
    """Each chunk renders '[{n}] Title: ... | Authors: ... | Tags: ... | Article ID:
    ...' on the header line, immediately followed by the raw chunk body next line."""
    out = render_user_prompt(
        "q", [_FakeChunk("42", "My Title", ["Jane Doe"], "the body", ["Health", "Science"])]
    )
    assert (
        "[1] Title: My Title | Authors: Jane Doe | Tags: Health, Science "
        "| Article ID: 42\nthe body" in out
    )


def test_user_prompt_authors_joined():
    """Multi-author list is comma-joined ('A, B'); empty/absent authors -> 'Unknown'."""
    out_multi = render_user_prompt("q", [_FakeChunk("1", "T", ["Ann", "Bob"], "x")])
    assert "Authors: Ann, Bob" in out_multi
    out_empty = render_user_prompt("q", [_FakeChunk("1", "T", [], "x")])
    assert "Authors: Unknown" in out_empty


def test_user_prompt_tags_joined():
    """Multi-tag list is comma-joined; empty/absent tags -> 'None' (keeps the
    header shape constant, mirroring the 'Unknown' authors convention)."""
    out_multi = render_user_prompt("q", [_FakeChunk("1", "T", ["A"], "x", ["Health", "Science"])])
    assert "Tags: Health, Science" in out_multi
    out_empty = render_user_prompt("q", [_FakeChunk("1", "T", ["A"], "x", [])])
    assert "Tags: None" in out_empty


def test_user_prompt_article_id_rendered_as_string():
    """A chunk's article_id appears as its string form in the header (string
    invariant)."""
    out = render_user_prompt("q", [_FakeChunk("12345", "T", ["A"], "x")])
    assert "Article ID: 12345" in out


def test_user_prompt_chunk_body_not_prefixed():
    """The emitted body equals the input .chunk exactly -- C9 adds no 'Title:'
    prefix."""
    body = "Start absurdly small and stack the habit onto an existing routine."
    out = render_user_prompt("q", [_FakeChunk("1", "Habits", ["A"], body)])
    # header ends the line, then the body verbatim on the next line
    assert f"| Article ID: 1\n{body}" in out
    # the body itself is not prefixed with the title
    assert "Title: Habits" not in body  # sanity
    assert body in out


def test_user_prompt_empty_chunks_marker():
    """Zero chunks -> '(no relevant context retrieved)' marker present AND the
    question line still rendered, so the model can fall back to 'I don't know'."""
    q = "Anything about quantum llamas?"
    out = render_user_prompt(q, [])
    assert f"Question: {q}" in out
    assert "Context from the Medium articles dataset:" in out
    assert "(no relevant context retrieved)" in out


# ---------- build_augmented_prompt --------------------------------------


def test_build_augmented_prompt_keys_and_values():
    """Returns exactly {'System', 'User'} (capital keys); System == SYSTEM_PROMPT;
    User == render_user_prompt(question, chunks)."""
    q = "What is X?"
    chunks = [_FakeChunk("1", "T", ["A"], "body")]
    ap = build_augmented_prompt(q, chunks)
    assert set(ap.keys()) == {"System", "User"}
    assert ap["System"] == SYSTEM_PROMPT
    assert ap["User"] == render_user_prompt(q, chunks)


# ---------- determinism --------------------------------------------------


def test_render_is_deterministic():
    """Same (question, chunks) -> byte-identical user prompt on repeat calls."""
    q = "q"
    chunks = [
        _FakeChunk("1", "First", ["A", "B"], "b1"),
        _FakeChunk("2", "Second", [], "b2"),
    ]
    assert render_user_prompt(q, chunks) == render_user_prompt(q, chunks)
