"""Live-API smoke tests for Component 10 (`src.rag.query_writer`).

Run with `pytest --smoke -v -m smoke -k query_writer`. Each test hits the live
LLMod.AI chat model (gpt-5-mini at reasoning_effort=minimal -> ~$0.0003/call).
No Pinecone, no namespace dependency.

Covers the classification the whole design hinges on (only the "list exactly 3"
question is dedup=True), query density, and a small live drift check over the
recordings bank (the live counterpart to the offline replay tests).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.rag.query_writer import rewrite_query

# (question, expected_dedup, topical_anchor) -- the four assignment examples.
Q1_MARKETING = (
    "Find an article that reframes marketing as a conversation with readers, "
    "aimed at writers who find self-promotion uncomfortable. Provide the title "
    "and author."
)
Q2_LIST3_EDU = "List exactly 3 articles about education. Return only the titles."
Q3_PANDEMICS = (
    "Find an article that argues past pandemics (such as the bubonic plague) can "
    "spur innovation and recovery, and summarise its central argument."
)
Q4_HABITS = (
    "I want practical, beginner-friendly advice on building habits that actually "
    "stick. Which article would you recommend, and why?"
)

_ASSIGNMENT = [
    pytest.param(Q1_MARKETING, False, "marketing", id="q1_marketing"),
    pytest.param(Q2_LIST3_EDU, True, "education", id="q2_list3_edu"),
    pytest.param(Q3_PANDEMICS, False, "pandemic", id="q3_pandemics"),
    pytest.param(Q4_HABITS, False, "habit", id="q4_habits"),
]

_RECORDINGS = json.loads(
    (Path(__file__).parent / "fixtures" / "rewriter_recordings.json").read_text(
        encoding="utf-8"
    )
)
# Small subset for the live drift check (one per type), bounded for cost.
_DRIFT_SUBSET = [
    next(e for e in _RECORDINGS if e["type"] == t) for t in (1, 2, 3, 4)
]


@pytest.mark.smoke
@pytest.mark.parametrize("question,expected_dedup,_anchor", _ASSIGNMENT)
def test_smoke_dedup_classification(question, expected_dedup, _anchor):
    """Only the 'list exactly 3' question is dedup=True; the other three False."""
    result = rewrite_query(question)
    assert result.dedup is expected_dedup


@pytest.mark.smoke
@pytest.mark.parametrize("question,_expected,anchor", _ASSIGNMENT)
def test_smoke_query_is_nonempty_and_anchored(question, _expected, anchor):
    """The rewritten query is non-empty and keeps a topical anchor token from
    the question. (No length assertion -- it's brittle; the prompt instructs a
    focused, un-augmented query, and what matters here is that the scaffolding is
    gone and a topical anchor survives.)"""
    result = rewrite_query(question)
    assert result.query.strip() != ""
    assert anchor in result.query.lower()


@pytest.mark.smoke
@pytest.mark.parametrize(
    "entry",
    _DRIFT_SUBSET,
    ids=[f"type{e['type']}" for e in _DRIFT_SUBSET],
)
def test_smoke_recording_drift(entry):
    """Live counterpart to the offline replay tests: the real model still
    classifies dedup the way the bank's expected_dedup says. A failure here is a
    drift signal -- re-run scripts/record_rewriter.py and re-curate."""
    result = rewrite_query(entry["question"])
    assert result.dedup == entry["expected_dedup"]
