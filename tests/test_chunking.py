"""Unit tests for `src.rag.chunking`.

Tests load real articles from the project CSV via Component 2's loader.
The golden numbers below were hand-verified against the actual splitter
output; they pin behavior so that a regression in either the tokenizer or the
splitter shows up as a failure here rather than as silently-different chunk
boundaries during ingest.
"""

from __future__ import annotations

import tiktoken
import pytest

from src.rag.chunking import _encoder, chunk_text, token_length


@pytest.fixture(scope="module")
def articles():
    """Module-scoped: load CSV once, share across every test in the file.
    Picked indices reach up to 6420, so we need a full load."""
    from src.data.csv_loader import load_articles
    arts = load_articles()
    return {a.row_idx: a for a in arts}


# --- Pairwise token-overlap helper (test-internal) -------------------------

def _pair_overlap(a: str, b: str, enc) -> int:
    """Longest token suffix of `a` that equals a token prefix of `b`."""
    ta, tb = enc.encode(a), enc.encode(b)
    for k in range(min(len(ta), len(tb)), 0, -1):
        if ta[-k:] == tb[:k]:
            return k
    return 0


# --- token_length ----------------------------------------------------------

def test_token_length_empty_is_zero():
    assert token_length("") == 0


def test_token_length_basic():
    assert token_length("hello") == 1
    assert token_length("hello world") == 2


def test_token_length_matches_tiktoken_directly():
    enc = tiktoken.get_encoding("cl100k_base")
    for s in ["hello", "Mind Your Nose", "The quick brown fox.", "  spaced  "]:
        assert token_length(s) == len(enc.encode(s)), s


# --- Empty / short input ---------------------------------------------------

def test_empty_string_returns_no_chunks():
    assert chunk_text("", 512, 0.1) == []


def test_whitespace_only_returns_no_chunks():
    assert chunk_text("   \n\n\t  ", 512, 0.1) == []


def test_short_text_returns_single_chunk():
    out = chunk_text("hello world", 512, 0.1)
    assert out == ["hello world"]


# --- Real-article snapshots -----------------------------------------------

def test_shortest_real_article_single_chunk_at_512(articles):
    a = articles[6420]
    assert token_length(a.text) == 280
    chunks = chunk_text(a.text, 512, 0.10)
    assert len(chunks) == 1
    assert chunks[0] == a.text
    assert token_length(chunks[0]) == 280


def test_near_median_article_golden_snapshot(articles):
    """row 2 ('Mind Your Nose', 1106 tokens) at cs=512, ov=0.10 yields
    exactly 3 chunks with token lengths [434, 456, 216]. The first chunk
    begins with the article title. Locks down tokenizer + splitter
    behavior; if any of them shifts, this is where we catch it."""
    a = articles[2]
    assert token_length(a.text) == 1106
    chunks = chunk_text(a.text, 512, 0.10)
    assert [token_length(c) for c in chunks] == [434, 456, 216]
    assert chunks[0].startswith("Mind Your Nose")


def test_near_median_natural_boundaries_dominate(articles):
    """Increasing overlap_ratio from 0.10 to 0.15 does NOT change the chunk
    boundaries on row 2 — the splitter snaps to natural paragraph breaks
    that fit within size limits, and overlap is a ceiling, not a target."""
    a = articles[2]
    lens_10 = [token_length(c) for c in chunk_text(a.text, 512, 0.10)]
    lens_15 = [token_length(c) for c in chunk_text(a.text, 512, 0.15)]
    assert lens_10 == lens_15 == [434, 456, 216]


def test_every_chunk_within_chunk_size(articles):
    """The one strict invariant: no chunk ever exceeds its target size, for
    any of our representative articles at any of the Phase-B chunk sizes."""
    for idx in (2, 6317, 4517):
        text = articles[idx].text
        for cs in (256, 512, 1024):
            for c in chunk_text(text, cs, 0.10):
                assert token_length(c) <= cs, (idx, cs, token_length(c))


def test_long_article_chunk_count_in_range(articles):
    """Loose bounds — tight enough to catch a regression that halves or
    doubles the count, loose enough to tolerate minor library updates.
    Observed: row 6317 -> 34 chunks; row 4517 -> 63 chunks."""
    n_6317 = len(chunk_text(articles[6317].text, 512, 0.10))
    n_4517 = len(chunk_text(articles[4517].text, 512, 0.10))
    assert 32 <= n_6317 <= 36, n_6317
    assert 60 <= n_4517 <= 66, n_4517


def test_average_overlap_at_or_below_target(articles):
    """For row 6317 at (cs=512, ov=0.10): average pairwise token-overlap is
    >0 and <= int(512*0.10)=51. Observed avg ~12. The splitter snaps to
    natural boundaries so actual overlap is typically a small fraction of
    the requested ceiling — this test encodes that property."""
    chunks = chunk_text(articles[6317].text, 512, 0.10)
    enc = _encoder()
    overlaps = [_pair_overlap(chunks[i], chunks[i + 1], enc) for i in range(len(chunks) - 1)]
    avg = sum(overlaps) / len(overlaps)
    target = int(512 * 0.10)
    assert avg > 0.0, overlaps
    assert avg <= target, (avg, target)
    assert max(overlaps) <= target, (max(overlaps), target)


def test_deterministic_on_real_article(articles):
    a = articles[2]
    first = chunk_text(a.text, 512, 0.1)
    second = chunk_text(a.text, 512, 0.1)
    assert first == second


# --- Validation ------------------------------------------------------------

def test_invalid_chunk_size_raises():
    with pytest.raises(ValueError):
        chunk_text("foo", 0, 0.1)
    with pytest.raises(ValueError):
        chunk_text("foo", -1, 0.1)


def test_invalid_overlap_ratio_raises():
    with pytest.raises(ValueError):
        chunk_text("foo", 512, -0.1)
    with pytest.raises(ValueError):
        chunk_text("foo", 512, 1.0)
    with pytest.raises(ValueError):
        chunk_text("foo", 512, 1.5)
