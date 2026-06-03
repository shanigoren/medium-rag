"""Stress / edge-case smoke tests for the full RAG loop (C11 `answer()`).

Money-costing (live): runs `answer()` against the `smoke` namespace for 8 hard
questions -- 2 per assignment question type -- focused on cases the happy-path
tests (CP-B, the demo) miss. The priority is the **IDK contract**: when NO
relevant article exists, the model's answer must contain the mandatory refusal
sentence rather than fabricate an answer from nearest-neighbour distractors.

These are QUALITY PROBES, not the CP-B integration gate. A failure here is a
finding about model/prompt behaviour to discuss (and possibly fix in the prompt),
not necessarily a code bug. Save the full run output to `run_logs/`.

Grounding (the 10-article `smoke` slice): 0 mental-health digest, 1 COVID+brain,
2 smell-training neuroplasticity, 3 Phineas Gage, 4 COVID mental health (young
adults), 5 blog->book writing, 6 Pakistan's first liver transplant, 7 sunlight+
mood, 8 Occam's dice (correlation vs causation), 9 origin-story pitching.
"""

from __future__ import annotations

import pytest

from src.config import load_config
from src.rag.chain import answer
from src.rag.vectorstore import namespace_stats

SMOKE_NS = "smoke"

# The mandatory refusal sentence; IDK-expected cases must contain it verbatim.
IDK = "I don't know based on the provided Medium articles data."


def _require_smoke(cfg) -> None:
    if namespace_stats(SMOKE_NS, cfg)["vector_count"] == 0:
        pytest.skip("run scripts/ingest.py --limit 10 --namespace smoke first")


def _answer(question: str):
    cfg = load_config()
    _require_smoke(cfg)
    return answer(question, cfg, namespace=SMOKE_NS)


def _ids(res) -> list[str]:
    return [row["article_id"] for row in res.context]


# --------------------------------------------------------------------------- #
# Type 1 -- precise fact retrieval
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_t1a_meditation_rewire_is_idk():
    """Near-miss IDK: article 2 is *smell* training rewiring the brain, and
    'meditation' is only name-dropped in article 7 -- no article is about
    meditation rewiring the brain. The model must refuse, not grab article 2."""
    res = _answer(
        "I'm looking for an article about how meditation can rewire the brain. "
        "Provide the title and author."
    )
    assert IDK in res.response


@pytest.mark.smoke
def test_t1b_liver_transplant_found():
    """Buried specific fact -> article 6 (Dr Faisal Dar, Fatima Arif). Under-
    specified on purpose (no '9-year-old boy' giveaway)."""
    res = _answer(
        "I'm looking for an article about Pakistan's first liver transplant. "
        "Provide the title and author."
    )
    assert "6" in _ids(res)
    assert IDK not in res.response
    assert "Faisal Dar" in res.response


# --------------------------------------------------------------------------- #
# Type 2 -- multi-result listing
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_t2a_entrepreneurship_under_supply():
    """Under-supply: only 2 entrepreneurship articles (5, 9) exist for a 'list 3'
    request. Decided behaviour (A): return the 2 real ones, do NOT fabricate a
    3rd, do NOT full-IDK. Asserts both real titles present and no IDK refusal.
    (Whether a fabricated 3rd appears is also eyeballed in run_logs.)"""
    res = _answer("List 3 articles about entrepreneurship. Return only the titles.")
    assert IDK not in res.response
    assert "How to Turn Your Popular Blog Series Into a Bestselling Book" in res.response
    assert "To Quickly Build Trust, Tell Your Origin Story" in res.response


@pytest.mark.smoke
def test_t2b_cooking_is_idk():
    """Zero relevant: no cooking/recipe article. The model must refuse with the
    exact sentinel, not invent three plausible-sounding titles to satisfy 'list 3'."""
    res = _answer("List 3 articles about cooking or food recipes. Return only the titles.")
    assert IDK in res.response


# --------------------------------------------------------------------------- #
# Type 3 -- key-idea summary
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_t3a_social_media_is_idk():
    """Near-miss IDK: article 4 (young adults, COVID, 'doomscroll') is adjacent
    but does NOT argue social media harms teenagers. The model must not summarize
    it as if it made that argument."""
    res = _answer(
        "Find an article that argues social media is harmful to teenagers, and "
        "summarize its central argument."
    )
    assert IDK in res.response


@pytest.mark.smoke
def test_t3b_correlation_causation_found():
    """Discriminate article 8 (Occam's dice) from the other neuro articles
    (1, 2, 3) and summarize it -- only 8 warns against correlation->causation."""
    res = _answer(
        "Find an article that warns against treating correlation as causation in "
        "neuroscience, and summarize its central argument."
    )
    assert "8" in _ids(res)
    assert IDK not in res.response


# --------------------------------------------------------------------------- #
# Type 4 -- recommendation with justification
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_t4a_investor_pitch_recommends_origin_story():
    """Recommendation discrimination: investor-pitch advice -> article 9 (origin
    story for fundraising), not article 5 (book writing/marketing distractor)."""
    res = _answer(
        "I'm a startup founder who needs to win over investors with a compelling "
        "pitch. Which article would you recommend, and why?"
    )
    assert "9" in _ids(res)
    assert IDK not in res.response
    assert "Origin Story" in res.response


@pytest.mark.smoke
def test_t4b_marathon_is_idk():
    """Near-miss IDK: fitness/wellness distractors exist (article 4's 'fitness
    challenge', article 7's wellbeing) but no article gives marathon-training
    advice. The model must refuse."""
    res = _answer(
        "I want practical advice on training for a marathon. Which article would "
        "you recommend, and why?"
    )
    assert IDK in res.response
