"""Validation for the curated Component 13 golden set.

The file under test is intentionally static: it was manually curated by
inspecting the first 100 CSV rows. These tests keep that artifact honest so later
eval code can trust its schema, IDs, titles, IDK cases, and body anchors.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.data.csv_loader import Article, load_articles


GOLDEN = Path(__file__).parent / "golden" / "subset100.json"
IDK = "I don't know based on the provided Medium articles data."


def _load_golden() -> dict[str, Any]:
    with GOLDEN.open(encoding="utf-8") as fh:
        return json.load(fh)


def _articles() -> dict[int, Article]:
    return {a.row_idx: a for a in load_articles(limit=100)}


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _metadata_text(article: Article) -> str:
    return _norm(" ".join([article.title, *article.tags, *article.authors]))


def _anchor_in_body_not_metadata(anchor: str, article: Article) -> bool:
    needle = _norm(anchor)
    return needle in _norm(article.text) and needle not in _metadata_text(article)


def test_subset100_metadata_is_fixed() -> None:
    data = _load_golden()
    assert data["dataset"] == "medium-english-50mb.csv"
    assert data["limit"] == 100
    assert data["selection_method"] == "first_100_rows_manual_curated"
    assert data["idk_sentence"] == IDK


def test_has_20_tests_with_5_per_assignment_type() -> None:
    tests = _load_golden()["tests"]
    assert len(tests) == 20
    assert Counter(t["question_type"] for t in tests) == {1: 5, 2: 5, 3: 5, 4: 5}
    assert len({t["question_id"] for t in tests}) == 20


def test_schema_and_idk_shape() -> None:
    required = {
        "question_id",
        "question_type",
        "question",
        "expected_article_idx",
        "acceptable_article_indices",
        "expected_titles",
        "expect_idk",
        "body_anchors",
        "rubric",
    }

    for item in _load_golden()["tests"]:
        assert set(item) == required
        assert isinstance(item["question_id"], str) and item["question_id"]
        assert item["question_type"] in {1, 2, 3, 4}
        assert isinstance(item["question"], str) and item["question"].strip()
        assert isinstance(item["acceptable_article_indices"], list)
        assert isinstance(item["expected_titles"], list)
        assert isinstance(item["expect_idk"], bool)
        assert isinstance(item["body_anchors"], list) and item["body_anchors"]
        assert isinstance(item["rubric"], str) and item["rubric"].strip()

        if item["expect_idk"]:
            assert item["expected_article_idx"] is None
            assert item["acceptable_article_indices"] == []
            assert item["expected_titles"] == []
            assert "exact IDK" in item["rubric"] or "IDK sentence" in item["rubric"]
        else:
            assert isinstance(item["expected_article_idx"], int)
            assert item["acceptable_article_indices"]
            assert item["expected_article_idx"] == item["acceptable_article_indices"][0]


def test_expected_ids_and_titles_match_first_100_csv_rows() -> None:
    articles = _articles()
    for item in _load_golden()["tests"]:
        for article_id in item["acceptable_article_indices"]:
            assert isinstance(article_id, int)
            assert 0 <= article_id < 100

        actual_titles = [articles[i].title for i in item["acceptable_article_indices"]]
        assert item["expected_titles"] == actual_titles


def test_type2_non_idk_cases_have_at_least_three_distinct_targets() -> None:
    for item in _load_golden()["tests"]:
        if item["question_type"] == 2 and not item["expect_idk"]:
            ids = item["acceptable_article_indices"]
            assert len(ids) >= 3
            assert len(set(ids)) == len(ids)


def test_every_assignment_type_has_an_idk_case() -> None:
    tests = _load_golden()["tests"]
    idk_types = {item["question_type"] for item in tests if item["expect_idk"]}
    assert idk_types == {1, 2, 3, 4}


def test_questions_do_not_copy_expected_full_titles() -> None:
    """Hardness guard: non-IDK questions should not contain the whole answer title."""
    for item in _load_golden()["tests"]:
        if item["expect_idk"]:
            continue
        question = _norm(item["question"])
        for title in item["expected_titles"]:
            assert _norm(title) not in question


def test_at_least_12_non_idk_tests_use_body_anchors_not_title_or_metadata() -> None:
    articles = _articles()
    hard_tests = 0

    for item in _load_golden()["tests"]:
        if item["expect_idk"]:
            continue
        found_body_only_anchor = False
        for article_id in item["acceptable_article_indices"]:
            article = articles[article_id]
            if any(
                _anchor_in_body_not_metadata(anchor, article)
                for anchor in item["body_anchors"]
            ):
                found_body_only_anchor = True
                break
        if found_body_only_anchor:
            hard_tests += 1

    assert hard_tests >= 12


def test_non_idk_body_anchors_exist_in_expected_article_text() -> None:
    articles = _articles()
    for item in _load_golden()["tests"]:
        if item["expect_idk"]:
            continue
        combined_text = _norm(
            " ".join(articles[i].text for i in item["acceptable_article_indices"])
        )
        assert any(_norm(anchor) in combined_text for anchor in item["body_anchors"])
