"""Loader and validation for curated golden tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GoldenTest:
    question_id: str
    question_type: int
    question: str
    expected_article_idx: int | None
    acceptable_article_indices: list[int]
    expected_titles: list[str]
    expect_idk: bool
    body_anchors: list[str]
    rubric: str

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "GoldenTest":
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
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"golden test missing field(s): {missing}")

        test = cls(
            question_id=str(row["question_id"]),
            question_type=int(row["question_type"]),
            question=str(row["question"]),
            expected_article_idx=(
                None if row["expected_article_idx"] is None else int(row["expected_article_idx"])
            ),
            acceptable_article_indices=[int(v) for v in row["acceptable_article_indices"]],
            expected_titles=[str(v) for v in row["expected_titles"]],
            expect_idk=bool(row["expect_idk"]),
            body_anchors=[str(v) for v in row["body_anchors"]],
            rubric=str(row["rubric"]),
        )
        test.validate()
        return test

    def validate(self) -> None:
        if not self.question_id:
            raise ValueError("question_id must be non-empty")
        if self.question_type not in {1, 2, 3, 4}:
            raise ValueError(f"{self.question_id}: question_type must be 1..4")
        if not self.question.strip():
            raise ValueError(f"{self.question_id}: question must be non-empty")
        if not self.rubric.strip():
            raise ValueError(f"{self.question_id}: rubric must be non-empty")
        if self.expect_idk:
            if self.expected_article_idx is not None:
                raise ValueError(f"{self.question_id}: IDK expected_article_idx must be null")
            if self.acceptable_article_indices:
                raise ValueError(f"{self.question_id}: IDK acceptable_article_indices must be empty")
            if self.expected_titles:
                raise ValueError(f"{self.question_id}: IDK expected_titles must be empty")
            return
        if self.expected_article_idx is None:
            raise ValueError(f"{self.question_id}: non-IDK expected_article_idx must be int")
        if not self.acceptable_article_indices:
            raise ValueError(f"{self.question_id}: non-IDK acceptable_article_indices required")
        if not self.expected_titles:
            raise ValueError(f"{self.question_id}: non-IDK expected_titles required")

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question_type": self.question_type,
            "question": self.question,
            "expected_article_idx": self.expected_article_idx,
            "acceptable_article_indices": list(self.acceptable_article_indices),
            "expected_titles": list(self.expected_titles),
            "expect_idk": self.expect_idk,
            "body_anchors": list(self.body_anchors),
            "rubric": self.rubric,
        }


def load_golden_tests(path: str | Path) -> list[GoldenTest]:
    """Load the committed curated golden set from JSON."""
    with Path(path).open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict) and isinstance(payload.get("tests"), list):
        payload = payload["tests"]
    if not isinstance(payload, list):
        raise ValueError("golden test file must contain a JSON list or an object with a tests list")
    tests = [GoldenTest.from_dict(row) for row in payload if isinstance(row, dict)]
    if len(tests) != len(payload):
        raise ValueError("all golden test entries must be objects")
    ids = [test.question_id for test in tests]
    dupes = sorted({qid for qid in ids if ids.count(qid) > 1})
    if dupes:
        raise ValueError(f"duplicate question_id(s): {dupes}")
    return tests
