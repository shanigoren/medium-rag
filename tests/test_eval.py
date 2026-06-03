"""Offline tests for Component 14 evaluation."""

from __future__ import annotations

import json
from src.eval.golden import GoldenTest, load_golden_tests
from src.eval.io import write_json, write_jsonl, write_summary
from src.eval.runner import IDK_SENTENCE, build_metrics, run_evaluation, score_idk, score_retrieval
from src.rag.chain import AnswerResult
from src.rag.query_writer import RewriteResult


def _test(
    *,
    qid: str = "q1",
    qtype: int = 1,
    expect_idk: bool = False,
    acceptable: list[int] | None = None,
) -> GoldenTest:
    return GoldenTest(
        question_id=qid,
        question_type=qtype,
        question=f"Find a relevant article for {qid}.",
        expected_article_idx=None if expect_idk else 3,
        acceptable_article_indices=[] if expect_idk else (acceptable or [3]),
        expected_titles=[] if expect_idk else ["A title"],
        expect_idk=expect_idk,
        body_anchors=["anchor"],
        rubric="Answer must satisfy the test.",
    )


def _answer(
    *,
    article_id: str = "3",
    response: str = "A title by An Author",
    query: str = "rewritten",
    dedup: bool = False,
) -> AnswerResult:
    return AnswerResult(
        response=response,
        context=[{"article_id": article_id, "title": "A title", "chunk": "body", "score": 0.9}],
        augmented_prompt={"System": "sys", "User": "user"},
        rewrite=RewriteResult(query=query, dedup=dedup),
    )


def test_load_golden_tests_parses_committed_subset() -> None:
    tests = load_golden_tests("tests/golden/subset100.json")
    assert len(tests) == 20
    assert {test.question_type for test in tests} == {1, 2, 3, 4}
    assert any(test.expect_idk for test in tests)


def test_score_retrieval_coerces_expected_ints_to_article_id_strings() -> None:
    test = _test(acceptable=[3, 7])
    assert score_retrieval(test, [{"article_id": "7"}]) is True
    assert score_retrieval(test, [{"article_id": "8"}]) is False
    assert score_retrieval(_test(expect_idk=True), [{"article_id": "3"}]) is None


def test_score_idk_requires_contained_mandatory_sentence() -> None:
    test = _test(expect_idk=True)
    assert score_idk(test, IDK_SENTENCE) is True
    assert score_idk(test, IDK_SENTENCE + " Extra.") is True
    assert score_idk(test, "No matching article was found.") is False
    assert score_idk(_test(), IDK_SENTENCE) is None


def test_run_evaluation_records_results_and_null_manual_review(cfg) -> None:
    tests = [
        _test(qid="type1", qtype=1, acceptable=[3]),
        _test(qid="type2", qtype=2, acceptable=[4]),
        _test(qid="idk", qtype=4, expect_idk=True),
    ]

    def answer_fn(question, cfg=None, *, namespace=None):
        if question == tests[0].question:
            return _answer(article_id="3", dedup=False)
        if namespace == "ns" and len(question) > 0 and question == tests[1].question:
            return _answer(article_id="4", dedup=True)
        return _answer(article_id="99", response=IDK_SENTENCE, dedup=False)

    summary = run_evaluation(
        tests,
        namespace="ns",
        cfg=cfg,
        answer_fn=answer_fn,
        experiment="exp",
        phase="phase_a",
    )

    assert len(summary.results) == 3
    assert summary.metrics["recall_at_k"] == 1.0
    assert summary.metrics["idk_pass_rate"] == 1.0
    assert summary.metrics["answer_pass_rate"] is None
    assert summary.metrics["combined_score"] is None
    assert all(row["manual_review"] is None for row in summary.results)
    assert summary.results[1]["dedup_pass"] is True


def test_run_evaluation_records_errors_without_stopping(cfg) -> None:
    tests = [_test(qid="ok"), _test(qid="boom")]
    calls = {"n": 0}

    def answer_fn(question, cfg=None, *, namespace=None):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("nope")
        return _answer()

    summary = run_evaluation(tests, namespace="ns", cfg=cfg, answer_fn=answer_fn)
    assert summary.metrics["errors"]["question_ids"] == ["boom"]
    assert summary.results[1]["retrieval_pass"] is False
    assert summary.results[1]["error"]["type"] == "RuntimeError"


def test_build_metrics_accepts_cost_block_and_keeps_answer_scores_null(cfg) -> None:
    rows = [
        {
            "question_id": "q",
            "question_type": 1,
            "retrieval_pass": True,
            "idk_pass": None,
            "dedup_pass": True,
            "error": None,
        }
    ]
    cost = {"spend_delta_usd": "0.01", "spend_logs_window": {"spend_usd": "0.01"}}
    metrics = build_metrics(rows, experiment="exp", phase=None, namespace="ns", cfg=cfg, cost=cost)
    assert metrics["cost"] == cost
    assert metrics["answer_pass_rate"] is None
    assert metrics["combined_score"] is None


def test_eval_io_writes_jsonl_json_and_summary(tmp_path) -> None:
    rows = [{"question_id": "q1", "value": 1}]
    metrics = {
        "experiment": "exp",
        "namespace": "ns",
        "n_tests": 1,
        "recall_at_k": 1.0,
        "idk_pass_rate": None,
        "dedup_accuracy": 1.0,
        "by_type": {
            "1": {"n": 1, "recall": 1.0, "idk_pass_rate": None, "dedup_accuracy": 1.0},
            "2": {"n": 0, "recall": None, "idk_pass_rate": None, "dedup_accuracy": None},
            "3": {"n": 0, "recall": None, "idk_pass_rate": None, "dedup_accuracy": None},
            "4": {"n": 0, "recall": None, "idk_pass_rate": None, "dedup_accuracy": None},
        },
    }
    write_jsonl(tmp_path / "results.jsonl", rows)
    write_json(tmp_path / "metrics.json", metrics)
    write_summary(tmp_path / "summary.md", metrics)

    assert json.loads((tmp_path / "results.jsonl").read_text(encoding="utf-8")) == rows[0]
    assert json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))["experiment"] == "exp"
    assert "pending manual review" in (tmp_path / "summary.md").read_text(encoding="utf-8")


def test_evaluate_ledger_row_contains_no_raw_secrets(tmp_path, monkeypatch, cfg) -> None:
    import scripts.evaluate as evaluate

    monkeypatch.setattr(evaluate, "LEDGER", tmp_path / "ledger.jsonl")
    row = {
        "label": "eval",
        "config": {"llmod_base_url": cfg.llmod_base_url},
        "llmod": {"spend_delta_usd": "0.01"},
    }
    evaluate._append_ledger(row, cfg)
    text = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")
    assert cfg.llmod_api_key not in text
    assert cfg.pinecone_api_key not in text
    assert json.loads(text)["label"] == "eval"
