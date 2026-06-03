"""Run and score the curated golden tests without any LLM judge calls."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from src.config import Config, load_config
from src.eval.golden import GoldenTest
from src.rag.chain import AnswerResult, answer

IDK_SENTENCE = "I don't know based on the provided Medium articles data."

AnswerFn = Callable[[str, Config | None], AnswerResult]
ProgressCallback = Callable[[dict[str, Any]], None]


@dataclass
class EvalSummary:
    results: list[dict[str, Any]]
    metrics: dict[str, Any]


def score_retrieval(test: GoldenTest, context: list[dict[str, Any]]) -> bool | None:
    """Recall@k for non-IDK tests; IDK tests are scored by answer behavior."""
    if test.expect_idk:
        return None
    acceptable = {str(idx) for idx in test.acceptable_article_indices}
    retrieved = {str(row.get("article_id")) for row in context}
    return bool(acceptable & retrieved)


def score_idk(test: GoldenTest, response: str) -> bool | None:
    """IDK pass if the mandatory refusal sentence appears in the answer."""
    if not test.expect_idk:
        return None
    return IDK_SENTENCE in response


def expected_dedup(test: GoldenTest) -> bool:
    return test.question_type == 2


def result_from_answer(test: GoldenTest, result: AnswerResult, duration_ms: int) -> dict[str, Any]:
    retrieval_pass = score_retrieval(test, result.context)
    idk_pass = score_idk(test, result.response)
    return {
        **test.to_dict(),
        "rewritten_query": result.rewrite.query,
        "dedup": result.rewrite.dedup,
        "expected_dedup": expected_dedup(test),
        "dedup_pass": result.rewrite.dedup == expected_dedup(test),
        "retrieved": result.context,
        "augmented_prompt": result.augmented_prompt,
        "answer": result.response,
        "manual_review": None,
        "retrieval_pass": retrieval_pass,
        "idk_pass": idk_pass,
        "duration_ms": duration_ms,
        "error": None,
    }


def result_from_error(test: GoldenTest, exc: Exception, duration_ms: int) -> dict[str, Any]:
    return {
        **test.to_dict(),
        "rewritten_query": "",
        "dedup": None,
        "expected_dedup": expected_dedup(test),
        "dedup_pass": False,
        "retrieved": [],
        "augmented_prompt": {},
        "answer": "",
        "manual_review": None,
        "retrieval_pass": False if not test.expect_idk else None,
        "idk_pass": False if test.expect_idk else None,
        "duration_ms": duration_ms,
        "error": {"type": type(exc).__name__, "message": str(exc)},
    }


def _rate(values: list[bool]) -> float | None:
    if not values:
        return None
    return sum(1 for value in values if value) / len(values)


def _by_type(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for qtype in (1, 2, 3, 4):
        rows = [row for row in results if row["question_type"] == qtype]
        recall_values = [row["retrieval_pass"] for row in rows if row["retrieval_pass"] is not None]
        idk_values = [row["idk_pass"] for row in rows if row["idk_pass"] is not None]
        dedup_values = [bool(row["dedup_pass"]) for row in rows]
        out[str(qtype)] = {
            "n": len(rows),
            "recall": _rate(recall_values),
            "idk_pass_rate": _rate(idk_values),
            "dedup_accuracy": _rate(dedup_values),
            "answer_pass_rate": None,
            "combined_score": None,
        }
    return out


def build_metrics(
    results: list[dict[str, Any]],
    *,
    experiment: str,
    phase: str | None,
    namespace: str,
    cfg: Config,
    cost: dict[str, Any] | None = None,
) -> dict[str, Any]:
    recall_values = [row["retrieval_pass"] for row in results if row["retrieval_pass"] is not None]
    idk_values = [row["idk_pass"] for row in results if row["idk_pass"] is not None]
    dedup_values = [bool(row["dedup_pass"]) for row in results]
    errors = [row for row in results if row["error"] is not None]
    return {
        "experiment": experiment,
        "phase": phase,
        "namespace": namespace,
        "n_tests": len(results),
        "n_non_idk": len(recall_values),
        "n_idk": len(idk_values),
        "recall_at_k": _rate(recall_values),
        "idk_pass_rate": _rate(idk_values),
        "dedup_accuracy": _rate(dedup_values),
        "answer_pass_rate": None,
        "combined_score": None,
        "by_type": _by_type(results),
        "errors": {
            "count": len(errors),
            "question_ids": [row["question_id"] for row in errors],
        },
        "cost": cost,
        "config": {
            "chunk_size": cfg.chunk_size,
            "overlap_ratio": cfg.overlap_ratio,
            "top_k": cfg.top_k,
            "retrieval_fetch_k": cfg.retrieval_fetch_k,
            "embed_content": cfg.embed_content,
            "embed_model": cfg.embed_model,
            "chat_model": cfg.chat_model,
            "reasoning_effort": cfg.reasoning_effort,
            "pinecone_index": cfg.pinecone_index,
        },
    }


def run_evaluation(
    tests: list[GoldenTest],
    *,
    namespace: str,
    cfg: Config | None = None,
    answer_fn: Callable[..., AnswerResult] = answer,
    experiment: str | None = None,
    phase: str | None = None,
    on_result: ProgressCallback | None = None,
) -> EvalSummary:
    """Run all tests through the chain and compute automatic metrics.

    `answer_fn` exists for offline tests. It must accept
    `(question, cfg, namespace=namespace)` like `src.rag.chain.answer`.
    """
    if cfg is None:
        cfg = load_config()
    rows: list[dict[str, Any]] = []
    for test in tests:
        started = time.monotonic()
        try:
            result = answer_fn(test.question, cfg, namespace=namespace)
            duration_ms = int((time.monotonic() - started) * 1000)
            row = result_from_answer(test, result, duration_ms)
        except Exception as exc:  # noqa: BLE001 - evaluation should record failures
            duration_ms = int((time.monotonic() - started) * 1000)
            row = result_from_error(test, exc, duration_ms)
        rows.append(row)
        if on_result is not None:
            on_result(row)

    metrics = build_metrics(
        rows,
        experiment=experiment or namespace,
        phase=phase,
        namespace=namespace,
        cfg=cfg,
    )
    return EvalSummary(results=rows, metrics=metrics)
