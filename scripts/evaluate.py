"""Run curated golden questions against one namespace with spend tracking.

No paid LLM judge exists here. The script runs the real RAG chain, scores
retrieval/IDK/dedup automatically, writes artifacts, and records LLMod spend
for the question run.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._console import to_ascii
from scripts.ingest import _load_with_overrides, _parse_overrides
from src.config import Config, load_config
from src.eval.golden import load_golden_tests
from src.eval.io import utc_stamp, write_json, write_jsonl, write_summary
from src.eval.runner import build_metrics, run_evaluation
from src.llm.cost_tracking import (
    SpendSnapshot,
    assert_no_secrets,
    filter_logs,
    get_spend_logs,
    get_spend_snapshot,
    public_config_snapshot,
    summarize_logs,
)
from src.rag.vectorstore import namespace_stats

LEDGER = Path("reports") / "costs" / "ledger.jsonl"


def _iso(dt) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _money_delta(before: SpendSnapshot, after: SpendSnapshot) -> Decimal:
    return after.user_spend - before.user_spend


def _append_ledger(row: dict[str, Any], cfg: Config) -> None:
    assert_no_secrets(row, cfg)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate curated golden questions.")
    parser.add_argument("--tests", default="tests/golden/subset100.json")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--label", required=True, help="Run label for artifacts and ledger.")
    parser.add_argument("--phase", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit-tests", type=int, default=None)
    parser.add_argument(
        "--no-cost-tracking",
        action="store_true",
        help="Offline/dev only: skip LLMod spend snapshots and ledger writes.",
    )
    parser.add_argument(
        "--allow-empty-namespace",
        action="store_true",
        help="Dev only: do not fail when namespace_stats reports zero vectors.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config field for this run only (repeatable).",
    )
    return parser


def _output_dir(label: str, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in label)
    return Path("reports") / "eval" / f"{safe}_{utc_stamp()}"


def _progress_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "question_id": row["question_id"],
        "question_type": row["question_type"],
        "expect_idk": row["expect_idk"],
        "rewritten_query": row["rewritten_query"],
        "dedup": row["dedup"],
        "expected_dedup": row["expected_dedup"],
        "dedup_pass": row["dedup_pass"],
        "retrieved": [
            {
                "rank": i + 1,
                "article_id": item.get("article_id"),
                "title": item.get("title"),
                "score": item.get("score"),
            }
            for i, item in enumerate(row["retrieved"])
        ],
        "retrieval_pass": row["retrieval_pass"],
        "idk_pass": row["idk_pass"],
        "answer_preview": row["answer"][:500],
        "duration_ms": row["duration_ms"],
        "error": row["error"],
    }


def _progress_writer(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("w", encoding="utf-8")

    def _write(row: dict[str, Any]) -> None:
        fh.write(json.dumps(_progress_row(row), ensure_ascii=True, sort_keys=True) + "\n")
        fh.flush()

    return fh, _write


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        overrides = _parse_overrides(args.override)
        cfg = _load_with_overrides(overrides) if overrides else load_config()
        tests = load_golden_tests(args.tests)
        if args.limit_tests is not None:
            tests = tests[: args.limit_tests]

        out_dir = _output_dir(args.label, args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        progress_path = out_dir / "progress.jsonl"

        vectors_before = namespace_stats(args.namespace, cfg)["vector_count"]
        if vectors_before == 0 and not args.allow_empty_namespace:
            print(
                "ERROR: namespace is empty; stopping before paid question run. "
                "Pass --allow-empty-namespace only for offline/dev stubs."
            )
            return 2

        print("Evaluation preflight")
        print(f"  label       = {to_ascii(args.label)}")
        print(f"  namespace   = {to_ascii(args.namespace)}")
        print(f"  tests       = {len(tests)}")
        print(f"  vectors     = {vectors_before}")
        print("  judge       = none (manual review only)")
        print(f"  output_dir  = {out_dir}")
        print(f"  progress    = {progress_path}")

        before = None if args.no_cost_tracking else get_spend_snapshot(cfg)
        started = before.taken_at if before is not None else None
        if before is not None:
            print(f"  llmod_spend_before = ${before.user_spend}")

        t0 = time.monotonic()
        progress_fh, on_result = _progress_writer(progress_path)
        try:
            summary = run_evaluation(
                tests,
                namespace=args.namespace,
                cfg=cfg,
                experiment=args.label,
                phase=args.phase,
                on_result=on_result,
            )
        finally:
            progress_fh.close()
        duration_ms = int((time.monotonic() - t0) * 1000)

        vectors_after = namespace_stats(args.namespace, cfg)["vector_count"]
        cost = None
        if before is not None and started is not None:
            after = get_spend_snapshot(cfg)
            ended = after.taken_at
            logs = filter_logs(get_spend_logs(cfg), start=started, end=ended)
            log_summary = summarize_logs(logs)
            spend_delta = _money_delta(before, after)
            cost = {
                "spend_before_usd": str(before.user_spend),
                "spend_after_usd": str(after.user_spend),
                "spend_delta_usd": str(spend_delta),
                "key_spend_before_usd": str(before.key_spend),
                "key_spend_after_usd": str(after.key_spend),
                "user_spend_before_usd": str(before.user_spend),
                "user_spend_after_usd": str(after.user_spend),
                "max_budget_usd": None if after.max_budget is None else str(after.max_budget),
                "remaining_budget_usd": (
                    None if after.max_budget is None else str(after.max_budget - after.user_spend)
                ),
                "spend_logs_window": log_summary,
            }
            summary.metrics = build_metrics(
                summary.results,
                experiment=args.label,
                phase=args.phase,
                namespace=args.namespace,
                cfg=cfg,
                cost=cost,
            )

            ledger_row = {
                "label": args.label,
                "kind": "evaluation",
                "started_at": _iso(started),
                "ended_at": _iso(ended),
                "duration_ms": duration_ms,
                "namespace": args.namespace,
                "tests": str(Path(args.tests)),
                "n_tests": len(tests),
                "namespace_vectors_before": vectors_before,
                "namespace_vectors_after": vectors_after,
                "reused_existing_namespace": vectors_before > 0 and vectors_after == vectors_before,
                "judge": "none_manual_review",
                "config": public_config_snapshot(cfg),
                "llmod": cost,
                "output_dir": str(out_dir),
                "progress": str(progress_path),
            }
            _append_ledger(ledger_row, cfg)

        write_jsonl(out_dir / "results.jsonl", summary.results)
        write_json(out_dir / "metrics.json", summary.metrics)
        write_summary(out_dir / "summary.md", summary.metrics)

        print("Evaluation complete")
        print(f"  recall_at_k       = {summary.metrics['recall_at_k']}")
        print(f"  idk_pass_rate     = {summary.metrics['idk_pass_rate']}")
        print(f"  dedup_accuracy    = {summary.metrics['dedup_accuracy']}")
        print("  answer_pass_rate  = pending manual review")
        print(f"  vectors_after     = {vectors_after}")
        if cost is not None:
            print(f"  llmod_spend_after = ${cost['spend_after_usd']}")
            print(f"  llmod_delta       = ${cost['spend_delta_usd']}")
            print(f"  spend_logs_delta  = ${cost['spend_logs_window']['spend_usd']}")
            print(f"  remaining_budget  = ${cost['remaining_budget_usd']}")
            print(f"  ledger            = {LEDGER}")
        print(f"  results           = {out_dir / 'results.jsonl'}")
        print(f"  metrics           = {out_dir / 'metrics.json'}")
        print(f"  progress          = {progress_path}")
        return 0
    except SystemExit as exc:
        print(f"ERROR: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - readable CLI failure
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
