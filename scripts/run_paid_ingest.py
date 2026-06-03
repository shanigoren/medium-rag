"""Run a paid ingest with LLMod spend tracking.

This wrapper is for measured money-spending runs. It snapshots LLMod spend
before/after ingestion, records `/spend/logs` rows inside the run window, and
writes a redacted ledger row to reports/costs/ledger.jsonl.
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
from scripts.ingest import _load_with_overrides, _parse_overrides, build_records, run_ingest
from src.config import Config, load_config
from src.data.csv_loader import load_articles
from src.llm.cost_tracking import (
    SpendSnapshot,
    assert_no_secrets,
    filter_logs,
    get_spend_logs,
    get_spend_snapshot,
    public_config_snapshot,
    summarize_logs,
)
from src.rag.chunking import token_length
from src.rag.vectorstore import namespace_stats

LEDGER = Path("reports") / "costs" / "ledger.jsonl"


def _iso(dt) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _preflight(cfg: Config, *, limit: int | None, csv_path: str | None) -> dict[str, Any]:
    articles = load_articles(csv_path, limit=limit)
    ids, texts, _metas = build_records(articles, cfg)
    token_counts = [token_length(text) for text in texts]
    return {
        "articles": len(articles),
        "chunks": len(ids),
        "vectors": len(ids),
        "embedding_input_tokens": sum(token_counts),
        "embedding_input_tokens_min": min(token_counts) if token_counts else 0,
        "embedding_input_tokens_max": max(token_counts) if token_counts else 0,
    }


def _money_delta(before: SpendSnapshot, after: SpendSnapshot) -> Decimal:
    """Primary run delta.

    LLMod's `/v2/user/info` updated immediately in practice, while `/key/info`
    can lag by a few seconds. Use the user-level delta as the source of truth for
    this key-scoped student account; still record key spend separately for audit.
    """
    return after.user_spend - before.user_spend


def _append_ledger(row: dict[str, Any], cfg: Config) -> None:
    assert_no_secrets(row, cfg)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measured paid ingest with LLMod cost tracking.")
    parser.add_argument("--label", required=True, help="Human-readable run label for the ledger.")
    parser.add_argument("--namespace", required=True, help="Pinecone namespace to ingest into.")
    parser.add_argument("--limit", type=int, default=None, help="Ingest only first N CSV rows.")
    parser.add_argument("--clean", action="store_true", help="Delete namespace before ingesting.")
    parser.add_argument("--csv", default=None, help="Optional CSV path override.")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config field for this run only (repeatable).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        overrides = _parse_overrides(args.override)
        cfg = _load_with_overrides(overrides) if overrides else load_config()

        print("Paid ingest preflight")
        print(f"  label       = {to_ascii(args.label)}")
        print(f"  namespace   = {to_ascii(args.namespace)}")
        print(f"  limit       = {args.limit}")
        print(f"  clean       = {args.clean}")
        print(
            "  config      = "
            f"chunk_size={cfg.chunk_size} overlap_ratio={cfg.overlap_ratio} "
            f"embed_content={cfg.embed_content} embed_model={cfg.embed_model}"
        )

        preflight = _preflight(cfg, limit=args.limit, csv_path=args.csv)
        print(
            "  preflight   = "
            f"articles={preflight['articles']} chunks={preflight['chunks']} "
            f"tokens={preflight['embedding_input_tokens']}"
        )

        existing = namespace_stats(args.namespace, cfg)["vector_count"]
        print(f"  namespace_vectors_before = {existing}")
        if existing > 0 and not args.clean:
            print(
                "ERROR: namespace is non-empty and --clean was not passed; "
                "stopping before paid embedding."
            )
            return 2

        before = get_spend_snapshot(cfg)
        print(f"  llmod_spend_before = ${before.key_spend}")

        started = before.taken_at
        t0 = time.monotonic()
        stats = run_ingest(
            args.namespace,
            cfg,
            limit=args.limit,
            clean=args.clean,
            csv_path=args.csv,
        )
        duration_ms = int((time.monotonic() - t0) * 1000)

        after = get_spend_snapshot(cfg)
        ended = after.taken_at
        logs = filter_logs(get_spend_logs(cfg), start=started, end=ended)
        log_summary = summarize_logs(logs)
        spend_delta = _money_delta(before, after)

        row = {
            "label": args.label,
            "kind": "ingest",
            "started_at": _iso(started),
            "ended_at": _iso(ended),
            "duration_ms": duration_ms,
            "namespace": args.namespace,
            "limit": args.limit,
            "clean": args.clean,
            "config": public_config_snapshot(cfg),
            "preflight": preflight,
            "ingest_stats": {
                "articles_total": stats.articles_total,
                "articles_chunked": stats.articles_chunked,
                "articles_skipped": stats.articles_skipped,
                "chunks_total": stats.chunks_total,
                "vectors_upserted": stats.vectors_upserted,
                "namespace": stats.namespace,
            },
            "llmod": {
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
            },
        }
        _append_ledger(row, cfg)

        print("Paid ingest complete")
        print(f"  llmod_spend_after  = ${after.user_spend}")
        print(f"  llmod_delta        = ${spend_delta}")
        print(f"  spend_logs_delta   = ${log_summary['spend_usd']}")
        print(f"  spend_logs_rows    = {log_summary['count']}")
        print(f"  vectors_upserted   = {stats.vectors_upserted}")
        print(f"  remaining_budget   = ${row['llmod']['remaining_budget_usd']}")
        print(f"  ledger             = {LEDGER}")
        print("STOP: measured embedding run finished; do not run eval/questions yet.")
        return 0
    except SystemExit as exc:
        print(f"ERROR: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 - CLI should fail readable/non-zero
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
