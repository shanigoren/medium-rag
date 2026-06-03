"""Offline tests for LLMod cost tracking and paid-ingest preflight."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

from scripts.run_paid_ingest import _append_ledger, _money_delta, _preflight
from src.llm.cost_tracking import (
    filter_logs,
    parse_spend_logs,
    parse_spend_snapshot,
    summarize_logs,
)


def test_parse_spend_snapshot_extracts_key_user_and_budget() -> None:
    snap = parse_spend_snapshot(
        {"info": {"spend": 0.125}},
        {"spend": "0.25", "max_budget": 5},
        taken_at=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )
    assert snap.key_spend == Decimal("0.125")
    assert snap.user_spend == Decimal("0.25")
    assert snap.max_budget == Decimal("5")


def test_spend_delta_math_is_decimal_exact() -> None:
    before = parse_spend_snapshot(
        {"info": {"spend": "0.10"}},
        {"spend": "0.10", "max_budget": "5.00"},
    )
    after = parse_spend_snapshot(
        {"info": {"spend": "0.13758249"}},
        {"spend": "0.13758249", "max_budget": "5.00"},
    )
    assert _money_delta(before, after) == Decimal("0.03758249")


def test_spend_logs_parse_filter_and_summarize() -> None:
    payload = [
        {
            "request_id": "before",
            "call_type": "aembedding",
            "model": "azure/embed",
            "model_group": "4UHRUIN-text-embedding-3-small",
            "spend": "0.01",
            "total_tokens": 10,
            "prompt_tokens": 10,
            "completion_tokens": 0,
            "startTime": "2026-06-01T09:59:59Z",
            "endTime": "2026-06-01T10:00:00Z",
        },
        {
            "request_id": "inside",
            "call_type": "aembedding",
            "model": "azure/embed",
            "model_group": "4UHRUIN-text-embedding-3-small",
            "spend": "0.02",
            "total_tokens": 20,
            "prompt_tokens": 20,
            "completion_tokens": 0,
            "startTime": "2026-06-01T10:00:01Z",
            "endTime": "2026-06-01T10:00:02Z",
        },
        {
            "request_id": "chat",
            "call_type": "acompletion",
            "model": "azure/chat",
            "model_group": "4UHRUIN-gpt-5-mini",
            "spend": "0.03",
            "total_tokens": 30,
            "prompt_tokens": 25,
            "completion_tokens": 5,
            "startTime": "2026-06-01T10:00:03Z",
            "endTime": "2026-06-01T10:00:04Z",
        },
    ]
    logs = parse_spend_logs(payload)
    window = filter_logs(
        logs,
        start=datetime(2026, 6, 1, 10, 0, 0, tzinfo=timezone.utc),
        end=datetime(2026, 6, 1, 10, 0, 3, tzinfo=timezone.utc),
    )
    assert [row.request_id for row in window] == ["inside", "chat"]

    summary = summarize_logs(window)
    assert summary["count"] == 2
    assert summary["spend_usd"] == "0.05"
    assert summary["total_tokens"] == 50
    assert {g["call_type"] for g in summary["by_call_type_model"]} == {
        "aembedding",
        "acompletion",
    }


def test_preflight_counts_articles_chunks_vectors_and_tokens(cfg) -> None:
    stats = _preflight(cfg, limit=3, csv_path=None)
    assert stats["articles"] == 3
    assert stats["chunks"] == stats["vectors"]
    assert stats["chunks"] >= 3
    assert stats["embedding_input_tokens"] > 0
    assert stats["embedding_input_tokens_max"] >= stats["embedding_input_tokens_min"]


def test_ledger_row_contains_no_raw_secrets(tmp_path, monkeypatch, cfg) -> None:
    import scripts.run_paid_ingest as paid

    monkeypatch.setattr(paid, "LEDGER", tmp_path / "ledger.jsonl")
    row = {
        "label": "test",
        "config": {"llmod_base_url": cfg.llmod_base_url},
        "llmod": {"spend_delta_usd": "0.01"},
    }
    _append_ledger(row, cfg)
    text = (tmp_path / "ledger.jsonl").read_text(encoding="utf-8")
    assert cfg.llmod_api_key not in text
    assert cfg.pinecone_api_key not in text
    assert json.loads(text)["label"] == "test"
