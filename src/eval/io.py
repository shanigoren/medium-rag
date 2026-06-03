"""Filesystem output helpers for evaluation artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")


def write_json(path: str | Path, obj: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=True, indent=2, sort_keys=True)
        fh.write("\n")


def write_summary(path: str | Path, metrics: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Evaluation: {metrics['experiment']}",
        "",
        f"- Namespace: `{metrics['namespace']}`",
        f"- Tests: {metrics['n_tests']}",
        f"- Recall@k: {metrics['recall_at_k']}",
        f"- IDK pass rate: {metrics['idk_pass_rate']}",
        f"- Dedup accuracy: {metrics['dedup_accuracy']}",
        "- Answer pass rate: pending manual review",
        "- Combined score: pending manual review",
        "",
        "| Type | N | Recall | IDK Pass | Dedup Accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for qtype, row in metrics["by_type"].items():
        lines.append(
            f"| {qtype} | {row['n']} | {row['recall']} | "
            f"{row['idk_pass_rate']} | {row['dedup_accuracy']} |"
        )
    cost = metrics.get("cost")
    if cost:
        lines.extend(
            [
                "",
                "## Cost",
                "",
                f"- LLMod delta: ${cost.get('spend_delta_usd')}",
                f"- Spend-log delta: ${cost.get('spend_logs_window', {}).get('spend_usd')}",
                f"- Remaining budget: ${cost.get('remaining_budget_usd')}",
            ]
        )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
