"""Regenerate tests/fixtures/rewriter_recordings.json from the live model.

MANUAL ONLY -- never invoked by pytest. Run when the rewriter prompt or the
model changes:

    conda activate medium-rag
    python scripts/record_rewriter.py

For each bank entry it calls the live rewriter chat path once on `question` and
captures the RAW model response verbatim into `recorded_response`. The
human-authored fields (`question`, `expected_dedup`, `type`, `note`) are left
untouched -- this script never decides the ground-truth label.

After capture it parses each fresh response and flags DRIFT when the parsed
`dedup` disagrees with the entry's `expected_dedup` (a model regression to
inspect, or a label to re-curate) and exits non-zero so a human must look.

ASCII-only stdout; masks the API key.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.llm.clients import get_chat
from src.rag.query_writer import _PROMPT, _parse

BANK = (
    Path(__file__).resolve().parent.parent
    / "tests"
    / "fixtures"
    / "rewriter_recordings.json"
)


def _mask(secret: str) -> str:
    if not secret:
        return "<empty>"
    if len(secret) <= 4:
        return "***"
    return f"***{secret[-4:]} ({len(secret)} chars)"


def main() -> int:
    try:
        cfg = load_config()
    except (FileNotFoundError, ValueError) as exc:
        print(f"FAILED to load config: {exc}", file=sys.stderr)
        return 1

    print("Recording rewriter responses from the live model")
    print(f"  chat_model = {cfg.chat_model}   reasoning_effort=minimal (forced)")
    print(f"  api_key    = {_mask(cfg.llmod_api_key)}")
    print(f"  bank       = {BANK}")

    entries = json.loads(BANK.read_text(encoding="utf-8"))
    chat = get_chat(cfg, reasoning_effort="minimal")

    drift: list[str] = []
    for e in entries:
        question = e["question"]
        raw = chat.invoke(_PROMPT.format_messages(question=question))
        text = raw.content if hasattr(raw, "content") else raw
        if not isinstance(text, str):
            text = str(text)
        e["recorded_response"] = text

        parsed = _parse(text)
        ok = parsed is not None and parsed.dedup == e["expected_dedup"]
        flag = "ok" if ok else "DRIFT"
        qy = parsed.query if parsed is not None else "<parse failed>"
        dd = parsed.dedup if parsed is not None else None
        print(f"\n  [{flag}] type {e['type']}")
        print(f"    Q     : {question}")
        print(f"    query : {qy}")
        print(f"    dedup : {dd}   (expected {e['expected_dedup']})")
        if not ok:
            drift.append(question)

    BANK.write_text(
        json.dumps(entries, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
    )
    print(f"\nWrote {len(entries)} recordings to {BANK}")

    if drift:
        print(
            f"\nWARNING: {len(drift)} entr(y/ies) drifted from expected_dedup "
            f"(parse failure or label disagreement):",
            file=sys.stderr,
        )
        for q in drift:
            print(f"  - {q}", file=sys.stderr)
        print(
            "Inspect each: either the model regressed, or expected_dedup needs "
            "re-curating. The recordings were still written.",
            file=sys.stderr,
        )
        return 1

    print("OK: all recordings parse and match expected_dedup.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
