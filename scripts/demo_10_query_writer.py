"""Demo for Component 10 (query rewriter).

Run:
    conda activate medium-rag
    python scripts/demo_10_query_writer.py
    python scripts/demo_10_query_writer.py --question "List 5 articles about AI"

With no --question it runs the four assignment example questions. For each it
prints the rewritten retrieval query and the dedup flag, so a human can see that
scaffolding/format instructions are stripped and only "list N" asks flip dedup
true. Hits the live API (gpt-5-mini at reasoning_effort=minimal). ASCII-only;
masks the API key. Exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.rag.query_writer import rewrite_query

FOUR_ASSIGNMENT_QUESTIONS = [
    "Find an article that reframes marketing as a conversation with readers, "
    "aimed at writers who find self-promotion uncomfortable. Provide the title "
    "and author.",
    "List exactly 3 articles about education. Return only the titles.",
    "Find an article that argues past pandemics (such as the bubonic plague) can "
    "spur innovation and recovery, and summarise its central argument.",
    "I want practical, beginner-friendly advice on building habits that actually "
    "stick. Which article would you recommend, and why?",
]


def _mask(secret: str) -> str:
    if not secret:
        return "<empty>"
    if len(secret) <= 4:
        return "***"
    return f"***{secret[-4:]} ({len(secret)} chars)"


def main() -> int:
    ap = argparse.ArgumentParser(description="Demo the C10 query rewriter.")
    ap.add_argument("--question", help="a single question (default: 4 assignment examples)")
    args = ap.parse_args()

    try:
        cfg = load_config()
    except (FileNotFoundError, ValueError) as exc:
        print(f"FAILED to load config: {exc}", file=sys.stderr)
        return 1

    print("Query rewriter (Component 10)")
    print(f"  chat_model = {cfg.chat_model}   reasoning_effort=minimal (forced)")
    print(f"  api_key    = {_mask(cfg.llmod_api_key)}")

    questions = [args.question] if args.question else FOUR_ASSIGNMENT_QUESTIONS
    for q in questions:
        try:
            r = rewrite_query(q, cfg)
        except Exception as exc:  # rewrite_query is best-effort, but be safe
            print(f"FAILED on question: {exc}", file=sys.stderr)
            return 1
        print(f"\n[input]  {q}")
        print(f"[query]  {r.query}   | dedup={r.dedup}")

    print("\nOK: query rewriter ran.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
