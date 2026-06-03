"""Demo for Component 11 (the Chain).

Run:
    conda activate medium-rag
    python scripts/demo_11_chain.py
    python scripts/demo_11_chain.py --question "List 3 articles about mental health"

Runs the full RAG loop (rewrite -> retrieve -> prompt -> answer) against the live
`smoke` namespace and, for each question, prints the retrieval query + dedup flag,
the model response, the full retrieved context (titles + full chunks), and a check
that the augmented prompt holds the ORIGINAL question. With no --question, runs
four questions grounded in the 10-article smoke slice -- one per assignment type.
Costs a few cents (one rewrite + one answer call per question). READ-ONLY.

All dynamic text is transliterated to ASCII via scripts/_console.to_ascii so it
renders on any Windows console / pipe / file, avoiding both cp1252 crashes and
mojibake when piped/redirected.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._console import to_ascii  # console-safe ASCII output
from src.config import load_config
from src.rag.chain import answer
from src.rag.vectorstore import namespace_stats

# Questions grounded in the 10-article `smoke` slice (CSV rows 0-9) -- one per
# assignment question type, each answerable from an article actually ingested:
#   T1 precise fact retrieval -> id 3 "Surviving a Rod Through the Head"
#   T2 multi-result listing   -> ids 0/1/4/7 (mental-health articles)
#   T3 key-idea summary       -> id 2 "Mind Your Nose"
#   T4 recommendation         -> id 5 "How to Turn Your Popular Blog Series ..."
SMOKE_DEMO_QUESTIONS = [
    "Find the article about Phineas Gage, the railroad worker who survived an iron "
    "rod piercing through his skull. Provide the title and author.",
    "List exactly 3 articles about mental health. Return only the titles.",
    "Find the article about how smell training can change your brain, and summarise "
    "its central argument.",
    "I want to turn my popular blog posts into a published book. Which article would "
    "you recommend, and why?",
]


def _mask(secret: str) -> str:
    if not secret or len(secret) < 4:
        return "****"
    return "..." + secret[-4:]


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo the C11 RAG chain.")
    parser.add_argument("--question", help="a single question (default: 4 smoke-grounded examples)")
    args = parser.parse_args()

    cfg = load_config()
    print()
    print("Answering from the 'smoke' namespace ...")
    print(f"  chat_model={cfg.chat_model}  reasoning_effort={cfg.reasoning_effort}  (answer call)")
    print(f"  top_k={cfg.top_k}  api_key={_mask(cfg.llmod_api_key)}   (last 4 chars only)")

    if namespace_stats("smoke", cfg)["vector_count"] == 0:
        print(
            "Namespace 'smoke' is empty. Run:\n"
            "  python scripts/ingest.py --limit 10 --namespace smoke",
            file=sys.stderr,
        )
        return 1

    questions = [args.question] if args.question else SMOKE_DEMO_QUESTIONS
    for q in questions:
        res = answer(q, cfg, namespace="smoke")
        print("\n" + "=" * 100)
        print(f"[Q] {to_ascii(q)}")
        print(f"[retrieval] query={to_ascii(res.rewrite.query)!r}  dedup={res.rewrite.dedup}")
        print(f"\n[A] {to_ascii(res.response)}")
        print(f"\n  context: {len(res.context)} chunks")
        for n, row in enumerate(res.context, start=1):
            print(f"\n   #{n}  score={row['score']:.3f}  id={row['article_id']}  title: {to_ascii(row['title'])}")
            print(f"   chunk:\n{to_ascii(row['chunk'])}")
        assert q in res.augmented_prompt["User"], "original question must appear in the prompt"
        print("\n  OK: original question present in augmented prompt")

    print("\n" + "=" * 100)
    print("OK: chain answered every question; augmented prompt held the original question.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- demo: surface any failure non-zero
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
