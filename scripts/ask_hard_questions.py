"""Run the 8 hard / edge-case questions through the full RAG chain and PRINT the
answers AND the retrieved chunks (unlike the pytest smoke tests, which only assert
pass/fail and show nothing).

Run:
    conda run -n medium-rag python scripts/ask_hard_questions.py
    # save the full output:
    conda run -n medium-rag python scripts/ask_hard_questions.py > run_logs/hard_answers.txt 2>&1

For each question prints: the question, the retrieval query + dedup flag, the
model's full answer, and EVERY retrieved chunk (rank, score, id, title, full
chunk text). Targets the live `smoke` namespace (read-only). All dynamic text is
transliterated to ASCII via scripts/_console.to_ascii so it renders correctly on
any Windows console / pipe / file.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._console import to_ascii  # console-safe ASCII output
from src.config import load_config
from src.rag.chain import answer
from src.rag.vectorstore import namespace_stats

SMOKE_NS = "smoke"

# (label, question) -- mirrors tests/test_smoke_hard_questions.py. IDK-expected
# cases are marked; the rest should name/recommend the indicated article.
HARD_QUESTIONS = [
    ("T1-a  [expect IDK]  meditation rewires the brain",
     "I'm looking for an article about how meditation can rewire the brain. Provide the title and author."),
    ("T1-b  [expect id 6]  Pakistan's first liver transplant",
     "I'm looking for an article about Pakistan's first liver transplant. Provide the title and author."),
    ("T2-a  [under-supply: only 2 exist]  entrepreneurship",
     "List 3 articles about entrepreneurship. Return only the titles."),
    ("T2-b  [expect IDK]  cooking / recipes",
     "List 3 articles about cooking or food recipes. Return only the titles."),
    ("T3-a  [expect IDK]  social media harms teenagers",
     "Find an article that argues social media is harmful to teenagers, and summarize its central argument."),
    ("T3-b  [expect id 8]  correlation vs causation",
     "Find an article that warns against treating correlation as causation in neuroscience, and summarize its central argument."),
    ("T4-a  [expect id 9]  investor pitch",
     "I'm a startup founder who needs to win over investors with a compelling pitch. Which article would you recommend, and why?"),
    ("T4-b  [expect IDK]  marathon training",
     "I want practical advice on training for a marathon. Which article would you recommend, and why?"),
]


def main() -> int:
    cfg = load_config()
    if namespace_stats(SMOKE_NS, cfg)["vector_count"] == 0:
        print(
            "Namespace 'smoke' is empty. Run:\n"
            "  python scripts/ingest.py --limit 10 --namespace smoke",
            file=sys.stderr,
        )
        return 1

    print(f"Asking {len(HARD_QUESTIONS)} hard questions against the 'smoke' namespace")
    print(f"  chat_model={cfg.chat_model}  reasoning_effort={cfg.reasoning_effort}")

    for label, q in HARD_QUESTIONS:
        res = answer(q, cfg, namespace=SMOKE_NS)
        ids = [row["article_id"] for row in res.context]
        print("\n" + "=" * 100)
        print(f"[{label}]")
        print(f"Q: {q}")
        print(f"retrieval: query={to_ascii(res.rewrite.query)!r}  dedup={res.rewrite.dedup}  context_ids={ids}")
        print(f"\nA: {to_ascii(res.response)}")
        print(f"\nretrieved context ({len(res.context)} chunks):")
        for n, row in enumerate(res.context, start=1):
            print(f"\n  #{n}  score={row['score']:.3f}  id={row['article_id']}  title: {to_ascii(row['title'])}")
            print(f"  chunk:\n{to_ascii(row['chunk'])}")

    print("\n" + "=" * 100)
    print("Done.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- surface any failure non-zero
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
