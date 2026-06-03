"""Demo for Component 8 (the retriever).

Run:
    conda activate medium-rag
    python scripts/demo_08_retriever.py

Runs the SAME query against the live `smoke` namespace in BOTH dedup modes so a
human can see the `dedup` flag change the result: dedup=True collapses to distinct
articles (type-2), dedup=False returns the top-k chunks as-is (type-3 depth,
possibly repeating an article). Costs a couple of cents (one embed_query per
mode). READ-ONLY: creates/deletes nothing. Exits non-zero on any exception or if
`smoke` is empty. ASCII-only output (this console is cp1252).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.rag.retriever import retrieve
from src.rag.vectorstore import namespace_stats


def _mask(secret: str) -> str:
    if not secret or len(secret) < 4:
        return "****"
    return "..." + secret[-4:]


def _show(label: str, results) -> None:
    print(f"  {label}: {len(results)} chunks")
    for r in results:
        preview = r.chunk[:90].replace("\n", "\\n")
        print(
            f"   #{r.rank}  score={r.score:.3f}  id={r.article_id}  "
            f"chunk_idx={r.chunk_idx}  title={r.title[:40]!r}"
        )
        print(f"        chunk: {preview}")


def main() -> int:
    cfg = load_config()
    print()
    print("Retrieving from the 'smoke' namespace ...")
    print(f"  top_k={cfg.top_k}  fetch_k={cfg.retrieval_fetch_k}  embed_dim={cfg.embed_dim}")
    print(f"  api_key={_mask(cfg.llmod_api_key)}   (last 4 chars only)")

    if namespace_stats("smoke", cfg)["vector_count"] == 0:
        print(
            "Namespace 'smoke' is empty. Run:\n"
            "  python scripts/ingest.py --limit 10 --namespace smoke",
            file=sys.stderr,
        )
        return 1

    q = "building habits that actually stick"
    print(f"\nQuery: {q!r}")

    deduped = retrieve(q, "smoke", cfg, dedup=True)
    _show("dedup=True (distinct articles)", deduped)
    assert len({r.article_id for r in deduped}) == len(deduped), (
        "dedup=True must be article-distinct"
    )

    raw = retrieve(q, "smoke", cfg, dedup=False)
    _show("dedup=False (top-k chunks as-is)", raw)

    print(
        "\nOK: retriever ran both modes; dedup=True returned distinct articles, "
        "dedup=False returned the raw top-k."
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- demo: surface any failure non-zero
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
