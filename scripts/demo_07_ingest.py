"""Demo for Component 7 (ingestion pipeline).

Run:
    conda activate medium-rag
    python scripts/demo_07_ingest.py

Ingests a tiny slice of real articles into a throwaway namespace, proves the
vectors are queryable with the correct metadata schema, shows the stored chunk
is clean prose (not the prefixed embed string), then deletes the namespace so
the demo leaves no residue. Costs a few cents (3 embedded articles + 1 query).
ASCII-only output (this console is cp1252).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.ingest import run_ingest
from src.config import load_config
from src.llm.clients import get_embeddings
from src.rag.vectorstore import (
    WRITE_CONSISTENCY_POLL_S,
    WRITE_CONSISTENCY_TIMEOUT_S,
    delete_namespace,
    namespace_stats,
    query,
)

NS = "demo_ingest"


def _mask(secret: str) -> str:
    if not secret or len(secret) < 4:
        return "****"
    return "..." + secret[-4:]


def _wait_for_count(ns, cfg, expected, timeout_s=WRITE_CONSISTENCY_TIMEOUT_S) -> int:
    import time

    deadline = time.monotonic() + timeout_s
    count = namespace_stats(ns, cfg)["vector_count"]
    while count != expected and time.monotonic() < deadline:
        time.sleep(WRITE_CONSISTENCY_POLL_S)
        count = namespace_stats(ns, cfg)["vector_count"]
    return count


def main() -> int:
    cfg = load_config()
    print()
    print("Ingesting a tiny slice to verify the full pipeline ...")
    print(
        f"  chunk_size={cfg.chunk_size}  overlap={cfg.overlap_ratio}  "
        f"embed_content={cfg.embed_content}"
    )
    print(f"  api_key={_mask(cfg.llmod_api_key)}   (last 4 chars only)")

    stats = run_ingest(NS, cfg, limit=3, clean=True)  # throwaway namespace
    print()
    print(
        f"Articles: {stats.articles_total}  chunked={stats.articles_chunked}  "
        f"skipped={stats.articles_skipped}  vectors={stats.vectors_upserted}"
    )

    count = _wait_for_count(NS, cfg, expected=stats.vectors_upserted)
    print(f"namespace '{NS}' now reports {count} vectors")

    qv = get_embeddings(cfg).embed_query("a sentence from the first article")
    matches = query(NS, qv, top_k=1, cfg=cfg)
    if not matches:
        print("FAILED: query returned no matches", file=sys.stderr)
        delete_namespace(NS, cfg)
        return 1
    m = matches[0].metadata

    print()
    print("Top match metadata:")
    print(f"  article_id={m['article_id']}  chunk_idx={m['chunk_idx']}")
    print(f"  title={m['title']!r}")
    print("  chunk (first 120 chars, raw, no Title: prefix):")
    print("  " + m["chunk"][:120].replace("\n", "\\n"))

    if m["chunk"].startswith("Title:"):
        print("FAILED: raw chunk must not carry the embed prefix", file=sys.stderr)
        delete_namespace(NS, cfg)
        return 1

    delete_namespace(NS, cfg)  # leave no residue
    print()
    print(f"OK: ingest pipeline verified; throwaway namespace '{NS}' deleted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
