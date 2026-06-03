"""Demo for Component 6 (embed builder).

Run:
    conda activate medium-rag
    python scripts/demo_06_embed.py

Loads one real article, chunks it, renders all three embed_content modes for
the first chunk, then embeds the three rendered strings and shows they come
back as real 1536-d vectors. Costs a few cents (one embed call on 3 short
strings). ASCII-only output (this console is cp1252).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.csv_loader import load_articles
from src.rag.chunking import chunk_text
from src.rag.embed import build_embed_text, embed_batch

MODES = ("chunk_only", "title_chunk", "title_tags_chunk")


def _mask(secret: str) -> str:
    if not secret or len(secret) < 4:
        return "****"
    return "..." + secret[-4:]


def main() -> int:
    cfg = load_config()
    print()
    print("Loading config and embeddings client ...")
    print(f"  embed_model = {cfg.embed_model}")
    print(f"  embed_dim   = {cfg.embed_dim}")
    print(f"  api_key     = {_mask(cfg.llmod_api_key)}   (last 4 chars only)")
    print()

    article = load_articles(limit=1)[0]  # needs medium-english-50mb.csv in repo root
    chunks = chunk_text(article.text, cfg.chunk_size, cfg.overlap_ratio)
    if not chunks:
        print("First article produced no chunks; nothing to embed.", file=sys.stderr)
        return 1
    first = chunks[0]

    print(f"Article: {article.title!r}  ({len(chunks)} chunks)")
    for mode in MODES:
        rendered = build_embed_text(article, first, mode)
        preview = rendered[:200].replace("\n", "\\n")
        print()
        print(f"[{mode}] embed string (first 200 chars):")
        print("  " + preview)
    print()

    print("Embedding 3 strings (one per mode) via embed_batch ...")
    texts = [build_embed_text(article, first, m) for m in MODES]
    vectors = embed_batch(texts, cfg)
    for mode, v in zip(MODES, vectors):
        print(
            f"  {mode:18s} -> dim={len(v)}  "
            f"first 3 = [{v[0]:.4f}, {v[1]:.4f}, {v[2]:.4f}]"
        )

    if not all(len(v) == cfg.embed_dim for v in vectors):
        print("FAILED: unexpected vector dimension", file=sys.stderr)
        return 1

    print()
    print(f"OK: embed builder produced {len(vectors)} vectors of dim {cfg.embed_dim}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
