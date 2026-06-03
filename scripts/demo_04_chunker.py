"""Demo for Component 4 (Chunker).

Run:
    conda activate medium-rag
    python scripts/demo_04_chunker.py

Loads one real article from the Medium CSV, chunks it at three sizes
(256 / 512 / 1024 tokens, all at overlap_ratio=0.10), and prints a readable
comparison. Exits non-zero if any chunk exceeds its target chunk_size.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tiktoken

from src.data.csv_loader import load_articles
from src.rag.chunking import chunk_text, token_length


def _pair_overlap(a: str, b: str, enc) -> int:
    ta, tb = enc.encode(a), enc.encode(b)
    for k in range(min(len(ta), len(tb)), 0, -1):
        if ta[-k:] == tb[:k]:
            return k
    return 0


def _preview(s: str, n: int = 60) -> str:
    # Collapse newlines so the preview fits on one line.
    flat = " ".join(s.split())
    return flat[:n]


def main() -> int:
    print("Loading 1 article from medium-english-50mb.csv ...")
    arts = load_articles(limit=1)
    article = arts[0]

    body_chars = len(article.text)
    body_tokens = token_length(article.text)
    print(f"  title          = {article.title!r}")
    print(f"  body length    = {body_chars:,} chars / {body_tokens:,} tokens (cl100k_base)")
    print()

    enc = tiktoken.get_encoding("cl100k_base")

    any_oversize = False
    for cs in (256, 512, 1024):
        overlap_ratio = 0.10
        target_overlap = int(cs * overlap_ratio)
        chunks = chunk_text(article.text, cs, overlap_ratio)
        lens = [token_length(c) for c in chunks]
        oversize = [n for n in lens if n > cs]
        if oversize:
            any_oversize = True

        print(
            f"Chunking at chunk_size={cs}, overlap_ratio={overlap_ratio:.2f}  "
            f"(overlap ceiling = {target_overlap} tokens):"
        )
        print(f"  num chunks     = {len(chunks)}")
        for i, c in enumerate(chunks):
            print(f"  chunk[{i}]: {lens[i]:>4} tokens | \"{_preview(c)}...\"")
        if len(chunks) >= 2:
            overlaps = [_pair_overlap(chunks[i], chunks[i + 1], enc) for i in range(len(chunks) - 1)]
            avg = sum(overlaps) / len(overlaps)
            print(f"  avg overlap    = {avg:.1f} tokens (ceiling = {target_overlap})")
        else:
            print("  avg overlap    = n/a (single chunk)")
        print()

    if any_oversize:
        print("FAIL: at least one chunk exceeded its target chunk_size.", file=sys.stderr)
        return 1

    print("OK: chunker produces sensible token-sized chunks at all three sizes.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
