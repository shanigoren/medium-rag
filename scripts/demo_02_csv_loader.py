"""Demo for Component 2 (CSV loader).

Run:
    conda activate medium-rag
    python scripts/demo_02_csv_loader.py            # default: 5 articles
    python scripts/demo_02_csv_loader.py --limit 10

Loads the Medium CSV from <repo_root>/medium-english-50mb.csv, prints each
article's row_idx, title, author(s), tag count, and a short text snippet.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.csv_loader import load_articles


def _snippet(text: str, n: int = 240, indent: str = "            ") -> str:
    """Return the first ~n chars of `text`, preserving newlines so the
    paragraph structure (which the chunker will see) is visible. Each
    line after the first is indented to align under the `text:` label.
    """
    truncated = text[:n]
    if len(text) > n:
        truncated = truncated.rstrip() + "..."
    lines = truncated.splitlines()
    if not lines:
        return ""
    return ("\n" + indent).join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Show the first N Medium articles.")
    parser.add_argument("--limit", type=int, default=5, help="how many articles to show (default 5)")
    args = parser.parse_args()

    try:
        articles = load_articles(limit=args.limit)
    except FileNotFoundError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"FAILED to parse CSV: {exc}", file=sys.stderr)
        return 1

    print(f"Loaded {len(articles)} article(s) from medium-english-50mb.csv\n")

    for art in articles:
        authors = ", ".join(art.authors) if art.authors else "<no authors>"
        tags = ", ".join(art.tags) if art.tags else "<no tags>"
        print(f"[row_idx={art.row_idx}]  {art.title}")
        print(f"  authors : {authors}")
        print(f"  tags ({len(art.tags)}): {tags}")
        print(f"  url     : {art.url}")
        print(f"  text    : {_snippet(art.text)}")
        print()

    print(f"OK: {len(articles)} article(s) parsed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
