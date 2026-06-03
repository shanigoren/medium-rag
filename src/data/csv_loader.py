"""CSV loader for the Medium articles dataset.

Reads `medium-english-50mb.csv` (or any compatible CSV) into a list of
typed `Article` instances. Deterministic order = CSV row order.

Pure pandas + stdlib — no LLM, no vector DB, no LangChain.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# Columns we require in the CSV (in any order).
_REQUIRED_COLUMNS = ("title", "text", "url", "authors", "timestamp", "tags")

# Default filename, relative to the repo root (parent of `src/`).
_DEFAULT_CSV_NAME = "medium-english-50mb.csv"


@dataclass(frozen=True)
class Article:
    """One row from the Medium CSV.

    `row_idx` is the 0-based position in the original CSV — stable across
    limit slicing, so it's our canonical article ID.
    """

    row_idx: int
    title: str
    text: str
    url: str
    authors: list[str]
    timestamp: str
    tags: list[str]


def _repo_root() -> Path:
    """Project root = parent of `src/` (this file lives in `src/data/`)."""
    return Path(__file__).resolve().parent.parent.parent


def _default_csv_path() -> Path:
    return _repo_root() / _DEFAULT_CSV_NAME


def _parse_list_field(raw: Any) -> list[str]:
    """Parse a stringified Python list cell into list[str].

    Falls back to `[]` on any parse failure or non-list result. Empty
    strings (post-`fillna`) also yield `[]`.
    """
    if raw is None:
        return []
    if not isinstance(raw, str):
        # Defensive: pandas with dtype=str + fillna("") shouldn't produce this,
        # but be robust if a caller passes a custom dataframe.
        return []
    s = raw.strip()
    if s == "":
        return []
    try:
        parsed = ast.literal_eval(s)
    except (ValueError, SyntaxError, MemoryError, TypeError):
        return []
    if not isinstance(parsed, (list, tuple)):
        return []
    # Coerce each element to str so downstream consumers don't have to type-check.
    return [str(item) for item in parsed]


def load_articles(
    csv_path: Path | str | None = None,
    *,
    limit: int | None = None,
) -> list[Article]:
    """Load articles from the Medium CSV.

    Args:
        csv_path: path to the CSV. Defaults to `<repo_root>/medium-english-50mb.csv`.
        limit: if given, return only the first N rows (in CSV order). None = all.

    Returns:
        A list of `Article`, ordered as in the CSV. `row_idx` is the 0-based
        position in the original CSV (so `limit=10` returns row_idx 0..9).

    Raises:
        FileNotFoundError: if csv_path is missing.
        ValueError: if required columns are absent or the file is empty.
    """
    path = Path(csv_path) if csv_path is not None else _default_csv_path()
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")

    # dtype=str avoids pandas inferring NaN-heavy list columns as float.
    # nrows=limit reads only what we need from disk.
    df = pd.read_csv(path, nrows=limit, dtype=str)
    df = df.fillna("")

    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {missing} (got {list(df.columns)})"
        )

    if len(df) == 0:
        raise ValueError(f"CSV has no data rows: {path}")

    articles: list[Article] = []
    # df.itertuples() preserves CSV row order; index=True gives us the 0-based row index
    # which (because we pass nrows=limit before any slicing) is exactly what we want
    # for row_idx — for limit=10, row_idx values are 0..9.
    for row in df.itertuples(index=True):
        articles.append(
            Article(
                row_idx=int(row.Index),
                title=str(row.title),
                text=str(row.text),
                url=str(row.url),
                authors=_parse_list_field(row.authors),
                timestamp=str(row.timestamp),
                tags=_parse_list_field(row.tags),
            )
        )

    return articles
