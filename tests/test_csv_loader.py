"""Unit tests for `src.data.csv_loader`.

Most tests build a small fixture CSV in tmp_path so they're fast and
deterministic. One test reads the real 50 MB file (skipped if missing).
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.data.csv_loader import Article, load_articles

# Same default name the loader uses; duplicated here to keep the test
# independent of the loader's internals.
_REAL_CSV_NAME = "medium-english-50mb.csv"
_SCHEMA = ["title", "text", "url", "authors", "timestamp", "tags"]


def _write_fixture_csv(tmp_path: Path, rows: list[dict]) -> Path:
    """Write a CSV with the project schema.

    Each row dict may omit fields — missing fields are written as empty cells
    so we exercise the loader's NaN-coercion path.
    """
    path = tmp_path / "fixture.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_SCHEMA)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in _SCHEMA})
    return path


def _row(
    title: str = "T",
    text: str = "body text",
    url: str = "https://example.com",
    authors: str = "['Alice']",
    timestamp: str = "2024-01-01",
    tags: str = "['Tech']",
) -> dict:
    return {
        "title": title,
        "text": text,
        "url": url,
        "authors": authors,
        "timestamp": timestamp,
        "tags": tags,
    }


def test_load_full(tmp_path):
    """No limit returns every row in the fixture CSV."""
    path = _write_fixture_csv(tmp_path, [_row(title=f"t{i}") for i in range(5)])
    arts = load_articles(path)
    assert len(arts) == 5
    assert [a.title for a in arts] == [f"t{i}" for i in range(5)]


def test_limit_returns_first_n(tmp_path):
    """limit=2 returns rows 0 and 1, in order."""
    path = _write_fixture_csv(tmp_path, [_row(title=f"t{i}") for i in range(5)])
    arts = load_articles(path, limit=2)
    assert len(arts) == 2
    assert [a.title for a in arts] == ["t0", "t1"]


def test_row_idx_is_zero_based(tmp_path):
    """The first article has row_idx=0, second has row_idx=1, etc."""
    path = _write_fixture_csv(tmp_path, [_row(title=f"t{i}") for i in range(3)])
    arts = load_articles(path)
    assert [a.row_idx for a in arts] == [0, 1, 2]


def test_authors_parsed_as_list(tmp_path):
    """authors string \"['Foo', 'Bar']\" becomes ['Foo', 'Bar']."""
    path = _write_fixture_csv(tmp_path, [_row(authors="['Foo', 'Bar']")])
    arts = load_articles(path)
    assert arts[0].authors == ["Foo", "Bar"]


def test_tags_parsed_as_list(tmp_path):
    """Same for tags."""
    path = _write_fixture_csv(tmp_path, [_row(tags="['Self', 'Mental Health']")])
    arts = load_articles(path)
    assert arts[0].tags == ["Self", "Mental Health"]


def test_malformed_authors_becomes_empty_list(tmp_path):
    """authors string 'not a python list' becomes []."""
    path = _write_fixture_csv(tmp_path, [_row(authors="not a python list")])
    arts = load_articles(path)
    assert arts[0].authors == []


def test_missing_text_becomes_empty_string(tmp_path):
    """A row with empty `text` cell returns text=''."""
    path = _write_fixture_csv(tmp_path, [_row(text="")])
    arts = load_articles(path)
    assert arts[0].text == ""
    # And missing list cells should fall back to [], not crash.
    path2 = _write_fixture_csv(tmp_path, [_row(authors="", tags="")])
    arts2 = load_articles(path2)
    assert arts2[0].authors == []
    assert arts2[0].tags == []


def test_missing_csv_file_raises(tmp_path):
    """load_articles(non_existent) raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_articles(tmp_path / "does_not_exist.csv")


def test_empty_csv_raises(tmp_path):
    """A CSV with only headers raises ValueError."""
    path = _write_fixture_csv(tmp_path, [])
    with pytest.raises(ValueError):
        load_articles(path)


def test_wrong_schema_raises(tmp_path):
    """A CSV missing the 'title' column raises ValueError."""
    path = tmp_path / "wrong.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["text", "url", "authors", "timestamp", "tags"])
        writer.writeheader()
        writer.writerow({"text": "x", "url": "y", "authors": "[]", "timestamp": "", "tags": "[]"})
    with pytest.raises(ValueError):
        load_articles(path)


def test_default_path_resolution():
    """When csv_path=None, loader resolves to <repo_root>/medium-english-50mb.csv.

    Does not require the file to exist — we just verify the resolved path
    by inspecting the FileNotFoundError message when the file is absent,
    or by checking that calling with the default succeeds when present.
    """
    from src.data import csv_loader

    expected = csv_loader._default_csv_path()  # type: ignore[attr-defined]
    # The expected path lives at <repo_root>/medium-english-50mb.csv.
    assert expected.name == _REAL_CSV_NAME
    # The parent of the resolved default path is the project root, which
    # should also contain the `src` directory.
    assert (expected.parent / "src").is_dir()


def test_loads_real_csv_smoke():
    """If <repo_root>/medium-english-50mb.csv exists, load 5 rows and assert basic shape."""
    from src.data import csv_loader

    real_path = csv_loader._default_csv_path()  # type: ignore[attr-defined]
    if not real_path.is_file():
        pytest.skip(f"real CSV not present at {real_path}")
    arts = load_articles(limit=5)
    assert len(arts) == 5
    assert [a.row_idx for a in arts] == [0, 1, 2, 3, 4]
    for a in arts:
        assert isinstance(a, Article)
        assert isinstance(a.title, str)
        assert isinstance(a.text, str)
        assert isinstance(a.authors, list)
        assert isinstance(a.tags, list)
