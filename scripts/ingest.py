"""Ingestion pipeline for the Medium RAG project (Component 7).

Composes the lower layers into one runnable pipeline:

    load_articles (C2) -> chunk_text (C4) -> build_embed_text (C6)
        -> embed_batch (C6) -> ensure_index + upsert (C5)

One CLI invocation = one ingest into one Pinecone namespace at one config:

    python scripts/ingest.py --namespace NAME [--limit N] [--override k=v ...] [--clean] [--csv PATH]

This component owns three things and nothing else:
  1. Orchestration of the C2->C4->C6->C5 chain.
  2. The per-chunk metadata schema C5 stores verbatim.
  3. The deterministic-ID / index-alignment contract (ids f"{row_idx}-{chunk_idx}",
     vectors[i] is the embedding of metadatas[i]).

It does no chunking math, no embed-string rendering/batching, and no Pinecone
client/upsert mechanics -- it wires those together. Import-safe: no work at
import time; the CLI is guarded by `if __name__ == "__main__"`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Make `src` importable when run as `python scripts/ingest.py` (the script dir,
# not the repo root, is sys.path[0] in that case). Harmless when imported as
# `scripts.ingest` under pytest, where the repo root is already on the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config, _ENV_OVERRIDES, load_config
from src.data.csv_loader import Article, load_articles
from src.rag.chunking import chunk_text
from src.rag.embed import build_embed_text, embed_batch
from src.rag.vectorstore import (
    WRITE_CONSISTENCY_POLL_S,
    WRITE_CONSISTENCY_TIMEOUT_S,
    delete_namespace,
    ensure_index,
    namespace_stats,
    upsert,
)

# Flush the buffer (embed_batch + upsert, then clear) every FLUSH_CHUNKS buffered
# chunks. Bounds peak resident embed strings AND vectors so the full ~7,600
# ingest never holds ~23k x 1536-float vectors (~280 MB) at once, and gives
# per-flush progress. Idempotency is unaffected: IDs are deterministic
# regardless of where flush boundaries fall.
FLUSH_CHUNKS = 1024


@dataclass(frozen=True)
class IngestStats:
    articles_total: int       # articles handed in
    articles_chunked: int     # produced >= 1 chunk
    articles_skipped: int     # produced 0 chunks (empty/whitespace body)
    chunks_total: int         # == vectors_upserted on success
    vectors_upserted: int     # sum of C5 upsert() return values
    namespace: str


def _records_for_article(
    article: Article, cfg: Config
) -> tuple[list[str], list[str], list[dict]]:
    """SINGLE SOURCE OF TRUTH for one article's records.

    chunk_text(...) -> for each chunk produce:
      id        = f"{article.row_idx}-{chunk_idx}"
      embed_txt = build_embed_text(article, chunk, cfg.embed_content)  # embedded
      metadata  = {article_id, title, authors, url, timestamp, tags, chunk, chunk_idx}

    `chunk` is the RAW chunk (no Title:/Tags: prefix) so the API returns clean
    text; embed_txt (the prefixed string) is NEVER stored. Empty `authors`/`tags`
    lists are OMITTED from the metadata entirely (Pinecone rejects null and may
    reject []); empty scalars (e.g. title="") are KEPT as "". Zero-chunk articles
    return ([], [], []).

    Both build_records and ingest_articles call THIS, so the pure-tested logic
    and the streaming loop cannot drift.
    """
    chunks = chunk_text(article.text, cfg.chunk_size, cfg.overlap_ratio)

    ids: list[str] = []
    embed_texts: list[str] = []
    metadatas: list[dict] = []
    for chunk_idx, chunk in enumerate(chunks):
        meta: dict = {
            "article_id": str(article.row_idx),
            "title": article.title,
            "url": article.url,
            "timestamp": article.timestamp,
            "chunk": chunk,
            "chunk_idx": chunk_idx,
        }
        # Omit empty lists -- Pinecone never sees [] (and never None). A missing
        # key is always valid; the retriever/API treat absent authors/tags as
        # empty. Non-empty scalars are kept above unconditionally.
        if article.authors:
            meta["authors"] = article.authors
        if article.tags:
            meta["tags"] = article.tags

        ids.append(f"{article.row_idx}-{chunk_idx}")
        embed_texts.append(build_embed_text(article, chunk, cfg.embed_content))
        metadatas.append(meta)

    return ids, embed_texts, metadatas


def build_records(
    articles: list[Article], cfg: Config
) -> tuple[list[str], list[str], list[dict]]:
    """Pure assembly over many articles: concatenate _records_for_article across
    `articles` into three index-aligned lists (ids, embed_texts, metadatas).

    No network, no embedding -- the snapshot-testable core. NOT used by
    ingest_articles (which streams via the same _records_for_article helper to
    bound memory). Articles yielding zero chunks contribute nothing.
    """
    ids: list[str] = []
    embed_texts: list[str] = []
    metadatas: list[dict] = []
    for article in articles:
        a_ids, a_texts, a_metas = _records_for_article(article, cfg)
        ids.extend(a_ids)
        embed_texts.extend(a_texts)
        metadatas.extend(a_metas)
    return ids, embed_texts, metadatas


def ingest_articles(
    articles: list[Article], namespace: str, cfg: Config
) -> IngestStats:
    """Embed + upsert `articles` into `namespace`. Assumes ensure_index() ran.

    Streams: for each article extend a buffer with _records_for_article(article,
    cfg); when the buffer reaches FLUSH_CHUNKS, embed_batch(embed_texts, cfg) then
    upsert(namespace, ids, vectors, metadatas, cfg) and clear it; flush the
    remainder after the loop. Bounding the buffer caps peak resident embed
    strings AND vectors. Prints ASCII-only per-flush progress. Returns IngestStats.
    """
    buf_ids: list[str] = []
    buf_texts: list[str] = []
    buf_metas: list[dict] = []

    articles_chunked = 0
    articles_skipped = 0
    chunks_total = 0
    vectors_upserted = 0

    def _flush() -> int:
        nonlocal buf_ids, buf_texts, buf_metas
        if not buf_ids:
            return 0
        vectors = embed_batch(buf_texts, cfg)
        written = upsert(namespace, buf_ids, vectors, buf_metas, cfg)
        buf_ids, buf_texts, buf_metas = [], [], []
        return written

    for article in articles:
        a_ids, a_texts, a_metas = _records_for_article(article, cfg)
        if a_ids:
            articles_chunked += 1
            chunks_total += len(a_ids)
        else:
            articles_skipped += 1
            continue

        buf_ids.extend(a_ids)
        buf_texts.extend(a_texts)
        buf_metas.extend(a_metas)

        # A long article may push the buffer a little over FLUSH_CHUNKS -- fine,
        # embed_batch re-batches internally.
        if len(buf_ids) >= FLUSH_CHUNKS:
            written = _flush()
            vectors_upserted += written
            print(f"  flushed {vectors_upserted} chunks -> {namespace}")

    written = _flush()
    if written:
        vectors_upserted += written
        print(f"  flushed {vectors_upserted} chunks -> {namespace}")

    return IngestStats(
        articles_total=len(articles),
        articles_chunked=articles_chunked,
        articles_skipped=articles_skipped,
        chunks_total=chunks_total,
        vectors_upserted=vectors_upserted,
        namespace=namespace,
    )


def run_ingest(
    namespace: str,
    cfg: Config | None = None,
    *,
    limit: int | None = None,
    clean: bool = False,
    csv_path=None,
) -> IngestStats:
    """End-to-end entry point used by the CLI, the demo, and C15.

    load_articles(limit) -> ensure_index(cfg) -> [if clean: delete_namespace then
    POLL namespace_stats until vector_count == 0] -> ingest_articles(...).

    The post-delete poll closes the Pinecone delete_all/upsert race: without it a
    still-in-flight delete_all can land AFTER the re-upsert and wipe fresh
    vectors. cfg defaults to load_config().
    """
    if cfg is None:
        cfg = load_config()

    articles = load_articles(csv_path, limit=limit)
    ensure_index(cfg)

    if clean:
        delete_namespace(namespace, cfg)
        _wait_for_empty(namespace, cfg)

    return ingest_articles(articles, namespace, cfg)


def _wait_for_empty(namespace: str, cfg: Config) -> None:
    """Poll namespace_stats until vector_count == 0 (bounded), then proceed.

    Pinecone delete_all is eventually consistent; an immediate re-upsert could be
    overwritten by the still-settling delete. We wait up to
    WRITE_CONSISTENCY_TIMEOUT_S, then proceed regardless (a fresh namespace reads
    0 on the first poll and this is a near-no-op).
    """
    deadline = time.monotonic() + WRITE_CONSISTENCY_TIMEOUT_S
    while True:
        if namespace_stats(namespace, cfg)["vector_count"] == 0:
            return
        if time.monotonic() >= deadline:
            return
        time.sleep(WRITE_CONSISTENCY_POLL_S)


def _load_with_overrides(overrides: dict[str, str]) -> Config:
    """Apply --override key=value through C1's validated reload.

    Build a reverse map from _ENV_OVERRIDES (field name -> env-var name), set the
    env vars inside a snapshot/restore guard, and call load_config() so C1's
    casters AND _validate run unchanged. The finally restores os.environ exactly
    as found, so calling main() in-process leaves no residue to leak into the
    next caller/test. Unknown keys raise SystemExit naming the allowed set.
    """
    reverse = {field: env for env, (field, _c) in _ENV_OVERRIDES.items()}
    bad = [k for k in overrides if k not in reverse]
    if bad:
        raise SystemExit(
            f"--override key(s) {bad} not allowed; choose from {sorted(reverse)}"
        )
    saved = {reverse[k]: os.environ.get(reverse[k]) for k in overrides}
    try:
        for k, v in overrides.items():
            os.environ[reverse[k]] = v
        return load_config()
    finally:
        for env_name, prev in saved.items():
            if prev is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = prev


def _parse_overrides(pairs: list[str]) -> dict[str, str]:
    """Parse repeated `key=value` flags into a dict. A token without '=' is a
    clean SystemExit, not an IndexError/ValueError traceback."""
    out: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(
                f"--override must be key=value (got {pair!r}); missing '='"
            )
        key, value = pair.split("=", 1)
        out[key] = value
    return out


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest Medium articles into a Pinecone namespace."
    )
    parser.add_argument(
        "--namespace", required=True, help="Pinecone namespace to write into."
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Ingest only the first N CSV rows."
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a config field for this run only (repeatable).",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="delete_namespace(NAME) before ingesting (waits for empty).",
    )
    parser.add_argument(
        "--csv", default=None, help="Override the CSV path (mainly for tests)."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """argparse CLI. Parses flags, applies --override, calls run_ingest, prints
    the one-line summary, returns an exit code (0 success, non-zero failure)."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        overrides = _parse_overrides(args.override)
        cfg = _load_with_overrides(overrides) if overrides else load_config()
        stats = run_ingest(
            args.namespace,
            cfg,
            limit=args.limit,
            clean=args.clean,
            csv_path=args.csv,
        )
    except SystemExit as exc:
        # argparse / override errors already carry a readable message.
        print(f"ERROR: {exc}")
        return 1
    except Exception as exc:  # noqa: BLE001 -- top-level CLI guard
        print(f"ERROR: {type(exc).__name__}: {exc}")
        return 1

    print(
        f"OK: namespace={stats.namespace} articles={stats.articles_total} "
        f"chunked={stats.articles_chunked} skipped={stats.articles_skipped} "
        f"chunks={stats.chunks_total} vectors={stats.vectors_upserted}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
