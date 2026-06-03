from __future__ import annotations

import argparse

from src.config import load_config
from src.llm.clients import get_embeddings
from src.rag.vectorstore import namespace_stats, query


QUERY = "past pandemics (such as the bubonic plague) can spur innovation and recovery"
TARGET_ARTICLE_ID = "6299"
NAMESPACES = [
    "prod",
    "probe_bubonic_c512_o10",
    "probe_bubonic_c512_o20",
    "probe_bubonic_c512_o30",
    "probe_bubonic_c1024_o10",
    "probe_bubonic_c1024_o20",
    "probe_bubonic_c1024_o30",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("namespaces", nargs="*", default=NAMESPACES)
    args = parser.parse_args()

    cfg = load_config()
    vector = get_embeddings(cfg).embed_query(QUERY)
    print(f"QUERY {QUERY}")

    for namespace in args.namespaces:
        stats = namespace_stats(namespace, cfg)
        count = int(stats.get("vector_count", 0) or 0)
        top_k = min(max(count, 1), 1000)
        hits = query(namespace, vector, top_k=top_k, cfg=cfg) if count else []

        target_hits = []
        first_distinct_pos = None
        seen_articles: list[str] = []
        seen = set()

        for rank, hit in enumerate(hits, start=1):
            metadata = hit.metadata or {}
            article_id = str(metadata.get("article_id", ""))
            if article_id not in seen:
                seen.add(article_id)
                seen_articles.append(article_id)
            if article_id == TARGET_ARTICLE_ID:
                if first_distinct_pos is None:
                    first_distinct_pos = seen_articles.index(article_id) + 1
                target_hits.append(
                    (
                        rank,
                        hit.score,
                        hit.id,
                        metadata.get("chunk_idx"),
                        metadata.get("title"),
                    )
                )

        print(
            f"\nNAMESPACE {namespace} vector_count={count} "
            f"queried_top_k={top_k} hits={len(hits)}"
        )
        if not target_hits:
            print("TARGET_NOT_FOUND_IN_WINDOW")
            if hits:
                last = hits[-1]
                print(
                    f"LOWEST_RETURNED rank={len(hits)} score={last.score:.6f} "
                    f"article_id={(last.metadata or {}).get('article_id')} id={last.id}"
                )
            continue

        print(
            f"TARGET_FIRST_CHUNK_RANK {target_hits[0][0]} "
            f"DISTINCT_ARTICLE_RANK {first_distinct_pos}"
        )
        for rank, score, vector_id, chunk_idx, title in target_hits:
            print(
                f"  rank={rank} score={score:.6f} id={vector_id} "
                f"chunk_idx={chunk_idx} title={title}"
            )

        first_rank = target_hits[0][0]
        lo = max(1, first_rank - 3)
        hi = min(len(hits), first_rank + 3)
        print(f"  NEIGHBORHOOD ranks {lo}-{hi}")
        for rank in range(lo, hi + 1):
            hit = hits[rank - 1]
            metadata = hit.metadata or {}
            print(
                f"    #{rank:03d} score={hit.score:.6f} "
                f"article_id={metadata.get('article_id')} id={hit.id} "
                f"title={metadata.get('title')}"
            )


if __name__ == "__main__":
    main()
