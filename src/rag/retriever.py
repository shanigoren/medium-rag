"""Component 8: the Retriever.

Turn a query string into a ranked list of chunks read back out of one Pinecone
namespace -- either article-deduplicated (`dedup=True`) or as-is (`dedup=False`),
depending on the caller-set flag. This is the first component on the *read* path:

    query string -> embed_query (C3) -> vectorstore.query (C5)
                 -> [dedup=True: collapse-by-article | dedup=False: top-k as-is]
                 -> list[RetrievedChunk]

The component embeds the query RAW (no title/tags prefix -- that prefix is a
document-side lever owned by C6) and obeys the `dedup` flag. It never rewrites
the query (C10) and never DECIDES `dedup` (also C10); it is a pure, deterministic
function of (query, namespace, cfg, top_k, fetch_k, dedup).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import Config, load_config
from src.llm.clients import get_embeddings
from src.rag.vectorstore import Match
from src.rag.vectorstore import query as _vs_query


@dataclass(frozen=True)
class RetrievedChunk:
    """One retrieval result. Under dedup=True it is the best chunk of one
    distinct article; under dedup=False it is simply one of the top-k chunks
    (and another record in the same list may share its article_id).

    Field provenance and downstream consumers:
      article_id  str        # from metadata['article_id'], re-str()'d (invariant)
      title       str        # from metadata['title']  -> API context + C9 header
      authors     list[str]  # from metadata.get('authors', []) -> C9 header only
      tags        list[str]  # from metadata.get('tags', []) -> C9 header only (topical
                             #   labels; help the model judge which article matches a
                             #   topic). NOT in the API context (wire schema is fixed).
      chunk       str        # RAW chunk (no embed prefix) -> API context + prompt
      score       float      # Match.score (cosine, desc) -> API context + eval
      chunk_idx   int        # from metadata['chunk_idx'] -> eval results.jsonl
      rank        int        # 1-based position in THIS returned list
    """

    article_id: str
    title: str
    authors: list[str]
    tags: list[str]
    chunk: str
    score: float
    chunk_idx: int
    rank: int


def _to_chunk(m: Match, rank: int) -> RetrievedChunk | None:
    """Map one C5 `Match` to a `RetrievedChunk`, or `None` if it is unusable.

    The single shared record builder for both dedup branches: the field
    assembly, the `str(article_id)` invariant, the `chunk_idx` int-coercion, and
    the missing-metadata skip all live here so the two paths cannot drift.

    Returns `None` when `m.metadata` lacks `article_id` or `chunk` (a vector
    upserted without full metadata is unusable for an answer). Other keys fall
    back to ""/[]/0.
    """
    md = m.metadata
    if "article_id" not in md or "chunk" not in md:
        return None
    return RetrievedChunk(
        article_id=str(md["article_id"]),
        title=md.get("title", ""),
        authors=list(md.get("authors", [])),
        tags=list(md.get("tags", [])),
        chunk=md["chunk"],
        score=m.score,
        chunk_idx=int(md.get("chunk_idx", 0)),
        rank=rank,
    )


def retrieve(
    query: str,
    namespace: str,
    cfg: Config | None = None,
    *,
    top_k: int | None = None,
    fetch_k: int | None = None,
    dedup: bool = True,
) -> list[RetrievedChunk]:
    """Embed `query` and return up to `top_k` RetrievedChunks from
    `namespace`, ordered by descending score.

    `dedup` (set per-question by the C10 rewriter; defaults True = safe baseline):
      - dedup=True : over-fetch `fetch_k` candidates, then collapse to one best
        chunk per article_id; return up to `top_k` DISTINCT articles. For type-2
        "list N distinct articles" questions.
      - dedup=False: return the `top_k` highest-scoring chunks as-is; two records
        may share an article_id. For type-1/3/4 questions, where depth (several
        chunks of one article) helps more than breadth.

    `cfg` defaults to load_config(). `top_k` defaults to cfg.top_k; `fetch_k`
    defaults to cfg.retrieval_fetch_k (used only when dedup=True). `query` is
    embedded RAW via embed_query (no title prefix) -- the retriever never rewrites
    and is unaware whether the string is the original question or C10's rewrite.

    Dedupe (dedup=True): vectorstore.query returns matches sorted by score desc,
    so the FIRST occurrence of each article_id is that article's best chunk. We
    keep first-seen per article_id and stop once we have top_k distinct articles.

    Returns [] for an empty/missing namespace (C5.query already degrades to []).
    Under dedup=True, returns fewer than top_k if the namespace holds fewer
    distinct articles. Raises ValueError on an empty/whitespace query.
    """
    # 1. Validate before any embedding call -- embedding an empty query is a
    #    caller bug; fail loud, don't waste an API call.
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty, non-whitespace string")

    # 2. Resolve params.
    if cfg is None:
        cfg = load_config()
    k = top_k if top_k is not None else cfg.top_k
    if dedup:
        fk = fetch_k if fetch_k is not None else cfg.retrieval_fetch_k
        fk = max(fk, k)  # clamp (not raise) so eval can sweep top_k
        query_top_k = fk
    else:
        query_top_k = k  # nothing to collapse -> query exactly top_k, no over-fetch

    # 3. Embed the query raw (single query -> embed_query, NOT embed_documents).
    vector = get_embeddings(cfg).embed_query(query)

    # 4. Query Pinecone (matches come back sorted by score desc; we do not re-sort).
    matches = _vs_query(namespace, vector, top_k=query_top_k, cfg=cfg)

    # 5. Build the result -- branch on dedup; both share _to_chunk.
    out: list[RetrievedChunk] = []
    if dedup:
        seen: set[str] = set()
        for m in matches:
            rc = _to_chunk(m, rank=len(out) + 1)
            if rc is None:
                continue  # skip BEFORE the seen bookkeeping -- no seen pollution
            if rc.article_id in seen:
                continue  # already kept this article's BEST (first-seen) chunk
            seen.add(rc.article_id)
            out.append(rc)
            if len(out) == k:
                break
    else:
        for m in matches:  # already <= k from the query call
            rc = _to_chunk(m, rank=len(out) + 1)
            if rc is None:
                continue  # skip unusable; do NOT over-fetch to backfill the slot
            out.append(rc)
    return out
