"""Embed builder for the Medium RAG pipeline.

Two steps sit between "we have chunks" and "we have vectors in Pinecone":

1. `build_embed_text(article, chunk, mode)` renders the exact string that gets
   embedded, for one of the three `embed_content` modes. The raw chunk (no
   prefix) is what the ingest pipeline stores in metadata so the API returns
   clean text; the prefixed string returned here is what the embedder sees.
2. `embed_batch(texts)` turns a list of strings into a list of 1536-d vectors
   via the LLMod-configured `OpenAIEmbeddings` client, preserving input order.

No chunking (Component 4 owns that) and no upserting (Component 5 owns that).
This module reaches the embedding API only through `src.llm.clients`; it never
imports the OpenAI SDK or pandas directly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.config import Config
from src.llm.clients import get_embeddings

if TYPE_CHECKING:
    # Type-hint only. Importing csv_loader.Article for real would drag pandas
    # into the embed path; with `from __future__ import annotations` the hint
    # is never evaluated at runtime, so we stay pandas-free and read
    # `.title` / `.tags` by duck typing.
    from src.data.csv_loader import Article


# How many texts go in one embed_documents() call.
# Safe at chunk_size<=512; at chunk_size=1024 a full batch nears the embeddings
# endpoint's aggregate per-request token budget. langchain-openai
# batches by text-count, not aggregate tokens, so it won't protect us here.
EMBED_BATCH_SIZE = 256

# The three embed_content modes. Mirrors src.config._VALID_EMBED_CONTENT;
# test_build_modes_match_config_enum guards against drift.
_VALID_MODES = ("chunk_only", "title_chunk", "title_tags_chunk")


def build_embed_text(article: "Article", chunk: str, mode: str) -> str:
    """Render the string to embed for one chunk under the given embed_content mode.

    The returned string is what gets EMBEDDED. The raw `chunk` (no prefix) is
    what Component 7 stores in metadata under the `chunk` key, so the API can
    return clean text. These are deliberately different strings.

    Modes (must match src.config._VALID_EMBED_CONTENT exactly):
      - "chunk_only"        -> chunk
      - "title_chunk"       -> f"Title: {title}\n\n{chunk}"
      - "title_tags_chunk"  -> f"Title: {title}\nTags: {tags}\n\n{chunk}"
                               where tags = ", ".join(article.tags)

    `article` only needs `.title: str` and `.tags: list[str]` attributes — any
    object with those works (tests pass a lightweight stub; production passes a
    csv_loader.Article). `mode` is an explicit argument, NOT read from config,
    so this function stays pure and trivially snapshot-testable; the caller
    (Component 7 / the experiment runner) passes `cfg.embed_content`.

    The chunk is rendered verbatim — no stripping or normalisation — so the
    embedded text stays in sync with the raw `chunk` stored in metadata.

    Raises:
        ValueError: if `mode` is not one of the three valid modes.
    """
    if mode == "chunk_only":
        return chunk
    if mode == "title_chunk":
        return f"Title: {article.title}\n\n{chunk}"
    if mode == "title_tags_chunk":
        tags = ", ".join(article.tags)
        return f"Title: {article.title}\nTags: {tags}\n\n{chunk}"
    raise ValueError(
        f"unknown embed mode {mode!r}; must be one of {list(_VALID_MODES)}"
    )


def embed_batch(texts: list[str], cfg: Config | None = None) -> list[list[float]]:
    """Embed `texts` into 1536-d vectors, preserving order 1:1 with the input.

    Slices `texts` into batches of <= EMBED_BATCH_SIZE and calls
    `embed_documents(batch)` once per batch on a single client, concatenating
    the results in order. Empty input -> returns [] and makes zero API calls.

    The returned list satisfies: len(result) == len(texts), and result[i] is
    the embedding of texts[i]. Each vector has length cfg.embed_dim (1536),
    guaranteed by the embedder configuration (Component 3 sets
    dimensions=cfg.embed_dim) — NOT re-validated here.

    Order-preservation is a hard contract: Component 7 zips the returned
    vectors against parallel `ids` / `metadatas` lists. The batches are issued
    sequentially and never reordered.
    """
    if not texts:
        return []

    # Build the client once — get_embeddings is not lru_cache'd, so constructing
    # it per batch would spin up a fresh HTTP session ~230x on the full ingest.
    client = get_embeddings(cfg)

    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start : start + EMBED_BATCH_SIZE]
        vectors.extend(client.embed_documents(batch))
    return vectors
