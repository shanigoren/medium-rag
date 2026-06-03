"""Component 9: Prompts.

A pure, dependency-free module that produces the two prompt strings the chat call
(C11) needs:

    (original question, list[RetrievedChunk])
        -> SYSTEM_PROMPT + render_user_prompt(...)
        -> {"System": ..., "User": ...}

It owns exactly one responsibility -- prompt text and formatting -- and nothing
else: no LLM call (C11), no retrieval (C8), no query rewriting (C10), no dedupe,
and it does not build the API `context` rows (C11 projects those).

The module is duck-typed on the chunk record (reads only `.article_id`, `.title`,
`.authors`, `.chunk`); it does NOT import `RetrievedChunk` at runtime (only under
TYPE_CHECKING for hints), so there is no coupling to `src.rag.retriever` and no
import cycle.

ASCII-only: the assignment's curly quotes are normalized to ASCII `"`/`'` so the
constant is cp1252-safe and the IDK sentinel string-matches the model's ASCII
output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:  # type-hints only; no runtime coupling to retriever
    from src.rag.retriever import RetrievedChunk


SYSTEM_PROMPT: str = (
    # --- mandatory assignment text (verbatim, ASCII-normalized) ---
    "You are a Medium-article assistant that answers questions strictly and only based on "
    "the Medium articles dataset context provided to you (metadata and article passages). "
    "You must not use any external knowledge, the open internet, or information that is not "
    "explicitly contained in the retrieved context. If the answer cannot be determined from "
    'the provided context, respond: "I don\'t know based on the provided Medium articles data."'
    "\n\n"
    "Always explain your answer using the given context, quoting or paraphrasing the relevant "
    "article passage or metadata when helpful."
    "\n\n"
    # --- permitted style appendix (clarifications, not constraint changes) ---
    "Style guidance: be concise. If asked to list N articles, return up to N distinct titles, "
    "one per line, that are actually supported by the retrieved context; if fewer than N "
    "relevant articles are available, return only those, and if none are relevant, respond that "
    "you don't know (as above) -- do not invent titles to reach N. If asked for a title and "
    "author, return both clearly labeled. If asked to summarize, write 3-5 sentences. If asked "
    "to recommend, name one article and justify it with a paraphrase or short quote from the "
    "retrieved context."
    "\n\n"
    "Each context item is shown as a header (Title, Authors, Tags, Article ID) followed by a "
    "passage. Tags are the article's topical labels: use the title, tags, and passage together "
    "to judge whether an article matches a requested topic (for example when listing or "
    "selecting articles by topic). Ground all factual statements, summaries, and recommendation "
    "justifications in the passage text itself -- tags indicate topic, not content."
)


def render_user_prompt(question: str, chunks: Sequence["RetrievedChunk"]) -> str:
    """Build the User prompt from the ORIGINAL question + retrieved chunks.

    Duck-typed on chunk attributes (.article_id, .title, .authors, .tags,
    .chunk) -- accepts any object exposing them, so no hard import of
    RetrievedChunk at runtime. Chunks are numbered [1..N] in the order given
    (already rank-sorted by C8; C9 does not re-sort). Empty `chunks` -> a
    "(no relevant context retrieved)" marker so the model can fall back to the
    mandatory "I don't know ..." response.
    """
    lines = [
        f"Question: {question}",
        "",
        "Context from the Medium articles dataset:",
        "",
    ]

    if not chunks:
        lines.append("(no relevant context retrieved)")
        return "\n".join(lines)

    blocks: list[str] = []
    for n, c in enumerate(chunks, start=1):
        authors = list(getattr(c, "authors", None) or [])
        authors_joined = ", ".join(authors) if authors else "Unknown"
        tags = list(getattr(c, "tags", None) or [])
        tags_joined = ", ".join(tags) if tags else "None"
        header = (
            f"[{n}] Title: {c.title} | Authors: {authors_joined} "
            f"| Tags: {tags_joined} | Article ID: {c.article_id}"
        )
        blocks.append(f"{header}\n{c.chunk}")

    # Blank line between chunk blocks; header section already ends with a blank line.
    lines.append("\n\n".join(blocks))
    return "\n".join(lines)


def build_augmented_prompt(question: str, chunks: Sequence["RetrievedChunk"]) -> dict:
    """Convenience for C11/C12: returns
        {"System": SYSTEM_PROMPT, "User": render_user_prompt(question, chunks)}
    with the EXACT capitalized keys the /api/prompt `Augmented_prompt` field
    requires. Single source of that casing so C11/C12/CP-C cannot drift.
    """
    return {
        "System": SYSTEM_PROMPT,
        "User": render_user_prompt(question, chunks),
    }
