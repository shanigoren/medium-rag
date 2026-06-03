"""Component 10: the Query rewriter.

A pre-retrieval LLM step. From the user's raw question it produces TWO
retrieval-only outputs in a SINGLE LLM call:

  - `query`: a short, dense query for vector search (conversational scaffolding
    and output-format instructions stripped; topical anchors / named entities
    kept).
  - `dedup`: whether the retriever (C8) should collapse to DISTINCT articles
    (`True`, for "list N" multi-result asks) or return the top-k chunks as-is
    (`False`, for find-one / summary / recommend asks).

The final answer call (C11) uses the ORIGINAL question, not `query`. Both outputs
are internal retrieval signals and never appear in the /api/prompt response.

On ANY failure -- model exception, empty/malformed/non-JSON output, missing
keys, a non-bool/non-string `dedup`, or a too-short query -- `rewrite_query`
falls back to `RewriteResult(query=question, dedup=True)`: the original question
unchanged with the safe always-distinct default the C8 retriever's `dedup=True`
default relies on. Retrieval must never fail because the rewriter struggled.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate

from src.config import Config, load_config
from src.llm.clients import get_chat

# Minimum non-whitespace chars for a rewritten query to be accepted (else the
# model effectively returned nothing useful and we fall back to the question).
_MIN_QUERY_CHARS = 3


@dataclass(frozen=True)
class RewriteResult:
    """The rewriter's two retrieval-only outputs from a single LLM call.

      query  str   # dense, scaffolding-stripped query for vector search
      dedup  bool  # True  -> C8 returns DISTINCT articles (type-2 "list N")
                   # False -> C8 returns top-k chunks as-is (type-1/3/4 depth)
    """

    query: str
    dedup: bool


# NOTE: braces in the JSON examples are DOUBLED ({{ }}) because this string is a
# ChatPromptTemplate template -- single braces would be parsed as variables.
REWRITER_SYSTEM_PROMPT = """Your task is to turn a user's question into a search query \
for a vector database of Medium articles, and to decide whether the search should return \
several distinct articles or passages from a single article.

Return ONLY compact JSON: {{"query": "...", "dedup": true|false}}
- "query": the search topic only -- keep the question's topical words and named entities; \
drop ALL framing and requests (e.g. "I'm looking for", "which article would you \
recommend", "provide the title and author"). Stay close to the original wording; do not \
add synonyms or extra keywords.
- "dedup": true only when the user asks for MULTIPLE DISTINCT articles (e.g. "list 3 \
articles about X"); false for a single article to find, summarise, or recommend.

Examples:
Q: Find an article that reframes marketing as a conversation with readers, aimed at writers \
who find self-promotion uncomfortable. Provide the title and author.
A: {{"query": "marketing as a conversation with readers, writers uncomfortable with self-promotion", "dedup": false}}
Q: List exactly 3 articles about education. Return only the titles.
A: {{"query": "education", "dedup": true}}
Q: Find an article that argues past pandemics (such as the bubonic plague) can spur \
innovation and recovery, and summarise its central argument.
A: {{"query": "past pandemics such as the bubonic plague spur innovation and recovery", "dedup": false}}
Q: I want practical, beginner-friendly advice on building habits that actually stick. \
Which article would you recommend, and why?
A: {{"query": "practical, beginner-friendly advice on building habits that actually stick", "dedup": false}}
Q: I have trouble sleeping - which article would you recommend, and why?
A: {{"query": "trouble sleeping", "dedup": false}}"""


_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", REWRITER_SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


def _strip_fence(text: str) -> str:
    """Remove a surrounding markdown code fence (``` or ```json) if present.

    gpt-5-mini sometimes wraps JSON in a fenced block. We drop the opening fence
    line and a trailing fence so `json.loads` sees clean JSON. A malformed fence
    is left as-is for `json.loads` to reject.
    """
    s = text.strip()
    if not s.startswith("```"):
        return s
    newline = s.find("\n")
    if newline == -1:
        return s  # single-line "```..." -- let json.loads fail
    s = s[newline + 1 :]
    s = s.rstrip()
    if s.endswith("```"):
        s = s[:-3]
    return s.strip()


def _coerce_dedup(value: object) -> bool | None:
    """Return a real bool from a JSON `dedup` value, or None if uncoercible.

    Accepts a real bool, or the case-insensitive strings "true"/"false".
    Anything else (int, null, other strings) -> None (treated as a parse
    failure upstream). Note: `bool` is checked first so a real bool is never
    mistaken for something else.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low == "true":
            return True
        if low == "false":
            return False
    return None


def _parse(text: object) -> RewriteResult | None:
    """Defensively parse the model's response into a RewriteResult, or None.

    Returns None (-> caller falls back) on: non-string input, empty output,
    non-JSON / non-object JSON, a missing `query`/`dedup` key, a non-string or
    too-short `query`, or a `dedup` that is neither bool nor "true"/"false".
    """
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        data = json.loads(_strip_fence(text))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    if "query" not in data or "dedup" not in data:
        return None
    query = data["query"]
    if not isinstance(query, str):
        return None
    query = query.strip()
    if len(query) < _MIN_QUERY_CHARS:
        return None
    dedup = _coerce_dedup(data["dedup"])
    if dedup is None:
        return None
    return RewriteResult(query=query, dedup=dedup)


def rewrite_query(question: str, cfg: Config | None = None) -> RewriteResult:
    """Rewrite `question` into a dense retrieval query and classify `dedup`.

    Single LLM call (gpt-5-mini, `reasoning_effort="minimal"` HARDCODED -- not
    `cfg.reasoning_effort`). On any failure returns the safe fallback
    `RewriteResult(query=question, dedup=True)`; no exception escapes.
    """
    if cfg is None:
        cfg = load_config()
    try:
        chat = get_chat(cfg, reasoning_effort="minimal")
        messages = _PROMPT.format_messages(question=question)
        raw = chat.invoke(messages)
        text = raw.content if hasattr(raw, "content") else raw
        result = _parse(text)
        if result is not None:
            return result
    except Exception:
        # Best-effort: any model/transport/parse error degrades to the baseline.
        pass
    return RewriteResult(query=question, dedup=True)
