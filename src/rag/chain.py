"""Component 11: the Chain.

Compose the read-path units (C10 rewriter, C8 retriever, C9 prompts, C3 chat
client) into the full RAG loop and produce the exact object the `/api/prompt`
endpoint (C12) serializes:

    question
       -> rewrite_query(question, cfg)             (C10) -> RewriteResult(query, dedup)
       -> retrieve(r.query, ns, cfg, dedup=r.dedup) (C8) -> list[RetrievedChunk]
       -> build_augmented_prompt(question, chunks)  (C9) -> {"System","User"}  (ORIGINAL question)
       -> get_chat(cfg).invoke([System, Human])     (C3) -> response text
       -> AnswerResult(response, context, augmented_prompt)

The chain is orchestration ONLY -- it owns no new behaviour. Two invariants it
must get right (the tests pin both):

  1. The answer call uses the ORIGINAL question, never the rewritten query. The
     rewritten query and `dedup` are retrieval-only signals; neither reaches the
     prompt, the chat call, or the response.
  2. The answer chat client uses `cfg.reasoning_effort` (NOT the rewriter's
     hardcoded "minimal") -- we call `get_chat(cfg)` with no override, so Phase C's
     reasoning-effort sweep actually moves the answer call.

The resolved `cfg` is threaded to ALL three sub-calls so eval's Phase B (`top_k`,
read by `retrieve`) and Phase C (`reasoning_effort`, read by `get_chat`) sweeps
take effect. `dedup` and the rewritten query never reach the client (they are
omitted from `to_api_dict()`); they ARE exposed on `AnswerResult.rewrite` for the
demo, server-side debug, and C14 eval's dedup-classification audit.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import Config, load_config
from src.llm.clients import get_chat
from src.prompts import build_augmented_prompt
from src.rag.query_writer import RewriteResult, rewrite_query
from src.rag.retriever import RetrievedChunk, retrieve

# Keys of one API `context` row, in wire order.
_CONTEXT_KEYS = ("article_id", "title", "chunk", "score")


@dataclass(frozen=True)
class AnswerResult:
    """The full RAG result for one question.

      response          str           # the model's answer text
      context           list[dict]    # API context rows, one per retrieved chunk,
                                       #   each EXACTLY {article_id, title, chunk, score}
      augmented_prompt  dict          # {"System": ..., "User": ...} (C9 casing) for the
                                       #   FINAL answer call; User holds the ORIGINAL question
      rewrite           RewriteResult # the ACTUAL (query, dedup) the rewriter produced and
                                       #   the retriever used -- INTERNAL (server-side debug,
                                       #   the demo, and C14 eval's dedup audit)

    The `rewrite` field is deliberately EXCLUDED from `to_api_dict()`: the rewritten
    query and `dedup` are internal retrieval signals and must never reach the client
    Exposing them on the in-process result -- rather
    than re-calling the rewriter downstream -- guarantees the demo/eval see the SAME
    values the answer was actually built from (the rewriter is non-deterministic, so a
    second call could diverge). `to_api_dict()` projects only the three wire keys, so
    C12 stays a thin wrapper.
    """

    response: str
    context: list[dict]
    augmented_prompt: dict
    rewrite: RewriteResult

    def to_api_dict(self) -> dict:
        """Exact /api/prompt wire shape.

        Single source of the capital-A `Augmented_prompt` key casing so C12/CP-C
        cannot drift (the inner "System"/"User" casing comes from C9). The `rewrite`
        field is intentionally omitted -- query/dedup never reach the client.
        """
        return {
            "response": self.response,
            "context": self.context,
            "Augmented_prompt": self.augmented_prompt,
        }


def _response_text(resp: object) -> str:
    """Extract the answer string from a chat result.

    Reads `resp.content`. gpt-5-mini via the proxy normally returns a plain
    string, but langchain message content can be a list of parts; we join the
    text parts defensively so a `str(list)` never leaks into the answer. Never
    `str(resp)` on the whole message or on a content list.
    """
    content = getattr(resp, "content", resp)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts).strip()
    return str(content).strip()


def _context_rows(chunks: list[RetrievedChunk]) -> list[dict]:
    """Project retrieved chunks to API context rows.

    Each row has EXACTLY the four wire keys (no `authors`/`chunk_idx`/`rank`
    leak), in `_CONTEXT_KEYS` order. `article_id` is re-`str()`'d and `score`
    re-`float()`'d so the contract types are locked at the boundary. Order is
    preserved (C8 already rank-sorted; duplicate article_ids under dedup=False
    are kept, not collapsed).
    """
    return [
        {
            "article_id": str(c.article_id),
            "title": c.title,
            "chunk": c.chunk,
            "score": float(c.score),
        }
        for c in chunks
    ]


def answer(
    question: str,
    cfg: Config | None = None,
    *,
    namespace: str | None = None,
) -> AnswerResult:
    """Run the full RAG read loop for one question.

    Pipeline: rewrite_query (C10) -> retrieve (C8) -> build_augmented_prompt with
    the ORIGINAL question (C9) -> get_chat(cfg).invoke (C3).

    `cfg` defaults to load_config(). `namespace` defaults to cfg.pinecone_namespace
    -- overridable so the eval harness (C14) can target experiment namespaces and
    the smoke tests/demo can target the `smoke` namespace without mutating config.

    The chat client uses cfg.reasoning_effort (NOT the rewriter's "minimal").
    `dedup` and the rewritten query never appear in the returned AnswerResult.

    Raises ValueError on an empty/whitespace question (before any LLM call). A
    failing chat call propagates -- the chain does not swallow it.
    """
    # 0. Validate before any work -- fail fast, do not waste a rewrite LLM call.
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty, non-whitespace string")

    # 1. Resolve inputs; thread this `cfg` to every downstream call.
    if cfg is None:
        cfg = load_config()
    ns = namespace if namespace is not None else cfg.pinecone_namespace

    # 2. Rewrite (C10): never raises (self-protecting fallback). Retrieval-only.
    r = rewrite_query(question, cfg)

    # 3. Retrieve (C8): embed r.query raw, obey r.dedup. top_k/fetch_k ride on cfg.
    chunks = retrieve(r.query, ns, cfg, dedup=r.dedup)

    # 4. Build the prompt (C9) with the ORIGINAL question -- never r.query.
    augmented = build_augmented_prompt(question, chunks)

    # 5. Answer (C3): cfg.reasoning_effort (no override). One invoke, two messages.
    chat = get_chat(cfg)
    resp = chat.invoke(
        [
            SystemMessage(content=augmented["System"]),
            HumanMessage(content=augmented["User"]),
        ]
    )

    return AnswerResult(
        response=_response_text(resp),
        context=_context_rows(chunks),
        augmented_prompt=augmented,
        rewrite=r,
    )
