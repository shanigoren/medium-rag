"""Demo for Component 12 (the HTTP API).

Run:
    conda activate medium-rag
    python scripts/demo_12_api.py

Drives the REAL FastAPI app in-process via Starlette's TestClient (a true ASGI
round-trip) but fakes the three money boundaries -- the LLM (rewrite + answer),
the query embedding, and Pinecone -- so it shows the live HTTP routing,
validation, error mapping, and byte-for-byte serialization with CANNED answers
over a seeded in-memory corpus. It COSTS NOTHING: no LLMod.AI / Pinecone calls.

(The API wrapper's job is the contract, not answer quality. Real end-to-end
answers over the corpus are demonstrated by scripts/demo_11_chain.py against the
live `smoke` namespace, and confirmed live at deploy / CP-F.)

Output is transliterated to plain ASCII (scripts/_console.to_ascii) so it renders
on any Windows console / pipe / file. Exits non-zero on any failure.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._console import to_ascii

# --- a tiny in-memory corpus (one article per assignment question type) -------
# Shaped like the C7 Pinecone metadata schema C8's retriever reads.
_SEED = [
    ("0", "Marketing as a Conversation", "Marketing is not shouting; it is a quiet "
     "conversation with readers who dislike self-promotion.", ["Dana Levin"], ["marketing", "writing"]),
    ("1", "Why Education Must Change", "Classrooms built for the last century no longer "
     "serve curious learners; education needs reinvention.", ["Sam Ortiz"], ["education"]),
    ("2", "Pandemics and Progress", "Past plagues, including the bubonic plague, repeatedly "
     "spurred innovation and economic recovery.", ["Lee Park"], ["history", "health"]),
    ("3", "Habits That Actually Stick", "Practical, beginner-friendly advice: start tiny, "
     "anchor a new habit to an old one, and forgive a missed day.", ["Ana Roy"], ["habits", "self-improvement"]),
]


def _matches() -> list[dict]:
    return [
        {
            "id": f"{aid}-0",
            "score": round(0.92 - 0.07 * i, 3),
            "metadata": {
                "article_id": aid, "title": title, "authors": list(authors),
                "tags": list(tags), "chunk": chunk, "chunk_idx": 0,
            },
        }
        for i, (aid, title, chunk, authors, tags) in enumerate(_SEED)
    ]


# --- fakes for the three money boundaries -------------------------------------


class _FakeEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        return [0.0] * 1536


class _RewriterChat:
    """Stands in for the rewriter LLM: returns the {query, dedup} JSON the real
    rewriter would. `dedup` is true only for 'list N' multi-result asks."""

    def invoke(self, messages):
        question = messages[-1].content
        dedup = "list" in question.lower()
        return SimpleNamespace(content=json.dumps({"query": question, "dedup": dedup}))


class _AnswerChat:
    """Stands in for the answer LLM: a fixed canned reply (the demo proves the
    HTTP contract, not answer quality)."""

    def invoke(self, messages):
        return SimpleNamespace(content="This is a canned answer (the LLM is faked in this demo).")


class _FakeIndex:
    def query(self, **_):
        return {"matches": _matches()}


class _FakePinecone:
    def __init__(self, api_key, **_):
        self._index = _FakeIndex()

    def Index(self, name):
        return self._index


def _install_fakes() -> None:
    """Patch the four money boundaries by reassigning the names AS IMPORTED into
    each consuming module (same targets the test fixtures use)."""
    import src.rag.chain as chain
    import src.rag.query_writer as query_writer
    import src.rag.retriever as retriever
    import src.rag.vectorstore as vectorstore

    vectorstore.Pinecone = _FakePinecone
    vectorstore._client.cache_clear()  # drop any real client cached under an api_key
    retriever.get_embeddings = lambda cfg=None: _FakeEmbeddings()
    query_writer.get_chat = lambda cfg=None, *, reasoning_effort=None: _RewriterChat()
    chain.get_chat = lambda cfg=None, *, reasoning_effort=None: _AnswerChat()


def _set_dummy_env_if_absent() -> None:
    """load_config() needs these present; the values are never used (every client
    is faked). Don't clobber a real .env if one is set."""
    for name, val in (
        ("LLMOD_API_KEY", "sk-demo-fake"),
        ("LLMOD_BASE_URL", "https://api.llmod.ai/v1"),
        ("PINECONE_API_KEY", "pc-demo-fake"),
    ):
        os.environ.setdefault(name, val)


def _mask(secret: str) -> str:
    if not secret or len(secret) < 4:
        return "****"
    return "..." + secret[-4:]


DEMO_QUESTIONS = [
    "Find an article that reframes marketing as a conversation with readers, aimed at "
    "writers who find self-promotion uncomfortable. Provide the title and author.",
    "List exactly 3 articles about education. Return only the titles.",
    "Find an article that argues past pandemics can spur innovation and recovery, and "
    "summarise its central argument.",
    "I want practical, beginner-friendly advice on building habits that stick. Which "
    "article would you recommend, and why?",
]


def main() -> int:
    _install_fakes()
    _set_dummy_env_if_absent()

    from fastapi.testclient import TestClient

    from api.index import app
    from src.config import load_config

    cfg = load_config()
    client = TestClient(app)

    print()
    print("Component 12 API demo -- real app, faked money boundaries (costs nothing).")
    print(f"  chat_model={cfg.chat_model}  api_key={_mask(cfg.llmod_api_key)}  (faked; last 4 chars)")

    stats = client.get("/api/stats")
    print(f"\n[GET /api/stats] status={stats.status_code}  body={stats.json()}   (from config.yaml)")

    for q in DEMO_QUESTIONS:
        resp = client.post("/api/prompt", json={"question": q})
        body = resp.json()
        print("\n" + "=" * 100)
        print(f"[POST /api/prompt] status={resp.status_code}")
        print(f"[Q] {to_ascii(q)}")
        print(f"[A] {to_ascii(body['response'])}   (canned)")
        print(f"\n  context: {len(body['context'])} chunks")
        for n, row in enumerate(body["context"], start=1):
            print(f"   #{n}  score={row['score']:.3f}  id={row['article_id']}  "
                  f"title={to_ascii(row['title'])!r}")
        assert q in body["Augmented_prompt"]["User"], "original question must appear in the prompt"
        print("  OK: original question present in augmented prompt")

    print("\n" + "=" * 100)
    print("[error paths]")
    missing = client.post("/api/prompt", json={})
    empty = client.post("/api/prompt", json={"question": "   "})
    print(f"  missing 'question' field -> status={missing.status_code}   (expect 422)")
    print(f"  empty / whitespace question -> status={empty.status_code}   (expect 400)")
    assert missing.status_code == 422 and empty.status_code == 400, "error mapping regressed"

    print("\nOK: API served the wire contract for every question and mapped both error paths.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:  # noqa: BLE001 -- demo: surface any failure non-zero
        print(f"FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
