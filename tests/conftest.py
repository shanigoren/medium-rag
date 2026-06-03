"""pytest configuration: registers the `--smoke` flag.

Smoke tests hit the live LLMod.AI / Pinecone APIs and cost real money. They
are skipped by default and only collected when `--smoke` is passed.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--smoke",
        action="store_true",
        default=False,
        help="run smoke tests that hit the live LLMod.AI / Pinecone APIs (costs money).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--smoke"):
        return
    skip = pytest.mark.skip(reason="needs --smoke")
    for item in items:
        if "smoke" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def cfg(monkeypatch):
    """A valid Config from default config.yaml + minimum required env.
    Shared across every test that needs a Config — do not duplicate this in
    individual test modules."""
    monkeypatch.setenv("LLMOD_API_KEY", "sk-test")
    monkeypatch.setenv("LLMOD_BASE_URL", "https://api.llmod.ai/v1")
    monkeypatch.setenv("PINECONE_API_KEY", "pc-test")
    from src.config import load_config

    return load_config()


@pytest.fixture
def fake_pc(monkeypatch):
    """Patch `src.rag.vectorstore.Pinecone` with the in-memory fake and reset
    the cached client. Yields the list of fakes constructed during the test
    so assertions can inspect recorded calls."""
    from src.rag import vectorstore
    from tests._fake_pinecone import _FakePinecone

    fakes: list = []

    def _factory(api_key, **kw):
        f = _FakePinecone(api_key=api_key, **kw)
        fakes.append(f)
        return f

    monkeypatch.setattr("src.rag.vectorstore.Pinecone", _factory)
    vectorstore._client.cache_clear()
    yield fakes
    vectorstore._client.cache_clear()


class _StubEmbeddings:
    """Recording stand-in for OpenAIEmbeddings. `embed_documents(texts)` returns
    one 1536-d vector per text whose element [0] encodes the text's GLOBAL input
    index (across all batches), so tests can prove output[i] lines up with
    input[i] across batch boundaries. Records each batch it was handed."""

    def __init__(self) -> None:
        self.batches: list[list[str]] = []
        self.queries: list[str] = []

    def embed_query(self, text: str) -> list[float]:
        """Record the query text and return a fixed 1536-d vector.

        The retriever (C8) calls `embed_query` for the single query string. We
        record `text` on `self.queries` so tests can assert the EXACT query was
        embedded raw (no title prefix). The vector content is irrelevant offline
        -- results are driven by `fakes[0]._index.query_response`."""
        self.queries.append(text)
        return [0.0] * 1536

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.batches.append(list(texts))
        # texts already appended above, so batches[:-1] is everything prior.
        start = sum(len(b) for b in self.batches[:-1])
        out: list[list[float]] = []
        for offset, _text in enumerate(texts):
            v = [0.0] * 1536
            v[0] = float(start + offset)  # global input index — order-revealing
            out.append(v)
        return out


@pytest.fixture
def fake_embeddings(monkeypatch):
    """Patch `src.rag.embed.get_embeddings` with a recording stub and yield it.

    Patches the name as imported into `src.rag.embed` (which does
    `from src.llm.clients import get_embeddings`), not the original in
    `src.llm.clients` — the local binding is what `embed_batch` calls. The stub
    ignores `cfg`. Read `stub.batches` to assert batching/order."""
    stub = _StubEmbeddings()
    monkeypatch.setattr("src.rag.embed.get_embeddings", lambda cfg=None: stub)
    return stub


@pytest.fixture
def fake_query_embeddings(monkeypatch):
    """Patch `src.rag.retriever.get_embeddings` with a recording stub and yield it.

    Patches the name as imported into `src.rag.retriever` (which does
    `from src.llm.clients import get_embeddings`), not the original in
    `src.llm.clients`. The stub's `embed_query` records each query string on
    `stub.queries`, so tests can assert the retriever embedded the EXACT query
    raw. The stub ignores `cfg`."""
    stub = _StubEmbeddings()
    monkeypatch.setattr("src.rag.retriever.get_embeddings", lambda cfg=None: stub)
    return stub


class _StubChat:
    """Recording stand-in for the chat client the query rewriter (C10) builds.

    `.invoke(messages)` records the messages on `self.calls`, then either raises
    `self.error` (if set) or returns an object whose `.content` is `self.content`
    (mirroring a LangChain AIMessage). `self.reasoning_efforts` records the
    `reasoning_effort` each `get_chat` call requested, so a test can assert the
    rewriter forced `"minimal"`. Reusable across parametrized cases via the
    setters."""

    def __init__(self, content: str = "", error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.calls: list = []
        self.reasoning_efforts: list[str | None] = []

    def set_content(self, content: str) -> None:
        self.content = content
        self.error = None

    def set_error(self, error: Exception) -> None:
        self.error = error

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error is not None:
            raise self.error
        return SimpleNamespace(content=self.content)


@pytest.fixture
def fake_chat(monkeypatch):
    """Patch `src.rag.query_writer.get_chat` with a recording `_StubChat`.

    Patches the name as imported into `src.rag.query_writer` (which does
    `from src.llm.clients import get_chat`). The factory records each requested
    `reasoning_effort` on `stub.reasoning_efforts` and returns the same stub, so
    tests can set `stub.content` / `stub.set_error(...)` and inspect
    `stub.calls`. No network."""
    stub = _StubChat()

    def _factory(cfg=None, *, reasoning_effort=None):
        stub.reasoning_efforts.append(reasoning_effort)
        return stub

    monkeypatch.setattr("src.rag.query_writer.get_chat", _factory)
    return stub


@pytest.fixture
def fake_chain_chat(monkeypatch):
    """Patch `src.rag.chain.get_chat` with a recording `_StubChat` for the C11
    answer call.

    Patches the name as imported into `src.rag.chain` (which does
    `from src.llm.clients import get_chat`). The factory records each call's `cfg`
    on `stub.cfgs` and `reasoning_effort` on `stub.reasoning_efforts`, so tests can
    assert the chain threaded the resolved cfg and passed NO reasoning_effort
    override (the answer call must use cfg.reasoning_effort, not the rewriter's
    "minimal"). Set `stub.content` / `stub.set_error(...)`; inspect `stub.calls`
    for the messages. No network."""
    stub = _StubChat()
    stub.cfgs = []

    def _factory(cfg=None, *, reasoning_effort=None):
        stub.cfgs.append(cfg)
        stub.reasoning_efforts.append(reasoning_effort)
        return stub

    monkeypatch.setattr("src.rag.chain.get_chat", _factory)
    return stub


@pytest.fixture
def rewriter_recordings():
    """Load the committed recordings bank (`tests/fixtures/rewriter_recordings.json`).

    Returns the list of entries (each: question, expected_dedup, type, note,
    recorded_response). Used by the offline replay tests and the live drift
    subset; refresh with `scripts/record_rewriter.py`."""
    path = Path(__file__).parent / "fixtures" / "rewriter_recordings.json"
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)
