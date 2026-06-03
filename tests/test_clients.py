"""Offline unit tests for `src.llm.clients`.

Verify the factories construct LangChain clients with the right model name,
base URL, embedding dimensions and reasoning_effort — without making any
network call. Attribute names follow langchain-openai 1.2.2 (`openai_api_base`,
`openai_api_key`, typed `reasoning_effort`).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from src.config import Config, load_config
from src.llm.clients import get_chat, get_embeddings


DEFAULT_YAML = """\
chunk_size: 512
overlap_ratio: 0.10
top_k: 5
retrieval_fetch_k: 30
embed_content: "title_chunk"
embed_model: "4UHRUIN-text-embedding-3-small"
embed_dim: 1536
chat_model: "4UHRUIN-gpt-5-mini"
reasoning_effort: "low"
pinecone_index: "medium-rag"
pinecone_namespace: "prod"
"""


def _write_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(DEFAULT_YAML, encoding="utf-8")
    return path


def _set_required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMOD_API_KEY", "test-key")
    monkeypatch.setenv("LLMOD_BASE_URL", "https://api.llmod.ai/v1")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")


def _clean_optional_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "PINECONE_INDEX",
        "PINECONE_NAMESPACE",
        "EMBED_CONTENT",
        "CHUNK_SIZE",
        "OVERLAP_RATIO",
        "TOP_K",
        "RETRIEVAL_FETCH_K",
    ):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def cfg(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Config:
    _set_required_env(monkeypatch)
    _clean_optional_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    yaml_path = _write_yaml(tmp_path)
    return load_config(yaml_path)


# --- get_embeddings --------------------------------------------------------


def test_get_embeddings_uses_configured_model(cfg: Config) -> None:
    client = get_embeddings(cfg)
    assert client.model == cfg.embed_model


def test_get_embeddings_uses_configured_base_url(cfg: Config) -> None:
    client = get_embeddings(cfg)
    assert client.openai_api_base == cfg.llmod_base_url


def test_get_embeddings_uses_configured_dimensions(cfg: Config) -> None:
    client = get_embeddings(cfg)
    assert client.dimensions == cfg.embed_dim == 1536


# --- get_chat --------------------------------------------------------------


def test_get_chat_uses_configured_model(cfg: Config) -> None:
    client = get_chat(cfg)
    assert client.model_name == cfg.chat_model


def test_get_chat_uses_configured_base_url(cfg: Config) -> None:
    client = get_chat(cfg)
    assert client.openai_api_base == cfg.llmod_base_url


def test_get_chat_forwards_reasoning_effort(cfg: Config) -> None:
    client = get_chat(cfg)
    assert client.reasoning_effort == cfg.reasoning_effort


# --- explicit-config path --------------------------------------------------


def test_factories_accept_explicit_config(cfg: Config) -> None:
    """Passing a custom Config wins over load_config(): no env / YAML access."""
    custom = dataclasses.replace(
        cfg,
        embed_model="custom-embed",
        chat_model="custom-chat",
        reasoning_effort="high",
        llmod_base_url="https://example.test/v1",
    )
    emb = get_embeddings(custom)
    chat = get_chat(custom)
    assert emb.model == "custom-embed"
    assert emb.openai_api_base == "https://example.test/v1"
    assert chat.model_name == "custom-chat"
    assert chat.reasoning_effort == "high"
    assert chat.openai_api_base == "https://example.test/v1"
