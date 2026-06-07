"""Unit tests for `src.config.load_config`.

Each test is independent: required env vars are set via `monkeypatch.setenv`
(auto-undone at teardown), and tests that need a custom YAML write it to
`tmp_path`. We also `monkeypatch.chdir(tmp_path)` for tests that do not load
the repo's real `config.yaml`, so a stray `.env` next to the repo cannot leak
into the test environment via `python-dotenv`.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from src.config import Config, load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DEFAULT_YAML = """\
chunk_size: 768
overlap_ratio: 0.10
top_k: 5
retrieval_fetch_k: 30
embed_content: "chunk_only"
embed_model: "4UHRUIN-text-embedding-3-small"
embed_dim: 1536
chat_model: "4UHRUIN-gpt-5-mini"
reasoning_effort: "low"
pinecone_index: "medium-rag"
pinecone_namespace: "prod"
"""


def _write_yaml(tmp_path: Path, content: str = DEFAULT_YAML) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")
    return path


def _set_required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set the three required secrets to non-empty placeholder values."""
    monkeypatch.setenv("LLMOD_API_KEY", "test-key")
    monkeypatch.setenv("LLMOD_BASE_URL", "https://api.llmod.ai/v1")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")


def _clean_optional_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure no optional env-var overrides leak from the host machine."""
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_load_defaults_from_repo_yaml(monkeypatch):
    """load_config() (no args) reads config.yaml from repo root + env, returns valid Config."""
    _set_required_env(monkeypatch)
    _clean_optional_env(monkeypatch)

    cfg = load_config()

    assert isinstance(cfg, Config)
    assert cfg.chunk_size == 768
    assert cfg.overlap_ratio == 0.10
    assert cfg.top_k == 20
    assert cfg.retrieval_fetch_k == 30
    assert cfg.embed_content == "chunk_only"
    assert cfg.embed_dim == 1536
    assert cfg.reasoning_effort == "low"
    assert cfg.pinecone_index == "medium-rag"
    assert cfg.pinecone_namespace == "prod"
    assert cfg.llmod_api_key == "test-key"
    assert cfg.pinecone_api_key == "test-pinecone-key"


def test_explicit_config_path(tmp_path, monkeypatch):
    """load_config(tmp_path / 'custom.yaml') reads from given path."""
    _set_required_env(monkeypatch)
    _clean_optional_env(monkeypatch)
    monkeypatch.chdir(tmp_path)  # avoid finding a stray .env

    yaml_path = tmp_path / "custom.yaml"
    yaml_path.write_text(
        DEFAULT_YAML.replace("chunk_size: 768", "chunk_size: 512"),
        encoding="utf-8",
    )

    cfg = load_config(yaml_path)
    assert cfg.chunk_size == 512


def test_env_override_pinecone_namespace(monkeypatch):
    """PINECONE_NAMESPACE env var wins over YAML value."""
    _set_required_env(monkeypatch)
    _clean_optional_env(monkeypatch)
    monkeypatch.setenv("PINECONE_NAMESPACE", "exp_c512_o10")

    cfg = load_config()
    assert cfg.pinecone_namespace == "exp_c512_o10"


def test_env_override_chunk_size_typed(monkeypatch):
    """CHUNK_SIZE='768' env var is cast to int 768."""
    _set_required_env(monkeypatch)
    _clean_optional_env(monkeypatch)
    monkeypatch.setenv("CHUNK_SIZE", "768")

    cfg = load_config()
    assert cfg.chunk_size == 768
    assert isinstance(cfg.chunk_size, int)


def test_chunk_size_above_cap_raises(tmp_path, monkeypatch):
    """chunk_size=2000 → ValueError mentioning 'chunk_size' and '1..1024'."""
    _set_required_env(monkeypatch)
    _clean_optional_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    yaml_path = _write_yaml(
        tmp_path, DEFAULT_YAML.replace("chunk_size: 768", "chunk_size: 2000")
    )

    with pytest.raises(ValueError, match=r"chunk_size.*1\.\.1024"):
        load_config(yaml_path)


def test_chunk_size_zero_raises(tmp_path, monkeypatch):
    """chunk_size=0 → ValueError."""
    _set_required_env(monkeypatch)
    _clean_optional_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    yaml_path = _write_yaml(
        tmp_path, DEFAULT_YAML.replace("chunk_size: 768", "chunk_size: 0")
    )

    with pytest.raises(ValueError, match=r"chunk_size"):
        load_config(yaml_path)


def test_overlap_ratio_above_cap_raises(tmp_path, monkeypatch):
    """overlap_ratio=0.4 → ValueError mentioning 'overlap_ratio'."""
    _set_required_env(monkeypatch)
    _clean_optional_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    yaml_path = _write_yaml(
        tmp_path, DEFAULT_YAML.replace("overlap_ratio: 0.10", "overlap_ratio: 0.4")
    )

    with pytest.raises(ValueError, match=r"overlap_ratio"):
        load_config(yaml_path)


def test_top_k_above_cap_raises(tmp_path, monkeypatch):
    """top_k=31 → ValueError mentioning 'top_k'."""
    _set_required_env(monkeypatch)
    _clean_optional_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    yaml_path = _write_yaml(
        tmp_path, DEFAULT_YAML.replace("top_k: 5", "top_k: 31")
    )

    with pytest.raises(ValueError, match=r"top_k"):
        load_config(yaml_path)


def test_top_k_zero_raises(tmp_path, monkeypatch):
    """top_k=0 → ValueError."""
    _set_required_env(monkeypatch)
    _clean_optional_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    yaml_path = _write_yaml(
        tmp_path, DEFAULT_YAML.replace("top_k: 5", "top_k: 0")
    )

    with pytest.raises(ValueError, match=r"top_k"):
        load_config(yaml_path)


def test_retrieval_fetch_k_below_top_k_raises(tmp_path, monkeypatch):
    """retrieval_fetch_k=3 with top_k=5 → ValueError."""
    _set_required_env(monkeypatch)
    _clean_optional_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    yaml_path = _write_yaml(
        tmp_path,
        DEFAULT_YAML.replace("retrieval_fetch_k: 30", "retrieval_fetch_k: 3"),
    )

    with pytest.raises(ValueError, match=r"retrieval_fetch_k"):
        load_config(yaml_path)


def test_invalid_embed_content_raises(tmp_path, monkeypatch):
    """embed_content='banana' → ValueError mentioning 'embed_content'."""
    _set_required_env(monkeypatch)
    _clean_optional_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    yaml_path = _write_yaml(
        tmp_path,
        DEFAULT_YAML.replace('embed_content: "chunk_only"', 'embed_content: "banana"'),
    )

    with pytest.raises(ValueError, match=r"embed_content"):
        load_config(yaml_path)


def test_invalid_reasoning_effort_raises(tmp_path, monkeypatch):
    """reasoning_effort='extreme' → ValueError."""
    _set_required_env(monkeypatch)
    _clean_optional_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    yaml_path = _write_yaml(
        tmp_path,
        DEFAULT_YAML.replace(
            'reasoning_effort: "low"', 'reasoning_effort: "extreme"'
        ),
    )

    with pytest.raises(ValueError, match=r"reasoning_effort"):
        load_config(yaml_path)


def test_missing_llmod_api_key_raises(monkeypatch, tmp_path):
    """Unset LLMOD_API_KEY → ValueError mentioning 'LLMOD_API_KEY'."""
    _clean_optional_env(monkeypatch)
    monkeypatch.chdir(tmp_path)  # avoid finding a stray .env that re-supplies the key
    monkeypatch.delenv("LLMOD_API_KEY", raising=False)
    monkeypatch.setenv("LLMOD_BASE_URL", "https://api.llmod.ai/v1")
    monkeypatch.setenv("PINECONE_API_KEY", "test-pinecone-key")
    yaml_path = _write_yaml(tmp_path)

    with pytest.raises(ValueError, match=r"LLMOD_API_KEY"):
        load_config(yaml_path)


def test_empty_pinecone_api_key_raises(monkeypatch, tmp_path):
    """PINECONE_API_KEY='' → ValueError mentioning 'PINECONE_API_KEY'."""
    _clean_optional_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLMOD_API_KEY", "test-key")
    monkeypatch.setenv("LLMOD_BASE_URL", "https://api.llmod.ai/v1")
    monkeypatch.setenv("PINECONE_API_KEY", "")
    yaml_path = _write_yaml(tmp_path)

    with pytest.raises(ValueError, match=r"PINECONE_API_KEY"):
        load_config(yaml_path)


def test_config_is_frozen():
    """Mutating a field on Config raises FrozenInstanceError."""
    cfg = Config(
        chunk_size=512,
        overlap_ratio=0.10,
        top_k=5,
        retrieval_fetch_k=30,
        embed_content="title_chunk",
        embed_model="m",
        embed_dim=1536,
        chat_model="c",
        reasoning_effort="low",
        pinecone_index="i",
        pinecone_namespace="n",
        llmod_api_key="k",
        llmod_base_url="u",
        pinecone_api_key="p",
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.chunk_size = 1024  # type: ignore[misc]
