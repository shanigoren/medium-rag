"""Configuration loader: YAML + env vars → frozen Config dataclass.

The single source of truth for runtime parameters. Loaded once at startup by
every other component. Must work with only stdlib + pyyaml + python-dotenv —
no LLM/Pinecone/HTTP access required.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml
from dotenv import find_dotenv, load_dotenv

# Allowed enums
_VALID_EMBED_CONTENT = {"chunk_only", "title_chunk", "title_tags_chunk"}
_VALID_REASONING_EFFORT = {"minimal", "low", "medium", "high"}

# Required env vars (must be present and non-empty)
_REQUIRED_ENV_VARS = ("LLMOD_API_KEY", "LLMOD_BASE_URL", "PINECONE_API_KEY")

# Optional env-var overrides for YAML fields: env name → (field name, caster)
_ENV_OVERRIDES: dict[str, tuple[str, type]] = {
    "PINECONE_INDEX": ("pinecone_index", str),
    "PINECONE_NAMESPACE": ("pinecone_namespace", str),
    "EMBED_CONTENT": ("embed_content", str),
    "CHUNK_SIZE": ("chunk_size", int),
    "OVERLAP_RATIO": ("overlap_ratio", float),
    "TOP_K": ("top_k", int),
    "RETRIEVAL_FETCH_K": ("retrieval_fetch_k", int),
}


@dataclass(frozen=True)
class Config:
    # RAG hyperparameters (assignment-caps enforced)
    chunk_size: int                 # tokens, [1, 1024]
    overlap_ratio: float            # [0, 0.3]
    top_k: int                      # [1, 30]
    retrieval_fetch_k: int          # [top_k, 30]
    embed_content: str              # "chunk_only" | "title_chunk" | "title_tags_chunk"

    # Models
    embed_model: str
    embed_dim: int
    chat_model: str
    reasoning_effort: str           # "minimal" | "low" | "medium" | "high"

    # Pinecone
    pinecone_index: str
    pinecone_namespace: str

    # Secrets (loaded from env)
    llmod_api_key: str
    llmod_base_url: str
    pinecone_api_key: str


def _repo_root() -> Path:
    """Project root = parent of this file's parent (src/)."""
    return Path(__file__).resolve().parent.parent


def _cast(value: str, caster: type, field_name: str) -> Any:
    try:
        return caster(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Failed to cast env override for '{field_name}' to {caster.__name__}: {value!r}"
        ) from exc


def _validate(cfg: Config) -> None:
    """Raise ValueError on any rule violation.

    Each message contains the offending field name so callers can grep for it.
    """
    if not (1 <= cfg.chunk_size <= 1024):
        raise ValueError(
            f"chunk_size must be in range 1..1024 (got {cfg.chunk_size})"
        )
    if not (0.0 <= cfg.overlap_ratio <= 0.3):
        raise ValueError(
            f"overlap_ratio must be in range 0..0.3 (got {cfg.overlap_ratio})"
        )
    if not (1 <= cfg.top_k <= 30):
        raise ValueError(
            f"top_k must be in range 1..30 (got {cfg.top_k})"
        )
    if not (cfg.top_k <= cfg.retrieval_fetch_k <= 30):
        raise ValueError(
            f"retrieval_fetch_k must be in range top_k..30 "
            f"(top_k={cfg.top_k}, retrieval_fetch_k={cfg.retrieval_fetch_k})"
        )
    if cfg.embed_content not in _VALID_EMBED_CONTENT:
        raise ValueError(
            f"embed_content must be one of {sorted(_VALID_EMBED_CONTENT)} "
            f"(got {cfg.embed_content!r})"
        )
    if cfg.reasoning_effort not in _VALID_REASONING_EFFORT:
        raise ValueError(
            f"reasoning_effort must be one of {sorted(_VALID_REASONING_EFFORT)} "
            f"(got {cfg.reasoning_effort!r})"
        )
    if cfg.embed_dim != 1536:
        raise ValueError(
            f"embed_dim must be 1536 (got {cfg.embed_dim})"
        )


def load_config(config_path: Path | str | None = None) -> Config:
    """Load config from YAML merged with environment variables.

    Args:
        config_path: Path to YAML file. Defaults to `<repo_root>/config.yaml`.

    Returns:
        Validated, frozen `Config` instance.

    Raises:
        FileNotFoundError: if the YAML file does not exist.
        ValueError: on any validation failure or missing required env var.
    """
    # 1. Load .env if one is found by walking up from CWD (not the source file's dir).
    # `find_dotenv(usecwd=True)` is important for test isolation: tests `monkeypatch.chdir(tmp_path)`
    # to ensure no stray developer .env re-supplies env vars they just deleted.
    # In production, callers run from the repo root, so the repo-level .env is picked up.
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path)

    # 2. Read YAML
    if config_path is None:
        yaml_path = _repo_root() / "config.yaml"
    else:
        yaml_path = Path(config_path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"config file not found: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"config.yaml must define a mapping at the top level "
            f"(got {type(data).__name__})"
        )

    # 3. Apply env overrides for the documented YAML fields
    for env_name, (field_name, caster) in _ENV_OVERRIDES.items():
        raw = os.environ.get(env_name)
        if raw is None or raw == "":
            continue
        if caster is str:
            data[field_name] = raw
        else:
            data[field_name] = _cast(raw, caster, field_name)

    # 4. Pull required secrets from env (non-empty)
    for var in _REQUIRED_ENV_VARS:
        raw = os.environ.get(var)
        if raw is None or raw == "":
            raise ValueError(
                f"required environment variable {var} is missing or empty"
            )
    data["llmod_api_key"] = os.environ["LLMOD_API_KEY"]
    data["llmod_base_url"] = os.environ["LLMOD_BASE_URL"]
    data["pinecone_api_key"] = os.environ["PINECONE_API_KEY"]

    # 5. Build the Config (catch missing/extra keys with a clear error)
    expected = {f.name for f in fields(Config)}
    missing = expected - data.keys()
    if missing:
        raise ValueError(
            f"config is missing required field(s): {sorted(missing)}"
        )
    extra = data.keys() - expected
    if extra:
        raise ValueError(
            f"config has unknown field(s): {sorted(extra)}"
        )

    cfg = Config(**{name: data[name] for name in expected})

    # 6. Validate
    _validate(cfg)
    return cfg
