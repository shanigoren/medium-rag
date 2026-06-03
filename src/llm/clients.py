"""LLMod.AI client factories.

Every component that needs to embed text or call gpt-5-mini goes through
`get_embeddings()` / `get_chat()`. Centralizes model name, base_url and API
key so future swaps are a single edit.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.config import Config, load_config


def get_embeddings(cfg: Config | None = None) -> OpenAIEmbeddings:
    """Return an `OpenAIEmbeddings` client configured for LLMod.AI."""
    if cfg is None:
        cfg = load_config()
    return OpenAIEmbeddings(
        model=cfg.embed_model,
        dimensions=cfg.embed_dim,
        api_key=cfg.llmod_api_key,
        base_url=cfg.llmod_base_url,
    )


def get_chat(
    cfg: Config | None = None,
    *,
    reasoning_effort: str | None = None,
) -> ChatOpenAI:
    """Return a `ChatOpenAI` client configured for LLMod.AI.

    `reasoning_effort` is passed as a top-level kwarg — langchain-openai 1.2.2
    exposes it as a typed parameter and would warn (and strip it) if passed
    via `model_kwargs`. gpt-5-mini is a reasoning model, so we do not pass
    `temperature`.

    The optional `reasoning_effort` argument overrides `cfg.reasoning_effort`
    when given (Component 10's query rewriter forces `"minimal"` this way); when
    omitted, the config value is used.
    """
    if cfg is None:
        cfg = load_config()
    return ChatOpenAI(
        model=cfg.chat_model,
        reasoning_effort=reasoning_effort or cfg.reasoning_effort,
        api_key=cfg.llmod_api_key,
        base_url=cfg.llmod_base_url,
    )
