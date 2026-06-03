"""Demo for Component 3 (LLMod.AI clients).

Run:
    conda activate medium-rag
    python scripts/demo_03_llmod_clients.py

Constructs both clients from the live config, embeds 'hello', and asks the
chat model for a one-word reply. Exits non-zero on any error.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import load_config
from src.llm.clients import get_chat, get_embeddings


def _mask(secret: str) -> str:
    if not secret:
        return "<empty>"
    if len(secret) <= 4:
        return "***"
    return f"***{secret[-4:]} ({len(secret)} chars)"


def main() -> int:
    print("Loading config and constructing clients ...")
    try:
        cfg = load_config()
    except (FileNotFoundError, ValueError) as exc:
        print(f"FAILED to load config: {exc}", file=sys.stderr)
        return 1

    print(f"  embed_model    = {cfg.embed_model}")
    print(f"  chat_model     = {cfg.chat_model}")
    print(f"  base_url       = {cfg.llmod_base_url}")
    print(f"  api_key        = {_mask(cfg.llmod_api_key)}")
    print()

    embeddings = get_embeddings(cfg)
    chat = get_chat(cfg)

    print("Embedding 'hello' ...")
    try:
        vec = embeddings.embed_query("hello")
    except Exception as exc:
        print(
            f"LLMod.AI rejected the embeddings request — check LLMOD_API_KEY "
            f"and LLMOD_BASE_URL.\n  detail: {exc}",
            file=sys.stderr,
        )
        return 1

    norm = math.sqrt(sum(v * v for v in vec))
    preview = ", ".join(f"{v:+.4f}" for v in vec[:8])
    print(f"  dim            = {len(vec)}")
    print(f"  first 8 values = [{preview}, ...]")
    print(f"  norm           = {norm:.4f}   (text-embedding-3-small returns unit vectors)")
    print()

    print("Chatting (system='you are terse', user='reply with the single word: pong') ...")
    try:
        response = chat.invoke([
            SystemMessage("you are terse"),
            HumanMessage("reply with the single word: pong"),
        ])
    except Exception as exc:
        print(
            f"LLMod.AI rejected the chat request — check LLMOD_API_KEY and "
            f"LLMOD_BASE_URL.\n  detail: {exc}",
            file=sys.stderr,
        )
        return 1

    text = response.content if hasattr(response, "content") else str(response)
    print(f"  response       = {text}")

    usage = getattr(response, "usage_metadata", None)
    if usage:
        print(
            f"  tokens         = prompt={usage.get('input_tokens', '?')}, "
            f"completion={usage.get('output_tokens', '?')}, "
            f"total={usage.get('total_tokens', '?')}"
        )
    print()
    print("OK: both LLMod.AI clients are working.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
