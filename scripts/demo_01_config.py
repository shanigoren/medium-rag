"""Demo for Component 1 (Config).

Run:
    conda activate medium-rag
    python scripts/demo_01_config.py

Loads config.yaml + .env, prints the resolved Config with secrets masked.
Exits non-zero on any error.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python scripts/demo_01_config.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config


def _mask(secret: str) -> str:
    if not secret:
        return "<empty>"
    if len(secret) <= 4:
        return "***"
    return f"***{secret[-4:]} ({len(secret)} chars)"


def main() -> int:
    print("Loading config from <repo_root>/config.yaml + .env ...\n")

    try:
        cfg = load_config()
    except (FileNotFoundError, ValueError) as exc:
        print(f"FAILED to load config: {exc}", file=sys.stderr)
        return 1

    print("RAG hyperparameters")
    print(f"  chunk_size         = {cfg.chunk_size}")
    print(f"  overlap_ratio      = {cfg.overlap_ratio}")
    print(f"  top_k              = {cfg.top_k}")
    print(f"  retrieval_fetch_k  = {cfg.retrieval_fetch_k}")
    print(f"  embed_content      = {cfg.embed_content}")
    print()
    print("Models (LLMod.AI)")
    print(f"  embed_model        = {cfg.embed_model}")
    print(f"  embed_dim          = {cfg.embed_dim}")
    print(f"  chat_model         = {cfg.chat_model}")
    print(f"  reasoning_effort   = {cfg.reasoning_effort}")
    print()
    print("Pinecone")
    print(f"  pinecone_index     = {cfg.pinecone_index}")
    print(f"  pinecone_namespace = {cfg.pinecone_namespace}")
    print()
    print("Secrets (masked)")
    print(f"  LLMOD_API_KEY      = {_mask(cfg.llmod_api_key)}")
    print(f"  LLMOD_BASE_URL     = {cfg.llmod_base_url}")
    print(f"  PINECONE_API_KEY   = {_mask(cfg.pinecone_api_key)}")
    print()
    print("OK: config loaded and validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
