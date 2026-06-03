"""Demo for Component 5 (Pinecone wrapper).

Run:
    conda activate medium-rag
    python scripts/demo_05_vectorstore.py

End-to-end visible roundtrip on the `_demo_vectorstore` namespace. Uses
deterministic one-hot vectors so the top-1 result is unambiguous. Cleans up
the namespace in a finally block regardless of failure. Costs $0.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.rag.vectorstore import (
    WRITE_CONSISTENCY_POLL_S,
    WRITE_CONSISTENCY_TIMEOUT_S,
    delete_namespace,
    ensure_index,
    namespace_stats,
    query,
    upsert,
)


NS = "_demo_vectorstore"
EMBED_DIM = 1536
LABELS = ["alpha", "beta", "gamma", "delta", "epsilon"]


def _one_hot(i: int, dim: int = EMBED_DIM) -> list[float]:
    v = [0.0] * dim
    v[i] = 1.0
    return v


def _mask(secret: str) -> str:
    if not secret or len(secret) < 4:
        return "****"
    return "..." + secret[-4:]


def _poll_until(predicate, timeout_s: float = WRITE_CONSISTENCY_TIMEOUT_S) -> float:
    start = time.monotonic()
    deadline = start + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return time.monotonic() - start
        time.sleep(WRITE_CONSISTENCY_POLL_S)
    raise TimeoutError(f"predicate never became true within {timeout_s}s")


def main() -> int:
    print()
    print("Loading config and Pinecone client ...")
    cfg = load_config()
    print(f"  index          = {cfg.pinecone_index}")
    print(f"  api_key        = {_mask(cfg.pinecone_api_key)}     (last 4 chars only)")
    print(f"  embed_dim      = {cfg.embed_dim}")
    print()

    print("ensure_index() ...")
    t0 = time.monotonic()
    ensure_index(cfg)
    elapsed = time.monotonic() - t0
    if elapsed < 1.0:
        print("  status         = ready (already existed)")
    else:
        print(f"  status         = ready (created in {elapsed:.0f}s)")
    print()

    try:
        # Wipe any leftovers from a prior failed run before writing.
        delete_namespace(NS, cfg)

        n = 5
        ids = [f"vec-{i}" for i in range(n)]
        vectors = [_one_hot(i) for i in range(n)]
        metadatas = [{"label": LABELS[i]} for i in range(n)]

        print(f"Upserting {n} deterministic vectors into {NS!r} ...")
        print(f"  vectors        = [vec-0 .. vec-{n - 1}], one-hot at distinct dims")
        print(
            "  metadata       = "
            + "{" + " | ".join(f'"{lbl}"' for lbl in LABELS[:n]) + "}"
        )
        written = upsert(NS, ids, vectors, metadatas, cfg)
        print(f"  count_written  = {written}")
        print()

        print("Waiting for write consistency ...", end=" ", flush=True)
        wait_elapsed = _poll_until(
            lambda: namespace_stats(NS, cfg)["vector_count"] == n
        )
        print(f"visible after {wait_elapsed:.1f}s")
        print()

        print(
            "Querying with the closest vector to vec-2 (one-hot at dim=2), top_k=3 ..."
        )
        results = query(NS, _one_hot(2), top_k=3, cfg=cfg)
        for rank, m in enumerate(results, start=1):
            print(
                f"  rank {rank} | id={m.id} | score={m.score:.4f} "
                f"| metadata={m.metadata}"
            )
        print()

        stats = namespace_stats(NS, cfg)
        print(f"namespace_stats({NS!r}) -> {stats}")
        print()
        print("OK: Pinecone wrapper roundtrip succeeded.")
        return 0

    except Exception as e:
        print(f"Pinecone roundtrip FAILED: {e}", file=sys.stderr)
        raise
    finally:
        try:
            delete_namespace(NS, cfg)
            print(f"Cleaned up namespace: {NS}")
        except Exception as cleanup_err:
            print(f"WARNING: cleanup failed: {cleanup_err}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
