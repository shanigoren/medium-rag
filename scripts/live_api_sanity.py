"""Live local API sanity check against the configured production namespace.

Run from the repo root and redirect output to run_logs because this calls the
real API stack and spends LLMod/Pinecone budget:

    $env:PYTHONPATH=(Get-Location).Path
    conda run -n medium-rag python scripts/live_api_sanity.py *> run_logs/live_api_sanity.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from api.index import app
from scripts._console import to_ascii


QUESTION = "List exactly 3 articles about education. Return only the titles."


def main() -> int:
    client = TestClient(app)

    print("LIVE_LOCAL_API_PROD_SANITY")

    stats = client.get("/api/stats")
    print("STATS_STATUS", stats.status_code)
    print("STATS_BODY", stats.text)

    print("QUESTION", QUESTION)
    resp = client.post("/api/prompt", json={"question": QUESTION})
    print("PROMPT_STATUS", resp.status_code)
    print("RAW_LEN", len(resp.text))

    body = resp.json()
    print("RESPONSE", to_ascii(body.get("response", "")))
    print("CONTEXT_COUNT", len(body.get("context", [])))
    print("BODY_KEYS", sorted(body.keys()))
    print("AUGMENTED_KEYS", sorted(body.get("Augmented_prompt", {}).keys()))

    for i, row in enumerate(body.get("context", []), start=1):
        print(
            "CTX",
            i,
            row.get("article_id"),
            f"{row.get('score'):.6f}",
            to_ascii(row.get("title", "")),
        )

    assert stats.status_code == 200
    assert resp.status_code == 200
    assert set(body) == {"response", "context", "Augmented_prompt"}
    assert set(stats.json()) == {"chunk_size", "overlap_ratio", "top_k"}
    print("OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
