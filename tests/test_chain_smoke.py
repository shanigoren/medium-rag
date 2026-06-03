"""C11 component smoke test (live).

Runs only with `pytest --smoke`. This is the COMPONENT's own live check -- does
the composed chain run against real LLMod.AI + Pinecone and return a well-formed
result? -- NOT the integration gate (that is CP-B, `tests/test_cp_b_read_path.py`,
which holds the cross-component seam assertions).

Read-only against the `smoke` namespace; deletes nothing (CP-B/C/D reuse it).
"""

from __future__ import annotations

import pytest

from src.config import load_config
from src.rag.chain import answer
from src.rag.vectorstore import namespace_stats


def _require_smoke(cfg) -> None:
    if namespace_stats("smoke", cfg)["vector_count"] == 0:
        pytest.skip("run scripts/ingest.py --limit 10 --namespace smoke first")


@pytest.mark.smoke
def test_smoke_chain_answers_live():
    """The composed loop runs end-to-end live and returns a well-formed
    AnswerResult: a non-empty response string, a non-empty context, and a
    well-formed to_api_dict(). Cross-component seams are CP-B's job."""
    cfg = load_config()
    _require_smoke(cfg)

    res = answer("What does the article say about building habits?", cfg, namespace="smoke")

    assert isinstance(res.response, str) and res.response.strip()
    assert res.context  # non-empty
    api = res.to_api_dict()
    assert set(api) == {"response", "context", "Augmented_prompt"}
    assert set(api["Augmented_prompt"]) == {"System", "User"}
