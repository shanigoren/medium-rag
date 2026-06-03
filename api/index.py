"""Component 12: the HTTP API (Vercel entry point).

A thin FastAPI wrapper over the C11 chain. Two routes, exactly the assignment's
wire contract:

    POST /api/prompt  {"question": "..."} -> answer(question).to_api_dict()
        = {"response", "context": [{article_id, title, chunk, score}],
           "Augmented_prompt": {"System", "User"}}
    GET  /api/stats                        -> {"chunk_size", "overlap_ratio", "top_k"}

The wrapper owns NO answer logic. C11's `to_api_dict()` owns the body shape and the
capital-A `Augmented_prompt` casing; C9 owns the inner System/User casing; C1 owns
the config values. We return a PLAIN dict (no `response_model`) so FastAPI
serializes the keys verbatim -- a `response_model` would risk renaming the
non-Pythonic `Augmented_prompt`.

Error policy mirrors the chain's honest-failure stance: an empty/whitespace
question (chain `ValueError`) -> 400; a malformed body (pydantic) -> 422; anything
else (proxy/auth/Pinecone) propagates -> 500. No catch-all hides a failure behind
an empty 200.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `import src.*` resolve both locally (`uvicorn api.index:app` from the repo
# root) and inside the Vercel function sandbox, whose CWD / packaging differ. This
# must run before the `src` imports below (hence the noqa on those).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from src.config import load_config  # noqa: E402
from src.rag.chain import answer  # noqa: E402

app = FastAPI()


class PromptRequest(BaseModel):
    """POST /api/prompt body -- exactly the assignment input schema."""

    question: str


@app.post("/api/prompt")
def prompt(req: PromptRequest) -> dict:
    """Answer one question over the corpus; return the assignment's strict output
    schema (C11's `to_api_dict()`) as a plain dict.

    An empty/whitespace question raises `ValueError` in the chain (before any LLM
    call) -> mapped to 400. Any other failure propagates -> 500 (no silent empty
    answer). Production answers from `cfg.pinecone_namespace`; the endpoint takes
    no namespace parameter.
    """
    try:
        return answer(req.question).to_api_dict()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/stats")
def stats() -> dict:
    """Return the live RAG hyperparameters -- EXACTLY the three assignment keys,
    read fresh from config each call so the endpoint always reflects the current
    values (assignment requirement). No LLM / Pinecone access.
    """
    cfg = load_config()
    return {
        "chunk_size": cfg.chunk_size,
        "overlap_ratio": cfg.overlap_ratio,
        "top_k": cfg.top_k,
    }
