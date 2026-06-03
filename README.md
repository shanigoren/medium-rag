# Medium Article RAG Assistant

RAG assistant over the course-provided `medium-english-50mb.csv` corpus of 7,682 Medium articles. The app answers from retrieved article metadata and passages only, using LLMod.AI models, Pinecone, and a FastAPI endpoint ready for Vercel deployment.

## Assignment Alignment

- Uses `4UHRUIN-text-embedding-3-small` embeddings with 1536 dimensions.
- Uses `4UHRUIN-gpt-5-mini` for query rewriting and final answers.
- Uses Pinecone as the vector database.
- Keeps the selected hyperparameters within the assignment caps: `chunk_size=768`, `overlap_ratio=0.10`, `top_k=5`.
- Exposes the required API routes:
  - `POST /api/prompt`
  - `GET /api/stats`
- The final answer prompt includes the required system-prompt constraints and instructs the model to answer only from retrieved context.

## Current Status

The project is in a working hand-in state:

- Full production corpus ingested into Pinecone namespace `prod`.
- Final config selected from subset experiments and hard add-on evaluation.
- API contract is covered by offline tests.
- Full local test suite passes: `312 passed, 50 skipped`.

Known limitation: the final production sanity check found one retrieval miss on the assignment's pandemic/bubonic-plague example. The relevant article exists in the full corpus but is ranked outside the allowed `top_k <= 30` retrieval window. This is documented in [reports/summary.md](reports/summary.md) and [reports/eval/prod_scale_sanity_20260602.md](reports/eval/prod_scale_sanity_20260602.md). No lexical rescue, hybrid search, reranker, or per-question special case is included because those would move beyond the current assignment design.

## Setup

Use the `medium-rag` conda environment. Do not use the repo-local `.venv`; it may be stale.

```powershell
conda create -n medium-rag python=3.11 -y
conda run -n medium-rag pip install -r requirements.txt
```

Create `.env` from `.env.example` and fill in:

```powershell
LLMOD_API_KEY=...
LLMOD_BASE_URL=https://api.llmod.ai/v1
PINECONE_API_KEY=...
```

## Run Tests

```powershell
conda run -n medium-rag pytest -q
```

## Run Locally

From the repo root:

```powershell
conda run -n medium-rag uvicorn api.index:app --reload
```

Then:

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/api/stats

Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/prompt `
  -ContentType "application/json" `
  -Body '{"question":"List exactly 3 articles about education. Return only the titles."}'
```

## Production Ingest

The current `config.yaml` points at the selected production namespace:

```yaml
chunk_size: 768
overlap_ratio: 0.10
top_k: 5
retrieval_fetch_k: 30
embed_content: "chunk_only"
pinecone_namespace: "prod"
```

The full ingest has already been run locally. To recreate it:

```powershell
$env:PYTHONPATH=(Get-Location).Path
conda run -n medium-rag python scripts/ingest.py --namespace prod
```

## Deploy To Vercel

Set these Vercel environment variables:

- `LLMOD_API_KEY`
- `LLMOD_BASE_URL`
- `PINECONE_API_KEY`
- Optional: `PINECONE_INDEX=medium-rag`
- Optional: `PINECONE_NAMESPACE=prod`

The included `vercel.json` routes all requests to `api/index.py`.

## Key Reports

- [reports/summary.md](reports/summary.md): short hand-in report and known limitation.
- [reports/eval/final_config_decision_20260602.md](reports/eval/final_config_decision_20260602.md): final hyperparameter decision.
- [reports/eval/prod_scale_sanity_20260602.md](reports/eval/prod_scale_sanity_20260602.md): full-corpus sanity check and retrieval shortcoming.
- [reports/eval/hard_addon_10_config_aggregate_20260602.md](reports/eval/hard_addon_10_config_aggregate_20260602.md): hard add-on comparison.

## Layout

```text
api/                 FastAPI app and Vercel entry point
src/                 Config, data loading, LLMod clients, RAG components, eval helpers
scripts/             Ingest, demos, evaluation, probes
tests/               Offline unit and integration tests
reports/             Hand-in reports and evaluation artifacts
config.yaml          Selected production hyperparameters
.env.example         Local/Vercel environment variable template
```
