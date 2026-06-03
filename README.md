# Medium Article RAG Assistant

RAG API over the course-provided Medium article corpus. The service retrieves article chunks from Pinecone and answers only from the retrieved context.

The current production configuration is:

```yaml
chunk_size: 768
overlap_ratio: 0.10
top_k: 5
retrieval_fetch_k: 30
embed_content: "chunk_only"
pinecone_namespace: "prod"
```

The public API exposes:

- `POST /api/prompt`
- `GET /api/stats`

## Reports

The main report is [reports/summary.md](reports/summary.md).

Experiment and evaluation notes are in [reports/eval](reports/eval):

- [phase_a_b_experiments_20260603.md](reports/eval/phase_a_b_experiments_20260603.md)
- [final_config_decision_20260602.md](reports/eval/final_config_decision_20260602.md)
- [hard_addon_10_config_aggregate_20260602.md](reports/eval/hard_addon_10_config_aggregate_20260602.md)
- [prod_scale_sanity_20260602.md](reports/eval/prod_scale_sanity_20260602.md)

## Setup

Use the `medium-rag` conda environment.

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

Optional overrides:

```powershell
PINECONE_INDEX=medium-rag
PINECONE_NAMESPACE=prod
```

## Run Locally

```powershell
conda run -n medium-rag uvicorn api.index:app --reload
```

Check the running API:

```powershell
Invoke-RestMethod -Method Get -Uri http://127.0.0.1:8000/api/stats
```

Ask a question:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/api/prompt `
  -ContentType "application/json" `
  -Body '{"question":"List exactly 3 articles about education. Return only the titles."}'
```

## API Shape

`POST /api/prompt`

Request:

```json
{
  "question": "List exactly 3 articles about education. Return only the titles."
}
```

Response:

```json
{
  "response": "...",
  "context": [
    {
      "article_id": "123",
      "title": "...",
      "chunk": "...",
      "score": 0.72
    }
  ],
  "Augmented_prompt": {
    "System": "...",
    "User": "..."
  }
}
```

`GET /api/stats`

Returns the active retrieval configuration, including chunk size, overlap ratio, and top-k.

## Tests

```powershell
conda run -n medium-rag pytest -q
```

## Ingest

The production namespace has already been ingested. To recreate it from the CSV:

```powershell
$env:PYTHONPATH=(Get-Location).Path
conda run -n medium-rag python scripts/ingest.py --namespace prod
```

The local corpus CSV and environment files are intentionally not committed.
