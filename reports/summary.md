# Medium Article RAG Assistant - Final Summary

## System

This project implements the required RAG assistant over the course-provided Medium article CSV. It uses:

- `4UHRUIN-text-embedding-3-small` for embeddings.
- `4UHRUIN-gpt-5-mini` for query rewriting and final answering.
- Pinecone serverless as the vector database.
- FastAPI with the required `POST /api/prompt` and `GET /api/stats` routes.
- Vercel-compatible routing through `vercel.json`.

The assistant answers only from retrieved Medium article metadata and passages. The final answer prompt includes the required assignment system-prompt constraints, plus concise formatting guidance.

## Final Hyperparameters

| Field | Value |
|---|---:|
| `chunk_size` | `768` |
| `overlap_ratio` | `0.10` |
| `top_k` | `5` |
| `retrieval_fetch_k` | `30` |
| `embed_content` | `chunk_only` |
| `reasoning_effort` | `low` |
| `pinecone_namespace` | `prod` |

These values satisfy the assignment caps: chunk size is at most 1024 tokens, overlap is at most 0.3, and top-k is at most 30.

## Experiment Summary

The workflow validated retrieval on small subsets before scaling:

1. Smoke-tested the pipeline on a tiny subset.
2. Compared embedding content variants on a deterministic 100-row subset.
3. Ran a chunk-size, overlap, and top-k grid over the 100-row subset.
4. Added a harder 100-row add-on evaluation because the original set was too easy to distinguish the top configs.
5. Selected `chunk_only`, `chunk_size=768`, `overlap_ratio=0.10`, `top_k=5`.
6. Ingested the full 7,682-article corpus into Pinecone namespace `prod`.

The selected config scored perfectly on the original curated 20-question subset and tied for the best reviewed score on the hard add-on set. Detailed experiment notes are in:

- `reports/eval/phase_a_b_experiments_20260603.md`
- `reports/eval/final_config_decision_20260602.md`
- `reports/eval/hard_addon_10_config_aggregate_20260602.md`
- `reports/eval/prod_scale_sanity_20260602.md`

Phase A compared embedded text formats. With fixed `chunk_size=512`, `overlap_ratio=0.10`, and `top_k=5`, `chunk_only` reached `1.0000` recall and `0.9833` combined score, while `title_tags_chunk` reached `0.9375` recall and `0.9458` combined score.

Phase B then swept chunk size, overlap, and top-k for `chunk_only`. The original 20-question set was useful as a filter but too easy as a final selector: 10 of 27 Phase B configurations scored perfectly. A harder add-on set was therefore run against only those 10 tied configs. The final selected config, `chunk_size=768`, `overlap_ratio=0.10`, `top_k=5`, tied for the best hard-add-on combined score while using less retrieved context than the `top_k=8` alternative and less overlap than the `0.15` alternative.

## Known Retrieval Limitation

The full-corpus production sanity check exposed a retrieval weakness on the assignment's pandemic example:

`Find an article that argues past pandemics (such as the bubonic plague) can spur innovation and recovery, and summarise its central argument.`

The relevant article exists in the full corpus as row `6299`, titled `Rebounding From The Pandemic... with AI`, but the production vector retriever does not surface it within the allowed retrieval window. A later probe found the target article at rank 219 for the original question, which is outside the assignment's maximum `top_k` of 30. Increasing `top_k` up to 30 and trying a full-corpus `title_tags_chunk` namespace did not fix this case.

I am leaving this as an acknowledged shortcoming rather than adding a lexical rescue, hybrid search layer, reranker over a larger candidate pool, or per-question special case. Those approaches may be reasonable future improvements, but they would change the retrieval design beyond the current assignment constraints and would make the reported `top_k` contract less straightforward.

For this version, the system remains a clean Pinecone vector-RAG implementation with the selected reported hyperparameters.

## Verification

Latest local verification:

```powershell
conda run -n medium-rag pytest -q
```

Result: `312 passed, 50 skipped`.

## Deployment Notes

Before submitting, deploy to Vercel and set:

- `LLMOD_API_KEY`
- `LLMOD_BASE_URL`
- `PINECONE_API_KEY`
- Optional: `PINECONE_INDEX=medium-rag`
- Optional: `PINECONE_NAMESPACE=prod`

Keep the Pinecone index active until grading is complete.
