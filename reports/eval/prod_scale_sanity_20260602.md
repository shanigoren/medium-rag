# Production Scale Sanity

Run date: 2026-06-02

## Final Selected 100-Row Config

| Field | Value |
|---|---|
| `embed_content` | `chunk_only` |
| `chunk_size` | `768` |
| `overlap_ratio` | `0.10` |
| `top_k` | `5` |
| `retrieval_fetch_k` | `30` |
| `reasoning_effort` | `low` |
| `namespace` | `prod` |

## Full Ingest

- Log: `run_logs/prod_ingest_20260602T133550Z.txt`
- Articles loaded: `7,682`
- Articles chunked: `7,682`
- Articles skipped: `0`
- Vectors upserted: `18,456`
- Post-ingest namespace stats: `{'vector_count': 18456}`

The vector count is within the expected production band of roughly `10k-30k`.

## Production Sanity Finding

The first production sanity run found a full-corpus retrieval miss on the assignment's pandemic example:

`Find an article that argues past pandemics (such as the bubonic plague) can spur innovation and recovery, and summarise its central argument.`

The target article exists in the full corpus as row `6299`, `Rebounding From The Pandemic... with AI`, but the selected `chunk_only` production retriever did not surface it. The answer model therefore returned an IDK response from the retrieved context.

## General Retuning Attempt

No targeted rewriter rescue is used.

Top-k sweep on existing `prod` / `chunk_only` vectors:

- Log: `run_logs/retrieval_topk_sweep_prod_20260602T153351Z.txt`
- Tested `top_k`: `8`, `12`, `20`, `30`
- Result: failed. Article `6299` was not retrieved for the pandemic example even at `top_k=30`.

Fallback full-corpus `title_tags_chunk` ingest:

- Namespace: `prod_title_tags_c768_o10`
- Ingest log: `run_logs/prod_title_tags_ingest_20260602T153707Z.txt`
- Articles loaded: `7,682`
- Vectors upserted: `18,456`
- LLMod spend delta: `$0.22871864`
- Remaining budget after ingest: `$3.39904864`

Top-k sweep on `prod_title_tags_c768_o10`:

- Log: `run_logs/retrieval_topk_sweep_prod_title_tags_20260602T155221Z.txt`
- Extra cap probe: `run_logs/retrieval_title_tags_k30_probe_20260602T155355Z.txt`
- Tested `top_k`: `5`, `8`, `12`, `20`, plus `30`
- Result: failed. Article `6299` was not retrieved for the pandemic example even at `top_k=30`.

## Current Decision

The planned general retuning path did not produce an acceptable replacement config. The committed production config should remain unchanged for now:

- `embed_content=chunk_only`
- `chunk_size=768`
- `overlap_ratio=0.10`
- `top_k=5`
- `retrieval_fetch_k=30`
- `pinecone_namespace=prod`

This miss is kept as a documented retrieval limitation for the hand-in version. No targeted rewriter rescue is used, and no lexical/hybrid retrieval, BM25-style candidate rescue, reranking over a larger candidate pool, or per-question special case is added. Those methods may be useful future work, but they would move the project beyond the current assignment design and make the reported Pinecone vector-retrieval setup less straightforward.

The hand-in version therefore remains a clean Pinecone vector-RAG system using the selected reported hyperparameters. For the missed pandemic example, the correct behavior is to answer from the retrieved context only; if the relevant article is not retrieved, the assistant should return the required "I don't know" response instead of fabricating an answer.

## Verification

- Focused rollback tests after removing the targeted rescue: `103 passed`
- Full-suite status after hand-in documentation updates: `312 passed, 50 skipped`
