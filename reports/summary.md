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

The experiments used a deterministic 100-row subset of the Medium corpus and a 20-question golden evaluation set. The questions covered factual article lookup, thematic lookup, multi-article listing/comparison, and unsupported questions where the assistant should answer that it does not know.

Each run used the same API chain:

1. Rewrite the user question for retrieval.
2. Embed the rewritten query.
3. Retrieve from Pinecone.
4. Deduplicate retrieved contexts by article.
5. Generate an answer from retrieved context only.
6. Score retrieval and answer quality against the golden expectations.

The main metrics were:

- `recall_at_k`: whether expected article IDs appeared in retrieved context for answerable questions.
- `answer_pass_rate`: reviewed answer correctness.
- `combined_score`: aggregate of retrieval, answer, deduplication, and unsupported-question behavior.
- `idk_pass_rate`: whether unsupported questions were rejected correctly.
- `retrieval_issues` and `answer_issues`: reviewed failure counts.

All Phase A and Phase B runs used `4UHRUIN-text-embedding-3-small`, `4UHRUIN-gpt-5-mini`, Pinecone index `medium-rag`, `retrieval_fetch_k=30`, and `reasoning_effort=low`.

Supporting experiment notes are in:

- `reports/eval/phase_a_b_experiments_20260603.md`
- `reports/eval/final_config_decision_20260602.md`
- `reports/eval/hard_addon_10_config_aggregate_20260602.md`
- `reports/eval/prod_scale_sanity_20260602.md`

### Phase A: Embedding Content

Phase A tested what text should be embedded for each chunk. The fixed baseline settings were `chunk_size=512`, `overlap_ratio=0.10`, and `top_k=5`.

| Config | Embed Content | Recall | Answer Pass | Combined | IDK | Retrieval Issues | Answer Issues | Cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `chunk_only_c512_o10` | `chunk_only` | 1.0000 | 0.9500 | 0.9833 | 1.0000 | 0 | 1 | $0.024121 |
| `title_tags_chunk_c512_o10` | `title_tags_chunk` | 0.9375 | 0.9000 | 0.9458 | 1.0000 | 2 | 1 | $0.022467 |

`chunk_only` performed better on this subset. It retrieved every expected answerable article and had only one answer-level issue. The `title_tags_chunk` variant looked attractive because article titles and tags can help keyword-style queries, but in this subset it introduced a retrieval miss and another retrieval-quality issue. The likely reason is that title and tag text can overweight broad topic labels relative to passage semantics.

Decision after Phase A: continue the grid search with `embed_content=chunk_only`.

### Phase B: Chunk, Overlap, and Top-k

Phase B held `chunk_only` fixed and swept chunk size, overlap, and top-k.

| Parameter | Values Tried |
|---|---|
| `chunk_size` | `512`, `768`, `1024` |
| `overlap_ratio` | `0.05`, `0.10`, `0.15` |
| `top_k` | `3`, `5`, `8` |

This produced 27 configurations. All settings were inside the assignment limits.

| Config | Chunk | Overlap | Top-k | Recall | Answer Pass | Combined | IDK | Retrieval Issues | Answer Issues | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `c512_o05_k3` | 512 | 0.05 | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | - | - | $0.020335 |
| `c512_o05_k5` | 512 | 0.05 | 5 | 1.0000 | 0.9500 | 0.9833 | 1.0000 | 0 | 1 | $0.020189 |
| `c512_o05_k8` | 512 | 0.05 | 8 | 1.0000 | 0.9000 | 0.9667 | 1.0000 | 0 | 2 | $0.020053 |
| `c512_o10_k3` | 512 | 0.10 | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | $0.018547 |
| `c512_o10_k5` | 512 | 0.10 | 5 | 0.9375 | 0.9500 | 0.9625 | 1.0000 | 1 | 1 | $0.020782 |
| `c512_o10_k8` | 512 | 0.10 | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | $0.018007 |
| `c512_o15_k3` | 512 | 0.15 | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | $0.015906 |
| `c512_o15_k5` | 512 | 0.15 | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | $0.018177 |
| `c512_o15_k8` | 512 | 0.15 | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | $0.020255 |
| `c768_o05_k3` | 768 | 0.05 | 3 | 0.9375 | 0.9500 | 0.9625 | 1.0000 | 1 | 1 | $0.023978 |
| `c768_o05_k5` | 768 | 0.05 | 5 | 0.9375 | 0.9500 | 0.9625 | 1.0000 | 1 | 1 | $0.019521 |
| `c768_o05_k8` | 768 | 0.05 | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | $0.025363 |
| `c768_o10_k3` | 768 | 0.10 | 3 | 0.9375 | 0.9500 | 0.9625 | 1.0000 | 1 | 1 | $0.016743 |
| `c768_o10_k5` | 768 | 0.10 | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | $0.020708 |
| `c768_o10_k8` | 768 | 0.10 | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | $0.025887 |
| `c768_o15_k3` | 768 | 0.15 | 3 | 0.9375 | 1.0000 | 0.9792 | 1.0000 | 1 | 0 | $0.019633 |
| `c768_o15_k5` | 768 | 0.15 | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 | $0.019272 |
| `c768_o15_k8` | 768 | 0.15 | 8 | 1.0000 | 0.9500 | 0.9833 | 1.0000 | 0 | 1 | $0.022865 |
| `c1024_o05_k3` | 1024 | 0.05 | 3 | 0.9375 | 0.9500 | 0.9625 | 1.0000 | 1 | 1 | $0.022564 |
| `c1024_o05_k5` | 1024 | 0.05 | 5 | 0.9375 | 0.9500 | 0.9625 | 1.0000 | - | - | $0.023274 |
| `c1024_o05_k8` | 1024 | 0.05 | 8 | 0.9375 | 0.9500 | 0.9625 | 1.0000 | 1 | 1 | $0.031442 |
| `c1024_o10_k3` | 1024 | 0.10 | 3 | 0.9375 | 0.9500 | 0.9625 | 1.0000 | 2 | 1 | $0.019251 |
| `c1024_o10_k5` | 1024 | 0.10 | 5 | 0.9375 | 0.9500 | 0.9625 | 1.0000 | 1 | 1 | $0.022930 |
| `c1024_o10_k8` | 1024 | 0.10 | 8 | 0.9375 | 0.9500 | 0.9625 | 1.0000 | 1 | 1 | $0.028829 |
| `c1024_o15_k3` | 1024 | 0.15 | 3 | 0.9375 | 0.9000 | 0.9458 | 1.0000 | 2 | 1 | $0.022317 |
| `c1024_o15_k5` | 1024 | 0.15 | 5 | 0.9375 | 0.9500 | 0.9625 | 1.0000 | - | - | $0.019358 |
| `c1024_o15_k8` | 1024 | 0.15 | 8 | 0.9375 | 0.9500 | 0.9625 | 1.0000 | - | - | $0.029029 |

The original 20-question set filtered out weak settings but was not hard enough to choose a single winner. Ten configurations achieved a perfect reviewed combined score. The 1024-token chunks were less reliable on this subset because every 1024-token run had recall below 1.0. The 512-token chunks were strong and cheap, but more fragmented. The 768-token chunks gave the best balance between semantic coverage and context size.

### Hard Add-on Check

The hard add-on set was run only on the ten Phase B configurations that scored perfectly on the original set, avoiding extra spend on already-weaker configs.

| Config | Recall | Answer Pass | Combined | IDK | Retrieval Issues | Answer Issues | Cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| `c768_o05_k8` | 0.9375 | 0.9500 | 0.9625 | 1.0000 | 2 | 1 | $0.036749 |
| `c768_o10_k5` | 0.9375 | 0.9500 | 0.9625 | 1.0000 | 1 | 1 | $0.025350 |
| `c768_o15_k5` | 0.9375 | 0.9500 | 0.9625 | 1.0000 | 1 | 1 | $0.023632 |
| `c512_o05_k3` | 0.9375 | 0.9000 | 0.9458 | 1.0000 | 3 | 2 | $0.017304 |
| `c512_o10_k8` | 0.9375 | 0.9000 | 0.9458 | 1.0000 | 2 | 2 | $0.024359 |
| `c512_o15_k3` | 0.9375 | 0.9000 | 0.9458 | 1.0000 | 2 | 2 | $0.015975 |
| `c512_o15_k5` | 0.9375 | 0.9000 | 0.9458 | 1.0000 | - | - | $0.019840 |
| `c512_o15_k8` | 0.9375 | 0.9000 | 0.9458 | 1.0000 | 2 | 2 | $0.021151 |
| `c768_o10_k8` | 0.9375 | 0.9000 | 0.9458 | 1.0000 | 2 | 1 | $0.025800 |
| `c512_o10_k3` | 0.9375 | 0.8500 | 0.9292 | 1.0000 | - | - | $0.018094 |

The hard add-on changed the decision. The cheaper 512-token candidates that looked perfect in Phase B fell behind. The top tier became `c768_o05_k8`, `c768_o10_k5`, and `c768_o15_k5`. All three tied on answer pass and combined score. `c768_o10_k5` was selected because it used less context than `c768_o05_k8` and less overlap than `c768_o15_k5`, while keeping the same hard-add-on score.

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
