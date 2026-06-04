# Medium Article RAG Assistant - Experiments and Final Report

## System Overview

This project implements a RAG assistant over the course-provided Medium article CSV. It uses:

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

## Experiment Overview

This report documents the subset experiments used to choose the production RAG configuration before the full-corpus ingest, plus later full-corpus checks against the production namespace.

## Evaluation Setup

The experiments used a deterministic 100-row subset of the Medium corpus and a 20-question golden evaluation set. The questions covered four categories:

- Type 1: factual article lookup where one target article should be retrieved.
- Type 2: thematic lookup where the answer depends on finding the right article.
- Type 3: multi-article listing and comparison.
- Type 4: out-of-corpus or unsupported questions where the system should answer that it does not know.

Each run used the same API chain:

1. Rewrite the user question for retrieval.
2. Embed the rewritten query.
3. Retrieve from Pinecone.
4. Deduplicate retrieved contexts by article.
5. Generate an answer from retrieved context only.
6. Score retrieval and answer quality against the golden expectations.

The main metrics were:

- `recall_at_k`: whether expected article IDs appeared in the retrieved context for answerable questions.
- `answer_pass_rate`: reviewed answer correctness.
- `combined_score`: aggregate of retrieval, answer, deduplication, and unsupported-question behavior.
- `idk_pass_rate`: whether unsupported questions were rejected correctly.
- `retrieval_issues` and `answer_issues`: manually reviewed failure counts.

All Phase A and Phase B runs used:

- Embedding model: `4UHRUIN-text-embedding-3-small`
- Chat model: `4UHRUIN-gpt-5-mini`
- Pinecone index: `medium-rag`
- `retrieval_fetch_k=30`
- `reasoning_effort=low`

## Phase A: Embedding Content

Phase A tested what text should be embedded for each chunk. The fixed baseline settings were `chunk_size=512`, `overlap_ratio=0.10`, and `top_k=5`.

Two embedding formats were compared:

- `chunk_only`: embed only the article passage chunk.
- `title_tags_chunk`: embed title, tags, and passage chunk together.

| Config | Embed Content | Recall | Answer Pass | Combined | IDK | Retrieval Issues | Answer Issues |
|---|---|---:|---:|---:|---:|---:|---:|
| `chunk_only_c512_o10` | `chunk_only` | 1.0000 | 0.9500 | 0.9833 | 1.0000 | 0 | 1 |
| `title_tags_chunk_c512_o10` | `title_tags_chunk` | 0.9375 | 0.9000 | 0.9458 | 1.0000 | 2 | 1 |

### Phase A Findings

`chunk_only` performed better on this subset. It retrieved every expected answerable article and had only one answer-level issue. The `title_tags_chunk` variant looked attractive because article titles and tags could help keyword-style queries, but in practice it introduced a retrieval miss and another retrieval-quality issue.

The likely reason is that title and tag text can overweight broad topic labels relative to the actual passage semantics. For this corpus, the passage chunk itself was a cleaner vector target than metadata-prefixed text.

Decision after Phase A: continue the grid search with `embed_content=chunk_only`.

## Phase B: Chunk, Overlap, and Top-k Grid

Phase B held `chunk_only` fixed and swept:

- `chunk_size`: 512, 768, 1024
- `overlap_ratio`: 0.05, 0.10, 0.15
- `top_k`: 3, 5, 8

This produced 27 configurations. All settings were inside the assignment limits.

| Config | Chunk | Overlap | Top-k | Recall | Answer Pass | Combined | IDK | Retrieval Issues | Answer Issues |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `c512_o05_k3` | 512 | 0.05 | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | - | - |
| `c512_o05_k5` | 512 | 0.05 | 5 | 1.0000 | 0.9500 | 0.9833 | 1.0000 | 0 | 1 |
| `c512_o05_k8` | 512 | 0.05 | 8 | 1.0000 | 0.9000 | 0.9667 | 1.0000 | 0 | 2 |
| `c512_o10_k3` | 512 | 0.10 | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| `c512_o10_k5` | 512 | 0.10 | 5 | 0.9375 | 0.9500 | 0.9625 | 1.0000 | 1 | 1 |
| `c512_o10_k8` | 512 | 0.10 | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| `c512_o15_k3` | 512 | 0.15 | 3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| `c512_o15_k5` | 512 | 0.15 | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| `c512_o15_k8` | 512 | 0.15 | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| `c768_o05_k3` | 768 | 0.05 | 3 | 0.9375 | 0.9500 | 0.9625 | 1.0000 | 1 | 1 |
| `c768_o05_k5` | 768 | 0.05 | 5 | 0.9375 | 0.9500 | 0.9625 | 1.0000 | 1 | 1 |
| `c768_o05_k8` | 768 | 0.05 | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| `c768_o10_k3` | 768 | 0.10 | 3 | 0.9375 | 0.9500 | 0.9625 | 1.0000 | 1 | 1 |
| `c768_o10_k5` | 768 | 0.10 | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| `c768_o10_k8` | 768 | 0.10 | 8 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| `c768_o15_k3` | 768 | 0.15 | 3 | 0.9375 | 1.0000 | 0.9792 | 1.0000 | 1 | 0 |
| `c768_o15_k5` | 768 | 0.15 | 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0 | 0 |
| `c768_o15_k8` | 768 | 0.15 | 8 | 1.0000 | 0.9500 | 0.9833 | 1.0000 | 0 | 1 |
| `c1024_o05_k3` | 1024 | 0.05 | 3 | 0.9375 | 0.9500 | 0.9625 | 1.0000 | 1 | 1 |
| `c1024_o05_k5` | 1024 | 0.05 | 5 | 0.9375 | 0.9500 | 0.9625 | 1.0000 | - | - |
| `c1024_o05_k8` | 1024 | 0.05 | 8 | 0.9375 | 0.9500 | 0.9625 | 1.0000 | 1 | 1 |
| `c1024_o10_k3` | 1024 | 0.10 | 3 | 0.9375 | 0.9500 | 0.9625 | 1.0000 | 2 | 1 |
| `c1024_o10_k5` | 1024 | 0.10 | 5 | 0.9375 | 0.9500 | 0.9625 | 1.0000 | 1 | 1 |
| `c1024_o10_k8` | 1024 | 0.10 | 8 | 0.9375 | 0.9500 | 0.9625 | 1.0000 | 1 | 1 |
| `c1024_o15_k3` | 1024 | 0.15 | 3 | 0.9375 | 0.9000 | 0.9458 | 1.0000 | 2 | 1 |
| `c1024_o15_k5` | 1024 | 0.15 | 5 | 0.9375 | 0.9500 | 0.9625 | 1.0000 | - | - |
| `c1024_o15_k8` | 1024 | 0.15 | 8 | 0.9375 | 0.9500 | 0.9625 | 1.0000 | - | - |

### Phase B Findings

The original 20-question set was useful for filtering out weak settings, but it was not hard enough to choose a single winner. Ten configurations achieved a perfect reviewed combined score.

Patterns observed:

- 1024-token chunks were less reliable on this subset. Every 1024-token run had recall below 1.0, and the strongest 1024-token runs still had lower combined scores than the best 512- and 768-token runs.
- 512-token chunks were strong and cheap, especially with 0.10 or 0.15 overlap. However, they were more fragmented, so a very low `top_k` could still be brittle on harder questions.
- 768-token chunks gave the best balance between semantic coverage and context size. Several 768-token configs scored perfectly, and `c768_o10_k5` did so without needing `top_k=8` or 0.15 overlap.
- Increasing `top_k` helped some retrieval cases but could also add distracting context. For example, `c768_o15_k8` had full recall but one answer issue.

The strongest Phase B candidates were all perfect on the original set, so a harder add-on evaluation was needed before final selection.

## Hard Add-on Check

The hard add-on set was run only on the ten Phase B configurations that scored perfectly on the original 20-question set. This avoided extra evaluation on already-weaker configurations.

| Config | Recall | Answer Pass | Combined | IDK | Retrieval Issues | Answer Issues |
|---|---:|---:|---:|---:|---:|---:|
| `c768_o05_k8` | 0.9375 | 0.9500 | 0.9625 | 1.0000 | 2 | 1 |
| `c768_o10_k5` | 0.9375 | 0.9500 | 0.9625 | 1.0000 | 1 | 1 |
| `c768_o15_k5` | 0.9375 | 0.9500 | 0.9625 | 1.0000 | 1 | 1 |
| `c512_o05_k3` | 0.9375 | 0.9000 | 0.9458 | 1.0000 | 3 | 2 |
| `c512_o10_k8` | 0.9375 | 0.9000 | 0.9458 | 1.0000 | 2 | 2 |
| `c512_o15_k3` | 0.9375 | 0.9000 | 0.9458 | 1.0000 | 2 | 2 |
| `c512_o15_k5` | 0.9375 | 0.9000 | 0.9458 | 1.0000 | - | - |
| `c512_o15_k8` | 0.9375 | 0.9000 | 0.9458 | 1.0000 | 2 | 2 |
| `c768_o10_k8` | 0.9375 | 0.9000 | 0.9458 | 1.0000 | 2 | 1 |
| `c512_o10_k3` | 0.9375 | 0.8500 | 0.9292 | 1.0000 | - | - |

The hard add-on changed the decision. The cheaper 512-token candidates that looked perfect in Phase B fell behind. The top tier became:

- `c768_o05_k8`
- `c768_o10_k5`
- `c768_o15_k5`

All three tied on answer pass and combined score. `c768_o10_k5` was selected because it used less context than `c768_o05_k8` and less overlap than `c768_o15_k5`, while keeping the same hard-add-on score.

## Phase C: Full-Corpus Checks

After deployment, the 100-row golden labels were no longer treated as complete ground truth because the production namespace contains the full Medium corpus. The hard add-on questions were therefore reused as a stress test, but the review focused on whether the responses were grounded in retrieved full-corpus context and whether increasing context helped or harmed answer quality.

This phase compared the selected production setting, `c768_o10_k5`, with the same production namespace and same ingest configuration but `top_k=8`.

The `top_k=8` run used the full production namespace (`prod`, 18,456 vectors). Four questions in the first pass hit transient provider authentication errors and were retried separately. The comparison below uses the completed results from the original pass plus the retry.

| Question | `top_k=5` Review | `top_k=8` Review | Effect |
|---|---|---|---|
| `hard_type1_001` energy/time management | Pass | Pass | No harm |
| `hard_type1_002` loss aversion/FOMO | Pass | Pass | No harm |
| `hard_type1_003` voltage imaging | Pass | Pass | No harm |
| `hard_type1_004` querying data sources | Pass | Pass | No harm |
| `hard_type1_005` blockchain artists | Plausible full-corpus answer | Plausible full-corpus answer | No clear change |
| `hard_type2_001` outside writing channels | Pass | Pass | No harm |
| `hard_type2_002` neuroscience methods | Pass | Pass | Slight retrieval improvement |
| `hard_type2_003` big tech/antitrust | Pass | Pass | No harm |
| `hard_type2_004` data/process systems | Pass | Pass | No harm |
| `hard_type2_005` elementary math lesson plans | Pass IDK | Pass IDK | No harm |
| `hard_type3_001` Occam's dice | Pass | Pass | No harm |
| `hard_type3_002` trust/origin story | Retrieval miss | Retrieval miss | Not fixed |
| `hard_type3_003` curiosity habit loop | Pass | Pass | No harm |
| `hard_type3_004` daily writing streak/health | Partial / near miss | Better answer | Improved |
| `hard_type3_005` anorexia/probiotics | Pass IDK | Pass IDK | No harm |
| `hard_type4_001` day job/art | Pass | Pass | No harm |
| `hard_type4_002` worldbuilding exercise | Pass | Pass | No harm |
| `hard_type4_003` old notebooks/drafts | Pass | Pass | No harm |
| `hard_type4_004` Fitzgerald dialogue | Pass | Pass | No harm |
| `hard_type4_005` home net-zero equipment | Pass IDK | Wrong recommendation with caveat | Harmed |

### Phase C Findings

Increasing `top_k` from 5 to 8 improved one partial case: the daily-writing-streak question received a more relevant full-corpus article. However, it did not fix the clearest retrieval miss, the origin-story trust question, because the intended article still was not retrieved. It also harmed one previously correct unsupported-question case: the home net-zero equipment question should have remained IDK, but the larger context window gave the assistant enough adjacent renewable-energy material to recommend an article while admitting it lacked the requested equipment-buying and sizing guidance.

The result supports keeping `top_k=5` for the final configuration. More retrieved context can help some borderline questions, but it also increases distractor pressure and does not solve deeper semantic retrieval failures.

### Reasoning-Effort Probe

A small follow-up probe tested whether increased answer-time reasoning could recover the benefits of `top_k=8` while reducing distractor-driven recommendations. The probe used four representative hard-add-on questions:

- `hard_type1_002`: a clean control that was already answered correctly.
- `hard_type3_002`: the clear trust/origin-story retrieval miss.
- `hard_type3_004`: the daily-writing-streak partial case that improved under `top_k=8`.
- `hard_type4_005`: the home net-zero equipment question where `top_k=8` harmed IDK behavior.

The test used the production namespace with `top_k=8` and `reasoning_effort=medium`.

| Question | Medium-Reasoning Result | Effect |
|---|---|---|
| `hard_type1_002` loss aversion/FOMO | Correct answer preserved | No harm |
| `hard_type3_002` trust/origin story | Still IDK because the intended article was not retrieved | Not fixed |
| `hard_type3_004` daily writing streak/health | Answered with a related habit-streak article, but less specifically than the earlier `top_k=8` low-reasoning answer | Worse |
| `hard_type4_005` home net-zero equipment | Recommended an adjacent renewable-energy article while caveating that equipment lists and sizing calculations were missing | Still harmed |

A final one-question probe tested `top_k=8` with `reasoning_effort=high` on `hard_type4_005`, the home net-zero equipment question. This also failed to recover the correct IDK behavior: the assistant still recommended the adjacent renewable-energy article `Why Integrate Diversity in the Energy Sector with AI and more?`.

These probes did not justify increasing `reasoning_effort`. They did not fix retrieval misses, and they did not reliably improve strict unsupported-question behavior. The final configuration therefore keeps `reasoning_effort=low`.

## Final Selection

Final selected configuration:

| Field | Value |
|---|---:|
| `embed_content` | `chunk_only` |
| `chunk_size` | `768` |
| `overlap_ratio` | `0.10` |
| `top_k` | `5` |
| `retrieval_fetch_k` | `30` |

This setting was then used for the full production ingest into Pinecone namespace `prod`.

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

The deployed Vercel API was also checked end to end on the hard add-on question set. `GET /api/stats` returned the selected production config, and all 20 `POST /api/prompt` calls returned HTTP 200.

## Deployment Notes

The public API exposes:

- `POST /api/prompt`
- `GET /api/stats`

Required deployment environment variables:

- `LLMOD_API_KEY`
- `LLMOD_BASE_URL`
- `PINECONE_API_KEY`

Optional deployment environment variables:

- `PINECONE_INDEX=medium-rag`
- `PINECONE_NAMESPACE=prod`

The Pinecone production namespace must remain active until grading is complete.
