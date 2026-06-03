# Phase A and Phase B Experiments

Run dates: 2026-06-01 to 2026-06-02

This report documents the subset experiments used to choose the production RAG configuration before the full-corpus ingest.

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

| Config | Embed Content | Recall | Answer Pass | Combined | IDK | Retrieval Issues | Answer Issues | Cost |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `chunk_only_c512_o10` | `chunk_only` | 1.0000 | 0.9500 | 0.9833 | 1.0000 | 0 | 1 | $0.024121 |
| `title_tags_chunk_c512_o10` | `title_tags_chunk` | 0.9375 | 0.9000 | 0.9458 | 1.0000 | 2 | 1 | $0.022467 |

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

### Phase B Findings

The original 20-question set was useful for filtering out weak settings, but it was not hard enough to choose a single winner. Ten configurations achieved a perfect reviewed combined score.

Patterns observed:

- 1024-token chunks were less reliable on this subset. Every 1024-token run had recall below 1.0, and the strongest 1024-token runs still had lower combined scores than the best 512- and 768-token runs.
- 512-token chunks were strong and cheap, especially with 0.10 or 0.15 overlap. However, they were more fragmented, so a very low `top_k` could still be brittle on harder questions.
- 768-token chunks gave the best balance between semantic coverage and context size. Several 768-token configs scored perfectly, and `c768_o10_k5` did so without needing `top_k=8` or 0.15 overlap.
- Increasing `top_k` helped some retrieval cases but could also add distracting context. For example, `c768_o15_k8` had full recall but one answer issue.

The strongest Phase B candidates were all perfect on the original set, so a harder add-on evaluation was needed before final selection.

## Hard Add-on Check

The hard add-on set was run only on the ten Phase B configurations that scored perfectly on the original 20-question set. This avoided spending budget on already-weaker configurations.

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

The hard add-on changed the decision. The cheaper 512-token candidates that looked perfect in Phase B fell behind. The top tier became:

- `c768_o05_k8`
- `c768_o10_k5`
- `c768_o15_k5`

All three tied on answer pass and combined score. `c768_o10_k5` was selected because it used less context than `c768_o05_k8` and less overlap than `c768_o15_k5`, while keeping the same hard-add-on score.

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

The later full-corpus sanity check found one retrieval miss on the pandemic/bubonic-plague example. That limitation is documented separately in `prod_scale_sanity_20260602.md`; it was not handled with a special-case rescue layer.
