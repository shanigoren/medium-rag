# Final Retrieval Config Decision

Decision date: 2026-06-02

## Selected Config

| Field | Value |
|---|---|
| `embed_content` | `chunk_only` |
| `chunk_size` | `768` |
| `overlap_ratio` | `0.10` |
| `top_k` | `5` |
| `retrieval_fetch_k` | `30` |
| `reasoning_effort` | `low` |
| Namespace used in 100-row experiments | `exp_c768_o10_chunk_only` |

This is the `c768_o10_k5` configuration.

## Evidence

Original curated set:

- Test file: `tests/golden/subset100.json`
- Reviewed recall: `1.0000`
- Reviewed answer pass rate: `1.0000`
- Reviewed combined score: `1.0000`
- IDK pass rate: `1.0000`

Hard add-on set:

- Test file: `tests/golden/subset100_hard_addon.json`
- Reviewed recall: `0.9375`
- Reviewed answer pass rate: `0.9500`
- Reviewed combined score: `0.9625`
- IDK pass rate: `1.0000`
- Failed question: `hard_type2_004`

The hard add-on was introduced because the original 20-question set was too coarse for final selection: 10 of 27 Phase B configs scored perfectly. The add-on was run only on those 10 tied configs, using existing namespaces only.

## Why This Config

The hard add-on top tier was:

| Config | Hard Add-on Answer Pass | Hard Add-on Combined | Notes |
|---|---:|---:|---|
| `c768_o05_k8` | `0.9500` | `0.9625` | Same failure count, more context than selected config |
| `c768_o10_k5` | `0.9500` | `0.9625` | Selected: tied best score, fewer retrieval issues than `c768_o05_k8`, less context than `k=8` |
| `c768_o15_k5` | `0.9500` | `0.9625` | Same score, higher overlap/index size than selected config |

`c512_o05_k3`, the earlier cheapest candidate, fell to `0.9000` answer pass and `0.9458` combined on the hard add-on, so it is no longer the best final choice.

## Cost

Hard add-on 10-config evals:

- LLMod key/user spend delta sum: `$0.22825414`
- Filtered spend-log sum: `$0.23299722`
- Remaining budget after last hard-add-on eval: `$3.86430475`

No extra combined final eval was run. This decision aggregates already-saved original Phase B and hard-add-on artifacts.

## Follow-Up Status

This config was used for the full-corpus `prod` ingest. The production scale sanity check is documented in `reports/eval/prod_scale_sanity_20260602.md`.

That sanity check found one full-corpus retrieval miss on the assignment's pandemic/bubonic-plague example. The miss is documented as a limitation rather than hidden behind an out-of-scope rescue layer. The hand-in version should proceed with this clean vector-RAG configuration, the documented limitation, and Vercel deployment.
