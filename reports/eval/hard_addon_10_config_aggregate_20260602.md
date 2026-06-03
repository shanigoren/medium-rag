# Hard Add-on 10-Config Aggregate

Separate aggregate for `tests/golden/subset100_hard_addon.json`; original Phase B artifacts are unchanged.

## Cost

- Eval ledger entries: 10
- LLMod key/user spend delta sum: `$0.22825414`
- Filtered spend-log sum: `$0.23299722`
- Remaining budget after last hard-add-on eval: `$3.86430475`

## Reviewed Leaderboard

| Config | Recall | Answer Pass | Combined | IDK | Retrieval Issues | Answer Issues | Failures |
|---|---:|---:|---:|---:|---:|---:|---|
| `c768_o05_k8` | 0.9375 | 0.9500 | 0.9625 | 1.0000 | 2 | 1 | hard_type2_004 |
| `c768_o10_k5` | 0.9375 | 0.9500 | 0.9625 | 1.0000 | 1 | 1 | hard_type2_004 |
| `c768_o15_k5` | 0.9375 | 0.9500 | 0.9625 | 1.0000 | 1 | 1 | hard_type2_004 |
| `c512_o05_k3` | 0.9375 | 0.9000 | 0.9458 | 1.0000 | 3 | 2 | hard_type2_001, hard_type2_004 |
| `c512_o10_k8` | 0.9375 | 0.9000 | 0.9458 | 1.0000 | 2 | 2 | hard_type1_001, hard_type2_004 |
| `c512_o15_k3` | 0.9375 | 0.9000 | 0.9458 | 1.0000 | 2 | 2 | hard_type2_001, hard_type2_004 |
| `c512_o15_k5` | 0.9375 | 0.9000 | 0.9458 | 1.0000 | 2 | 2 | hard_type2_001, hard_type2_004 |
| `c512_o15_k8` | 0.9375 | 0.9000 | 0.9458 | 1.0000 | 2 | 2 | hard_type1_001, hard_type2_004 |
| `c768_o10_k8` | 0.9375 | 0.9000 | 0.9458 | 1.0000 | 2 | 1 | hard_type1_001, hard_type2_004 |
| `c512_o10_k3` | 0.9375 | 0.8500 | 0.9292 | 1.0000 | 3 | 3 | hard_type1_001, hard_type2_001, hard_type2_004 |

## Notes

- All 10 tied Phase B configs were evaluated against the hard add-on only; no re-embedding was performed.
- All hard-add-on runs had raw `recall_at_k = 0.9375`, `idk_pass_rate = 1.0`, and `dedup_accuracy = 1.0`.
- The strongest reviewed hard-add-on configs were `c768_o05_k8`, `c768_o10_k5`, and `c768_o15_k5`, each with one answer failure on `hard_type2_004`.
- The previously cheapest Phase B winner candidate, `c512_o05_k3`, had two answer failures on the hard add-on (`hard_type2_001`, `hard_type2_004`).
