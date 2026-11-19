# Per-type evaluation — `test_baseline_coref_strong_gold`

*Split: `test` — exact matching — coref unmatched policy: `singleton`*

Coref documents: 38 | URI-linking documents: 38 (`test_gold_resolution_rule`)

## Coreference (EXACT, micro)

| type | docs | gold mentions | sys mentions | matched | MUC F1 % | B3 F1 % | CEAF_e F1 % | LEA F1 % | CoNLL F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Decision | 38 | 3902 | 3902 | 3902 | 92.08 | 84.23 | 83.79 | 78.35 | 86.70 |
| Legislation | 38 | 3560 | 3560 | 3560 | 95.61 | 71.01 | 62.53 | 68.31 | 76.38 |
| Secondary | 38 | 456 | 456 | 456 | 84.17 | 83.23 | 78.32 | 72.97 | 81.91 |

LEA detail:

| type | LEA P % | LEA R % | LEA F1 % |
| --- | --- | --- | --- |
| Decision | 75.12 | 81.87 | 78.35 |
| Legislation | 55.97 | 87.65 | 68.31 |
| Secondary | 69.34 | 76.99 | 72.97 |

## URI linking (EXACT, micro)

| type | TP | FN | FP | Precision % | Recall % | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| Decision | 3341 | 557 | 253 | 92.96 | 85.71 | 89.19 |
| Legislation | 2600 | 956 | 843 | 75.52 | 73.12 | 74.30 |
| Secondary | 387 | 68 | 21 | 94.85 | 85.05 | 89.69 |

Overall URI linking F1: 82.43 (P 85.00, R 80.01)
