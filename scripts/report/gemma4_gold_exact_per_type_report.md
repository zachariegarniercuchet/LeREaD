# Per-type evaluation — `test_gemma4_coref_fs6_random_gold`

*Split: `test` — exact matching — coref unmatched policy: `singleton`*

Coref documents: 38 | URI-linking documents: 38 (`test_gemma4_coref_fs6_random_gold`)

## Coreference (EXACT, micro)

| type | docs | gold mentions | sys mentions | matched | MUC F1 % | B3 F1 % | CEAF_e F1 % | LEA F1 % | CoNLL F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Decision | 38 | 3902 | 3899 | 3899 | 94.26 | 90.64 | 89.44 | 87.14 | 91.45 |
| Legislation | 38 | 3560 | 3559 | 3559 | 97.12 | 92.58 | 81.31 | 90.91 | 90.34 |
| Secondary | 38 | 456 | 456 | 456 | 89.60 | 90.65 | 83.68 | 81.85 | 87.98 |

LEA detail:

| type | LEA P % | LEA R % | LEA F1 % |
| --- | --- | --- | --- |
| Decision | 85.78 | 88.55 | 87.14 |
| Legislation | 88.85 | 93.07 | 90.91 |
| Secondary | 85.20 | 78.76 | 81.85 |

## URI linking (EXACT, micro)

| type | TP | FN | FP | Precision % | Recall % | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| Decision | 3035 | 863 | 861 | 77.90 | 77.86 | 77.88 |
| Legislation | 2988 | 568 | 567 | 84.05 | 84.03 | 84.04 |
| Secondary | 343 | 112 | 112 | 75.38 | 75.38 | 75.38 |

Overall URI linking F1: 80.51 (P 80.52, R 80.49)
