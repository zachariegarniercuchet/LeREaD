# Per-type evaluation — `test_baseline_coref_strong_from-test_baseline_rules-final`

*Split: `test` — overlap matching — coref unmatched policy: `singleton`*

Coref documents: 38 | URI-linking documents: 38 (`test_baseline_coref_strong_from-test_baseline_rules-final`)

## Coreference (OVERLAP, micro)

| type | docs | gold mentions | sys mentions | matched | MUC F1 % | B3 F1 % | CEAF_e F1 % | LEA F1 % | CoNLL F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Decision | 38 | 3902 | 3726 | 2549 | 50.66 | 66.66 | 46.47 | 36.30 | 54.60 |
| Legislation | 38 | 3560 | 2795 | 2281 | 64.91 | 57.32 | 17.22 | 30.78 | 46.48 |
| Secondary | 38 | 456 | 938 | 164 | 13.33 | 54.07 | 29.32 | 14.42 | 32.24 |

LEA detail:

| type | LEA P % | LEA R % | LEA F1 % |
| --- | --- | --- | --- |
| Decision | 35.25 | 37.41 | 36.30 |
| Legislation | 26.07 | 37.57 | 30.78 |
| Secondary | 12.78 | 16.53 | 14.42 |

## URI linking (OVERLAP, micro)

| type | TP | FN | FP | Precision % | Recall % | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| Decision | 1389 | 2509 | 1431 | 49.26 | 35.63 | 41.35 |
| Legislation | 379 | 3177 | 2024 | 15.77 | 10.66 | 12.72 |
| Secondary | 64 | 391 | 579 | 9.95 | 14.07 | 11.66 |

Overall URI linking F1: 26.60 (P 31.23, R 23.16)
