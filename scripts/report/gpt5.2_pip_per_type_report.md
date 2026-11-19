# Per-type evaluation — `gpt5.2_pip`

*Split: `test` — overlap matching — coref unmatched policy: `singleton`*

Coref documents: 24 | URI-linking documents: 24 (`gpt5.2_pip`)

## Coreference (OVERLAP, micro)

| type | docs | gold mentions | sys mentions | matched | MUC F1 % | B3 F1 % | CEAF_e F1 % | LEA F1 % | CoNLL F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Decision | 24 | 2169 | 2396 | 1764 | 66.28 | 78.07 | 61.85 | 55.93 | 68.74 |
| Legislation | 24 | 1647 | 1759 | 1428 | 81.63 | 77.85 | 32.21 | 62.01 | 63.89 |
| Secondary | 24 | 280 | 573 | 221 | 35.57 | 66.55 | 42.62 | 32.65 | 48.25 |

LEA detail:

| type | LEA P % | LEA R % | LEA F1 % |
| --- | --- | --- | --- |
| Decision | 55.99 | 55.88 | 55.93 |
| Legislation | 60.18 | 63.95 | 62.01 |
| Secondary | 33.74 | 31.64 | 32.65 |

## URI linking (OVERLAP, micro)

| type | TP | FN | FP | Precision % | Recall % | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| Decision | 1280 | 886 | 1114 | 53.47 | 59.10 | 56.14 |
| Legislation | 978 | 669 | 780 | 55.63 | 59.38 | 57.44 |
| Secondary | 29 | 250 | 543 | 5.07 | 10.39 | 6.82 |

Overall URI linking F1: 51.88 (P 48.41, R 55.89)
