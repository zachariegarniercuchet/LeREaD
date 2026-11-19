# Reflex (CanLII citator) — coreference resolution evaluation

*Generated 2026-11-17 20:24 — `python scripts/reflex/evaluate_reflex_coref_resolution.py`*

**Scope**: coreference resolution only. Reflex anchors sharing the same ``concatenated-id`` form one cluster; gold mentions sharing the same ``docid`` form one cluster.  Spans are aligned into gold coordinates (token-diff Alignment) before clustering.  Only the grouping by docid is scored — the docid strings themselves are never compared.

**Candidate pool**: `gold`
**Merge gap**: `15` normalised characters
**Unmatched policy**: `singleton` (end-to-end / twinless — comparable with pipeline systems)

**Documents**: 100 reflex export(s) paired with gold.

## EXACT — EXACT span match (label + identical offsets)

| Metric | Precision % | Recall % | F1 % |
| --- | --- | --- | --- |
| MUC | 0.00 | 0.00 | 0.00 |
| B3 | 90.98 | 44.38 | 57.50 |
| CEAF_e | 32.29 | 65.32 | 41.74 |
| LEA | 25.00 | 25.00 | 25.00 |
| **CoNLL (MUC/B3/CEAF_e)** | - | - | 33.08 |

Alignment stats — gold mentions: 17119 | system mentions: 6346 | matched: 3 | gold-only: 17116 | system-only: 6343

## OVERLAP — OVERLAP span match (label + intersecting offsets)

| Metric | Precision % | Recall % | F1 % |
| --- | --- | --- | --- |
| MUC | 3.69 | 1.06 | 1.58 |
| B3 | 90.70 | 42.26 | 55.46 |
| CEAF_e | 30.57 | 64.39 | 39.95 |
| LEA | 23.31 | 23.23 | 23.27 |
| **CoNLL (MUC/B3/CEAF_e)** | - | - | 32.33 |

Alignment stats — gold mentions: 17119 | system mentions: 6346 | matched: 1379 | gold-only: 15740 | system-only: 4967

## Results per split

### EXACT

| split | docs | gold mentions | sys mentions | matched | MUC F1 % | B3 F1 % | CEAF_e F1 % | LEA F1 % | CoNLL F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 38 | 7625 | 2919 | 1 | 0.00 | 58.08 | 42.16 | 24.98 | 33.41 |
| dev | 3 | 298 | 57 | 0 | 0.00 | 55.17 | 41.10 | 22.08 | 32.09 |
| train | 4 | 2694 | 1096 | 2 | 0.00 | 54.28 | 30.26 | 15.03 | 28.18 |
| silver | 55 | 6502 | 2275 | 0 | 0.00 | 57.46 | 42.31 | 25.89 | 33.26 |

### OVERLAP

| split | docs | gold mentions | sys mentions | matched | MUC F1 % | B3 F1 % | CEAF_e F1 % | LEA F1 % | CoNLL F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 38 | 7625 | 2919 | 1 | 1.58 | 55.64 | 39.95 | 22.71 | 32.39 |
| dev | 3 | 298 | 57 | 0 | 0.00 | 53.01 | 38.50 | 19.45 | 30.50 |
| train | 4 | 2694 | 1096 | 2 | 7.75 | 47.69 | 24.24 | 10.41 | 26.56 |
| silver | 55 | 6502 | 2275 | 0 | 1.21 | 56.04 | 41.17 | 24.79 | 32.81 |

## Per mention type

Scores are macro-averaged across documents, separately for decisions and legislation. `Secondary` is excluded: Reflex never links secondary sources, so that row would be 0/0 by construction (it is still present in the JSON).

### EXACT

| type | docs | gold mentions | sys mentions | matched | MUC F1 % | B3 F1 % | CEAF_e F1 % | LEA F1 % | CoNLL F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Decision | 100 | 7794 | 3118 | 0 | 0.00 | 74.94 | 67.04 | 49.86 | 47.33 |
| Legislation | 100 | 8282 | 3229 | 3 | 0.00 | 41.07 | 14.99 | 5.62 | 18.69 |

### OVERLAP

| type | docs | gold mentions | sys mentions | matched | MUC F1 % | B3 F1 % | CEAF_e F1 % | LEA F1 % | CoNLL F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Decision | 100 | 7794 | 3118 | 909 | 0.00 | 72.85 | 64.26 | 46.69 | 45.70 |
| Legislation | 100 | 8282 | 3229 | 470 | 2.37 | 39.84 | 15.17 | 6.01 | 19.13 |

### Per split and mention type

OVERLAP span match (the meaningful mode for a citator):

| split | type | gold mentions | sys mentions | matched | MUC F1 % | B3 F1 % | CEAF_e F1 % | LEA F1 % | CoNLL F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | Decision | 3719 | 1458 | 416 | 0.00 | 71.86 | 63.01 | 44.49 | 44.96 |
| test | Legislation | 3482 | 1461 | 217 | 2.45 | 40.33 | 15.56 | 6.20 | 19.44 |
| dev | Decision | 78 | 36 | 15 | 0.00 | 77.50 | 67.73 | 47.42 | 48.41 |
| dev | Legislation | 182 | 21 | 1 | 0.00 | 29.99 | 11.79 | 3.69 | 13.93 |
| train | Decision | 1577 | 668 | 300 | 0.00 | 55.07 | 42.82 | 22.51 | 32.63 |
| train | Legislation | 883 | 428 | 95 | 9.68 | 43.74 | 5.69 | 1.60 | 19.70 |
| silver | Decision | 2420 | 956 | 178 | 0.00 | 74.57 | 66.48 | 49.93 | 47.02 |
| silver | Legislation | 3735 | 1319 | 157 | 1.92 | 39.76 | 15.78 | 6.34 | 19.15 |

## Per-document breakdown (exact match)

| document | split | gold | system | matched | MUC F1 % | B3 F1 % | CEAF_e F1 % | LEA F1 % | CoNLL F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1989CanLII1415ONCA | dev | 75 | 3 | 0 | 0.00 | 51.43 | 36.38 | 12.82 | 29.27 |
| 2016NBOMB12 | dev | 128 | 19 | 0 | 0.00 | 47.43 | 30.05 | 14.97 | 25.83 |
| 2021QCCA1675 | dev | 95 | 35 | 0 | 0.00 | 66.63 | 56.87 | 38.46 | 41.17 |
| 1982CanLII3042NBCA | silver | 103 | 31 | 0 | 0.00 | 61.02 | 43.99 | 23.13 | 35.01 |
| 1982CanLII883ONLRB | silver | 69 | 3 | 0 | 0.00 | 43.48 | 39.22 | 23.61 | 27.57 |
| 1983CanLII2475SKQB | silver | 476 | 142 | 0 | 0.00 | 48.87 | 26.26 | 13.11 | 25.05 |
| 1993CanLII2395BCCA | silver | 84 | 13 | 0 | 0.00 | 48.44 | 41.67 | 23.71 | 30.04 |
| 1994CanLII8687PESCAD | silver | 59 | 18 | 0 | 0.00 | 57.71 | 38.27 | 22.08 | 31.99 |
| 1998CanLII18739NLSCTD | silver | 118 | 25 | 0 | 0.00 | 61.84 | 51.96 | 32.87 | 37.93 |
| 1999CanLII18647MBCA | silver | 86 | 53 | 0 | 0.00 | 76.16 | 66.33 | 48.20 | 47.50 |
| 2002CanLII49492ONSC | silver | 163 | 52 | 0 | 0.00 | 46.74 | 20.45 | 8.84 | 22.40 |
| 2002MBCA125 | silver | 93 | 50 | 0 | 0.00 | 84.37 | 77.41 | 60.84 | 53.93 |
| 2002NFCA43 | silver | 165 | 21 | 0 | 0.00 | 40.98 | 31.93 | 16.67 | 24.30 |
| 2003ABCA256 | silver | 187 | 129 | 0 | 0.00 | 60.11 | 31.60 | 17.09 | 30.57 |
| 2003FC1056 | silver | 135 | 78 | 0 | 0.00 | 75.35 | 66.13 | 48.36 | 47.16 |
| 2003SKAIA1 | silver | 23 | 1 | 0 | 0.00 | 22.22 | 9.98 | 4.17 | 10.73 |
| 2004CanLII14579ONCA | silver | 44 | 31 | 0 | 0.00 | 75.36 | 58.61 | 36.00 | 44.66 |
| 2005NBCA70 | silver | 99 | 39 | 0 | 0.00 | 74.55 | 65.23 | 47.83 | 46.59 |
| 2006NBCA75 | silver | 8 | 1 | 0 | 0.00 | 61.54 | 46.15 | 22.22 | 35.90 |
| 2006NUCJ23 | silver | 54 | 29 | 0 | 0.00 | 71.00 | 51.50 | 31.33 | 40.83 |
| 2007NUCA2 | silver | 50 | 11 | 0 | 0.00 | 67.01 | 56.27 | 34.43 | 41.09 |
| 2008CanLII89121MBLB | silver | 37 | 1 | 0 | 0.00 | 23.26 | 14.56 | 5.26 | 12.61 |
| 2008NLCA4 | silver | 60 | 26 | 0 | 0.00 | 78.97 | 67.17 | 43.02 | 48.71 |
| 2009CanLII72366SKLRB | silver | 30 | 18 | 0 | 0.00 | 70.77 | 49.86 | 29.17 | 40.21 |
| 2009KQB465 | silver | 8 | 7 | 0 | 0.00 | 93.33 | 88.10 | 73.33 | 60.48 |
| 2009NBQB60 | silver | 77 | 25 | 0 | 0.00 | 76.08 | 66.64 | 45.10 | 47.57 |
| 2010CanLII27153NBWCAT | silver | 23 | 1 | 0 | 0.00 | 45.16 | 33.12 | 16.67 | 26.09 |
| 2011CanLII 91839PEIPC | silver | 262 | 92 | 0 | 0.00 | 43.34 | 6.89 | 1.98 | 16.74 |
| 2011FC18 | silver | 29 | 11 | 0 | 0.00 | 70.33 | 63.21 | 45.00 | 44.51 |
| 2011NBCA76 | silver | 170 | 64 | 0 | 0.00 | 63.66 | 47.74 | 29.06 | 37.13 |
| 2011TMOB65 | silver | 46 | 21 | 0 | 0.00 | 67.99 | 58.54 | 40.30 | 42.17 |
| 2012PECA3 | silver | 88 | 20 | 0 | 0.00 | 49.52 | 35.98 | 18.52 | 28.50 |
| 2012QCCA1663 | silver | 69 | 10 | 0 | 0.00 | 66.10 | 58.72 | 39.24 | 41.61 |
| 2012SKQB62 | silver | 800 | 388 | 0 | 0.00 | 51.42 | 16.87 | 7.74 | 22.76 |
| 2014FC638 | silver | 317 | 52 | 0 | 0.00 | 48.61 | 35.86 | 19.78 | 28.16 |
| 2015NLCA24 | silver | 46 | 11 | 0 | 0.00 | 67.04 | 55.13 | 33.33 | 40.72 |
| 2016NSCA89 | silver | 118 | 31 | 0 | 0.00 | 61.91 | 55.11 | 34.23 | 39.01 |
| 2016PESC4 | silver | 101 | 55 | 0 | 0.00 | 83.13 | 77.30 | 60.26 | 53.48 |
| 2016QCCA76 | silver | 119 | 66 | 0 | 0.00 | 70.30 | 50.81 | 29.19 | 40.37 |
| 2018BCHPRB52 | silver | 25 | 21 | 0 | 0.00 | 61.78 | 27.67 | 15.22 | 29.82 |
| 2018FCA69 | silver | 227 | 107 | 0 | 0.00 | 62.67 | 45.45 | 23.05 | 36.04 |
| 2018MBQB64 | silver | 73 | 23 | 0 | 0.00 | 72.00 | 64.47 | 44.79 | 45.49 |
| 2018NBOMB6 | silver | 105 | 63 | 0 | 0.00 | 53.23 | 15.93 | 6.55 | 23.05 |
| 2018SKCA43 | silver | 661 | 199 | 0 | 0.00 | 50.89 | 33.07 | 18.32 | 27.99 |
| 2019CHRT30 | silver | 44 | 12 | 0 | 0.00 | 61.38 | 45.25 | 23.21 | 35.54 |
| 2020ABSRB85 | silver | 24 | 10 | 0 | 0.00 | 58.16 | 32.68 | 17.65 | 30.28 |
| 2020CanLII55888MBLB | silver | 16 | 0 | 0 | 0.00 | 22.22 | 12.50 | 6.25 | 11.57 |
| 2020ONAFRAAT8 | silver | 232 | 38 | 0 | 0.00 | 34.49 | 9.70 | 3.70 | 14.73 |
| 2021NBOMB1 | silver | 76 | 15 | 0 | 0.00 | 47.06 | 41.35 | 24.18 | 29.47 |
| 2021NUIPC10 | silver | 74 | 9 | 0 | 0.00 | 41.90 | 38.79 | 22.89 | 26.90 |
| 2021NUIPC8 | silver | 49 | 0 | 0 | 0.00 | 33.90 | 28.42 | 14.29 | 20.77 |
| 2022QCCS1670 | silver | 16 | 18 | 0 | 0.00 | 69.09 | 45.45 | 26.47 | 38.18 |
| 2023ABCA35 | silver | 44 | 11 | 0 | 0.00 | 52.24 | 32.76 | 16.36 | 28.33 |
| 2023FCA24 | silver | 253 | 84 | 0 | 0.00 | 55.34 | 34.76 | 18.69 | 30.03 |
| 2024ONCMT19 | silver | 51 | 28 | 0 | 0.00 | 56.23 | 24.41 | 11.39 | 26.88 |
| 2025MBKB113 | silver | 50 | 11 | 0 | 0.00 | 57.86 | 45.84 | 24.59 | 34.57 |
| 2025NBEUB2 | silver | 21 | 0 | 0 | 0.00 | 32.00 | 22.22 | 9.52 | 18.07 |
| 2026ALRB5 | silver | 45 | 0 | 0 | 0.00 | 30.19 | 21.70 | 11.11 | 17.30 |
| 1983CanLII2967L | test | 50 | 11 | 0 | 0.00 | 74.23 | 67.91 | 49.18 | 47.38 |
| 1993CanLII1889PESCTSD | test | 138 | 30 | 0 | 0.00 | 60.00 | 45.60 | 21.43 | 35.20 |
| 1993CanLII3004FCA | test | 549 | 233 | 0 | 0.00 | 59.22 | 38.51 | 18.03 | 32.58 |
| 1993NSSC71 | test | 70 | 19 | 0 | 0.00 | 77.03 | 64.85 | 40.45 | 47.29 |
| 1997CanLII11497 | test | 99 | 6 | 0 | 0.00 | 18.91 | 10.77 | 4.76 | 9.90 |
| 1999CanLII7320(ONCA) | test | 264 | 110 | 0 | 0.00 | 68.91 | 54.80 | 28.34 | 41.24 |
| 2001BCSC1342 | test | 95 | 14 | 0 | 0.00 | 60.62 | 46.81 | 24.77 | 35.81 |
| 2001CanLII21117 | test | 296 | 150 | 0 | 0.00 | 55.05 | 21.62 | 9.87 | 25.56 |
| 2002SCC33 | test | 315 | 144 | 0 | 0.00 | 62.35 | 48.75 | 28.32 | 37.03 |
| 2003MBCA71 | test | 47 | 20 | 0 | 0.00 | 76.62 | 67.85 | 49.25 | 48.16 |
| 2005NLTD57 | test | 79 | 16 | 0 | 0.00 | 50.26 | 35.69 | 17.89 | 28.65 |
| 2005QCCA437 | test | 96 | 24 | 0 | 0.00 | 54.43 | 36.52 | 20.00 | 30.32 |
| 2009BCHRT256 | test | 157 | 42 | 0 | 0.00 | 60.26 | 55.17 | 35.68 | 38.48 |
| 2009NUCA3 | test | 52 | 12 | 0 | 0.00 | 69.01 | 63.37 | 43.75 | 44.13 |
| 2009PECA12 | test | 86 | 50 | 0 | 0.00 | 62.98 | 46.21 | 27.21 | 36.40 |
| 2010SKQB255 | test | 97 | 62 | 0 | 0.00 | 66.54 | 44.83 | 26.42 | 37.12 |
| 2012SKCA24 | test | 44 | 19 | 0 | 0.00 | 64.22 | 48.89 | 30.16 | 37.70 |
| 2013ONAFRAAT18 | test | 39 | 2 | 0 | 0.00 | 45.28 | 31.33 | 14.63 | 25.54 |
| 2014CanLII36807PELRB | test | 238 | 85 | 0 | 0.00 | 50.15 | 14.89 | 4.33 | 21.68 |
| 2015NSSC25 | test | 226 | 110 | 0 | 0.00 | 59.89 | 37.35 | 20.54 | 32.41 |
| 2016QCCS1184 | test | 66 | 18 | 0 | 0.00 | 72.41 | 63.96 | 42.86 | 45.46 |
| 2017ABCA157 | test | 297 | 82 | 0 | 0.00 | 63.26 | 57.44 | 38.26 | 40.23 |
| 2018NSOIPC2 | test | 263 | 114 | 0 | 0.00 | 56.58 | 43.04 | 24.14 | 33.21 |
| 2019QCCA1492 | test | 1404 | 695 | 0 | 0.00 | 59.70 | 40.16 | 23.15 | 33.29 |
| 2020ABPLAB11 | test | 38 | 44 | 0 | 0.00 | 60.31 | 23.39 | 9.76 | 27.90 |
| 2020BCHPRB101 | test | 147 | 71 | 1 | 0.00 | 55.69 | 27.61 | 14.29 | 27.76 |
| 2020FCA63 | test | 162 | 108 | 0 | 0.00 | 65.21 | 45.05 | 23.33 | 36.75 |
| 2022BCSC135 | test | 1134 | 312 | 0 | 0.00 | 42.79 | 16.76 | 8.09 | 19.85 |
| 2022ONCMT32 | test | 129 | 1 | 0 | 0.00 | 39.51 | 28.95 | 11.54 | 22.82 |
| 2024ABOHSAB19 | test | 52 | 32 | 0 | 0.00 | 57.14 | 22.77 | 10.71 | 26.64 |
| 2024BCSC764 | test | 231 | 101 | 0 | 0.00 | 57.45 | 33.30 | 17.77 | 30.25 |
| 2024NBKB203 | test | 239 | 70 | 0 | 0.00 | 58.90 | 51.45 | 32.69 | 36.78 |
| 2024NUCA9 | test | 113 | 49 | 0 | 0.00 | 63.68 | 48.33 | 30.86 | 37.34 |
| 2025ABAER7 | test | 143 | 5 | 0 | 0.00 | 21.69 | 14.31 | 6.08 | 12.00 |
| 2025ABKB246 | test | 58 | 8 | 0 | 0.00 | 50.00 | 42.95 | 27.27 | 30.98 |
| 2025MBKB136 | test | 45 | 21 | 0 | 0.00 | 74.29 | 66.87 | 45.45 | 47.05 |
| 2025NBEUB1 | test | 24 | 0 | 0 | 0.00 | 28.57 | 17.89 | 8.33 | 15.49 |
| 2026ABCA2L | test | 43 | 29 | 0 | 0.00 | 83.90 | 76.32 | 59.72 | 53.41 |
| 1994CanLII4528NLCA | train | 167 | 74 | 1 | 0.00 | 47.38 | 11.67 | 5.83 | 19.69 |
| 1997CanLII16226ONCA | train | 623 | 291 | 1 | 0.00 | 54.15 | 27.01 | 12.27 | 27.05 |
| 2008CSC9 | train | 470 | 210 | 0 | 0.00 | 58.35 | 39.16 | 19.56 | 32.50 |
| 2019SCC65 | train | 1434 | 521 | 0 | 0.00 | 57.26 | 43.19 | 22.46 | 33.48 |

## Notes

* The `singleton` unmatched policy (default) counts extraction errors: mentions found by only one side become singletons on the other side. This makes Reflex's scores directly comparable with the LLM/pipeline systems scored with `evaluate_coref.py --match-mode span --unmatched singleton`.
* The `drop` policy isolates coreference from extraction errors and is an upper bound — not comparable with pipeline end-to-end scores.
* Reflex never links secondary sources, so those clusters are empty on the system side by construction.
* Reflex only exposes a ``concatenated-id`` on legislation anchors. Decision anchors carry no ``concatenated-id``, so every decision prediction is a singleton cluster: decision MUC / CEAF_e / LEA are 0 by construction, and only B3 is non-zero. Decision coreference cannot be measured from these exports as they stand.
* Coref metrics (MUC / B3 / CEAF_e / LEA) are implemented inline, equivalent to the `coval` library used by `src/evaluation/`.
