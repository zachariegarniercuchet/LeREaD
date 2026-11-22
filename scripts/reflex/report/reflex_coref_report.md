# Reflex (CanLII citator) — coreference resolution evaluation

*Generated 2026-11-20 16:21 — `python scripts/reflex/evaluate_reflex_coref_resolution.py`*

**Scope**: coreference resolution only. Reflex anchors sharing the same ``concatenated-id`` form one cluster; gold mentions sharing the same ``docid`` form one cluster.  Spans are aligned into gold coordinates (token-diff Alignment) before clustering.  Only the grouping by docid is scored — the docid strings themselves are never compared.

**Candidate pool**: `gold`
**Merge gap**: `15` normalised characters
**Unmatched policy**: `singleton` (end-to-end / twinless — comparable with pipeline systems)

**Documents**: 100 reflex export(s) paired with gold.

## EXACT — EXACT span match (label + identical offsets)

| Metric | Precision % | Recall % | F1 % |
| --- | --- | --- | --- |
| MUC | 48.39 | 12.05 | 17.10 |
| B3 | 95.27 | 42.88 | 57.12 |
| CEAF_e | 34.37 | 75.82 | 45.63 |
| LEA | 31.79 | 29.73 | 30.65 |
| **CoNLL (MUC/B3/CEAF_e)** | - | - | 39.95 |

Alignment stats — gold mentions: 17759 | system mentions: 6346 | matched: 2363 | gold-only: 15396 | system-only: 3983

## OVERLAP — OVERLAP span match (label + intersecting offsets)

| Metric | Precision % | Recall % | F1 % |
| --- | --- | --- | --- |
| MUC | 68.50 | 16.68 | 23.77 |
| B3 | 98.45 | 36.03 | 50.60 |
| CEAF_e | 25.79 | 73.13 | 36.31 |
| LEA | 28.15 | 23.88 | 25.63 |
| **CoNLL (MUC/B3/CEAF_e)** | - | - | 36.89 |

Alignment stats — gold mentions: 17759 | system mentions: 6346 | matched: 6135 | gold-only: 11624 | system-only: 212

## Results per split

### EXACT

| split | docs | gold mentions | sys mentions | matched | MUC F1 % | B3 F1 % | CEAF_e F1 % | LEA F1 % | CoNLL F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 38 | 7917 | 2919 | 1040 | 16.11 | 57.72 | 45.61 | 30.17 | 39.81 |
| dev | 3 | 330 | 57 | 15 | 8.20 | 51.37 | 38.95 | 21.09 | 32.84 |
| train | 4 | 2717 | 1096 | 342 | 31.85 | 54.09 | 37.72 | 28.02 | 41.22 |
| silver | 55 | 6795 | 2275 | 966 | 17.19 | 57.24 | 46.58 | 31.70 | 40.34 |

### OVERLAP

| split | docs | gold mentions | sys mentions | matched | MUC F1 % | B3 F1 % | CEAF_e F1 % | LEA F1 % | CoNLL F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 38 | 7917 | 2919 | 1040 | 22.31 | 50.31 | 34.20 | 23.58 | 35.61 |
| dev | 3 | 330 | 57 | 15 | 10.42 | 44.50 | 30.41 | 14.66 | 28.45 |
| train | 4 | 2717 | 1096 | 342 | 36.89 | 41.97 | 17.19 | 17.75 | 32.02 |
| silver | 55 | 6795 | 2275 | 966 | 24.54 | 51.76 | 39.48 | 28.22 | 38.60 |

## Per mention type

Scores are macro-averaged across documents, separately for decisions and legislation. `Secondary` is excluded: Reflex never links secondary sources, so that row would be 0/0 by construction (it is still present in the JSON).

### EXACT

| type | docs | gold mentions | sys mentions | matched | MUC F1 % | B3 F1 % | CEAF_e F1 % | LEA F1 % | CoNLL F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Decision | 100 | 8098 | 3118 | 30 | 0.00 | 73.72 | 65.34 | 47.67 | 46.35 |
| Legislation | 100 | 8556 | 3229 | 2333 | 25.09 | 38.77 | 18.90 | 16.26 | 27.59 |

### OVERLAP

| type | docs | gold mentions | sys mentions | matched | MUC F1 % | B3 F1 % | CEAF_e F1 % | LEA F1 % | CoNLL F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Decision | 100 | 8098 | 3118 | 3019 | 0.00 | 63.30 | 51.42 | 32.04 | 38.24 |
| Legislation | 100 | 8556 | 3229 | 3116 | 35.78 | 40.26 | 21.35 | 24.47 | 32.46 |

### Per split and mention type

OVERLAP span match (the meaningful mode for a citator):

| split | type | gold mentions | sys mentions | matched | MUC F1 % | B3 F1 % | CEAF_e F1 % | LEA F1 % | CoNLL F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | Decision | 3901 | 1458 | 1409 | 0.00 | 60.88 | 48.37 | 28.13 | 36.42 |
| test | Legislation | 3560 | 1461 | 1404 | 38.64 | 43.12 | 23.34 | 27.43 | 35.03 |
| dev | Decision | 96 | 36 | 36 | 0.00 | 59.53 | 42.47 | 17.71 | 34.00 |
| dev | Legislation | 192 | 21 | 21 | 13.27 | 23.78 | 11.03 | 7.95 | 16.03 |
| train | Decision | 1580 | 668 | 665 | 0.00 | 40.27 | 24.10 | 6.92 | 21.46 |
| train | Legislation | 900 | 428 | 422 | 61.80 | 46.87 | 12.70 | 34.40 | 40.45 |
| silver | Decision | 2521 | 956 | 909 | 0.00 | 66.85 | 56.00 | 37.35 | 40.95 |
| silver | Legislation | 3904 | 1319 | 1269 | 33.14 | 38.70 | 21.17 | 22.61 | 31.00 |

## Per-document breakdown (exact match)

| document | split | gold | system | matched | MUC F1 % | B3 F1 % | CEAF_e F1 % | LEA F1 % | CoNLL F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1989CanLII1415ONCA | dev | 95 | 3 | 0 | 0.00 | 44.44 | 28.60 | 8.16 | 24.35 |
| 2016NBOMB12 | dev | 136 | 19 | 9 | 13.33 | 44.09 | 30.26 | 15.89 | 29.23 |
| 2021QCCA1675 | dev | 99 | 35 | 6 | 11.27 | 65.57 | 58.00 | 39.23 | 44.95 |
| 1982CanLII3042NBCA | silver | 105 | 31 | 11 | 14.12 | 58.23 | 47.50 | 28.21 | 39.95 |
| 1982CanLII883ONLRB | silver | 69 | 3 | 0 | 0.00 | 43.48 | 39.22 | 23.61 | 27.57 |
| 1983CanLII2475SKQB | silver | 480 | 142 | 86 | 33.07 | 45.80 | 34.26 | 24.11 | 37.71 |
| 1993CanLII2395BCCA | silver | 87 | 13 | 0 | 0.00 | 47.33 | 40.72 | 23.00 | 29.35 |
| 1994CanLII8687PESCAD | silver | 59 | 18 | 12 | 30.19 | 56.37 | 43.47 | 31.50 | 43.35 |
| 1998CanLII18739NLSCTD | silver | 126 | 25 | 0 | 0.00 | 59.53 | 49.87 | 31.13 | 36.47 |
| 1999CanLII18647MBCA | silver | 86 | 53 | 10 | 28.13 | 76.64 | 73.37 | 54.56 | 59.38 |
| 2002CanLII49492ONSC | silver | 169 | 52 | 28 | 27.66 | 39.10 | 25.01 | 15.45 | 30.59 |
| 2002MBCA125 | silver | 97 | 50 | 1 | 0.00 | 84.29 | 77.99 | 60.96 | 54.09 |
| 2002NFCA43 | silver | 168 | 21 | 3 | 1.42 | 40.00 | 31.86 | 17.38 | 24.42 |
| 2003ABCA256 | silver | 193 | 129 | 77 | 57.03 | 64.98 | 51.12 | 41.67 | 57.71 |
| 2003FC1056 | silver | 141 | 78 | 17 | 31.37 | 76.02 | 73.36 | 55.91 | 60.25 |
| 2003SKAIA1 | silver | 30 | 1 | 0 | 0.00 | 36.84 | 28.04 | 16.13 | 21.63 |
| 2004CanLII14579ONCA | silver | 44 | 31 | 9 | 28.57 | 76.86 | 67.42 | 46.68 | 57.62 |
| 2005NBCA70 | silver | 110 | 39 | 12 | 22.86 | 72.95 | 68.55 | 51.37 | 54.79 |
| 2006NBCA75 | silver | 10 | 1 | 0 | 0.00 | 70.59 | 58.82 | 36.36 | 43.14 |
| 2006NUCJ23 | silver | 54 | 29 | 9 | 30.43 | 74.51 | 59.44 | 41.77 | 54.79 |
| 2007NUCA2 | silver | 50 | 11 | 3 | 6.45 | 66.67 | 57.10 | 37.93 | 43.41 |
| 2008CanLII89121MBLB | silver | 38 | 1 | 0 | 0.00 | 22.73 | 14.22 | 5.13 | 12.32 |
| 2008NLCA4 | silver | 63 | 26 | 4 | 12.12 | 79.43 | 68.40 | 45.88 | 53.32 |
| 2009CanLII72366SKLRB | silver | 31 | 18 | 6 | 34.48 | 71.47 | 58.93 | 36.08 | 54.96 |
| 2009KQB465 | silver | 11 | 7 | 2 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| 2009NBQB60 | silver | 82 | 25 | 4 | 4.55 | 74.39 | 65.19 | 44.66 | 48.04 |
| 2010CanLII27153NBWCAT | silver | 23 | 1 | 0 | 0.00 | 45.16 | 33.12 | 16.67 | 26.09 |
| 2011CanLII 91839PEIPC | silver | 269 | 92 | 73 | 41.07 | 35.85 | 12.56 | 16.76 | 29.83 |
| 2011FC18 | silver | 29 | 11 | 1 | 0.00 | 69.36 | 65.35 | 46.15 | 44.90 |
| 2011NBCA76 | silver | 176 | 64 | 30 | 35.76 | 63.63 | 57.48 | 41.09 | 52.29 |
| 2011TMOB65 | silver | 48 | 21 | 5 | 21.62 | 66.95 | 65.23 | 47.25 | 51.27 |
| 2012PECA3 | silver | 88 | 20 | 3 | 5.26 | 48.33 | 37.59 | 20.17 | 30.39 |
| 2012QCCA1663 | silver | 77 | 10 | 2 | 0.00 | 64.00 | 55.68 | 35.29 | 39.89 |
| 2012SKQB62 | silver | 879 | 388 | 303 | 51.65 | 56.25 | 29.70 | 33.03 | 45.87 |
| 2014FC638 | silver | 320 | 52 | 13 | 8.37 | 48.40 | 38.93 | 23.30 | 31.90 |
| 2015NLCA24 | silver | 47 | 11 | 1 | 0.00 | 65.50 | 53.38 | 31.58 | 39.63 |
| 2016NSCA89 | silver | 121 | 31 | 3 | 2.35 | 61.40 | 55.00 | 34.90 | 39.58 |
| 2016PESC4 | silver | 104 | 55 | 3 | 8.16 | 83.15 | 77.93 | 61.54 | 56.41 |
| 2016QCCA76 | silver | 150 | 66 | 27 | 30.65 | 68.81 | 57.03 | 36.92 | 52.16 |
| 2018BCHPRB52 | silver | 26 | 21 | 11 | 44.44 | 61.12 | 39.84 | 29.51 | 48.47 |
| 2018FCA69 | silver | 227 | 107 | 23 | 18.00 | 63.72 | 50.16 | 30.23 | 43.96 |
| 2018MBQB64 | silver | 75 | 23 | 0 | 0.00 | 71.90 | 64.08 | 43.88 | 45.32 |
| 2018NBOMB6 | silver | 109 | 63 | 41 | 49.67 | 50.71 | 25.71 | 24.26 | 42.03 |
| 2018SKCA43 | silver | 681 | 199 | 30 | 6.91 | 48.09 | 34.50 | 19.21 | 29.83 |
| 2019CHRT30 | silver | 44 | 12 | 1 | 0.00 | 60.42 | 46.41 | 23.64 | 35.61 |
| 2020ABSRB85 | silver | 25 | 10 | 4 | 24.00 | 53.95 | 38.80 | 24.63 | 38.92 |
| 2020CanLII55888MBLB | silver | 16 | 0 | 0 | 0.00 | 22.22 | 12.50 | 6.25 | 11.57 |
| 2020ONAFRAAT8 | silver | 238 | 38 | 28 | 21.43 | 32.40 | 11.89 | 11.88 | 21.91 |
| 2021NBOMB1 | silver | 84 | 15 | 6 | 0.00 | 38.26 | 30.53 | 15.05 | 22.93 |
| 2021NUIPC10 | silver | 74 | 9 | 0 | 0.00 | 41.90 | 38.79 | 22.89 | 26.90 |
| 2021NUIPC8 | silver | 50 | 0 | 0 | 0.00 | 33.33 | 27.94 | 14.00 | 20.43 |
| 2022QCCS1670 | silver | 24 | 18 | 12 | 60.87 | 79.41 | 72.59 | 53.02 | 70.95 |
| 2023ABCA35 | silver | 44 | 11 | 6 | 15.38 | 46.46 | 35.79 | 20.99 | 32.55 |
| 2023FCA24 | silver | 258 | 84 | 42 | 29.15 | 55.34 | 42.28 | 29.46 | 42.26 |
| 2024ONCMT19 | silver | 66 | 28 | 2 | 2.70 | 54.79 | 28.28 | 14.21 | 28.59 |
| 2025MBKB113 | silver | 50 | 11 | 2 | 5.41 | 57.83 | 47.28 | 28.81 | 36.84 |
| 2025NBEUB2 | silver | 24 | 0 | 0 | 0.00 | 28.57 | 17.38 | 4.17 | 15.32 |
| 2026ALRB5 | silver | 46 | 0 | 0 | 0.00 | 32.73 | 24.55 | 13.04 | 19.09 |
| 1983CanLII2967L | test | 52 | 11 | 0 | 0.00 | 72.73 | 65.86 | 46.03 | 46.20 |
| 1993CanLII1889PESCTSD | test | 138 | 30 | 0 | 0.00 | 60.00 | 45.60 | 21.43 | 35.20 |
| 1993CanLII3004FCA | test | 549 | 233 | 31 | 7.68 | 58.56 | 40.84 | 20.46 | 35.69 |
| 1993NSSC71 | test | 71 | 19 | 4 | 17.14 | 78.87 | 69.32 | 46.73 | 55.11 |
| 1997CanLII11497 | test | 100 | 6 | 1 | 0.00 | 17.35 | 10.87 | 4.76 | 9.41 |
| 1999CanLII7320(ONCA) | test | 271 | 110 | 4 | 3.19 | 68.02 | 54.63 | 28.39 | 41.95 |
| 2001BCSC1342 | test | 95 | 14 | 3 | 6.25 | 60.72 | 48.71 | 27.36 | 38.56 |
| 2001CanLII21117 | test | 322 | 150 | 110 | 50.51 | 54.18 | 33.08 | 28.97 | 45.92 |
| 2002SCC33 | test | 328 | 144 | 16 | 9.25 | 61.22 | 50.12 | 29.98 | 40.20 |
| 2003MBCA71 | test | 49 | 20 | 1 | 0.00 | 77.03 | 70.25 | 51.47 | 49.09 |
| 2005NLTD57 | test | 80 | 16 | 3 | 3.12 | 49.60 | 36.26 | 19.88 | 29.66 |
| 2005QCCA437 | test | 116 | 24 | 14 | 22.00 | 56.45 | 41.38 | 26.58 | 39.94 |
| 2009BCHRT256 | test | 157 | 42 | 0 | 0.00 | 60.26 | 55.17 | 35.68 | 38.48 |
| 2009NUCA3 | test | 52 | 12 | 5 | 0.00 | 65.54 | 62.23 | 42.37 | 42.59 |
| 2009PECA12 | test | 87 | 50 | 14 | 25.88 | 63.87 | 52.91 | 35.68 | 47.56 |
| 2010SKQB255 | test | 102 | 62 | 27 | 45.28 | 69.39 | 60.64 | 43.02 | 58.44 |
| 2012SKCA24 | test | 44 | 19 | 6 | 15.38 | 60.78 | 51.38 | 31.49 | 42.52 |
| 2013ONAFRAAT18 | test | 40 | 2 | 0 | 0.00 | 44.44 | 29.51 | 11.90 | 24.65 |
| 2014CanLII36807PELRB | test | 243 | 85 | 70 | 46.32 | 50.28 | 22.48 | 22.24 | 39.69 |
| 2015NSSC25 | test | 296 | 110 | 75 | 40.13 | 58.20 | 40.39 | 32.65 | 46.24 |
| 2016QCCS1184 | test | 87 | 18 | 3 | 3.85 | 67.53 | 57.89 | 37.25 | 43.09 |
| 2017ABCA157 | test | 302 | 82 | 5 | 1.94 | 63.70 | 58.76 | 40.04 | 41.47 |
| 2018NSOIPC2 | test | 271 | 114 | 26 | 19.16 | 53.45 | 46.96 | 28.47 | 39.86 |
| 2019QCCA1492 | test | 1423 | 695 | 232 | 24.81 | 59.04 | 46.09 | 31.14 | 43.31 |
| 2020ABPLAB11 | test | 64 | 44 | 26 | 51.16 | 62.16 | 39.71 | 30.16 | 51.01 |
| 2020BCHPRB101 | test | 149 | 71 | 47 | 48.00 | 58.14 | 39.25 | 33.75 | 48.46 |
| 2020FCA63 | test | 162 | 108 | 22 | 22.62 | 65.85 | 51.57 | 29.88 | 46.68 |
| 2022BCSC135 | test | 1145 | 312 | 194 | 28.28 | 40.09 | 21.62 | 18.84 | 30.00 |
| 2022ONCMT32 | test | 162 | 1 | 0 | 0.00 | 43.27 | 32.28 | 13.50 | 25.18 |
| 2024ABOHSAB19 | test | 52 | 32 | 23 | 52.94 | 58.63 | 32.96 | 30.98 | 48.18 |
| 2024BCSC764 | test | 232 | 101 | 55 | 40.16 | 58.65 | 44.19 | 33.11 | 47.67 |
| 2024NBKB203 | test | 244 | 70 | 3 | 0.00 | 57.99 | 50.53 | 31.83 | 36.17 |
| 2024NUCA9 | test | 115 | 49 | 18 | 27.18 | 62.10 | 53.85 | 36.76 | 47.71 |
| 2025ABAER7 | test | 144 | 5 | 0 | 0.00 | 21.56 | 14.23 | 6.04 | 11.93 |
| 2025ABKB246 | test | 58 | 8 | 0 | 0.00 | 50.00 | 42.95 | 27.27 | 30.98 |
| 2025MBKB136 | test | 47 | 21 | 1 | 0.00 | 72.38 | 64.01 | 41.79 | 45.46 |
| 2025NBEUB1 | test | 25 | 0 | 0 | 0.00 | 27.59 | 17.24 | 8.00 | 14.94 |
| 2026ABCA2L | test | 43 | 29 | 1 | 0.00 | 83.66 | 77.59 | 60.56 | 53.75 |
| 1994CanLII4528NLCA | train | 167 | 74 | 60 | 52.73 | 46.33 | 22.68 | 30.05 | 40.58 |
| 1997CanLII16226ONCA | train | 623 | 291 | 162 | 42.88 | 54.47 | 38.34 | 28.35 | 45.23 |
| 2008CSC9 | train | 484 | 210 | 68 | 25.53 | 58.51 | 45.25 | 29.11 | 43.09 |
| 2019SCC65 | train | 1443 | 521 | 52 | 6.26 | 57.04 | 44.62 | 24.58 | 35.97 |

## Notes

* The `singleton` unmatched policy (default) counts extraction errors: mentions found by only one side become singletons on the other side. This makes Reflex's scores directly comparable with the LLM/pipeline systems scored with `evaluate_coref.py --match-mode span --unmatched singleton`.
* The `drop` policy isolates coreference from extraction errors and is an upper bound — not comparable with pipeline end-to-end scores.
* Reflex never links secondary sources, so those clusters are empty on the system side by construction.
* Reflex only exposes a ``concatenated-id`` on legislation anchors. Decision anchors carry no ``concatenated-id``, so every decision prediction is a singleton cluster: decision MUC / CEAF_e / LEA are 0 by construction, and only B3 is non-zero. Decision coreference cannot be measured from these exports as they stand.
* Coref metrics (MUC / B3 / CEAF_e / LEA) are implemented inline, equivalent to the `coval` library used by `src/evaluation/`.
