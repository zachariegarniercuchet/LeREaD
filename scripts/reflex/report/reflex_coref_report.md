# Reflex (CanLII citator) — coreference resolution evaluation

*Generated 2026-11-08 15:29 — `python scripts/reflex/evaluate_reflex_coref_resolution.py`*

**Scope**: coreference resolution only. Reflex anchors sharing the same ``concatenated-id`` form one cluster; gold mentions sharing the same ``docid`` form one cluster.  Spans are aligned into gold coordinates (token-diff Alignment) before clustering.  Only the grouping by docid is scored — the docid strings themselves are never compared.

**Candidate pool**: `gold`
**Merge gap**: `15` normalised characters
**Unmatched policy**: `singleton` (end-to-end / twinless — comparable with pipeline systems)

**Documents**: 45 reflex export(s) paired with gold.

## EXACT — EXACT span match (label + identical offsets)

| Metric | Precision % | Recall % | F1 % |
| --- | --- | --- | --- |
| MUC | 0.00 | 0.00 | 0.00 |
| B3 | 59.12 | 9.75 | 15.47 |
| CEAF_e | 0.04 | 0.03 | 0.04 |
| LEA | 0.00 | 0.00 | 0.00 |
| **CoNLL (MUC/B3/CEAF_e)** | - | - | 5.17 |

Alignment stats — gold mentions: 10617 | system mentions: 4072 | matched: 3 | gold-only: 10614 | system-only: 4069

## OVERLAP — OVERLAP span match (label + intersecting offsets)

| Metric | Precision % | Recall % | F1 % |
| --- | --- | --- | --- |
| MUC | 4.29 | 1.38 | 2.02 |
| B3 | 59.56 | 9.63 | 15.35 |
| CEAF_e | 4.29 | 1.69 | 2.26 |
| LEA | 1.27 | 0.26 | 0.42 |
| **CoNLL (MUC/B3/CEAF_e)** | - | - | 6.54 |

Alignment stats — gold mentions: 10617 | system mentions: 4072 | matched: 1044 | gold-only: 9573 | system-only: 3028

## Results per split

### EXACT

| split | docs | gold mentions | sys mentions | matched | MUC F1 % | B3 F1 % | CEAF_e F1 % | LEA F1 % | CoNLL F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 38 | 7625 | 2919 | 1 | 0.00 | 16.40 | 0.03 | 0.00 | 5.48 |
| dev | 3 | 298 | 57 | 0 | 0.00 | 16.51 | 0.00 | 0.00 | 5.50 |
| train | 4 | 2694 | 1096 | 2 | 0.00 | 5.81 | 0.17 | 0.00 | 1.99 |

### OVERLAP

| split | docs | gold mentions | sys mentions | matched | MUC F1 % | B3 F1 % | CEAF_e F1 % | LEA F1 % | CoNLL F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| test | 38 | 7625 | 2919 | 1 | 1.58 | 16.19 | 1.96 | 0.37 | 6.58 |
| dev | 3 | 298 | 57 | 0 | 0.00 | 15.83 | 0.51 | 0.00 | 5.44 |
| train | 4 | 2694 | 1096 | 2 | 7.75 | 6.95 | 6.47 | 1.26 | 7.06 |

## Per-document breakdown (exact match)

| document | split | gold | system | matched | MUC F1 % | B3 F1 % | CEAF_e F1 % | LEA F1 % | CoNLL F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1989CanLII1415ONCA | dev | 75 | 3 | 0 | 0.00 | 19.45 | 0.00 | 0.00 | 6.48 |
| 2016NBOMB12 | dev | 128 | 19 | 0 | 0.00 | 7.15 | 0.00 | 0.00 | 2.38 |
| 2021QCCA1675 | dev | 95 | 35 | 0 | 0.00 | 22.93 | 0.00 | 0.00 | 7.64 |
| 1983CanLII2967L | test | 50 | 11 | 0 | 0.00 | 27.11 | 0.00 | 0.00 | 9.04 |
| 1993CanLII1889PESCTSD | test | 138 | 30 | 0 | 0.00 | 27.63 | 0.00 | 0.00 | 9.21 |
| 1993CanLII3004FCA | test | 549 | 233 | 0 | 0.00 | 11.53 | 0.00 | 0.00 | 3.84 |
| 1993NSSC71 | test | 70 | 19 | 0 | 0.00 | 50.57 | 0.00 | 0.00 | 16.86 |
| 1997CanLII11497 | test | 99 | 6 | 0 | 0.00 | 2.34 | 0.00 | 0.00 | 0.78 |
| 1999CanLII7320(ONCA) | test | 264 | 110 | 0 | 0.00 | 43.69 | 0.00 | 0.00 | 14.56 |
| 2001BCSC1342 | test | 95 | 14 | 0 | 0.00 | 23.96 | 0.00 | 0.00 | 7.99 |
| 2001CanLII21117 | test | 296 | 150 | 0 | 0.00 | 4.47 | 0.00 | 0.00 | 1.49 |
| 2002SCC33 | test | 315 | 144 | 0 | 0.00 | 19.46 | 0.00 | 0.00 | 6.49 |
| 2003MBCA71 | test | 47 | 20 | 0 | 0.00 | 47.18 | 0.00 | 0.00 | 15.73 |
| 2005NLTD57 | test | 79 | 16 | 0 | 0.00 | 25.20 | 0.00 | 0.00 | 8.40 |
| 2005QCCA437 | test | 96 | 24 | 0 | 0.00 | 7.92 | 0.00 | 0.00 | 2.64 |
| 2009BCHRT256 | test | 157 | 42 | 0 | 0.00 | 7.15 | 0.00 | 0.00 | 2.38 |
| 2009NUCA3 | test | 52 | 12 | 0 | 0.00 | 14.35 | 0.00 | 0.00 | 4.78 |
| 2009PECA12 | test | 86 | 50 | 0 | 0.00 | 18.35 | 0.00 | 0.00 | 6.12 |
| 2010SKQB255 | test | 97 | 62 | 0 | 0.00 | 13.90 | 0.00 | 0.00 | 4.63 |
| 2012SKCA24 | test | 44 | 19 | 0 | 0.00 | 12.40 | 0.00 | 0.00 | 4.13 |
| 2013ONAFRAAT18 | test | 39 | 2 | 0 | 0.00 | 24.40 | 0.00 | 0.00 | 8.13 |
| 2014CanLII36807PELRB | test | 238 | 85 | 0 | 0.00 | 3.11 | 0.00 | 0.00 | 1.04 |
| 2015NSSC25 | test | 226 | 110 | 0 | 0.00 | 8.24 | 0.00 | 0.00 | 2.75 |
| 2016QCCS1184 | test | 66 | 18 | 0 | 0.00 | 26.01 | 0.00 | 0.00 | 8.67 |
| 2017ABCA157 | test | 297 | 82 | 0 | 0.00 | 5.13 | 0.00 | 0.00 | 1.71 |
| 2018NSOIPC2 | test | 263 | 114 | 0 | 0.00 | 3.00 | 0.00 | 0.00 | 1.00 |
| 2019QCCA1492 | test | 1404 | 695 | 0 | 0.00 | 5.38 | 0.00 | 0.00 | 1.79 |
| 2020ABPLAB11 | test | 38 | 44 | 0 | 0.00 | 11.88 | 0.00 | 0.00 | 3.96 |
| 2020BCHPRB101 | test | 147 | 71 | 1 | 0.00 | 5.10 | 0.95 | 0.00 | 2.02 |
| 2020FCA63 | test | 162 | 108 | 0 | 0.00 | 13.80 | 0.00 | 0.00 | 4.60 |
| 2022BCSC135 | test | 1134 | 312 | 0 | 0.00 | 3.51 | 0.00 | 0.00 | 1.17 |
| 2022ONCMT32 | test | 129 | 1 | 0 | 0.00 | 6.32 | 0.00 | 0.00 | 2.11 |
| 2024ABOHSAB19 | test | 52 | 32 | 0 | 0.00 | 10.58 | 0.00 | 0.00 | 3.53 |
| 2024BCSC764 | test | 231 | 101 | 0 | 0.00 | 7.14 | 0.00 | 0.00 | 2.38 |
| 2024NBKB203 | test | 239 | 70 | 0 | 0.00 | 13.22 | 0.00 | 0.00 | 4.41 |
| 2024NUCA9 | test | 113 | 49 | 0 | 0.00 | 11.51 | 0.00 | 0.00 | 3.84 |
| 2025ABAER7 | test | 143 | 5 | 0 | 0.00 | 4.27 | 0.00 | 0.00 | 1.42 |
| 2025ABKB246 | test | 58 | 8 | 0 | 0.00 | 16.30 | 0.00 | 0.00 | 5.43 |
| 2025MBKB136 | test | 45 | 21 | 0 | 0.00 | 28.57 | 0.00 | 0.00 | 9.52 |
| 2025NBEUB1 | test | 24 | 0 | 0 | 0.00 | 13.11 | 0.00 | 0.00 | 4.37 |
| 2026ABCA2L | test | 43 | 29 | 0 | 0.00 | 45.57 | 0.00 | 0.00 | 15.19 |
| 1994CanLII4528NLCA | train | 167 | 74 | 1 | 0.00 | 3.55 | 0.47 | 0.00 | 1.34 |
| 1997CanLII16226ONCA | train | 623 | 291 | 1 | 0.00 | 1.99 | 0.21 | 0.00 | 0.73 |
| 2008CSC9 | train | 470 | 210 | 0 | 0.00 | 10.92 | 0.00 | 0.00 | 3.64 |
| 2019SCC65 | train | 1434 | 521 | 0 | 0.00 | 6.79 | 0.00 | 0.00 | 2.26 |

## Notes

* The `singleton` unmatched policy (default) counts extraction errors: mentions found by only one side become singletons on the other side. This makes Reflex's scores directly comparable with the LLM/pipeline systems scored with `evaluate_coref.py --match-mode span --unmatched singleton`.
* The `drop` policy isolates coreference from extraction errors and is an upper bound — not comparable with pipeline end-to-end scores.
* Reflex never links secondary sources, so those clusters are empty on the system side by construction.
* Coref metrics (MUC / B3 / CEAF_e / LEA) are implemented inline, equivalent to the `coval` library used by `src/evaluation/`.
