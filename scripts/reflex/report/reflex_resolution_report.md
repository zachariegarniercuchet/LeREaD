# Reflex (CanLII citator) — URI linking (resolution) evaluation

*Generated 2026-11-20 16:21 — `python scripts/reflex/evaluate_reflex_coref_resolution.py`*

**Scope**: URI linking (stage 3). Each reflex anchor's ``concatenated-id`` is mapped to a URI via the candidate pool, then the predicted URI is compared against the gold ``uri`` attribute.  Only gold spans carrying a real uri value (not ``uri="none"``, not absent) are counted as instances.

**Candidate pool**: `gold`
**Merge gap**: `15` normalised characters

**Documents**: 100 reflex export(s) paired with gold.

**URI lookup**: 2596 of 6347 concatenated-ids mapped to a URI (3751 unmapped).

## EXACT — EXACT span match

| Measure | Count |
| --- | --- |
| TP (correct uri) | 1642 |
| FN — wrong uri (matched) | 249 |
| FN — no uri produced | 472 |
| FN — mention never extracted | 15379 |
| FP — spurious span w/ uri | 705 |
| Skipped (gold uri=none) | 0 |

| Measure | Value |
| --- | --- |
| Precision % | 63.25 |
| Recall % | 9.25 |
| F1 % | 16.15 |
| Resolver-only accuracy % (Level-1-matched spans only) | 69.49 |
| Matched real-uri instances | 2363 |

## OVERLAP — OVERLAP span match

| Measure | Count |
| --- | --- |
| TP (correct uri) | 2110 |
| FN — wrong uri (matched) | 402 |
| FN — no uri produced | 3620 |
| FN — mention never extracted | 11610 |
| FP — spurious span w/ uri | 83 |
| Skipped (gold uri=none) | 3 |

| Measure | Value |
| --- | --- |
| Precision % | 81.31 |
| Recall % | 11.89 |
| F1 % | 20.75 |
| Resolver-only accuracy % (Level-1-matched spans only) | 34.41 |
| Matched real-uri instances | 6132 |

## Results per split (overlap match)

| split | docs | TP | FN | FP | Precision % | Recall % | F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test | 38 | 1049 | 6860 | 384 | 73.20 | 13.26 | 22.46 |
| dev | 3 | 21 | 307 | 0 | 100.00 | 6.40 | 12.03 |
| train | 4 | 403 | 2307 | 23 | 94.60 | 14.87 | 25.70 |
| silver | 55 | 637 | 6158 | 78 | 89.09 | 9.37 | 16.96 |

## Per-type breakdown (micro)

Micro-aggregated across documents, separately for decisions and legislation. `Secondary` is excluded because Reflex never links secondary sources; the full per-label table remains in the JSON.

**EXACT span match**

| type | TP | FN | FP | Precision % | Recall % | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| Decision | 0 | 8090 | 0 | 0.00 | 0.00 | 0.00 |
| Legislation | 1642 | 6908 | 954 | 63.25 | 19.20 | 29.46 |

**OVERLAP span match**

| type | TP | FN | FP | Precision % | Recall % | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| Decision | 0 | 8090 | 0 | 0.00 | 0.00 | 0.00 |
| Legislation | 2110 | 6440 | 485 | 81.31 | 24.68 | 37.86 |

## Per split and mention type (overlap match)

| split | type | TP | FN | FP | Precision % | Recall % | F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test | Decision | 0 | 3898 | 0 | 0.00 | 0.00 | 0.00 |
| test | Legislation | 1049 | 2507 | 384 | 73.20 | 29.50 | 42.05 |
| dev | Decision | 0 | 96 | 0 | 0.00 | 0.00 | 0.00 |
| dev | Legislation | 21 | 171 | 0 | 100.00 | 10.94 | 19.72 |
| train | Decision | 0 | 1575 | 0 | 0.00 | 0.00 | 0.00 |
| train | Legislation | 403 | 495 | 23 | 94.60 | 44.88 | 60.88 |
| silver | Decision | 0 | 2521 | 0 | 0.00 | 0.00 | 0.00 |
| silver | Legislation | 637 | 3267 | 78 | 89.09 | 16.32 | 27.58 |

## Per-document breakdown (exact match)

| document | split | Precision % | Recall % | F1 % | Accuracy % | TP | FN-wrong | FN-no-uri | FN-missed | FP-spurious |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1989CanLII1415ONCA | dev | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 95 | 1 |
| 2016NBOMB12 | dev | 64.29 | 6.72 | 12.16 | 100.00 | 9 | 0 | 0 | 125 | 5 |
| 2021QCCA1675 | dev | 100.00 | 6.06 | 11.43 | 100.00 | 6 | 0 | 0 | 93 | 0 |
| 1982CanLII3042NBCA | silver | 80.00 | 3.81 | 7.27 | 36.36 | 4 | 0 | 7 | 94 | 1 |
| 1982CanLII883ONLRB | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 69 | 0 |
| 1983CanLII2475SKQB | silver | 79.00 | 16.46 | 27.24 | 91.86 | 79 | 7 | 0 | 394 | 14 |
| 1993CanLII2395BCCA | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 87 | 1 |
| 1994CanLII8687PESCAD | silver | 91.67 | 18.64 | 30.99 | 91.67 | 11 | 0 | 1 | 47 | 1 |
| 1998CanLII18739NLSCTD | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 126 | 0 |
| 1999CanLII18647MBCA | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 10 | 76 | 0 |
| 2002CanLII49492ONSC | silver | 100.00 | 1.18 | 2.34 | 7.14 | 2 | 0 | 26 | 141 | 0 |
| 2002MBCA125 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 1 | 0 | 96 | 2 |
| 2002NFCA43 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 2 | 1 | 165 | 0 |
| 2003ABCA256 | silver | 66.67 | 26.94 | 38.38 | 67.53 | 52 | 13 | 12 | 116 | 13 |
| 2003FC1056 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 17 | 124 | 0 |
| 2003SKAIA1 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 30 | 0 |
| 2004CanLII14579ONCA | silver | 57.14 | 9.09 | 15.69 | 44.44 | 4 | 0 | 5 | 35 | 3 |
| 2005NBCA70 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 12 | 98 | 0 |
| 2006NBCA75 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 10 | 0 |
| 2006NUCJ23 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 1 | 8 | 45 | 7 |
| 2007NUCA2 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 2 | 1 | 47 | 1 |
| 2008CanLII89121MBLB | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 38 | 0 |
| 2008NLCA4 | silver | 100.00 | 6.35 | 11.94 | 100.00 | 4 | 0 | 0 | 59 | 0 |
| 2009CanLII72366SKLRB | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 6 | 25 | 0 |
| 2009KQB465 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 2 | 9 | 0 |
| 2009NBQB60 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 2 | 2 | 78 | 0 |
| 2010CanLII27153NBWCAT | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 23 | 0 |
| 2011CanLII 91839PEIPC | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 73 | 196 | 0 |
| 2011FC18 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 1 | 28 | 0 |
| 2011NBCA76 | silver | 100.00 | 0.57 | 1.13 | 3.33 | 1 | 0 | 29 | 146 | 0 |
| 2011TMOB65 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 5 | 43 | 0 |
| 2012PECA3 | silver | 60.00 | 3.41 | 6.45 | 100.00 | 3 | 0 | 0 | 85 | 2 |
| 2012QCCA1663 | silver | 100.00 | 1.30 | 2.56 | 50.00 | 1 | 0 | 1 | 75 | 0 |
| 2012SKQB62 | silver | 89.60 | 17.63 | 29.47 | 51.16 | 155 | 2 | 146 | 576 | 16 |
| 2014FC638 | silver | 81.25 | 4.06 | 7.74 | 100.00 | 13 | 0 | 0 | 307 | 3 |
| 2015NLCA24 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 1 | 46 | 0 |
| 2016NSCA89 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 3 | 118 | 0 |
| 2016PESC4 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 3 | 101 | 0 |
| 2016QCCA76 | silver | 53.12 | 11.33 | 18.68 | 62.96 | 17 | 7 | 3 | 123 | 8 |
| 2018BCHPRB52 | silver | 57.14 | 30.77 | 40.00 | 72.73 | 8 | 1 | 2 | 15 | 5 |
| 2018FCA69 | silver | 80.77 | 9.25 | 16.60 | 91.30 | 21 | 0 | 2 | 204 | 5 |
| 2018MBQB64 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 75 | 1 |
| 2018NBOMB6 | silver | 67.86 | 34.86 | 46.06 | 92.68 | 38 | 1 | 2 | 68 | 17 |
| 2018SKCA43 | silver | 31.17 | 3.52 | 6.33 | 80.00 | 24 | 0 | 6 | 651 | 53 |
| 2019CHRT30 | silver | 50.00 | 2.27 | 4.35 | 100.00 | 1 | 0 | 0 | 43 | 1 |
| 2020ABSRB85 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 4 | 21 | 1 |
| 2020CanLII55888MBLB | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 16 | 0 |
| 2020ONAFRAAT8 | silver | 80.00 | 11.76 | 20.51 | 100.00 | 28 | 0 | 0 | 210 | 7 |
| 2021NBOMB1 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 6 | 78 | 0 |
| 2021NUIPC10 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 74 | 0 |
| 2021NUIPC8 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 50 | 0 |
| 2022QCCS1670 | silver | 83.33 | 41.67 | 55.56 | 83.33 | 10 | 0 | 2 | 12 | 2 |
| 2023ABCA35 | silver | 100.00 | 9.09 | 16.67 | 66.67 | 4 | 0 | 2 | 38 | 0 |
| 2023FCA24 | silver | 50.00 | 1.94 | 3.73 | 11.90 | 5 | 0 | 37 | 216 | 5 |
| 2024ONCMT19 | silver | 9.09 | 3.03 | 4.55 | 100.00 | 2 | 0 | 0 | 64 | 20 |
| 2025MBKB113 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 2 | 48 | 0 |
| 2025NBEUB2 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 24 | 0 |
| 2026ALRB5 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 46 | 0 |
| 1983CanLII2967L | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 51 | 0 |
| 1993CanLII1889PESCTSD | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 138 | 0 |
| 1993CanLII3004FCA | test | 25.26 | 4.37 | 7.45 | 77.42 | 24 | 7 | 0 | 518 | 64 |
| 1993NSSC71 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 4 | 67 | 0 |
| 1997CanLII11497 | test | 25.00 | 1.00 | 1.92 | 100.00 | 1 | 0 | 0 | 99 | 3 |
| 1999CanLII7320(ONCA) | test | 80.00 | 1.48 | 2.90 | 100.00 | 4 | 0 | 0 | 267 | 1 |
| 2001BCSC1342 | test | 60.00 | 3.19 | 6.06 | 100.00 | 3 | 0 | 0 | 91 | 2 |
| 2001CanLII21117 | test | 78.52 | 32.92 | 46.39 | 96.36 | 106 | 4 | 0 | 212 | 25 |
| 2002SCC33 | test | 61.54 | 4.88 | 9.04 | 100.00 | 16 | 0 | 0 | 312 | 10 |
| 2003MBCA71 | test | 50.00 | 2.08 | 4.00 | 100.00 | 1 | 0 | 0 | 47 | 1 |
| 2005NLTD57 | test | 100.00 | 3.75 | 7.23 | 100.00 | 3 | 0 | 0 | 77 | 0 |
| 2005QCCA437 | test | 86.67 | 11.30 | 20.00 | 92.86 | 13 | 1 | 0 | 101 | 1 |
| 2009BCHRT256 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 157 | 2 |
| 2009NUCA3 | test | 50.00 | 3.85 | 7.14 | 40.00 | 2 | 2 | 1 | 47 | 0 |
| 2009PECA12 | test | 86.67 | 14.94 | 25.49 | 92.86 | 13 | 0 | 1 | 73 | 2 |
| 2010SKQB255 | test | 65.85 | 26.47 | 37.76 | 100.00 | 27 | 0 | 0 | 75 | 14 |
| 2012SKCA24 | test | 50.00 | 13.64 | 21.43 | 100.00 | 6 | 0 | 0 | 38 | 6 |
| 2013ONAFRAAT18 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 40 | 0 |
| 2014CanLII36807PELRB | test | 83.95 | 27.98 | 41.98 | 97.14 | 68 | 2 | 0 | 173 | 11 |
| 2015NSSC25 | test | 89.71 | 20.61 | 33.52 | 81.33 | 61 | 5 | 9 | 221 | 2 |
| 2016QCCS1184 | test | 66.67 | 2.30 | 4.44 | 66.67 | 2 | 0 | 1 | 84 | 1 |
| 2017ABCA157 | test | 28.57 | 0.66 | 1.29 | 40.00 | 2 | 2 | 1 | 297 | 3 |
| 2018NSOIPC2 | test | 83.87 | 9.59 | 17.22 | 100.00 | 26 | 0 | 0 | 245 | 5 |
| 2019QCCA1492 | test | 19.73 | 5.21 | 8.25 | 31.90 | 74 | 156 | 2 | 1187 | 145 |
| 2020ABPLAB11 | test | 67.57 | 39.06 | 49.50 | 96.15 | 25 | 1 | 0 | 38 | 11 |
| 2020BCHPRB101 | test | 83.02 | 29.53 | 43.56 | 93.62 | 44 | 0 | 3 | 102 | 9 |
| 2020FCA63 | test | 32.61 | 9.26 | 14.42 | 68.18 | 15 | 7 | 0 | 140 | 24 |
| 2022BCSC135 | test | 74.90 | 15.90 | 26.22 | 93.81 | 182 | 11 | 1 | 951 | 50 |
| 2022ONCMT32 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 162 | 0 |
| 2024ABOHSAB19 | test | 74.07 | 38.46 | 50.63 | 86.96 | 20 | 0 | 3 | 29 | 7 |
| 2024BCSC764 | test | 85.71 | 23.28 | 36.61 | 98.18 | 54 | 0 | 1 | 177 | 9 |
| 2024NBKB203 | test | 100.00 | 0.41 | 0.82 | 33.33 | 1 | 0 | 2 | 241 | 0 |
| 2024NUCA9 | test | 69.57 | 13.91 | 23.19 | 88.89 | 16 | 0 | 2 | 97 | 7 |
| 2025ABAER7 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 144 | 0 |
| 2025ABKB246 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 58 | 0 |
| 2025MBKB136 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 1 | 46 | 1 |
| 2025NBEUB1 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 25 | 0 |
| 2026ABCA2L | test | 9.09 | 2.33 | 3.70 | 100.00 | 1 | 0 | 0 | 42 | 10 |
| 1994CanLII4528NLCA | train | 90.91 | 36.36 | 51.95 | 100.00 | 60 | 0 | 0 | 105 | 6 |
| 1997CanLII16226ONCA | train | 83.51 | 25.40 | 38.96 | 96.91 | 157 | 5 | 0 | 456 | 26 |
| 2008CSC9 | train | 80.77 | 13.02 | 22.42 | 92.65 | 63 | 5 | 0 | 416 | 10 |
| 2019SCC65 | train | 53.19 | 3.47 | 6.51 | 96.15 | 50 | 2 | 0 | 1391 | 42 |

## URI lookup diagnostics

* Total concatenated-ids looked up: 6347
* Mapped to a URI: 2596
* Unmapped (no candidate-pool entry): 3751

| type | lookups | mapped | unmapped |
| --- | --- | --- | --- |
| Decision | 3118 | 0 | 3118 |
| Legislation | 3229 | 2596 | 633 |

* Reflex only exposes a ``concatenated-id`` on legislation anchors. Decision anchors have no ``concatenated-id``, so the current lookup cannot map them and their URI-linking score is 0 by construction.

* Unmapped concatenated-ids produce system spans with no uri, which count as false negatives (no uri produced) when the corresponding gold span has a real uri.
