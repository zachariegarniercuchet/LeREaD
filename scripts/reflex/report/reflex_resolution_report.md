# Reflex (CanLII citator) — URI linking (resolution) evaluation

*Generated 2026-11-17 20:24 — `python scripts/reflex/evaluate_reflex_coref_resolution.py`*

**Scope**: URI linking (stage 3). Each reflex anchor's ``concatenated-id`` is mapped to a URI via the candidate pool, then the predicted URI is compared against the gold ``uri`` attribute.  Only gold spans carrying a real uri value (not ``uri="none"``, not absent) are counted as instances.

**Candidate pool**: `gold`
**Merge gap**: `15` normalised characters

**Documents**: 100 reflex export(s) paired with gold.

**URI lookup**: 2596 of 6347 concatenated-ids mapped to a URI (3751 unmapped).

## EXACT — EXACT span match

| Measure | Count |
| --- | --- |
| TP (correct uri) | 2 |
| FN — wrong uri (matched) | 0 |
| FN — no uri produced | 0 |
| FN — mention never extracted | 17100 |
| FP — spurious span w/ uri | 2594 |
| Skipped (gold uri=none) | 0 |

| Measure | Value |
| --- | --- |
| Precision % | 0.08 |
| Recall % | 0.01 |
| F1 % | 0.02 |
| Resolver-only accuracy % (Level-1-matched spans only) | 100.00 |
| Matched real-uri instances | 2 |

## OVERLAP — OVERLAP span match

| Measure | Count |
| --- | --- |
| TP (correct uri) | 196 |
| FN — wrong uri (matched) | 204 |
| FN — no uri produced | 978 |
| FN — mention never extracted | 15724 |
| FP — spurious span w/ uri | 2195 |
| Skipped (gold uri=none) | 1 |

| Measure | Value |
| --- | --- |
| Precision % | 7.55 |
| Recall % | 1.15 |
| F1 % | 1.99 |
| Resolver-only accuracy % (Level-1-matched spans only) | 14.22 |
| Matched real-uri instances | 1378 |

## Results per split (overlap match)

| split | docs | TP | FN | FP | Precision % | Recall % | F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test | 38 | 63 | 7554 | 1370 | 4.40 | 0.83 | 1.39 |
| dev | 3 | 1 | 295 | 20 | 4.76 | 0.34 | 0.63 |
| train | 4 | 70 | 2617 | 356 | 16.43 | 2.61 | 4.50 |
| silver | 55 | 62 | 6440 | 653 | 8.67 | 0.95 | 1.72 |

## Per-type breakdown (micro)

Micro-aggregated across documents, separately for decisions and legislation. `Secondary` is excluded because Reflex never links secondary sources; the full per-label table remains in the JSON.

**EXACT span match**

| type | TP | FN | FP | Precision % | Recall % | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| Decision | 0 | 7786 | 0 | 0.00 | 0.00 | 0.00 |
| Legislation | 2 | 8274 | 2594 | 0.08 | 0.02 | 0.04 |

**OVERLAP span match**

| type | TP | FN | FP | Precision % | Recall % | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| Decision | 0 | 7786 | 0 | 0.00 | 0.00 | 0.00 |
| Legislation | 196 | 8080 | 2399 | 7.55 | 2.37 | 3.61 |

## Per split and mention type (overlap match)

| split | type | TP | FN | FP | Precision % | Recall % | F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test | Decision | 0 | 3716 | 0 | 0.00 | 0.00 | 0.00 |
| test | Legislation | 63 | 3415 | 1370 | 4.40 | 1.81 | 2.57 |
| dev | Decision | 0 | 78 | 0 | 0.00 | 0.00 | 0.00 |
| dev | Legislation | 1 | 181 | 20 | 4.76 | 0.55 | 0.99 |
| train | Decision | 0 | 1572 | 0 | 0.00 | 0.00 | 0.00 |
| train | Legislation | 70 | 811 | 356 | 16.43 | 7.95 | 10.71 |
| silver | Decision | 0 | 2420 | 0 | 0.00 | 0.00 | 0.00 |
| silver | Legislation | 62 | 3673 | 653 | 8.67 | 1.66 | 2.79 |

## Per-document breakdown (exact match)

| document | split | Precision % | Recall % | F1 % | Accuracy % | TP | FN-wrong | FN-no-uri | FN-missed | FP-spurious |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1989CanLII1415ONCA | dev | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 75 | 1 |
| 2016NBOMB12 | dev | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 126 | 14 |
| 2021QCCA1675 | dev | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 95 | 6 |
| 1982CanLII3042NBCA | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 103 | 5 |
| 1982CanLII883ONLRB | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 69 | 0 |
| 1983CanLII2475SKQB | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 476 | 100 |
| 1993CanLII2395BCCA | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 84 | 1 |
| 1994CanLII8687PESCAD | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 59 | 12 |
| 1998CanLII18739NLSCTD | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 118 | 0 |
| 1999CanLII18647MBCA | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 86 | 0 |
| 2002CanLII49492ONSC | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 163 | 2 |
| 2002MBCA125 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 93 | 3 |
| 2002NFCA43 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 165 | 2 |
| 2003ABCA256 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 187 | 78 |
| 2003FC1056 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 135 | 0 |
| 2003SKAIA1 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 23 | 0 |
| 2004CanLII14579ONCA | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 44 | 7 |
| 2005NBCA70 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 99 | 0 |
| 2006NBCA75 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 8 | 0 |
| 2006NUCJ23 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 54 | 8 |
| 2007NUCA2 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 50 | 3 |
| 2008CanLII89121MBLB | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 37 | 0 |
| 2008NLCA4 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 60 | 4 |
| 2009CanLII72366SKLRB | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 30 | 0 |
| 2009KQB465 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 8 | 0 |
| 2009NBQB60 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 77 | 2 |
| 2010CanLII27153NBWCAT | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 23 | 0 |
| 2011CanLII 91839PEIPC | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 262 | 0 |
| 2011FC18 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 29 | 0 |
| 2011NBCA76 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 170 | 1 |
| 2011TMOB65 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 46 | 0 |
| 2012PECA3 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 88 | 5 |
| 2012QCCA1663 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 69 | 1 |
| 2012SKQB62 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 800 | 173 |
| 2014FC638 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 317 | 16 |
| 2015NLCA24 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 46 | 0 |
| 2016NSCA89 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 118 | 0 |
| 2016PESC4 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 101 | 0 |
| 2016QCCA76 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 119 | 32 |
| 2018BCHPRB52 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 25 | 14 |
| 2018FCA69 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 227 | 26 |
| 2018MBQB64 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 73 | 1 |
| 2018NBOMB6 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 105 | 56 |
| 2018SKCA43 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 661 | 77 |
| 2019CHRT30 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 44 | 2 |
| 2020ABSRB85 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 24 | 1 |
| 2020CanLII55888MBLB | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 16 | 0 |
| 2020ONAFRAAT8 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 232 | 35 |
| 2021NBOMB1 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 76 | 0 |
| 2021NUIPC10 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 74 | 0 |
| 2021NUIPC8 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 49 | 0 |
| 2022QCCS1670 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 16 | 12 |
| 2023ABCA35 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 44 | 4 |
| 2023FCA24 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 253 | 10 |
| 2024ONCMT19 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 51 | 22 |
| 2025MBKB113 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 50 | 0 |
| 2025NBEUB2 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 21 | 0 |
| 2026ALRB5 | silver | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 45 | 0 |
| 1983CanLII2967L | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 49 | 0 |
| 1993CanLII1889PESCTSD | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 138 | 0 |
| 1993CanLII3004FCA | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 549 | 95 |
| 1993NSSC71 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 70 | 0 |
| 1997CanLII11497 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 99 | 4 |
| 1999CanLII7320(ONCA) | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 264 | 5 |
| 2001BCSC1342 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 94 | 5 |
| 2001CanLII21117 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 296 | 135 |
| 2002SCC33 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 315 | 26 |
| 2003MBCA71 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 46 | 2 |
| 2005NLTD57 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 79 | 3 |
| 2005QCCA437 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 95 | 15 |
| 2009BCHRT256 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 157 | 2 |
| 2009NUCA3 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 52 | 4 |
| 2009PECA12 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 86 | 15 |
| 2010SKQB255 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 97 | 41 |
| 2012SKCA24 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 44 | 12 |
| 2013ONAFRAAT18 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 39 | 0 |
| 2014CanLII36807PELRB | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 238 | 81 |
| 2015NSSC25 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 226 | 68 |
| 2016QCCS1184 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 66 | 3 |
| 2017ABCA157 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 297 | 7 |
| 2018NSOIPC2 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 263 | 31 |
| 2019QCCA1492 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 1400 | 375 |
| 2020ABPLAB11 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 38 | 37 |
| 2020BCHPRB101 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 147 | 53 |
| 2020FCA63 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 162 | 46 |
| 2022BCSC135 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 1134 | 243 |
| 2022ONCMT32 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 129 | 0 |
| 2024ABOHSAB19 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 52 | 27 |
| 2024BCSC764 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 231 | 63 |
| 2024NBKB203 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 239 | 1 |
| 2024NUCA9 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 113 | 23 |
| 2025ABAER7 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 143 | 0 |
| 2025ABKB246 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 58 | 0 |
| 2025MBKB136 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 45 | 1 |
| 2025NBEUB1 | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 24 | 0 |
| 2026ABCA2L | test | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 43 | 11 |
| 1994CanLII4528NLCA | train | 1.52 | 0.61 | 0.87 | 100.00 | 1 | 0 | 0 | 164 | 65 |
| 1997CanLII16226ONCA | train | 0.53 | 0.16 | 0.25 | 100.00 | 1 | 0 | 0 | 617 | 187 |
| 2008CSC9 | train | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 470 | 78 |
| 2019SCC65 | train | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 1434 | 94 |

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
