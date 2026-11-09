# Reflex (CanLII citator) — URI linking (resolution) evaluation

*Generated 2026-11-08 15:29 — `python scripts/reflex/evaluate_reflex_coref_resolution.py`*

**Scope**: URI linking (stage 3). Each reflex anchor's ``concatenated-id`` is mapped to a URI via the candidate pool, then the predicted URI is compared against the gold ``uri`` attribute.  Only gold spans carrying a real uri value (not ``uri="none"``, not absent) are counted as instances.

**Candidate pool**: `gold`
**Merge gap**: `15` normalised characters

**Documents**: 45 reflex export(s) paired with gold.

**URI lookup**: 1881 of 4072 concatenated-ids mapped to a URI (2191 unmapped).

## EXACT — EXACT span match

| Measure | Count |
| --- | --- |
| TP (correct uri) | 2 |
| FN — wrong uri (matched) | 0 |
| FN — no uri produced | 0 |
| FN — mention never extracted | 10598 |
| FP — spurious span w/ uri | 1879 |
| Skipped (gold uri=none) | 0 |

| Measure | Value |
| --- | --- |
| Precision % | 0.11 |
| Recall % | 0.02 |
| F1 % | 0.03 |
| Resolver-only accuracy % (Level-1-matched spans only) | 100.00 |
| Matched real-uri instances | 2 |

## OVERLAP — OVERLAP span match

| Measure | Count |
| --- | --- |
| TP (correct uri) | 134 |
| FN — wrong uri (matched) | 172 |
| FN — no uri produced | 737 |
| FN — mention never extracted | 9557 |
| FP — spurious span w/ uri | 1574 |
| Skipped (gold uri=none) | 1 |

| Measure | Value |
| --- | --- |
| Precision % | 7.13 |
| Recall % | 1.26 |
| F1 % | 2.15 |
| Resolver-only accuracy % (Level-1-matched spans only) | 12.85 |
| Matched real-uri instances | 1043 |

## Results per split (overlap match)

| split | docs | TP | FN | FP | Precision % | Recall % | F1 % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test | 38 | 63 | 7554 | 1370 | 4.40 | 0.83 | 1.39 |
| dev | 3 | 1 | 295 | 20 | 4.76 | 0.34 | 0.63 |
| train | 4 | 70 | 2617 | 356 | 16.43 | 2.61 | 4.50 |

## Per-label breakdown (exact match, micro)

## Per-document breakdown (exact match)

| document | split | Precision % | Recall % | F1 % | Accuracy % | TP | FN-wrong | FN-no-uri | FN-missed | FP-spurious |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1989CanLII1415ONCA | dev | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 75 | 1 |
| 2016NBOMB12 | dev | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 126 | 14 |
| 2021QCCA1675 | dev | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 | 0 | 95 | 6 |
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

* Total concatenated-ids looked up: 4072
* Mapped to a URI: 1881
* Unmapped (no candidate-pool entry): 2191

* Unmapped concatenated-ids produce system spans with no uri, which count as false negatives (no uri produced) when the corresponding gold span has a real uri.
