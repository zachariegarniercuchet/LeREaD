# Reflex (CanLII citator) extraction — evaluation against LeREaD gold

*Generated 2026-10-22 09:53 — `python scripts/reflex/evaluate_reflex.py --compare "rules=output/test_baseline_rules/**/*_final.html" --compare "gpt-5.2=output/test_gpt-5.2_DEC_fs6_greedy_sentence_long/**/*_final.html" --compare "gemma4=output/test_gemma4_DEC_fs6_greedy_sentence_long/**/*_final.html" --compare "qwen38-27b=output/test_qwen38-27b_DEC_fs6_greedy_sentence_long/**/*_final.html"`*

**Scope**: parent mentions only (`decision` / `legislation` / `secondary sources`). Components (title, citation, fragment, authors, source) are **not** scored — Reflex does not produce them.

## 1. Documents

* 100 reflex export(s) paired with a gold document (3 dev, 55 silver, 38 test, 4 train).
* Raw reflex anchors: 8670 -> 6347 predicted mentions after merging (gap = 15).
* Anchor types seen: decision=3132, legislation=5538; 3 http link(s) with an unrecognised target.

**Fuzzy id matches** (id differs by the court suffix, documents verified identical by content-token similarity):

| reflex file | gold doc | split | similarity |
| --- | --- | --- | --- |
| 1983CanLII2967MBCA_reflex.html | 1983CanLII2967L | test | 0.972 |
| 1989CanLII1415CACITT_reflex.html | 1989CanLII1415ONCA | dev | 0.998 |
| 1993CanLII1889PESCTD_reflex.html | 1993CanLII1889PESCTSD | test | 0.983 |
| 1997CanLII11497NSSC_reflex.html | 1997CanLII11497 | test | 0.998 |
| 1999CanLII7320ONCA_reflex.html | 1999CanLII7320(ONCA) | test | 0.975 |
| 2001CanLII21117QCTDP_reflex.html | 2001CanLII21117 | test | 0.996 |
| 2005NBCA70._reflex.html | 2005NBCA70 | silver | 0.990 |
| 2009SKQB465_reflex.html | 2009KQB465 | silver | 0.991 |
| 2011CanLII91839PEIPC_reflex.html | 2011CanLII 91839PEIPC | silver | 0.999 |
| 2026ABCA2_reflex.html | 2026ABCA2L | test | 0.994 |

Similarity = content-token Jaccard (same decision re-rendered ~0.96-1.00, two different decisions ~0.10).

## 2. How a Reflex prediction is built

1. **What counts as a mention.** `reflex2-link` anchors whose href contains `/laws/` -> **legislation**; `reflex3-caselaw` anchors with href `/.../doc/` -> **decision**; `/jnl/ /book/ /comm/` -> **secondary sources**. Internal navigation (`reflex-parag`, footnote `#_edn…`) is ignored.
2. **Grouping.** Consecutive anchors resolving to the same reference (same base href) within **15** normalised characters become one prediction, so `Federal Court Act` + `s. 18` is the single mention `Federal Court Act, s. 18`, as in the gold.
3. **Alignment.** The reflex rendering is token-diffed onto the gold one. Reflex *injects* neutral citations (`1982 CanLII 19 (SCC)`) that the gold text never contains; they map to their insertion point, normally inside the gold decision mention that names the same case.
4. **Scoring** re-uses `src/evaluation/evaluation_l1_util.py` on gold coordinates: **EXACT** = label + boundary text + 200-char context (the Table-1 metric); **OVERLAP** = label + character overlap (the lenient comparison metric `evaluate.py` already prints).

## 3. Merge-gap sweep

| merge gap | exact P | exact R | exact F1 | overlap P | overlap R | overlap F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 14.8 | 6.9 | 9.4 | 75.6 | 35.0 | 47.9 |
| 5 | 17.2 | 7.4 | 10.4 | 80.9 | 34.9 | 48.8 |
| 10 | 34.8 | 12.7 | 18.6 | 94.8 | 34.6 | 50.7 |
| 15 | 37.2 | 13.3 | 19.6 | 96.7 | 34.5 | 50.9 |
| 20 | 37.2 | 13.2 | 19.5 | 96.8 | 34.5 | 50.8 |
| 30 | 36.6 | 13.0 | 19.2 | 96.9 | 34.3 | 50.7 |
| 45 | 35.8 | 12.6 | 18.6 | 96.9 | 34.0 | 50.4 |
| 60 | 35.3 | 12.3 | 18.3 | 97.0 | 33.8 | 50.2 |

**Selected gap: 15** (best overlap micro-F1; the plateau is wide, 10-30 all within ~0.5 F1).

## 4. Results per split and mention type

### Decision + legislation only, all splits

Secondary sources are left out: Reflex never links one, so the category is 0/0 by construction and carries no signal.

**EXACT — EXACT boundaries (Table-1 metric)**

| split | mention type | P % | R % | F1 % |
| --- | --- | --- | --- | --- |
| train | decision | 0.0 | 0.0 | 0.0 |
| train | legislation | 79.9 | 38.0 | 51.5 |
| dev | decision | 0.0 | 0.0 | 0.0 |
| dev | legislation | 71.4 | 7.8 | 14.1 |
| test | decision | 1.3 | 0.5 | 0.7 |
| test | legislation | 69.9 | 28.7 | 40.7 |
| silver | decision | 1.2 | 0.4 | 0.6 |
| silver | legislation | 72.4 | 24.5 | 36.6 |

**OVERLAP — OVERLAP matching (lenient)**

| split | mention type | P % | R % | F1 % |
| --- | --- | --- | --- | --- |
| train | decision | 99.6 | 42.1 | 59.2 |
| train | legislation | 98.6 | 46.9 | 63.6 |
| dev | decision | 100.0 | 37.5 | 54.5 |
| dev | legislation | 100.0 | 10.9 | 19.7 |
| test | decision | 96.6 | 36.1 | 52.6 |
| test | legislation | 96.1 | 39.4 | 55.9 |
| silver | decision | 95.1 | 36.1 | 52.3 |
| silver | legislation | 96.2 | 32.5 | 48.6 |

### Per-split detail (all mention types)

### split `dev` — human gold — 3 documents

**EXACT — EXACT boundaries (Table-1 metric)**

| mention type | gold | reflex | TP | P % | R % | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| decision | 96 | 36 | 0 | 0.0 | 0.0 | 0.0 |
| legislation | 192 | 21 | 15 | 71.4 | 7.8 | 14.1 |
| secondary sources | 42 | 0 | 0 | 0.0 | 0.0 | 0.0 |
| **MICRO (all types)** | 330 | 57 | 15 | 26.3 | 4.5 | 7.8 |

**OVERLAP — OVERLAP matching (lenient)**

| mention type | gold | reflex | TP | P % | R % | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| decision | 96 | 36 | 36 | 100.0 | 37.5 | 54.5 |
| legislation | 192 | 21 | 21 | 100.0 | 10.9 | 19.7 |
| secondary sources | 42 | 0 | 0 | 0.0 | 0.0 | 0.0 |
| **MICRO (all types)** | 330 | 57 | 57 | 100.0 | 17.3 | 29.5 |

### split `silver` — **silver** gold (auto-annotated), use with care — 55 documents

**EXACT — EXACT boundaries (Table-1 metric)**

| mention type | gold | reflex | TP | P % | R % | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| decision | 2521 | 956 | 11 | 1.2 | 0.4 | 0.6 |
| legislation | 3904 | 1319 | 955 | 72.4 | 24.5 | 36.6 |
| secondary sources | 370 | 0 | 0 | 0.0 | 0.0 | 0.0 |
| **MICRO (all types)** | 6795 | 2275 | 966 | 42.5 | 14.2 | 21.3 |

**OVERLAP — OVERLAP matching (lenient)**

| mention type | gold | reflex | TP | P % | R % | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| decision | 2521 | 956 | 909 | 95.1 | 36.1 | 52.3 |
| legislation | 3904 | 1319 | 1269 | 96.2 | 32.5 | 48.6 |
| secondary sources | 370 | 0 | 0 | 0.0 | 0.0 | 0.0 |
| **MICRO (all types)** | 6795 | 2275 | 2178 | 95.7 | 32.1 | 48.0 |

### split `test` — human gold, behind Table 1 — 38 documents

**EXACT — EXACT boundaries (Table-1 metric)**

| mention type | gold | reflex | TP | P % | R % | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| decision | 3901 | 1458 | 19 | 1.3 | 0.5 | 0.7 |
| legislation | 3560 | 1461 | 1021 | 69.9 | 28.7 | 40.7 |
| secondary sources | 456 | 0 | 0 | 0.0 | 0.0 | 0.0 |
| **MICRO (all types)** | 7917 | 2919 | 1040 | 35.6 | 13.1 | 19.2 |

**OVERLAP — OVERLAP matching (lenient)**

| mention type | gold | reflex | TP | P % | R % | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| decision | 3901 | 1458 | 1409 | 96.6 | 36.1 | 52.6 |
| legislation | 3560 | 1461 | 1404 | 96.1 | 39.4 | 55.9 |
| secondary sources | 456 | 0 | 0 | 0.0 | 0.0 | 0.0 |
| **MICRO (all types)** | 7917 | 2919 | 2813 | 96.4 | 35.5 | 51.9 |

### split `train` — human gold — 4 documents

**EXACT — EXACT boundaries (Table-1 metric)**

| mention type | gold | reflex | TP | P % | R % | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| decision | 1580 | 668 | 0 | 0.0 | 0.0 | 0.0 |
| legislation | 900 | 428 | 342 | 79.9 | 38.0 | 51.5 |
| secondary sources | 237 | 0 | 0 | 0.0 | 0.0 | 0.0 |
| **MICRO (all types)** | 2717 | 1096 | 342 | 31.2 | 12.6 | 17.9 |

**OVERLAP — OVERLAP matching (lenient)**

| mention type | gold | reflex | TP | P % | R % | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| decision | 1580 | 668 | 665 | 99.6 | 42.1 | 59.2 |
| legislation | 900 | 428 | 422 | 98.6 | 46.9 | 63.6 |
| secondary sources | 237 | 0 | 0 | 0.0 | 0.0 | 0.0 |
| **MICRO (all types)** | 2717 | 1096 | 1087 | 99.2 | 40.0 | 57.0 |

## 5. Recall chain and why Reflex misses mentions (split `test`)

| mention type | gold | detected (overlap) | exact boundaries | missed: short form (≤30 ch) | missed: full mention | missed but contains a citation |
| --- | --- | --- | --- | --- | --- | --- |
| decision | 3901 | 1409 (36.1%) | 19 (0.5%) | 1936 | 556 | 254 |
| legislation | 3560 | 1404 (39.4%) | 1021 (28.7%) | 1840 | 316 | 62 |
| secondary sources | 456 | 0 (0.0%) | 0 (0.0%) | 151 | 305 | 42 |

*`short form` = gold mention of ≤30 characters (`Newport`, `idem, s. 3`, `Bill C-91`, `Act`) — a citator has nothing to resolve for most of them. `contains a citation` = the gold mention does carry a citation string, so a citator could in principle have linked it (these are the real Reflex misses worth arguing about).*

Where is the nearest Reflex prediction of the same type for each missed gold mention?

| mention type | no reflex link nearby (genuine miss) | no reflex link of this type in the document | touching (boundary mismatch) | within 40 chars |
| --- | --- | --- | --- | --- |
| decision | 2162 | 6 | 156 | 168 |
| legislation | 1675 | 263 | 32 | 186 |
| secondary sources | 0 | 456 | 0 | 0 |

### Examples of missed gold mentions

* **decision — no reflex link nearby (genuine miss)**: `Manitoba v. Manitoba Human Rights Commission`; `Newport`; `Hiscott v. General Electric Co. (1975), 521 F. 2d 632`
* **decision — no reflex link of this type in the document**: `Citation: MegumaGold Corp. (Re), 2025 NBEUB 1`; `Cadillac Ventures Inc. and Ring of Fire Metals PYT Ltd. v. The Recorder MN-001-2023`; `Stewart v. New Brunswick (Mining Commissioner) (1995), N.B.R. (2d) 353 (Q.B.); [1995] N.B.J. No. 191`
* **decision — touching (boundary mismatch)**: `p. 185`; `Third Eye Capital`; `para. 1`
* **decision — within 40 chars**: `Re Manitoba Human Rights Com'n`; `Canamerican Auto Lease and Rental Limited v. Canada, (unreported, March 4, 1985, Fed. T.D.)`; `Ellis-Don Construction Ltd. v. Canada (Minister of Public Works), (unreported, April 16, 1992, Fed. `
* **legislation — no reflex link nearby (genuine miss)**: `Food and Drugs Act`; `Patented Medicines Regulations`; `Patented Medicines Regulations`
* **legislation — no reflex link of this type in the document**: `Human Rights Act, 1974 (Man.), c. 65`; `C.C.S.M., H175`; `Act`
* **legislation — touching (boundary mismatch)**: `Subsection 55.2(5) of the Patent Act`; `Bill C-91`; `subsection 12(1)`
* **legislation — within 40 chars**: `Regulations, ss. 5(1),(2)`; `Patented Medicines Regulations`; `Patent Act`
* **secondary sources — no reflex link of this type in the document**: `Maxwell on Interpretation of Statutes, 12th ed. (1969), at p. 320`; `Maxwell`; `Cheshire, Fifoot and Furmston, Law of Con- tract, 12th ed. (1991)`

## 6. Reflex vs the other systems (same documents, same scorer)

### `rules` — 38 shared document(s): 1983CanLII2967L, 1993CanLII1889PESCTSD, 1993CanLII3004FCA, 1993NSSC71, 1997CanLII11497, 1999CanLII7320(ONCA), 2001BCSC1342, 2001CanLII21117, 2002SCC33, 2003MBCA71, 2005NLTD57, 2005QCCA437, 2009BCHRT256, 2009NUCA3, 2009PECA12, 2010SKQB255, 2012SKCA24, 2013ONAFRAAT18, 2014CanLII36807PELRB, 2015NSSC25, 2016QCCS1184, 2017ABCA157, 2018NSOIPC2, 2019QCCA1492, 2020ABPLAB11, 2020BCHPRB101, 2020FCA63, 2022BCSC135, 2022ONCMT32, 2024ABOHSAB19, 2024BCSC764, 2024NBKB203, 2024NUCA9, 2025ABAER7, 2025ABKB246, 2025MBKB136, 2025NBEUB1, 2026ABCA2L

**EXACT** — EXACT boundaries (Table-1 metric)

| system | gold | spans | TP | P % | R % | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| **Reflex (CanLII citator)** | 7917 | 2919 | 1040 | 35.6 | 13.1 | 19.2 |
| rules | 7917 | 7459 | 2430 | 32.6 | 30.7 | 31.6 |

**OVERLAP** — OVERLAP matching (lenient)

| system | gold | spans | TP | P % | R % | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| **Reflex (CanLII citator)** | 7917 | 2919 | 2813 | 96.4 | 35.5 | 51.9 |
| rules | 7917 | 7459 | 4994 | 67.0 | 63.1 | 65.0 |

### `gpt-5.2` — 4 shared document(s): 2001CanLII21117, 2002SCC33, 2005QCCA437, 2024NBKB203

**EXACT** — EXACT boundaries (Table-1 metric)

| system | gold | spans | TP | P % | R % | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| **Reflex (CanLII citator)** | 1010 | 388 | 143 | 36.9 | 14.2 | 20.5 |
| gpt-5.2 | 1010 | 1026 | 828 | 80.7 | 82.0 | 81.3 |

**OVERLAP** — OVERLAP matching (lenient)

| system | gold | spans | TP | P % | R % | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| **Reflex (CanLII citator)** | 1010 | 388 | 357 | 92.0 | 35.3 | 51.1 |
| gpt-5.2 | 1010 | 1026 | 911 | 88.8 | 90.2 | 89.5 |

### `gemma4` — 4 shared document(s): 2001CanLII21117, 2002SCC33, 2005QCCA437, 2024NBKB203

**EXACT** — EXACT boundaries (Table-1 metric)

| system | gold | spans | TP | P % | R % | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| **Reflex (CanLII citator)** | 1010 | 388 | 143 | 36.9 | 14.2 | 20.5 |
| gemma4 | 1010 | 895 | 781 | 87.3 | 77.3 | 82.0 |

**OVERLAP** — OVERLAP matching (lenient)

| system | gold | spans | TP | P % | R % | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| **Reflex (CanLII citator)** | 1010 | 388 | 357 | 92.0 | 35.3 | 51.1 |
| gemma4 | 1010 | 895 | 854 | 95.4 | 84.6 | 89.7 |

### `qwen38-27b` — 4 shared document(s): 2001CanLII21117, 2002SCC33, 2005QCCA437, 2024NBKB203

**EXACT** — EXACT boundaries (Table-1 metric)

| system | gold | spans | TP | P % | R % | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| **Reflex (CanLII citator)** | 1010 | 388 | 143 | 36.9 | 14.2 | 20.5 |
| qwen38-27b | 1010 | 751 | 642 | 85.5 | 63.6 | 72.9 |

**OVERLAP** — OVERLAP matching (lenient)

| system | gold | spans | TP | P % | R % | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| **Reflex (CanLII citator)** | 1010 | 388 | 357 | 92.0 | 35.3 | 51.1 |
| qwen38-27b | 1010 | 751 | 707 | 94.1 | 70.0 | 80.3 |

Reference — Table 1 of the paper (mention-level micro-F1, test set, exact boundaries; different document set, listed for orientation):

| model | mention F1 % |
| --- | --- |
| GPT-5.2 | 76 |
| Gemma-4-31B-it | 78 |
| Qwen3.8-27B | 71 |
| Muse-Glimmer-30B | 65 |
| SaulLM-54B-Inst. | 6 |

## 7. Per-document breakdown

| document | split | gold | reflex | exact P/R/F1 | overlap P/R/F1 | text align |
| --- | --- | --- | --- | --- | --- | --- |
| 1989CanLII1415ONCA | dev | 95 | 3 | 0.0/0.0/0.0 | 100.0/3.2/6.1 | 0.999 |
| 2016NBOMB12 | dev | 136 | 19 | 47.4/6.6/11.6 | 100.0/14.0/24.5 | 0.999 |
| 2021QCCA1675 | dev | 99 | 35 | 17.1/6.1/9.0 | 100.0/35.4/52.2 | 0.998 |
| 1982CanLII3042NBCA | silver | 105 | 31 | 35.5/10.5/16.2 | 100.0/29.5/45.6 | 0.996 |
| 1982CanLII883ONLRB | silver | 69 | 3 | 0.0/0.0/0.0 | 100.0/4.3/8.3 | 0.998 |
| 1983CanLII2475SKQB | silver | 480 | 142 | 60.6/17.9/27.7 | 96.5/28.5/44.1 | 0.995 |
| 1993CanLII2395BCCA | silver | 87 | 13 | 0.0/0.0/0.0 | 100.0/14.9/26.0 | 0.994 |
| 1994CanLII8687PESCAD | silver | 59 | 18 | 66.7/20.3/31.2 | 100.0/30.5/46.8 | 0.996 |
| 1998CanLII18739NLSCTD | silver | 126 | 25 | 0.0/0.0/0.0 | 100.0/19.8/33.1 | 0.997 |
| 1999CanLII18647MBCA | silver | 86 | 53 | 18.9/11.6/14.4 | 96.2/59.3/73.4 | 0.993 |
| 2002CanLII49492ONSC | silver | 169 | 52 | 53.8/16.6/25.3 | 98.1/30.2/46.2 | 0.995 |
| 2002MBCA125 | silver | 97 | 50 | 2.0/1.0/1.4 | 76.0/39.2/51.7 | 0.990 |
| 2002NFCA43 | silver | 168 | 21 | 14.3/1.8/3.2 | 100.0/12.5/22.2 | 0.997 |
| 2003ABCA256 | silver | 193 | 129 | 59.7/39.9/47.8 | 91.5/61.1/73.3 | 0.993 |
| 2003FC1056 | silver | 141 | 78 | 21.8/12.1/15.5 | 96.2/53.2/68.5 | 0.994 |
| 2003SKAIA1 | silver | 30 | 1 | 0.0/0.0/0.0 | 100.0/3.3/6.5 | 1.000 |
| 2004CanLII14579ONCA | silver | 44 | 31 | 29.0/20.5/24.0 | 58.1/40.9/48.0 | 0.992 |
| 2005NBCA70 | silver | 110 | 39 | 30.8/10.9/16.1 | 100.0/35.5/52.3 | 0.996 |
| 2006NBCA75 | silver | 10 | 1 | 0.0/0.0/0.0 | 100.0/10.0/18.2 | 1.000 |
| 2006NUCJ23 | silver | 54 | 29 | 31.0/16.7/21.7 | 93.1/50.0/65.1 | 0.999 |
| 2007NUCA2 | silver | 50 | 11 | 27.3/6.0/9.8 | 100.0/22.0/36.1 | 0.998 |
| 2008CanLII89121MBLB | silver | 38 | 1 | 0.0/0.0/0.0 | 100.0/2.6/5.1 | 0.999 |
| 2008NLCA4 | silver | 63 | 26 | 15.4/6.3/9.0 | 100.0/41.3/58.4 | 0.994 |
| 2009CanLII72366SKLRB | silver | 31 | 18 | 33.3/19.4/24.5 | 88.9/51.6/65.3 | 1.000 |
| 2009KQB465 | silver | 11 | 7 | 28.6/18.2/22.2 | 100.0/63.6/77.8 | 0.996 |
| 2009NBQB60 | silver | 82 | 25 | 16.0/4.9/7.5 | 100.0/30.5/46.7 | 0.995 |
| 2010CanLII27153NBWCAT | silver | 23 | 1 | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 1.000 |
| 2011CanLII 91839PEIPC | silver | 269 | 92 | 79.3/27.1/40.4 | 98.9/33.8/50.4 | 1.000 |
| 2011FC18 | silver | 29 | 11 | 9.1/3.4/5.0 | 100.0/37.9/55.0 | 0.974 |
| 2011NBCA76 | silver | 176 | 64 | 46.9/17.0/25.0 | 98.4/35.8/52.5 | 0.995 |
| 2011TMOB65 | silver | 48 | 21 | 23.8/10.4/14.5 | 100.0/43.8/60.9 | 0.989 |
| 2012PECA3 | silver | 88 | 20 | 15.0/3.4/5.6 | 100.0/22.7/37.0 | 0.999 |
| 2012QCCA1663 | silver | 77 | 10 | 20.0/2.6/4.6 | 100.0/13.0/23.0 | 0.999 |
| 2012SKQB62 | silver | 879 | 388 | 78.1/34.5/47.8 | 98.7/43.6/60.5 | 0.999 |
| 2014FC638 | silver | 320 | 52 | 25.0/4.1/7.0 | 100.0/16.2/28.0 | 1.000 |
| 2015NLCA24 | silver | 47 | 11 | 9.1/2.1/3.4 | 100.0/23.4/37.9 | 0.999 |
| 2016NSCA89 | silver | 121 | 31 | 9.7/2.5/3.9 | 100.0/25.6/40.8 | 0.992 |
| 2016PESC4 | silver | 104 | 55 | 5.5/2.9/3.8 | 67.3/35.6/46.5 | 0.995 |
| 2016QCCA76 | silver | 150 | 66 | 40.9/18.0/25.0 | 98.5/43.3/60.2 | 0.998 |
| 2018BCHPRB52 | silver | 26 | 21 | 52.4/42.3/46.8 | 95.2/76.9/85.1 | 1.000 |
| 2018FCA69 | silver | 227 | 107 | 21.5/10.1/13.8 | 98.1/46.3/62.9 | 0.999 |
| 2018MBQB64 | silver | 75 | 23 | 0.0/0.0/0.0 | 100.0/30.7/46.9 | 0.998 |
| 2018NBOMB6 | silver | 109 | 63 | 65.1/37.6/47.7 | 92.1/53.2/67.4 | 0.999 |
| 2018SKCA43 | silver | 681 | 199 | 15.1/4.4/6.8 | 99.5/29.1/45.0 | 0.999 |
| 2019CHRT30 | silver | 44 | 12 | 8.3/2.3/3.6 | 100.0/27.3/42.9 | 1.000 |
| 2020ABSRB85 | silver | 25 | 10 | 40.0/16.0/22.9 | 100.0/40.0/57.1 | 1.000 |
| 2020CanLII55888MBLB | silver | 16 | 0 | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 1.000 |
| 2020ONAFRAAT8 | silver | 238 | 38 | 73.7/11.8/20.3 | 89.5/14.3/24.6 | 1.000 |
| 2021NBOMB1 | silver | 84 | 15 | 40.0/7.1/12.1 | 66.7/11.9/20.2 | 1.000 |
| 2021NUIPC10 | silver | 74 | 9 | 0.0/0.0/0.0 | 100.0/12.2/21.7 | 1.000 |
| 2021NUIPC8 | silver | 50 | 0 | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 1.000 |
| 2022QCCS1670 | silver | 24 | 18 | 66.7/50.0/57.1 | 94.4/70.8/81.0 | 1.000 |
| 2023ABCA35 | silver | 44 | 11 | 54.5/13.6/21.8 | 100.0/25.0/40.0 | 1.000 |
| 2023FCA24 | silver | 258 | 84 | 50.0/16.3/24.6 | 100.0/32.6/49.1 | 0.999 |
| 2024ONCMT19 | silver | 66 | 28 | 7.1/3.0/4.3 | 100.0/42.4/59.6 | 1.000 |
| 2025MBKB113 | silver | 50 | 11 | 18.2/4.0/6.6 | 100.0/22.0/36.1 | 1.000 |
| 2025NBEUB2 | silver | 24 | 0 | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 1.000 |
| 2026ALRB5 | silver | 46 | 0 | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 1.000 |
| 1983CanLII2967L | test | 52 | 11 | 0.0/0.0/0.0 | 100.0/21.2/34.9 | 0.988 |
| 1993CanLII1889PESCTSD | test | 138 | 30 | 0.0/0.0/0.0 | 93.3/20.3/33.3 | 0.993 |
| 1993CanLII3004FCA | test | 549 | 233 | 13.3/5.6/7.9 | 96.6/41.0/57.5 | 0.987 |
| 1993NSSC71 | test | 71 | 19 | 21.1/5.6/8.9 | 100.0/26.8/42.2 | 0.994 |
| 1997CanLII11497 | test | 100 | 6 | 16.7/1.0/1.9 | 100.0/6.0/11.3 | 0.999 |
| 1999CanLII7320(ONCA) | test | 271 | 110 | 3.6/1.5/2.1 | 98.2/39.9/56.7 | 0.990 |
| 2001BCSC1342 | test | 95 | 14 | 21.4/3.2/5.5 | 100.0/14.7/25.7 | 0.995 |
| 2001CanLII21117 | test | 322 | 150 | 73.3/34.2/46.6 | 95.3/44.4/60.6 | 0.999 |
| 2002SCC33 | test | 328 | 144 | 11.1/4.9/6.8 | 100.0/43.9/61.0 | 0.993 |
| 2003MBCA71 | test | 49 | 20 | 5.0/2.0/2.9 | 100.0/40.8/58.0 | 0.994 |
| 2005NLTD57 | test | 80 | 16 | 18.8/3.8/6.2 | 100.0/20.0/33.3 | 0.996 |
| 2005QCCA437 | test | 116 | 24 | 58.3/12.1/20.0 | 100.0/20.7/34.3 | 0.997 |
| 2009BCHRT256 | test | 157 | 42 | 0.0/0.0/0.0 | 100.0/26.8/42.2 | 1.000 |
| 2009NUCA3 | test | 52 | 12 | 41.7/9.6/15.6 | 100.0/23.1/37.5 | 0.993 |
| 2009PECA12 | test | 87 | 50 | 28.0/16.1/20.4 | 100.0/57.5/73.0 | 0.997 |
| 2010SKQB255 | test | 102 | 62 | 43.5/26.5/32.9 | 98.4/59.8/74.4 | 0.995 |
| 2012SKCA24 | test | 44 | 19 | 31.6/13.6/19.0 | 100.0/43.2/60.3 | 1.000 |
| 2013ONAFRAAT18 | test | 40 | 2 | 0.0/0.0/0.0 | 100.0/5.0/9.5 | 1.000 |
| 2014CanLII36807PELRB | test | 243 | 85 | 82.4/28.8/42.7 | 100.0/35.0/51.8 | 1.000 |
| 2015NSSC25 | test | 296 | 110 | 68.2/25.3/36.9 | 100.0/37.2/54.2 | 0.999 |
| 2016QCCS1184 | test | 87 | 18 | 16.7/3.4/5.7 | 100.0/20.7/34.3 | 0.999 |
| 2017ABCA157 | test | 302 | 82 | 6.1/1.7/2.6 | 100.0/27.2/42.7 | 0.999 |
| 2018NSOIPC2 | test | 271 | 114 | 22.8/9.6/13.5 | 100.0/42.1/59.2 | 0.999 |
| 2019QCCA1492 | test | 1423 | 695 | 33.4/16.3/21.9 | 98.8/48.3/64.9 | 0.992 |
| 2020ABPLAB11 | test | 64 | 44 | 59.1/40.6/48.1 | 95.5/65.6/77.8 | 0.999 |
| 2020BCHPRB101 | test | 149 | 71 | 66.2/31.5/42.7 | 100.0/47.7/64.5 | 0.999 |
| 2020FCA63 | test | 162 | 108 | 20.4/13.6/16.3 | 94.4/63.0/75.6 | 0.996 |
| 2022BCSC135 | test | 1145 | 312 | 62.2/16.9/26.6 | 93.6/25.5/40.1 | 0.999 |
| 2022ONCMT32 | test | 162 | 1 | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 0.035 |
| 2024ABOHSAB19 | test | 52 | 32 | 71.9/44.2/54.8 | 93.8/57.7/71.4 | 1.000 |
| 2024BCSC764 | test | 232 | 101 | 54.5/23.7/33.0 | 90.1/39.2/54.7 | 1.000 |
| 2024NBKB203 | test | 244 | 70 | 4.3/1.2/1.9 | 65.7/18.9/29.3 | 0.999 |
| 2024NUCA9 | test | 115 | 49 | 36.7/15.7/22.0 | 93.9/40.0/56.1 | 0.997 |
| 2025ABAER7 | test | 144 | 5 | 0.0/0.0/0.0 | 100.0/3.5/6.7 | 1.000 |
| 2025ABKB246 | test | 58 | 8 | 0.0/0.0/0.0 | 100.0/13.8/24.2 | 0.999 |
| 2025MBKB136 | test | 47 | 21 | 4.8/2.1/2.9 | 71.4/31.9/44.1 | 0.998 |
| 2025NBEUB1 | test | 25 | 0 | 0.0/0.0/0.0 | 0.0/0.0/0.0 | 1.000 |
| 2026ABCA2L | test | 43 | 29 | 3.4/2.3/2.8 | 86.2/58.1/69.4 | 0.998 |
| 1994CanLII4528NLCA | train | 167 | 74 | 81.1/35.9/49.8 | 98.6/43.7/60.6 | 0.998 |
| 1997CanLII16226ONCA | train | 623 | 291 | 55.7/26.0/35.4 | 97.9/45.7/62.4 | 0.994 |
| 2008CSC9 | train | 484 | 210 | 32.4/14.0/19.6 | 99.5/43.2/60.2 | 0.994 |
| 2019SCC65 | train | 1443 | 521 | 10.0/3.6/5.3 | 99.8/36.0/53.0 | 0.995 |

`text align` = token-level similarity of the two renderings of the same document (1.0 = identical text).

## 8. Error examples

First document, **1982CanLII3042NBCA**:

* EXACT: 94 missed / 20 spurious — the reflex side is dominated by injected neutral citations (`1982 CanLII 19 (SCC)`), which can never equal a gold boundary because the gold text does not contain them.
* OVERLAP: 74 missed / 0 spurious.

**Gold mentions Reflex never touches (overlap FN)**

* `[legislation]` s. 19(1)
* `[secondary sources]` E. A. Driedger, The Construction of Statutes (1974), p. 138
* `[legislation]` s. 12 of the above Act
* `[legislation]` s. 19
* `[legislation]` s. 38 of the Act
* `[legislation]` s. 19.1 of the Act
* … and 68 more

**Reflex predictions with no gold counterpart (overlap FP)**

* none — every reflex prediction landed inside a gold mention

---
**Reading these numbers.**

* **EXACT is structurally harsh for Reflex**: it links only the *citation* part of a case reference (`2002 SCC 33`), while the gold parent span covers the whole reference (`MacDonald v. Smith, 2002 SCC 33`), and for pre-1999 citations the string it links does not exist in the gold text at all. Legislation boundaries, where Reflex and gold agree on the span, score far better.
* **OVERLAP is the fair detection metric** for this system: precision stays at ~98% (Reflex only links what it resolved) and recall ~35-40%, because gold also annotates short forms, fragments and secondary sources that a citator never links.
* **`secondary sources` recall is 0 by construction** — no reflex export contains a single non-`/laws/` `/doc/` link, so books, dictionaries and law reviews are simply out of Reflex's scope.
* On the same documents and the same scorer the LLM/rules baselines sit well above these numbers (see section 6), which is exactly the claim the paper makes: Reflex is a respectable high-precision extractor, but its recall ceiling keeps it clearly below the LLM results.
