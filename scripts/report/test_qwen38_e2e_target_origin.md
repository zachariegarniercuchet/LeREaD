# Target-origin stratified linking evaluation — `test_qwen38-27b_coref_fs6_random_from-test_qwen38-27b_DEC_fs6_greedy_sentence_long-final`

*Split: `test` — exact span matching — end-to-end pipeline (extraction + coref + resolution)*

Documents evaluated: 38 | documents without a resolution mapping: 0

Gold targets: **404 synthetic** (1331 mentions) vs **1142 CanLII** (6578 mentions).

Gold URIs are classified by the gold `uri` attribute: `https://synthetic.leread.ca/gold/*` = synthetic, everything else = CanLII. TP/FN/FP are charged to the *gold* target's class (spurious spans without a gold counterpart are charged to the predicted URI's class).

## Overall (mention-level micro)

TP 3821 | FN 4088 | FP 2251 | **P 62.93 / R 48.31 / F1 54.66** | resolver-only accuracy 77.22 (4948 spans)

## By candidate origin (mention-level micro)

| origin | unique targets | mentions | TP | FN | FP | Precision % | Recall % | F1 % | Resolver acc % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic | 404 | 1331 | 559 | 772 | 313 | 64.11 | 42.00 | 50.75 | 81.13 |
| canlii | 1142 | 6578 | 3262 | 3316 | 1938 | 62.73 | 49.59 | 55.39 | 76.59 |

**Synthetic - CanLII F1 difference: -4.64 points** (95% CI -11.52 to 3.04; paired cluster bootstrap over documents).
Resolver-accuracy difference: 4.54 points (95% CI -3.24 to 14.93).

Document-cluster bootstrap 95% CIs (primary; targets within a document are correlated):

| origin | F1 95% CI % | resolver accuracy 95% CI % |
| --- | --- | --- |
| synthetic | 44.80 - 55.96 | 75.19 - 87.25 |
| canlii | 48.13 - 60.75 | 67.89 - 83.10 |

## Impact on the headline number

Synthetic targets are 16.83% of real-uri mentions and 26.13% of unique cited targets in this split.

| setting | TP | FN | FP | Precision % | Recall % | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| All targets (headline) | 3821 | 4088 | 2251 | 62.93 | 48.31 | 54.66 |
| CanLII only (synthetic removed) | 3262 | 3316 | 1938 | 62.73 | 49.59 | 55.39 |
| Synthetic only | 559 | 772 | 313 | 64.11 | 42.00 | 50.75 |
| Bound: all synthetic wrong | 3262 | 4647 | 2251 | 59.17 | 41.24 | 48.61 |

Headline minus CanLII-only F1: **-0.73 points**.

## By label × candidate origin (mention-level micro)

| label | origin | mentions | TP | FN | FP | Precision % | Recall % | F1 % | Resolver acc % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Decision | synthetic | 445 | 185 | 260 | 78 | 70.34 | 41.57 | 52.26 | 84.09 |
| Decision | canlii | 3453 | 1621 | 1832 | 745 | 68.51 | 46.94 | 55.71 | 85.54 |
| Legislation | synthetic | 498 | 215 | 283 | 147 | 59.39 | 43.17 | 50.00 | 77.06 |
| Legislation | canlii | 3058 | 1610 | 1448 | 1129 | 58.78 | 52.65 | 55.55 | 69.16 |
| Secondary sources | synthetic | 388 | 159 | 229 | 88 | 64.37 | 40.98 | 50.08 | 83.68 |
| Secondary sources | canlii | 67 | 31 | 36 | 64 | 32.63 | 46.27 | 38.27 | 86.11 |

## Per unique cited target (each target weighted equally)

| origin | targets | Mean accuracy % | Median accuracy % | % targets fully correct | % targets fully wrong |
| --- | --- | --- | --- | --- | --- |
| synthetic | 404 | 45.86 | 50.00 | 28.71 | 36.88 |
| canlii | 1142 | 49.65 | 50.00 | 29.42 | 32.92 |

## Near-exact string-match diagnostic

Built from each run's `*_resolution_mapping.json` (`query_text`) compared against the gold candidate row's `docTitle` + `autoCompletions`.

| origin | spans with features | mean target-token containment % | % with exact metadata match | % with exact title match | accuracy | exact match % | accuracy | no exact match % | accuracy | hard (no exact, containment<0.8) % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic | 146 | 70.24 | 59.59 | 49.32 | 88.51 (87) | 96.61 (59) | 96.61 (59) |
| canlii | 757 | 57.35 | 82.56 | 81.11 | 88.96 (625) | 83.33 (132) | 83.33 (132) |

Retrieval of the gold target (from the saved top-k lists):

| origin | spans with top-k | Top-1 hit % | Top-k hit % |
| --- | --- | --- | --- |
| synthetic | 146 | 97.95 | 98.63 |
| canlii | 757 | 87.71 | 96.83 |

_Note: top-k lists were saved by the original run (k = debug_topk), so `Top-k hit %` is a lower bound when k < 10._

Spurious system assertions (no gold counterpart, class of asserted URI): canlii=948, synthetic=185

