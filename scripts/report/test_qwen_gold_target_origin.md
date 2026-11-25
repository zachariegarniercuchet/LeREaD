# Target-origin stratified linking evaluation — `test_gold_resolution_qwen3-8b`

*Split: `test` — exact span matching — gold-span (oracle) resolution*

Documents evaluated: 38 | documents without a resolution mapping: 0

Gold targets: **404 synthetic** (1331 mentions) vs **1142 CanLII** (6578 mentions).

Gold URIs are classified by the gold `uri` attribute: `https://synthetic.leread.ca/gold/*` = synthetic, everything else = CanLII. TP/FN/FP are charged to the *gold* target's class (spurious spans without a gold counterpart are charged to the predicted URI's class).

## Overall (mention-level micro)

TP 6813 | FN 1096 | FP 959 | **P 87.66 / R 86.14 / F1 86.89** | resolver-only accuracy 86.14 (7909 spans)

## By candidate origin (mention-level micro)

| origin | unique targets | mentions | TP | FN | FP | Precision % | Recall % | F1 % | Resolver acc % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic | 404 | 1331 | 1165 | 166 | 125 | 90.31 | 87.53 | 88.90 | 87.53 |
| canlii | 1142 | 6578 | 5648 | 930 | 834 | 87.13 | 85.86 | 86.49 | 85.86 |

**Synthetic - CanLII F1 difference: 2.40 points** (95% CI -7.98 to 9.86; paired cluster bootstrap over documents).
Resolver-accuracy difference: 1.67 points (95% CI -8.09 to 9.32).

Document-cluster bootstrap 95% CIs (primary; targets within a document are correlated):

| origin | F1 95% CI % | resolver accuracy 95% CI % |
| --- | --- | --- |
| synthetic | 81.16 - 94.51 | 80.19 - 93.56 |
| canlii | 82.80 - 91.35 | 82.03 - 90.61 |

## Impact on the headline number

Synthetic targets are 16.83% of real-uri mentions and 26.13% of unique cited targets in this split.

| setting | TP | FN | FP | Precision % | Recall % | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| All targets (headline) | 6813 | 1096 | 959 | 87.66 | 86.14 | 86.89 |
| CanLII only (synthetic removed) | 5648 | 930 | 834 | 87.13 | 85.86 | 86.49 |
| Synthetic only | 1165 | 166 | 125 | 90.31 | 87.53 | 88.90 |
| Bound: all synthetic wrong | 5648 | 2261 | 959 | 85.49 | 71.41 | 77.82 |

Headline minus CanLII-only F1: **0.40 points**.

## By label × candidate origin (mention-level micro)

| label | origin | mentions | TP | FN | FP | Precision % | Recall % | F1 % | Resolver acc % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Decision | synthetic | 445 | 407 | 38 | 37 | 91.67 | 91.46 | 91.56 | 91.46 |
| Decision | canlii | 3453 | 2865 | 588 | 519 | 84.66 | 82.97 | 83.81 | 82.97 |
| Legislation | synthetic | 498 | 419 | 79 | 70 | 85.69 | 84.14 | 84.90 | 84.14 |
| Legislation | canlii | 3058 | 2727 | 331 | 309 | 89.82 | 89.18 | 89.50 | 89.18 |
| Secondary sources | synthetic | 388 | 339 | 49 | 18 | 94.96 | 87.37 | 91.01 | 87.37 |
| Secondary sources | canlii | 67 | 56 | 11 | 6 | 90.32 | 83.58 | 86.82 | 83.58 |

## Per unique cited target (each target weighted equally)

| origin | targets | Mean accuracy % | Median accuracy % | % targets fully correct | % targets fully wrong |
| --- | --- | --- | --- | --- | --- |
| synthetic | 404 | 84.28 | 100.00 | 83.42 | 15.35 |
| canlii | 1142 | 88.60 | 100.00 | 87.74 | 10.68 |

## Near-exact string-match diagnostic

Built from each run's `*_resolution_mapping.json` (`query_text`) compared against the gold candidate row's `docTitle` + `autoCompletions`.

| origin | spans with features | mean target-token containment % | % with exact metadata match | % with exact title match | accuracy | exact match % | accuracy | no exact match % | accuracy | hard (no exact, containment<0.8) % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic | 1290 | 96.04 | 98.14 | 88.84 | 91.71 (1266) | 16.67 (24) | 6.67 (15) |
| canlii | 6482 | 42.86 | 77.28 | 62.82 | 90.98 (5009) | 74.07 (1473) | 73.94 (1466) |

Retrieval of the gold target (from the saved top-k lists):

| origin | spans with top-k | Top-1 hit % | Top-k hit % |
| --- | --- | --- | --- |
| synthetic | 1290 | 90.31 | 95.04 |
| canlii | 6482 | 87.13 | 92.90 |

_Note: top-k lists were saved by the original run (k = debug_topk), so `Top-k hit %` is a lower bound when k < 10._

