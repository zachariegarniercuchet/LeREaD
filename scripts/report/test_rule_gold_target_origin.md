# Target-origin stratified linking evaluation — `test_gold_resolution_rule`

*Split: `test` — exact span matching — gold-span (oracle) resolution*

Documents evaluated: 38 | documents without a resolution mapping: 0

Gold targets: **404 synthetic** (1331 mentions) vs **1142 CanLII** (6578 mentions).

Gold URIs are classified by the gold `uri` attribute: `https://synthetic.leread.ca/gold/*` = synthetic, everything else = CanLII. TP/FN/FP are charged to the *gold* target's class (spurious spans without a gold counterpart are charged to the predicted URI's class).

## Overall (mention-level micro)

TP 6328 | FN 1581 | FP 1117 | **P 85.00 / R 80.01 / F1 82.43** | resolver-only accuracy 80.01 (7909 spans)

## By candidate origin (mention-level micro)

| origin | unique targets | mentions | TP | FN | FP | Precision % | Recall % | F1 % | Resolver acc % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic | 404 | 1331 | 1188 | 143 | 79 | 93.76 | 89.26 | 91.45 | 89.26 |
| canlii | 1142 | 6578 | 5140 | 1438 | 1038 | 83.20 | 78.14 | 80.59 | 78.14 |

**Synthetic - CanLII F1 difference: 10.87 points** (95% CI 0.22 to 18.23; paired cluster bootstrap over documents).
Resolver-accuracy difference: 11.12 points (95% CI 1.27 to 17.99).

Document-cluster bootstrap 95% CIs (primary; targets within a document are correlated):

| origin | F1 95% CI % | resolver accuracy 95% CI % |
| --- | --- | --- |
| synthetic | 86.48 - 95.56 | 84.19 - 93.72 |
| canlii | 72.63 - 90.97 | 70.47 - 88.30 |

## Impact on the headline number

Synthetic targets are 16.83% of real-uri mentions and 26.13% of unique cited targets in this split.

| setting | TP | FN | FP | Precision % | Recall % | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| All targets (headline) | 6328 | 1581 | 1117 | 85.00 | 80.01 | 82.43 |
| CanLII only (synthetic removed) | 5140 | 1438 | 1038 | 83.20 | 78.14 | 80.59 |
| Synthetic only | 1188 | 143 | 79 | 93.76 | 89.26 | 91.45 |
| Bound: all synthetic wrong | 5140 | 2769 | 1117 | 82.15 | 64.99 | 72.57 |

Headline minus CanLII-only F1: **1.84 points**.

## By label × candidate origin (mention-level micro)

| label | origin | mentions | TP | FN | FP | Precision % | Recall % | F1 % | Resolver acc % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Decision | synthetic | 445 | 401 | 44 | 39 | 91.14 | 90.11 | 90.62 | 90.11 |
| Decision | canlii | 3453 | 2940 | 513 | 214 | 93.21 | 85.14 | 89.00 | 85.14 |
| Legislation | synthetic | 498 | 456 | 42 | 19 | 96.00 | 91.57 | 93.73 | 91.57 |
| Legislation | canlii | 3058 | 2144 | 914 | 824 | 72.24 | 70.11 | 71.16 | 70.11 |
| Secondary sources | synthetic | 388 | 331 | 57 | 21 | 94.03 | 85.31 | 89.46 | 85.31 |
| Secondary sources | canlii | 67 | 56 | 11 | 0 | 100.00 | 83.58 | 91.06 | 83.58 |

## Per unique cited target (each target weighted equally)

| origin | targets | Mean accuracy % | Median accuracy % | % targets fully correct | % targets fully wrong |
| --- | --- | --- | --- | --- | --- |
| synthetic | 404 | 87.13 | 100.00 | 86.39 | 12.62 |
| canlii | 1142 | 84.60 | 100.00 | 83.80 | 14.54 |

## Near-exact string-match diagnostic

Built from each run's `*_resolution_mapping.json` (`query_text`) compared against the gold candidate row's `docTitle` + `autoCompletions`.

| origin | spans with features | mean target-token containment % | % with exact metadata match | % with exact title match | accuracy | exact match % | accuracy | no exact match % | accuracy | hard (no exact, containment<0.8) % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic | 1267 | 97.12 | 99.68 | 90.45 | 93.75 (1263) | 100.00 (4) | 100.00 (1) |
| canlii | 6178 | 44.14 | 80.74 | 65.90 | 90.48 (4988) | 52.69 (1190) | 52.41 (1183) |

Retrieval of the gold target (from the saved top-k lists):

| origin | spans with top-k | Top-1 hit % | Top-k hit % |
| --- | --- | --- | --- |
| synthetic | 1267 | 93.76 | 99.68 |
| canlii | 6178 | 83.20 | 94.56 |

_Note: top-k lists were saved by the original run (k = debug_topk), so `Top-k hit %` is a lower bound when k < 10._

