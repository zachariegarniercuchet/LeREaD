# Extraction error analysis

Suite: `baseline`; gold: `data\annotated\test`; matching = same label + same text + same context (eval L1).

## baseline-rules — 38 docs evaluated (0 skipped)

### 1. Miss rate by mention type (parent) and component type (child)

| label | level | gold | sys | TP | miss rate (FN/gold) | FDR (FP/sys) | R | F1 |
|---|---|---|---|---|---|---|---|---|
| authors | child | 341 | 663 | 61 | 82.1% | 90.8% | 17.9% | 12.2% |
| citation | child | 2967 | 1739 | 1451 | 51.1% | 16.6% | 48.9% | 61.7% |
| decision | parent | 3901 | 3726 | 1211 | 69.0% | 67.5% | 31.0% | 31.8% |
| fragment | child | 4547 | 3406 | 2769 | 39.1% | 18.7% | 60.9% | 69.6% |
| legislation | parent | 3560 | 2795 | 1241 | 65.1% | 55.6% | 34.9% | 39.1% |
| secondary sources | parent | 456 | 938 | 33 | 92.8% | 96.5% | 7.2% | 4.7% |
| source | child | 270 | 29 | 15 | 94.4% | 48.3% | 5.6% | 10.0% |
| title | child | 6261 | 4272 | 2341 | 62.6% | 45.2% | 37.4% | 44.5% |
| unable to classify | child | 88 | 0 | 0 | 100.0% | 0.0% | 0.0% | 0.0% |

### 2. Boundary match -> correct mention type? (parents only)
- Gold parent mentions: 7917; with a system span on the exact same boundary: 2597 (32.8% of gold)
- Of those, correct mention type: 2476 (95.34%) — wrong type: 121 (4.66%)
- Confusion (gold x system, boundary-matched pairs):

| gold \ sys | decision | legislation | secondary sources | missed (no boundary) |
|---|---|---|---|---|
| decision | 1202 | 68 | 24 | 2607 |
| legislation | 0 | 1241 | 3 | 2316 |
| secondary sources | 21 | 5 | 33 | 397 |
| spurious (no boundary) | 2503 | 1481 | 878 | - |

### 3. Miss rate by mention pattern (parent level)
_Alignment mismatches mention->span: 0/7917_

| mention | pattern | gold | miss rate |
|---|---|---|---|
| decision | title | 1279 | 78.1% |
| legislation | title | 1157 | 84.7% |
| legislation | fragment | 870 | 35.3% |
| legislation | fragment -> title | 821 | 51.6% |
| decision | title -> citation | 603 | 58.5% |
| decision | title -> citation -> fragment | 476 | 65.1% |
| decision | fragment | 442 | 41.4% |
| decision | title -> fragment | 290 | 60.3% |
| decision | title -> citation -> citation | 170 | 71.8% |
| legislation | title -> citation | 147 | 87.1% |
| secondary sources | authors -> title -> source -> fragment | 119 | 95.8% |
| decision | citation | 104 | 53.8% |
| legislation | title -> citation -> fragment | 103 | 87.4% |
| legislation | fragment -> title -> citation | 89 | 71.9% |
| decision | title -> citation -> fragment -> citation | 72 | 87.5% |
| legislation | fragment -> fragment | 60 | 81.7% |
| legislation | title -> fragment | 60 | 78.3% |
| legislation | fragment -> fragment -> title | 58 | 89.7% |
| secondary sources | authors | 56 | 94.6% |
| secondary sources | authors -> title -> source | 51 | 88.2% |

## Cross-model comparison

See `comparison_miss_rate_by_label.csv` and `comparison_boundary_type.csv`.

- baseline-rules: boundary recall 32.8% (N=7917), type accuracy on matched 95.34%