# Extraction error analysis

Gold: `private/dataset/Annotated`; matching = same label + same text + same context (eval L1).

## gemma4 — 33 docs evaluated (0 skipped)

### 1. Miss rate by mention type (parent) and component type (child)

| label | level | gold | sys | TP | miss rate (FN/gold) | FDR (FP/sys) | R | F1 |
|---|---|---|---|---|---|---|---|---|
| authors | child | 187 | 155 | 132 | 29.4% | 14.8% | 70.6% | 77.2% |
| citation | child | 2074 | 1439 | 1190 | 42.6% | 17.3% | 57.4% | 67.7% |
| decision | parent | 2790 | 2133 | 1697 | 39.2% | 20.4% | 60.8% | 68.9% |
| fragment | child | 2909 | 2587 | 2374 | 18.4% | 8.2% | 81.6% | 86.4% |
| legislation | parent | 2437 | 2378 | 1946 | 20.1% | 18.2% | 79.9% | 80.8% |
| secondary sources | parent | 257 | 274 | 172 | 33.1% | 37.2% | 66.9% | 64.8% |
| source | child | 147 | 143 | 107 | 27.2% | 25.2% | 72.8% | 73.8% |
| title | child | 4363 | 3960 | 3349 | 23.2% | 15.4% | 76.8% | 80.5% |
| unable to classify | child | 87 | 0 | 0 | 100.0% | 0.0% | 0.0% | 0.0% |

### 2. Boundary match -> correct mention type? (parents only)
- Gold parent mentions: 5484; with a system span on the exact same boundary: 3841 (70.0% of gold)
- Of those, correct mention type: 3814 (99.30%) — wrong type: 27 (0.70%)
- Confusion (gold x system, boundary-matched pairs):

| gold \ sys | decision | legislation | secondary sources | missed (no boundary) |
|---|---|---|---|---|
| decision | 1697 | 5 | 1 | 1087 |
| legislation | 0 | 1945 | 5 | 487 |
| secondary sources | 16 | 0 | 172 | 69 |
| spurious (no boundary) | 420 | 428 | 96 | - |

### 3. Miss rate by mention pattern (parent level)
_Alignment mismatches mention->span: 0/5484_

| mention | pattern | gold | miss rate |
|---|---|---|---|
| decision | title | 924 | 35.4% |
| legislation | title | 924 | 26.9% |
| legislation | fragment | 585 | 15.9% |
| legislation | fragment -> title | 568 | 3.0% |
| decision | title -> citation | 434 | 24.4% |
| decision | fragment | 312 | 84.9% |
| decision | title -> fragment | 223 | 13.5% |
| decision | title -> citation -> fragment | 178 | 14.6% |
| decision | title -> citation -> citation | 157 | 36.9% |
| decision | citation | 90 | 46.7% |
| decision | title -> citation -> fragment -> citation | 67 | 3.0% |
| secondary sources | authors -> title -> source -> fragment | 60 | 6.7% |
| legislation | title -> fragment | 59 | 37.3% |
| decision | citation -> citation | 47 | 70.2% |
| legislation | title -> citation | 45 | 13.3% |
| legislation | fragment -> fragment -> title | 41 | 7.3% |
| legislation | fragment -> fragment | 39 | 41.0% |
| decision | title -> citation -> citation -> citation | 33 | 90.9% |
| decision | fragment -> fragment | 32 | 90.6% |
| legislation | fragment -> title -> citation | 31 | 16.1% |

## gpt-5.2 — 34 docs evaluated (0 skipped)

### 1. Miss rate by mention type (parent) and component type (child)

| label | level | gold | sys | TP | miss rate (FN/gold) | FDR (FP/sys) | R | F1 |
|---|---|---|---|---|---|---|---|---|
| authors | child | 292 | 274 | 222 | 24.0% | 19.0% | 76.0% | 78.4% |
| citation | child | 2677 | 2563 | 2377 | 11.2% | 7.3% | 88.8% | 90.7% |
| decision | parent | 3401 | 3652 | 2635 | 22.5% | 27.8% | 77.5% | 74.7% |
| fragment | child | 3900 | 3980 | 3419 | 12.3% | 14.1% | 87.7% | 86.8% |
| legislation | parent | 3119 | 3315 | 2612 | 16.3% | 21.2% | 83.7% | 81.2% |
| secondary sources | parent | 387 | 771 | 273 | 29.5% | 64.6% | 70.5% | 47.2% |
| source | child | 233 | 260 | 177 | 24.0% | 31.9% | 76.0% | 71.8% |
| title | child | 5584 | 6043 | 4900 | 12.2% | 18.9% | 87.8% | 84.3% |
| unable to classify | child | 87 | 0 | 0 | 100.0% | 0.0% | 0.0% | 0.0% |

### 2. Boundary match -> correct mention type? (parents only)
- Gold parent mentions: 6907; with a system span on the exact same boundary: 5554 (80.4% of gold)
- Of those, correct mention type: 5519 (99.37%) — wrong type: 35 (0.63%)
- Confusion (gold x system, boundary-matched pairs):

| gold \ sys | decision | legislation | secondary sources | missed (no boundary) |
|---|---|---|---|---|
| decision | 2635 | 4 | 6 | 756 |
| legislation | 0 | 2611 | 14 | 494 |
| secondary sources | 11 | 0 | 273 | 103 |
| spurious (no boundary) | 1006 | 700 | 478 | - |

### 3. Miss rate by mention pattern (parent level)
_Alignment mismatches mention->span: 0/6907_

| mention | pattern | gold | miss rate |
|---|---|---|---|
| decision | title | 1179 | 18.2% |
| legislation | title | 1046 | 27.1% |
| legislation | fragment | 727 | 9.1% |
| legislation | fragment -> title | 704 | 2.8% |
| decision | title -> citation | 492 | 4.7% |
| decision | title -> citation -> fragment | 413 | 6.1% |
| decision | fragment | 331 | 76.4% |
| decision | title -> fragment | 227 | 9.3% |
| decision | title -> citation -> citation | 157 | 15.3% |
| legislation | title -> citation | 130 | 8.5% |
| secondary sources | authors -> title -> source -> fragment | 101 | 8.9% |
| legislation | title -> citation -> fragment | 100 | 5.0% |
| decision | citation | 92 | 30.4% |
| legislation | fragment -> title -> citation | 82 | 14.6% |
| decision | title -> citation -> fragment -> citation | 72 | 2.8% |
| legislation | title -> fragment | 60 | 11.7% |
| legislation | fragment -> fragment | 51 | 45.1% |
| legislation | fragment -> fragment -> title | 49 | 20.4% |
| decision | citation -> citation | 47 | 31.9% |
| secondary sources | authors | 45 | 46.7% |

## muse — 34 docs evaluated (0 skipped)

### 1. Miss rate by mention type (parent) and component type (child)

| label | level | gold | sys | TP | miss rate (FN/gold) | FDR (FP/sys) | R | F1 |
|---|---|---|---|---|---|---|---|---|
| authors | child | 292 | 165 | 148 | 49.3% | 10.3% | 50.7% | 64.8% |
| citation | child | 2677 | 1391 | 1227 | 54.2% | 11.8% | 45.8% | 60.3% |
| decision | parent | 3401 | 1818 | 1505 | 55.7% | 17.2% | 44.3% | 57.7% |
| fragment | child | 3900 | 2414 | 2245 | 42.4% | 7.0% | 57.6% | 71.1% |
| legislation | parent | 3119 | 2169 | 1910 | 38.8% | 11.9% | 61.2% | 72.2% |
| secondary sources | parent | 387 | 223 | 162 | 58.1% | 27.4% | 41.9% | 53.1% |
| source | child | 233 | 145 | 111 | 52.4% | 23.4% | 47.6% | 58.7% |
| title | child | 5584 | 3413 | 3112 | 44.3% | 8.8% | 55.7% | 69.2% |
| unable to classify | child | 87 | 0 | 0 | 100.0% | 0.0% | 0.0% | 0.0% |

### 2. Boundary match -> correct mention type? (parents only)
- Gold parent mentions: 6907; with a system span on the exact same boundary: 3584 (51.9% of gold)
- Of those, correct mention type: 3577 (99.80%) — wrong type: 7 (0.20%)
- Confusion (gold x system, boundary-matched pairs):

| gold \ sys | decision | legislation | secondary sources | missed (no boundary) |
|---|---|---|---|---|
| decision | 1505 | 0 | 0 | 1896 |
| legislation | 0 | 1910 | 1 | 1208 |
| secondary sources | 6 | 0 | 162 | 219 |
| spurious (no boundary) | 307 | 259 | 60 | - |

### 3. Miss rate by mention pattern (parent level)
_Alignment mismatches mention->span: 0/6907_

| mention | pattern | gold | miss rate |
|---|---|---|---|
| decision | title | 1179 | 46.4% |
| legislation | title | 1046 | 54.9% |
| legislation | fragment | 727 | 32.6% |
| legislation | fragment -> title | 704 | 19.7% |
| decision | title -> citation | 492 | 47.0% |
| decision | title -> citation -> fragment | 413 | 48.4% |
| decision | fragment | 331 | 98.8% |
| decision | title -> fragment | 227 | 44.9% |
| decision | title -> citation -> citation | 157 | 55.4% |
| legislation | title -> citation | 130 | 28.5% |
| secondary sources | authors -> title -> source -> fragment | 101 | 32.7% |
| legislation | title -> citation -> fragment | 100 | 38.0% |
| decision | citation | 92 | 75.0% |
| legislation | fragment -> title -> citation | 82 | 36.6% |
| decision | title -> citation -> fragment -> citation | 72 | 26.4% |
| legislation | title -> fragment | 60 | 55.0% |
| legislation | fragment -> fragment | 51 | 56.9% |
| legislation | fragment -> fragment -> title | 49 | 26.5% |
| decision | citation -> citation | 47 | 72.3% |
| secondary sources | authors | 45 | 84.4% |

## qwen38-27b — 34 docs evaluated (0 skipped)

### 1. Miss rate by mention type (parent) and component type (child)

| label | level | gold | sys | TP | miss rate (FN/gold) | FDR (FP/sys) | R | F1 |
|---|---|---|---|---|---|---|---|---|
| authors | child | 292 | 185 | 163 | 44.2% | 11.9% | 55.8% | 68.3% |
| citation | child | 2677 | 1517 | 1364 | 49.0% | 10.1% | 51.0% | 65.0% |
| decision | parent | 3401 | 2312 | 1833 | 46.1% | 20.7% | 53.9% | 64.2% |
| fragment | child | 3900 | 2972 | 2671 | 31.5% | 10.1% | 68.5% | 77.7% |
| legislation | parent | 3119 | 2730 | 2272 | 27.2% | 16.8% | 72.8% | 77.7% |
| secondary sources | parent | 387 | 304 | 197 | 49.1% | 35.2% | 50.9% | 57.0% |
| source | child | 233 | 157 | 126 | 45.9% | 19.7% | 54.1% | 64.6% |
| title | child | 5584 | 4409 | 3781 | 32.3% | 14.2% | 67.7% | 75.7% |
| unable to classify | child | 87 | 0 | 0 | 100.0% | 0.0% | 0.0% | 0.0% |

### 2. Boundary match -> correct mention type? (parents only)
- Gold parent mentions: 6907; with a system span on the exact same boundary: 4317 (62.5% of gold)
- Of those, correct mention type: 4302 (99.65%) — wrong type: 15 (0.35%)
- Confusion (gold x system, boundary-matched pairs):

| gold \ sys | decision | legislation | secondary sources | missed (no boundary) |
|---|---|---|---|---|
| decision | 1833 | 4 | 2 | 1562 |
| legislation | 0 | 2272 | 0 | 847 |
| secondary sources | 9 | 0 | 197 | 181 |
| spurious (no boundary) | 470 | 454 | 105 | - |

### 3. Miss rate by mention pattern (parent level)
_Alignment mismatches mention->span: 0/6907_

| mention | pattern | gold | miss rate |
|---|---|---|---|
| decision | title | 1179 | 33.2% |
| legislation | title | 1046 | 33.1% |
| legislation | fragment | 727 | 24.2% |
| legislation | fragment -> title | 704 | 12.1% |
| decision | title -> citation | 492 | 40.0% |
| decision | title -> citation -> fragment | 413 | 40.9% |
| decision | fragment | 331 | 86.4% |
| decision | title -> fragment | 227 | 35.2% |
| decision | title -> citation -> citation | 157 | 51.6% |
| legislation | title -> citation | 130 | 30.8% |
| secondary sources | authors -> title -> source -> fragment | 101 | 24.8% |
| legislation | title -> citation -> fragment | 100 | 39.0% |
| decision | citation | 92 | 67.4% |
| legislation | fragment -> title -> citation | 82 | 32.9% |
| decision | title -> citation -> fragment -> citation | 72 | 30.6% |
| legislation | title -> fragment | 60 | 43.3% |
| legislation | fragment -> fragment | 51 | 41.2% |
| legislation | fragment -> fragment -> title | 49 | 14.3% |
| decision | citation -> citation | 47 | 83.0% |
| secondary sources | authors | 45 | 68.9% |

## saul-54b — 34 docs evaluated (0 skipped)

### 1. Miss rate by mention type (parent) and component type (child)

| label | level | gold | sys | TP | miss rate (FN/gold) | FDR (FP/sys) | R | F1 |
|---|---|---|---|---|---|---|---|---|
| authors | child | 292 | 7 | 6 | 97.9% | 14.3% | 2.1% | 4.0% |
| citation | child | 2677 | 202 | 144 | 94.6% | 28.7% | 5.4% | 10.0% |
| decision | parent | 3401 | 393 | 191 | 94.4% | 51.4% | 5.6% | 10.1% |
| fragment | child | 3900 | 201 | 116 | 97.0% | 42.3% | 3.0% | 5.7% |
| legislation | parent | 3119 | 122 | 15 | 99.5% | 87.7% | 0.5% | 0.9% |
| secondary sources | parent | 387 | 34 | 8 | 97.9% | 76.5% | 2.1% | 3.8% |
| source | child | 233 | 3 | 2 | 99.1% | 33.3% | 0.9% | 1.7% |
| title | child | 5584 | 325 | 248 | 95.6% | 23.7% | 4.4% | 8.4% |
| unable to classify | child | 87 | 0 | 0 | 100.0% | 0.0% | 0.0% | 0.0% |

### 2. Boundary match -> correct mention type? (parents only)
- Gold parent mentions: 6907; with a system span on the exact same boundary: 217 (3.1% of gold)
- Of those, correct mention type: 214 (98.62%) — wrong type: 3 (1.38%)
- Confusion (gold x system, boundary-matched pairs):

| gold \ sys | decision | legislation | secondary sources | missed (no boundary) |
|---|---|---|---|---|
| decision | 191 | 1 | 0 | 3209 |
| legislation | 0 | 15 | 0 | 3104 |
| secondary sources | 2 | 0 | 8 | 377 |
| spurious (no boundary) | 200 | 106 | 26 | - |

### 3. Miss rate by mention pattern (parent level)
_Alignment mismatches mention->span: 0/6907_

| mention | pattern | gold | miss rate |
|---|---|---|---|
| decision | title | 1179 | 93.9% |
| legislation | title | 1046 | 99.8% |
| legislation | fragment | 727 | 99.2% |
| legislation | fragment -> title | 704 | 99.1% |
| decision | title -> citation | 492 | 94.5% |
| decision | title -> citation -> fragment | 413 | 94.7% |
| decision | fragment | 331 | 100.0% |
| decision | title -> fragment | 227 | 93.0% |
| decision | title -> citation -> citation | 157 | 91.1% |
| legislation | title -> citation | 130 | 100.0% |
| secondary sources | authors -> title -> source -> fragment | 101 | 99.0% |
| legislation | title -> citation -> fragment | 100 | 100.0% |
| decision | citation | 92 | 97.8% |
| legislation | fragment -> title -> citation | 82 | 100.0% |
| decision | title -> citation -> fragment -> citation | 72 | 87.5% |
| legislation | title -> fragment | 60 | 98.3% |
| legislation | fragment -> fragment | 51 | 100.0% |
| legislation | fragment -> fragment -> title | 49 | 100.0% |
| decision | citation -> citation | 47 | 97.9% |
| secondary sources | authors | 45 | 100.0% |

## Cross-model comparison

See `comparison_miss_rate_by_label.csv` and `comparison_boundary_type.csv`.

- gemma4: boundary recall 70.0% (N=5484), type accuracy on matched 99.30%
- gpt-5.2: boundary recall 80.4% (N=6907), type accuracy on matched 99.37%
- muse: boundary recall 51.9% (N=6907), type accuracy on matched 99.80%
- qwen38-27b: boundary recall 62.5% (N=6907), type accuracy on matched 99.65%
- saul-54b: boundary recall 3.1% (N=6907), type accuracy on matched 98.62%