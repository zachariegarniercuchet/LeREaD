# Extraction error analysis

Suite: `aafinal`; gold: `data\annotated\test`; matching = same label + same text + same context (eval L1).

## gemma4 — 38 docs evaluated (0 skipped)

### 1. Miss rate by mention type (parent) and component type (child)

| label | level | gold | sys | TP | miss rate (FN/gold) | FDR (FP/sys) | R | F1 |
|---|---|---|---|---|---|---|---|---|
| authors | child | 341 | 305 | 261 | 23.5% | 14.4% | 76.5% | 80.8% |
| citation | child | 2967 | 2282 | 1944 | 34.5% | 14.8% | 65.5% | 74.1% |
| decision | parent | 3901 | 3111 | 2598 | 33.4% | 16.5% | 66.6% | 74.1% |
| fragment | child | 4547 | 4205 | 3824 | 15.9% | 9.1% | 84.1% | 87.4% |
| legislation | parent | 3560 | 3485 | 2906 | 18.4% | 16.6% | 81.6% | 82.5% |
| secondary sources | parent | 456 | 466 | 305 | 33.1% | 34.5% | 66.9% | 66.2% |
| source | child | 270 | 273 | 203 | 24.8% | 25.6% | 75.2% | 74.8% |
| title | child | 6261 | 5820 | 5025 | 19.7% | 13.7% | 80.3% | 83.2% |
| unable to classify | child | 88 | 0 | 0 | 100.0% | 0.0% | 0.0% | 0.0% |

### 2. Boundary match -> correct mention type? (parents only)
- Gold parent mentions: 7917; with a system span on the exact same boundary: 5838 (73.7% of gold)
- Of those, correct mention type: 5808 (99.49%) — wrong type: 30 (0.51%)
- Confusion (gold x system, boundary-matched pairs):

| gold \ sys | decision | legislation | secondary sources | missed (no boundary) |
|---|---|---|---|---|
| decision | 2598 | 5 | 1 | 1297 |
| legislation | 0 | 2905 | 6 | 649 |
| secondary sources | 18 | 0 | 305 | 133 |
| spurious (no boundary) | 495 | 575 | 154 | - |

### 3. Miss rate by mention pattern (parent level)
_Alignment mismatches mention->span: 0/7917_

| mention | pattern | gold | miss rate |
|---|---|---|---|
| decision | title | 1279 | 28.1% |
| legislation | title | 1157 | 25.4% |
| legislation | fragment | 870 | 16.7% |
| legislation | fragment -> title | 821 | 4.4% |
| decision | title -> citation | 603 | 27.7% |
| decision | title -> citation -> fragment | 476 | 6.7% |
| decision | fragment | 442 | 78.5% |
| decision | title -> fragment | 290 | 11.7% |
| decision | title -> citation -> citation | 170 | 37.6% |
| legislation | title -> citation | 147 | 14.3% |
| secondary sources | authors -> title -> source -> fragment | 119 | 8.4% |
| decision | citation | 104 | 45.2% |
| legislation | title -> citation -> fragment | 103 | 12.6% |
| legislation | fragment -> title -> citation | 89 | 13.5% |
| decision | title -> citation -> fragment -> citation | 72 | 2.8% |
| legislation | fragment -> fragment | 60 | 38.3% |
| legislation | title -> fragment | 60 | 36.7% |
| legislation | fragment -> fragment -> title | 58 | 5.2% |
| secondary sources | authors | 56 | 66.1% |
| secondary sources | authors -> title -> source | 51 | 19.6% |

## gpt5.2 — 38 docs evaluated (0 skipped)

### 1. Miss rate by mention type (parent) and component type (child)

| label | level | gold | sys | TP | miss rate (FN/gold) | FDR (FP/sys) | R | F1 |
|---|---|---|---|---|---|---|---|---|
| authors | child | 341 | 318 | 262 | 23.2% | 17.6% | 76.8% | 79.5% |
| citation | child | 2967 | 2856 | 2620 | 11.7% | 8.3% | 88.3% | 90.0% |
| decision | parent | 3901 | 4140 | 3038 | 22.1% | 26.6% | 77.9% | 75.6% |
| fragment | child | 4547 | 4599 | 3956 | 13.0% | 14.0% | 87.0% | 86.5% |
| legislation | parent | 3560 | 3771 | 2993 | 15.9% | 20.6% | 84.1% | 81.7% |
| secondary sources | parent | 456 | 853 | 323 | 29.2% | 62.1% | 70.8% | 49.4% |
| source | child | 270 | 300 | 210 | 22.2% | 30.0% | 77.8% | 73.7% |
| title | child | 6261 | 6792 | 5511 | 12.0% | 18.9% | 88.0% | 84.4% |
| unable to classify | child | 88 | 0 | 0 | 100.0% | 0.0% | 0.0% | 0.0% |

### 2. Boundary match -> correct mention type? (parents only)
- Gold parent mentions: 7917; with a system span on the exact same boundary: 6396 (80.8% of gold)
- Of those, correct mention type: 6353 (99.33%) — wrong type: 43 (0.67%)
- Confusion (gold x system, boundary-matched pairs):

| gold \ sys | decision | legislation | secondary sources | missed (no boundary) |
|---|---|---|---|---|
| decision | 3038 | 4 | 10 | 849 |
| legislation | 0 | 2992 | 15 | 553 |
| secondary sources | 14 | 0 | 323 | 119 |
| spurious (no boundary) | 1088 | 775 | 505 | - |

### 3. Miss rate by mention pattern (parent level)
_Alignment mismatches mention->span: 0/7917_

| mention | pattern | gold | miss rate |
|---|---|---|---|
| decision | title | 1279 | 17.2% |
| legislation | title | 1157 | 26.1% |
| legislation | fragment | 870 | 9.5% |
| legislation | fragment -> title | 821 | 3.7% |
| decision | title -> citation | 603 | 5.0% |
| decision | title -> citation -> fragment | 476 | 5.7% |
| decision | fragment | 442 | 71.3% |
| decision | title -> fragment | 290 | 9.0% |
| decision | title -> citation -> citation | 170 | 14.7% |
| legislation | title -> citation | 147 | 8.2% |
| secondary sources | authors -> title -> source -> fragment | 119 | 8.4% |
| decision | citation | 104 | 30.8% |
| legislation | title -> citation -> fragment | 103 | 5.8% |
| legislation | fragment -> title -> citation | 89 | 14.6% |
| decision | title -> citation -> fragment -> citation | 72 | 2.8% |
| legislation | fragment -> fragment | 60 | 45.0% |
| legislation | title -> fragment | 60 | 11.7% |
| legislation | fragment -> fragment -> title | 58 | 17.2% |
| secondary sources | authors | 56 | 48.2% |
| secondary sources | authors -> title -> source | 51 | 21.6% |

## muse — 38 docs evaluated (0 skipped)

### 1. Miss rate by mention type (parent) and component type (child)

| label | level | gold | sys | TP | miss rate (FN/gold) | FDR (FP/sys) | R | F1 |
|---|---|---|---|---|---|---|---|---|
| authors | child | 341 | 193 | 175 | 48.7% | 9.3% | 51.3% | 65.5% |
| citation | child | 2967 | 1560 | 1369 | 53.9% | 12.2% | 46.1% | 60.5% |
| decision | parent | 3901 | 2103 | 1752 | 55.1% | 16.7% | 44.9% | 58.4% |
| fragment | child | 4547 | 2835 | 2628 | 42.2% | 7.3% | 57.8% | 71.2% |
| legislation | parent | 3560 | 2498 | 2201 | 38.2% | 11.9% | 61.8% | 72.7% |
| secondary sources | parent | 456 | 264 | 193 | 57.7% | 26.9% | 42.3% | 53.6% |
| source | child | 270 | 176 | 132 | 51.1% | 25.0% | 48.9% | 59.2% |
| title | child | 6261 | 3904 | 3547 | 43.3% | 9.1% | 56.7% | 69.8% |
| unable to classify | child | 88 | 0 | 0 | 100.0% | 0.0% | 0.0% | 0.0% |

### 2. Boundary match -> correct mention type? (parents only)
- Gold parent mentions: 7917; with a system span on the exact same boundary: 4156 (52.5% of gold)
- Of those, correct mention type: 4146 (99.76%) — wrong type: 10 (0.24%)
- Confusion (gold x system, boundary-matched pairs):

| gold \ sys | decision | legislation | secondary sources | missed (no boundary) |
|---|---|---|---|---|
| decision | 1752 | 0 | 0 | 2149 |
| legislation | 0 | 2201 | 1 | 1358 |
| secondary sources | 9 | 0 | 193 | 254 |
| spurious (no boundary) | 342 | 297 | 70 | - |

### 3. Miss rate by mention pattern (parent level)
_Alignment mismatches mention->span: 0/7917_

| mention | pattern | gold | miss rate |
|---|---|---|---|
| decision | title | 1279 | 44.2% |
| legislation | title | 1157 | 53.1% |
| legislation | fragment | 870 | 32.3% |
| legislation | fragment -> title | 821 | 21.6% |
| decision | title -> citation | 603 | 49.8% |
| decision | title -> citation -> fragment | 476 | 45.4% |
| decision | fragment | 442 | 98.0% |
| decision | title -> fragment | 290 | 40.0% |
| decision | title -> citation -> citation | 170 | 54.7% |
| legislation | title -> citation | 147 | 27.9% |
| secondary sources | authors -> title -> source -> fragment | 119 | 33.6% |
| decision | citation | 104 | 73.1% |
| legislation | title -> citation -> fragment | 103 | 37.9% |
| legislation | fragment -> title -> citation | 89 | 37.1% |
| decision | title -> citation -> fragment -> citation | 72 | 26.4% |
| legislation | fragment -> fragment | 60 | 55.0% |
| legislation | title -> fragment | 60 | 55.0% |
| legislation | fragment -> fragment -> title | 58 | 25.9% |
| secondary sources | authors | 56 | 82.1% |
| secondary sources | authors -> title -> source | 51 | 56.9% |

## qwen38 — 38 docs evaluated (0 skipped)

### 1. Miss rate by mention type (parent) and component type (child)

| label | level | gold | sys | TP | miss rate (FN/gold) | FDR (FP/sys) | R | F1 |
|---|---|---|---|---|---|---|---|---|
| authors | child | 341 | 214 | 189 | 44.6% | 11.7% | 55.4% | 68.1% |
| citation | child | 2967 | 1696 | 1506 | 49.2% | 11.2% | 50.8% | 64.6% |
| decision | parent | 3901 | 2641 | 2117 | 45.7% | 19.8% | 54.3% | 64.7% |
| fragment | child | 4547 | 3450 | 3107 | 31.7% | 9.9% | 68.3% | 77.7% |
| legislation | parent | 3560 | 3111 | 2607 | 26.8% | 16.2% | 73.2% | 78.2% |
| secondary sources | parent | 456 | 345 | 226 | 50.4% | 34.5% | 49.6% | 56.4% |
| source | child | 270 | 184 | 147 | 45.6% | 20.1% | 54.4% | 64.8% |
| title | child | 6261 | 4960 | 4266 | 31.9% | 14.0% | 68.1% | 76.0% |
| unable to classify | child | 88 | 0 | 0 | 100.0% | 0.0% | 0.0% | 0.0% |

### 2. Boundary match -> correct mention type? (parents only)
- Gold parent mentions: 7917; with a system span on the exact same boundary: 4971 (62.8% of gold)
- Of those, correct mention type: 4950 (99.58%) — wrong type: 21 (0.42%)
- Confusion (gold x system, boundary-matched pairs):

| gold \ sys | decision | legislation | secondary sources | missed (no boundary) |
|---|---|---|---|---|
| decision | 2117 | 4 | 2 | 1778 |
| legislation | 0 | 2607 | 0 | 953 |
| secondary sources | 15 | 0 | 226 | 215 |
| spurious (no boundary) | 509 | 500 | 117 | - |

### 3. Miss rate by mention pattern (parent level)
_Alignment mismatches mention->span: 0/7917_

| mention | pattern | gold | miss rate |
|---|---|---|---|
| decision | title | 1279 | 31.6% |
| legislation | title | 1157 | 31.7% |
| legislation | fragment | 870 | 23.6% |
| legislation | fragment -> title | 821 | 13.0% |
| decision | title -> citation | 603 | 44.4% |
| decision | title -> citation -> fragment | 476 | 37.4% |
| decision | fragment | 442 | 84.8% |
| decision | title -> fragment | 290 | 30.0% |
| decision | title -> citation -> citation | 170 | 51.2% |
| legislation | title -> citation | 147 | 31.3% |
| secondary sources | authors -> title -> source -> fragment | 119 | 25.2% |
| decision | citation | 104 | 69.2% |
| legislation | title -> citation -> fragment | 103 | 38.8% |
| legislation | fragment -> title -> citation | 89 | 32.6% |
| decision | title -> citation -> fragment -> citation | 72 | 30.6% |
| legislation | fragment -> fragment | 60 | 43.3% |
| legislation | title -> fragment | 60 | 43.3% |
| legislation | fragment -> fragment -> title | 58 | 15.5% |
| secondary sources | authors | 56 | 71.4% |
| secondary sources | authors -> title -> source | 51 | 60.8% |

## saullm — 38 docs evaluated (0 skipped)

### 1. Miss rate by mention type (parent) and component type (child)

| label | level | gold | sys | TP | miss rate (FN/gold) | FDR (FP/sys) | R | F1 |
|---|---|---|---|---|---|---|---|---|
| authors | child | 341 | 8 | 7 | 97.9% | 12.5% | 2.1% | 4.0% |
| citation | child | 2967 | 239 | 175 | 94.1% | 26.8% | 5.9% | 10.9% |
| decision | parent | 3901 | 489 | 244 | 93.7% | 50.1% | 6.3% | 11.1% |
| fragment | child | 4547 | 268 | 165 | 96.4% | 38.4% | 3.6% | 6.9% |
| legislation | parent | 3560 | 144 | 18 | 99.5% | 87.5% | 0.5% | 1.0% |
| secondary sources | parent | 456 | 38 | 9 | 98.0% | 76.3% | 2.0% | 3.6% |
| source | child | 270 | 3 | 2 | 99.3% | 33.3% | 0.7% | 1.5% |
| title | child | 6261 | 402 | 313 | 95.0% | 22.1% | 5.0% | 9.4% |
| unable to classify | child | 88 | 0 | 0 | 100.0% | 0.0% | 0.0% | 0.0% |

### 2. Boundary match -> correct mention type? (parents only)
- Gold parent mentions: 7917; with a system span on the exact same boundary: 274 (3.5% of gold)
- Of those, correct mention type: 271 (98.91%) — wrong type: 3 (1.09%)
- Confusion (gold x system, boundary-matched pairs):

| gold \ sys | decision | legislation | secondary sources | missed (no boundary) |
|---|---|---|---|---|
| decision | 244 | 1 | 0 | 3656 |
| legislation | 0 | 18 | 0 | 3542 |
| secondary sources | 2 | 0 | 9 | 445 |
| spurious (no boundary) | 243 | 125 | 29 | - |

### 3. Miss rate by mention pattern (parent level)
_Alignment mismatches mention->span: 0/7917_

| mention | pattern | gold | miss rate |
|---|---|---|---|
| decision | title | 1279 | 93.3% |
| legislation | title | 1157 | 99.7% |
| legislation | fragment | 870 | 99.2% |
| legislation | fragment -> title | 821 | 99.3% |
| decision | title -> citation | 603 | 93.9% |
| decision | title -> citation -> fragment | 476 | 91.8% |
| decision | fragment | 442 | 100.0% |
| decision | title -> fragment | 290 | 91.4% |
| decision | title -> citation -> citation | 170 | 91.2% |
| legislation | title -> citation | 147 | 100.0% |
| secondary sources | authors -> title -> source -> fragment | 119 | 98.3% |
| decision | citation | 104 | 98.1% |
| legislation | title -> citation -> fragment | 103 | 100.0% |
| legislation | fragment -> title -> citation | 89 | 100.0% |
| decision | title -> citation -> fragment -> citation | 72 | 87.5% |
| legislation | fragment -> fragment | 60 | 100.0% |
| legislation | title -> fragment | 60 | 98.3% |
| legislation | fragment -> fragment -> title | 58 | 100.0% |
| secondary sources | authors | 56 | 100.0% |
| secondary sources | authors -> title -> source | 51 | 98.0% |

## Cross-model comparison

See `comparison_miss_rate_by_label.csv` and `comparison_boundary_type.csv`.

- gemma4: boundary recall 73.7% (N=7917), type accuracy on matched 99.49%
- gpt5.2: boundary recall 80.8% (N=7917), type accuracy on matched 99.33%
- muse: boundary recall 52.5% (N=7917), type accuracy on matched 99.76%
- qwen38: boundary recall 62.8% (N=7917), type accuracy on matched 99.58%
- saullm: boundary recall 3.5% (N=7917), type accuracy on matched 98.91%