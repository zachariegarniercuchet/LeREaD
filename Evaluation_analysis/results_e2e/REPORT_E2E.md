# End-to-end URI resolution failure analysis (by grouped pattern)

Runs: `output_coref/1-final/*_pip`; gold: `data/annotated/test`.
Pattern grouping collapses consecutive duplicates (decision -> title -> title = decision -> title).

## gemma4 — 37 docs (align mismatches: 85)

### URI F1 overall (end-to-end, real-uri instances only)

| mention | gold | TP | FN_ext | FN_nouri | FN_wrong | FP_spur | P | R | F1 | extr_R | resolv_acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decision | 2843 | 1508 | 1050 | 3 | 282 | 478 | 66.5% | 53.0% | 59.0% | 63.1% | 84.1% |
| legislation | 2445 | 1560 | 399 | 0 | 486 | 482 | 61.7% | 63.8% | 62.7% | 83.7% | 76.2% |
| secondary sources | 51 | 29 | 16 | 0 | 6 | 120 | 18.7% | 56.9% | 28.2% | 68.6% | 82.9% |
| ALL | - | 3097 | - | - | - | - | 62.6% | 58.0% | 60.2% | - | - |

### Top failure patterns (by failure count)

| mention | pattern | gold | failures | fail rate | extr_R | resolv_acc |
|---|---|---|---|---|---|---|
| decision | fragment | 410 | 367 | 89.5% | 20.7% | 50.6% |
| decision | title | 847 | 341 | 40.3% | 70.4% | 84.9% |
| legislation | fragment | 721 | 302 | 41.9% | 84.9% | 68.5% |
| decision | title -> citation | 660 | 280 | 42.4% | 60.8% | 94.8% |
| legislation | title | 772 | 265 | 34.3% | 78.1% | 84.1% |
| legislation | fragment -> title | 721 | 208 | 28.8% | 95.1% | 74.8% |
| decision | citation | 147 | 126 | 85.7% | 48.3% | 29.6% |
| decision | title -> fragment | 306 | 72 | 23.5% | 88.2% | 86.7% |
| decision | title -> citation -> fragment | 259 | 34 | 13.1% | 91.1% | 95.3% |
| legislation | title -> fragment | 61 | 31 | 50.8% | 60.7% | 81.1% |
| decision | title -> citation -> title | 33 | 31 | 93.9% | 9.1% | 66.7% |
| decision | citation -> fragment | 32 | 28 | 87.5% | 90.6% | 13.8% |
| legislation | title -> citation -> title | 16 | 14 | 87.5% | 12.5% | 100.0% |
| legislation | title -> citation | 45 | 13 | 28.9% | 84.4% | 84.2% |
| decision | title -> citation -> title -> fragment | 19 | 13 | 68.4% | 31.6% | 100.0% |

## gpt5.2 — 24 docs (align mismatches: 171)

### URI F1 overall (end-to-end, real-uri instances only)

| mention | gold | TP | FN_ext | FN_nouri | FN_wrong | FP_spur | P | R | F1 | extr_R | resolv_acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decision | 1796 | 1152 | 395 | 0 | 249 | 735 | 53.9% | 64.1% | 58.6% | 78.0% | 82.2% |
| legislation | 1366 | 939 | 200 | 0 | 227 | 406 | 59.7% | 68.7% | 63.9% | 85.4% | 80.5% |
| secondary sources | 37 | 23 | 12 | 0 | 2 | 368 | 5.9% | 62.2% | 10.7% | 67.6% | 92.0% |
| ALL | - | 2114 | - | - | - | - | 51.5% | 66.1% | 57.9% | - | - |

### Top failure patterns (by failure count)

| mention | pattern | gold | failures | fail rate | extr_R | resolv_acc |
|---|---|---|---|---|---|---|
| decision | fragment | 222 | 179 | 80.6% | 43.7% | 44.3% |
| decision | title | 432 | 145 | 33.6% | 81.5% | 81.5% |
| legislation | title | 425 | 143 | 33.6% | 77.9% | 85.2% |
| legislation | fragment | 459 | 120 | 26.1% | 87.8% | 84.1% |
| legislation | fragment -> title | 365 | 118 | 32.3% | 94.2% | 71.8% |
| decision | citation | 127 | 102 | 80.3% | 68.5% | 28.7% |
| decision | title -> citation | 525 | 80 | 15.2% | 87.4% | 96.9% |
| decision | title -> fragment | 176 | 42 | 23.9% | 86.9% | 87.6% |
| decision | citation -> fragment | 31 | 26 | 83.9% | 90.3% | 17.9% |
| decision | title -> citation -> title | 21 | 18 | 85.7% | 14.3% | 100.0% |
| decision | title -> citation -> fragment | 151 | 12 | 7.9% | 94.7% | 97.2% |
| decision | fragment -> title | 11 | 11 | 100.0% | 54.5% | 0.0% |
| legislation | citation | 17 | 9 | 52.9% | 94.1% | 50.0% |
| legislation | title -> fragment | 16 | 9 | 56.2% | 75.0% | 58.3% |
| legislation | title -> citation | 27 | 6 | 22.2% | 85.2% | 91.3% |

## muse — 38 docs (align mismatches: 807)

### URI F1 overall (end-to-end, real-uri instances only)

| mention | gold | TP | FN_ext | FN_nouri | FN_wrong | FP_spur | P | R | F1 | extr_R | resolv_acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decision | 3449 | 1368 | 1897 | 8 | 176 | 350 | 72.2% | 39.7% | 51.2% | 45.0% | 88.1% |
| legislation | 3047 | 1426 | 1037 | 1 | 583 | 297 | 61.8% | 46.8% | 53.3% | 66.0% | 70.9% |
| secondary sources | 67 | 24 | 40 | 0 | 3 | 71 | 24.5% | 35.8% | 29.1% | 40.3% | 88.9% |
| ALL | - | 2818 | - | - | - | - | 65.6% | 42.9% | 51.9% | - | - |

### Top failure patterns (by failure count)

| mention | pattern | gold | failures | fail rate | extr_R | resolv_acc |
|---|---|---|---|---|---|---|
| decision | title | 1103 | 567 | 51.4% | 56.1% | 86.6% |
| legislation | fragment | 869 | 545 | 62.7% | 68.2% | 54.6% |
| legislation | title | 886 | 477 | 53.8% | 53.7% | 85.9% |
| decision | fragment | 432 | 431 | 99.8% | 1.6% | 14.3% |
| decision | title -> citation | 718 | 415 | 57.8% | 45.4% | 92.9% |
| legislation | fragment -> title | 861 | 374 | 43.4% | 78.5% | 72.0% |
| decision | title -> citation -> fragment | 518 | 254 | 49.0% | 53.9% | 94.6% |
| decision | citation | 147 | 135 | 91.8% | 27.9% | 29.3% |
| decision | title -> fragment | 308 | 132 | 42.9% | 59.7% | 95.7% |
| legislation | title -> citation | 116 | 55 | 47.4% | 73.3% | 71.8% |
| legislation | title -> fragment | 63 | 44 | 69.8% | 44.4% | 67.9% |
| legislation | fragment -> title -> citation | 89 | 43 | 48.3% | 64.0% | 80.7% |
| legislation | title -> citation -> fragment | 100 | 42 | 42.0% | 67.0% | 86.6% |
| decision | citation -> fragment | 32 | 30 | 93.8% | 53.1% | 11.8% |
| decision | title -> citation -> title | 35 | 26 | 74.3% | 28.6% | 90.0% |

## qwen38 — 38 docs (align mismatches: 1199)

### URI F1 overall (end-to-end, real-uri instances only)

| mention | gold | TP | FN_ext | FN_nouri | FN_wrong | FP_spur | P | R | F1 | extr_R | resolv_acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decision | 3449 | 1620 | 1556 | 1 | 272 | 515 | 67.3% | 47.0% | 55.3% | 54.9% | 85.6% |
| legislation | 3047 | 1737 | 721 | 6 | 583 | 501 | 61.6% | 57.0% | 59.2% | 76.3% | 74.7% |
| secondary sources | 67 | 31 | 31 | 0 | 5 | 117 | 20.3% | 46.3% | 28.2% | 53.7% | 86.1% |
| ALL | - | 3388 | - | - | - | - | 63.0% | 51.6% | 56.7% | - | - |

### Top failure patterns (by failure count)

| mention | pattern | gold | failures | fail rate | extr_R | resolv_acc |
|---|---|---|---|---|---|---|
| legislation | fragment | 869 | 482 | 55.5% | 76.1% | 58.5% |
| decision | title | 1103 | 445 | 40.3% | 70.4% | 84.7% |
| decision | fragment | 432 | 403 | 93.3% | 15.0% | 44.6% |
| decision | title -> citation | 718 | 385 | 53.6% | 49.9% | 93.0% |
| legislation | title | 886 | 349 | 39.4% | 74.6% | 81.2% |
| legislation | fragment -> title | 861 | 259 | 30.1% | 86.3% | 81.0% |
| decision | title -> citation -> fragment | 518 | 207 | 40.0% | 62.9% | 95.4% |
| decision | citation | 147 | 131 | 89.1% | 27.2% | 40.0% |
| decision | title -> fragment | 308 | 120 | 39.0% | 70.5% | 86.6% |
| legislation | title -> citation | 116 | 56 | 48.3% | 70.7% | 73.2% |
| legislation | title -> citation -> fragment | 100 | 56 | 56.0% | 50.0% | 88.0% |
| legislation | fragment -> title -> citation | 89 | 38 | 42.7% | 67.4% | 85.0% |
| legislation | title -> fragment | 63 | 35 | 55.6% | 54.0% | 82.4% |
| decision | citation -> fragment | 32 | 31 | 96.9% | 40.6% | 7.7% |
| decision | title -> citation -> title | 35 | 24 | 68.6% | 37.1% | 84.6% |

## saullm — 38 docs (align mismatches: 365)

### URI F1 overall (end-to-end, real-uri instances only)

| mention | gold | TP | FN_ext | FN_nouri | FN_wrong | FP_spur | P | R | F1 | extr_R | resolv_acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| decision | 3449 | 179 | 3232 | 15 | 23 | 143 | 51.9% | 5.2% | 9.4% | 6.3% | 82.5% |
| legislation | 3047 | 5 | 3030 | 5 | 7 | 92 | 4.8% | 0.2% | 0.3% | 0.6% | 29.4% |
| secondary sources | 67 | 1 | 65 | 1 | 0 | 2 | 33.3% | 1.5% | 2.9% | 3.0% | 50.0% |
| ALL | - | 185 | - | - | - | - | 40.9% | 2.8% | 5.3% | - | - |

### Top failure patterns (by failure count)

| mention | pattern | gold | failures | fail rate | extr_R | resolv_acc |
|---|---|---|---|---|---|---|
| decision | title | 1103 | 1048 | 95.0% | 7.0% | 71.4% |
| legislation | title | 886 | 884 | 99.8% | 0.5% | 50.0% |
| legislation | fragment | 869 | 869 | 100.0% | 0.8% | 0.0% |
| legislation | fragment -> title | 861 | 859 | 99.8% | 0.6% | 40.0% |
| decision | title -> citation | 718 | 678 | 94.4% | 6.1% | 90.9% |
| decision | title -> citation -> fragment | 518 | 473 | 91.3% | 9.1% | 95.7% |
| decision | fragment | 432 | 432 | 100.0% | 0.0% | - |
| decision | title -> fragment | 308 | 290 | 94.2% | 7.8% | 75.0% |
| decision | citation | 147 | 147 | 100.0% | 2.0% | 0.0% |
| legislation | title -> citation | 116 | 116 | 100.0% | 0.0% | - |
| legislation | title -> citation -> fragment | 100 | 100 | 100.0% | 0.0% | - |
| legislation | fragment -> title -> citation | 89 | 89 | 100.0% | 0.0% | - |
| legislation | title -> fragment | 63 | 62 | 98.4% | 1.6% | 100.0% |
| decision | title -> citation -> fragment -> citation | 70 | 60 | 85.7% | 14.3% | 100.0% |
| decision | title -> citation -> title | 35 | 33 | 94.3% | 5.7% | 100.0% |

## Most failures by mention type (4 models aggregated, excl. saullm)

### decision

| pattern | failures (all models) |
|---|---|
| title | 1498 |
| fragment | 1380 |
| title -> citation | 1160 |
| title -> citation -> fragment | 507 |
| citation | 494 |
| title -> fragment | 366 |
| citation -> fragment | 115 |
| title -> citation -> title | 99 |
| fragment -> title | 56 |
| title -> citation -> fragment -> citation | 48 |

### legislation

| pattern | failures (all models) |
|---|---|
| fragment | 1449 |
| title | 1234 |
| fragment -> title | 959 |
| title -> citation | 130 |
| title -> fragment | 119 |
| title -> citation -> fragment | 115 |
| fragment -> title -> citation | 94 |
| citation | 47 |
| title -> citation -> title | 29 |
| fragment -> title -> fragment | 24 |

### secondary sources

| pattern | failures (all models) |
|---|---|
| authors -> title -> source -> fragment | 30 |
| authors | 23 |
| authors -> title -> source | 19 |
| authors -> fragment | 16 |
| fragment | 14 |
| title -> source -> fragment | 5 |
| fragment -> authors | 4 |
| authors -> title | 2 |
| authors -> title -> fragment | 1 |
| title -> source | 1 |

## Common-docs comparison (24 docs present in every pipeline run)

See `comparison_common24_e2e_uri_f1.csv` and `comparison_common24_e2e_pattern_fail_rate.csv`.