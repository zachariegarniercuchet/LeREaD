# Strict end-to-end pattern ranking (exact match, worst -> best F1)

Aggregate: gemma4, gpt5.2, muse, qwen38 on 24 common docs; gold instances >= 100.
Level-1 matching: same label + same text + same context (exact boundaries).
TP = every span of the mention matched AND uri correct.

| rank | mention | pattern | gold | TP | sys | TP_sys | R | P | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | decision | citation -> fragment | 124 | 4 | 35 | 8 | 3.2% | 22.9% | 5.7% |
| 2 | decision | citation | 508 | 38 | 334 | 36 | 7.5% | 10.8% | 8.8% |
| 3 | decision | fragment | 888 | 102 | 393 | 103 | 11.5% | 26.2% | 16.0% |
| 4 | legislation | fragment | 1836 | 748 | 1959 | 748 | 40.7% | 38.2% | 39.4% |
| 5 | legislation | title -> citation | 108 | 55 | 175 | 60 | 50.9% | 34.3% | 41.0% |
| 6 | decision | title -> citation | 2100 | 803 | 1827 | 924 | 38.2% | 50.6% | 43.5% |
| 7 | legislation | fragment -> title | 1460 | 664 | 1488 | 677 | 45.5% | 45.5% | 45.5% |
| 8 | decision | title -> citation -> fragment | 604 | 253 | 661 | 363 | 41.9% | 54.9% | 47.5% |
| 9 | decision | title | 1728 | 956 | 2331 | 992 | 55.3% | 42.6% | 48.1% |
| 10 | legislation | title | 1700 | 888 | 1958 | 923 | 52.2% | 47.1% | 49.6% |
| 11 | legislation | fragment -> title -> citation | 100 | 63 | 111 | 65 | 63.0% | 58.6% | 60.7% |
| 12 | decision | title -> citation -> fragment -> citation | 224 | 128 | 250 | 165 | 57.1% | 66.0% | 61.3% |
| 13 | decision | title -> fragment | 704 | 406 | 681 | 458 | 57.7% | 67.3% | 62.1% |