# Strict end-to-end pattern ranking (overlap match, worst -> best F1)

Aggregate: gemma4, gpt5.2, muse, qwen38 on 24 common docs; gold instances >= 100.
Level-1 matching: same label + intersecting character offsets (overlap).
TP = every span of the mention matched AND uri correct.

| rank | mention | pattern | gold | TP | sys | TP_sys | R | P | F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | decision | citation -> fragment | 124 | 4 | 35 | 8 | 3.2% | 22.9% | 5.7% |
| 2 | decision | citation | 508 | 49 | 334 | 44 | 9.6% | 13.2% | 11.1% |
| 3 | decision | fragment | 888 | 126 | 393 | 129 | 14.2% | 32.8% | 19.8% |
| 4 | legislation | fragment | 1836 | 796 | 1959 | 777 | 43.4% | 39.7% | 41.4% |
| 5 | legislation | fragment -> title | 1460 | 666 | 1488 | 705 | 45.6% | 47.4% | 46.5% |
| 6 | legislation | title | 1700 | 896 | 1958 | 947 | 52.7% | 48.4% | 50.4% |
| 7 | decision | title | 1728 | 931 | 2331 | 1113 | 53.9% | 47.7% | 50.6% |
| 8 | legislation | title -> citation | 108 | 69 | 175 | 83 | 63.9% | 47.4% | 54.4% |
| 9 | decision | title -> citation | 2100 | 1119 | 1827 | 1243 | 53.3% | 68.0% | 59.8% |
| 10 | decision | title -> citation -> fragment | 604 | 332 | 661 | 475 | 55.0% | 71.9% | 62.3% |
| 11 | decision | title -> fragment | 704 | 410 | 681 | 493 | 58.2% | 72.4% | 64.5% |
| 12 | legislation | fragment -> title -> citation | 100 | 71 | 111 | 69 | 71.0% | 62.2% | 66.3% |
| 13 | decision | title -> citation -> fragment -> citation | 224 | 140 | 250 | 193 | 62.5% | 77.2% | 69.1% |