# Reviewer W1 — synthetic vs CanLII linking on the test set

**Setting.** Gold-span (oracle) resolution on `data/annotated/test` (38 documents),
i.e. exactly the setting of the paper's Table 3 but on **test**. Gold `uri`
attributes were stripped before resolution, so there is no extraction or coref
error in these numbers (0 missed extractions, 0 spurious spans in both runs).

**Runs reused (no new GPU run).**

| system | run folder |
| --- | --- |
| Qwen3-8B embedding | `output_coref/test_gold_resolution_qwen3-8b` |
| Rule-based linker | `output_coref/2-resolution_gold/test_gold_resolution_rule` |

Both use the same gold candidate pool (`data/gold_candidate_pool.csv`: 6,976
CSV rows, expanded to **7,058 candidate entries** by the loaders because some
rows carry more than one metadata entry). It includes the **539 synthetic
rows** — the reviewer's "27%" (539/1,976 cited documents).

Gold-URI classification: `https://synthetic.leread.ca/gold/*` = synthetic;
everything else = CanLII (`canlii.ca` / `canlii.org`).

---

## 1. Test gold-span linking (overall)

| system | P % | R % | F1 % |
| --- | --- | --- | --- |
| Qwen3-8B | 87.66 | 86.14 | **86.89** |
| Rule-based | 85.00 | 80.01 | **82.43** |

This is the number missing from Table 3 (which reported train+dev only).

## 2. Test gold-span linking by target origin

| system | origin | unique targets | mentions | P % | R % | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| Qwen3-8B | CanLII | 1,142 | 6,578 | 87.13 | 85.86 | **86.49** |
| Qwen3-8B | synthetic | 404 | 1,331 | 90.31 | 87.53 | **88.90** |
| Rule | CanLII | 1,142 | 6,578 | 83.20 | 78.14 | **80.59** |
| Rule | synthetic | 404 | 1,331 | 93.76 | 89.26 | **91.45** |

Synthetic targets are **26.1% of unique targets** but only **16.8% of
real-uri mentions** in test.

Per unique cited target (each target weighted equally, Qwen): synthetic mean
accuracy **84.28%** vs CanLII **88.60%**; 15.35% of synthetic targets are fully
wrong vs 10.68% of CanLII targets. So at the cited-document level, synthetic
targets are not uniformly easier either.

Paired document-cluster bootstrap (95% CI):

| system | synthetic − CanLII F1 |
| --- | --- |
| Qwen3-8B | +2.40 pts (−7.98 to +9.86) — **not significant** |
| Rule | +10.87 pts (+0.22 to +18.23) |

## 2b. End-to-end test check (existing Qwen38-27B pipeline)

To cover the reviewer's first request (test *linking* accuracy, not only gold
spans), the same stratification was run on the existing end-to-end test run
`test_qwen38-27b_coref_fs6_random_from-test_qwen38-27b_DEC_fs6_greedy_sentence_long-final`
(38 documents, extraction + coref + resolution):

| origin | unique targets | mentions | P % | R % | F1 % | resolver-only acc % |
| --- | --- | --- | --- | --- | --- | --- |
| CanLII | 1,142 | 6,578 | 62.73 | 49.59 | **55.39** | 76.59 |
| synthetic | 404 | 1,331 | 64.11 | 42.00 | **50.75** | 81.13 |

End-to-end, synthetic targets are **4.64 F1 points worse**, not better
(95% CI −11.52 to +3.04). Once the mention is found, synthetic is slightly
easier (resolver-only accuracy 81.13 vs 76.59), but extraction/coref recall on
synthetic mentions is lower. The end-to-end headline is 54.66 with all targets
and 55.39 with synthetic removed.

## 3. By label (controls for the origin/label confound)

| system | label | CanLII F1 % | synthetic F1 % |
| --- | --- | --- | --- |
| Qwen3-8B | Decision | 83.81 | 91.56 |
| Qwen3-8B | Legislation | **89.50** | **84.90** |
| Qwen3-8B | Secondary | 86.82 | 91.01 |
| Rule | Decision | 89.00 | 90.62 |
| Rule | Legislation | **71.16** | **93.73** |
| Rule | Secondary | 91.06 | 89.46 |

For the strong retriever, synthetic **legislation** is *harder* than CanLII
legislation. The rule-based gap is overwhelmingly a legislation effect: its
lexical linker fails on CanLII legislation (71.16) but almost always hits the
verbatim synthetic title (93.73).

## 4. How much does it change the headline?

| setting | Qwen F1 % | Rule F1 % |
| --- | --- | --- |
| All targets (headline) | 86.89 | 82.43 |
| CanLII only (synthetic removed) | 86.49 | 80.59 |
| Synthetic only | 88.90 | 91.45 |
| Headline − CanLII-only | **+0.40** | **+1.84** |

Even the absurd worst case ("every synthetic mention is wrong") gives F1 77.82
(Qwen) and 72.57 (rule). The realistic bound is the CanLII-only row.

## 5. Is synthetic "near-exact"? (shortcut diagnostic)

For each matched span we compared the run's saved `query_text`
(`*_resolution_mapping.json`) against the gold candidate row's `docTitle` +
`autoCompletions`.

| system | origin | mean target-token containment % | % spans with exact metadata match |
| --- | --- | --- | --- |
| Qwen3-8B | CanLII | 42.86 | 77.28 |
| Qwen3-8B | synthetic | **96.04** | **98.14** |
| Rule | CanLII | 44.14 | 80.74 |
| Rule | synthetic | **97.12** | **99.68** |

Yes: synthetic queries are much closer to a verbatim copy of the candidate row.
But the linking accuracy conditional on an exact metadata match is similar
(Qwen: 90.98 CanLII vs 91.71 synthetic), and Qwen still links non-exact CanLII
targets at 74.07. The synthetic *non-exact* subset is tiny (24 spans Qwen,
4 rule; 15 / 1 "hard" spans), so these runs cannot estimate synthetic
performance without the shortcut.

Top-1 retrieval hit rate of the gold target: Qwen 87.13 CanLII vs 90.31
synthetic; rule 83.20 vs 93.76.

## 6. Bottom line for the response

1. On test gold spans, Qwen links synthetic targets only **+2.4 F1 points**
   above CanLII (CI crosses 0); for legislation it is actually **4.6 points
   worse**. Removing synthetic targets moves the Qwen headline from 86.89 to
   86.49 (**0.40 points**).
2. End-to-end, synthetic targets are **worse** (50.75 vs 55.39 F1), so the
   shortcut does not inflate the pipeline's test score.
3. The rule-based linker does benefit strongly (+10.9), but that is a property
   of a purely lexical linker on legislation, not of the benchmark's headline
   (1.84-point effect).
4. The near-exact construction is real (98–100% of synthetic spans have an
   exact metadata match), but it does not translate into a large advantage for
   the embedding linker, which is what the paper's claims rest on.
5. If reviewers want the strict causal test, the clean one is a
   document-disjoint synthetic pool (build each synthetic entry only from
   mentions outside the evaluated document, rebuild the index, re-resolve).
   It requires a Qwen re-index + re-run on Tamia. The relevant bound is already
   visible: the *inflation* attributable to synthetic targets is at most the
   headline minus the CanLII-only score (0.40 points for Qwen). A disjoint pool
   could make synthetic *harder* than CanLII, which would make the benchmark
   conservative, not inflated. So it is probably not worth a new experiment
   unless the reviewer insists.

---

Detailed per-run reports and machine-readable results:

- `scripts/report/test_qwen_gold_target_origin.md` / `.json`
- `scripts/report/test_rule_gold_target_origin.md` / `.json`
- `scripts/report/test_qwen38_e2e_target_origin.md` / `.json`

Reproduce with:

```bash
python scripts/evaluate_resolution_target_origin.py \
    --batch-folder output_coref/test_gold_resolution_qwen3-8b --split test \
    --out scripts/report/test_qwen_gold_target_origin.md \
    --json-out scripts/report/test_qwen_gold_target_origin.json

python scripts/evaluate_resolution_target_origin.py \
    --batch-folder output_coref/2-resolution_gold/test_gold_resolution_rule --split test \
    --out scripts/report/test_rule_gold_target_origin.md \
    --json-out scripts/report/test_rule_gold_target_origin.json

python scripts/evaluate_resolution_target_origin.py \
    --batch-folder 'output_coref/test_qwen38-27b_coref_fs6_random_from-test_qwen38-27b_DEC_fs6_greedy_sentence_long-final' \
    --split test \
    --out scripts/report/test_qwen38_e2e_target_origin.md \
    --json-out scripts/report/test_qwen38_e2e_target_origin.json
```
