# Reflex (CanLII citator) — overall evaluation (no per-type breakdown)

*Generated 2026-11-25 08:28 — `python scripts/reflex/evaluate_reflex_overall.py`*

**Scope.** All results are pooled over mention types; there is no decision / legislation / secondary breakdown. Extraction is reported for parent mentions, child components and the total (parent + children), exactly as `evaluate_batch` does. Coreference and URI linking are reported overall.

Headline numbers come from the unmodified stock evaluators. Where the stock code reconstructs span offsets heuristically (extraction and linking), a clearly-labelled *DOM-faithful offset* cross-check is given alongside; coreference computes real offsets, so it needs none.

* Documents: 38 paired Reflex export(s) (38 test).
* Merge gap: `15` normalised characters.
* Candidate pool for URI lookup: `gold`.
* Exported run folders: `C:\Users\zakga\OneDrive\Documents\code\LeREaD\output\test_reflex` and `C:\Users\zakga\OneDrive\Documents\code\LeREaD\output_coref\test_reflex`.

The same scripts the paper uses score Reflex here: `evaluation_l1_util.evaluate_batch` / `..._overlap` for extraction, `evaluate_coref.py` span mode for coreference, and `evaluate_resolution_batch` / `..._overlap` for linking. Reflex spans are aligned into gold coordinates first, so no scorer is re-implemented.

## 1. Extraction (standard `evaluate.py` machinery)

**EXACT** — label + boundary text + 200-char context (the Table-1 metric)

| mentions | gold | system | TP | P % | R % | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| Parent | 7917 | 2919 | 1081 | 37.03 | 13.65 | 19.95 |
| Children | 14474 | 0 | 0 | 0.00 | 0.00 | 0.00 |
| **Total (parent + children)** | 22391 | 2919 | 1081 | 37.03 | 4.83 | 8.54 |

**OVERLAP** — label + character overlap (lenient comparison)

| mentions | gold | system | TP | P % | R % | F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| Parent | 7917 | 2919 | 2135 | 73.14 | 26.97 | 39.41 |
| Children | 14474 | 0 | 0 | 0.00 | 0.00 | 0.00 |
| **Total (parent + children)** | 22391 | 2919 | 2135 | 73.14 | 9.54 | 16.87 |

*Reflex produces no child components (`title`, `citation`, `fragment`, `authors`, `source`), so the children row is 0/0 by construction and the total is the parent score diluted by every gold child. The parent row is the fair, citator-scope number; the total row is the number comparable with a full pipeline that annotates both levels.*

**Cross-check — DOM-faithful offsets.** The stock `extract_spans` does not read a tag's real position: it looks up the *n*-th occurrence of the tag's surface text (processing `manual_label` tags before `auto_label` ones). On nested, highly repetitive legal text a span can land on a different occurrence of the same string — and Reflex's alignment puts many anchors on 1-character gold insertion points whose text repeats everywhere. Re-running the **same stock matching and metrics** with true DOM offsets isolates that scorer effect:

| metric | TP stock | TP true | F1 stock % | F1 true % | delta |
| --- | --- | --- | --- | --- | --- |
| Parent EXACT | 1081 | 1040 | 19.95 | 19.20 | -0.76 |
| Parent OVERLAP | 2135 | 2813 | 39.41 | 51.92 | +12.51 |
| Total EXACT | 1081 | 1040 | 8.54 | 8.22 | -0.32 |
| Total OVERLAP | 2135 | 2813 | 16.87 | 22.23 | +5.36 |

## 2. Coreference (standard `evaluate_coref.py`, span mode)

Unmatched policy: `singleton` (end-to-end / twinless). 38 document(s).

| metric | exact P % | exact R % | exact F1 % | overlap P % | overlap R % | overlap F1 % |
| --- | --- | --- | --- | --- | --- | --- |
| MUC | 58.09 | 11.73 | 19.52 | 88.82 | 17.94 | 29.85 |
| B3 | 93.20 | 40.87 | 56.83 | 96.62 | 31.75 | 47.79 |
| CEAF_e | 30.53 | 72.09 | 42.89 | 18.57 | 68.36 | 29.21 |
| LEA | 28.01 | 25.84 | 26.88 | 22.79 | 19.02 | 20.74 |
| CoNLL | - | - | 39.75 | - | - | 35.62 |

Alignment (exact): gold mentions 8006 | system mentions 2919 | matched 882.
Alignment (overlap): gold mentions 8006 | system mentions 2919 | matched 2708.

*Only parent-level mentions carry a gold `docid`; child components are not coreference instances. Reflex exposes a ``concatenated-id`` only on legislation anchors, so decision predictions are singleton clusters. Offsets here are read from the DOM, so no cross-check is needed.*

## 3. URI linking (standard `evaluate_resolution.py` machinery)

Only gold mentions with a real `uri` are counted; child components carry no `uri`, so linking is effectively parent-level.

| match | P % | R % | F1 % |
| --- | --- | --- | --- |
| exact | 58.86 | 10.67 | 18.07 |
| overlap | 70.48 | 12.77 | 21.62 |

Counts behind these numbers: Reflex emits **1434** URI(s) (1434 legislation), while **7909** gold mentions need one (3898 decision, 3556 legislation, 455 secondary sources). Precision is over Reflex's emitted URIs; recall is over all gold real-uri mentions, including the decision/secondary mentions that Reflex cannot link at all. That asymmetry is why the overall F1 is low despite reasonable precision.

Spans with a Reflex-derived uri: 1434. Anchors without a ``concatenated-id`` (all decisions) produce no uri and therefore count as `FN — no uri produced` when the gold span needs one.

**Cross-check — DOM-faithful offsets** (same stock resolution matching, true positions):

| match | P stock % | R stock % | F1 stock % | P true % | R true % | F1 true % |
| --- | --- | --- | --- | --- | --- | --- |
| EXACT | 58.86 | 10.67 | 18.07 | 56.49 | 10.24 | 17.34 |
| OVERLAP | 70.48 | 12.77 | 21.62 | 73.20 | 13.26 | 22.46 |
