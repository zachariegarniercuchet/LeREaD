# Reflex evaluation (`scripts/reflex/`)

Scores the **Reflex (CanLII citator) exports** in `data/reflex/` (`<id>_reflex.html`, 100 files)
against LeREaD gold (`data/annotated/<split>/`), at the **parent/mention level only**
(`decision` / `legislation` / `secondary sources`). Components (title, citation, fragment,
authors, source) are *not* scored: Reflex does not produce them.

The scorer itself is the paper's own `src/evaluation/evaluation_l1_util.py`
(`Span`, `compute_metrics`, `compute_metrics_overlap`), so numbers are directly comparable
with Table 1 of the manuscript; only the Reflex-to-gold adaptation layer is new code here.

## Quick start (from repo root)

```bash
# main evaluation -> writes report/ (markdown + json)
python scripts/reflex/evaluate_reflex.py

# with side-by-side baselines on the shared documents
python scripts/reflex/evaluate_reflex.py ^
    --compare rules="output/test_baseline_rules/**/*_final.html" ^
    --compare gpt-5.2="output/test_gpt-5.2_DEC_fs6_greedy_sentence_long/**/*_final.html" ^
    --compare gemma4="output/test_gemma4_DEC_fs6_greedy_sentence_long/**/*_final.html" ^
    --compare qwen38-27b="output/test_qwen38-27b_DEC_fs6_greedy_sentence_long/**/*_final.html"

# inventory / format exploration only (no scoring)
python scripts/reflex/explore.py
```

Useful flags of `evaluate_reflex.py`:

| flag | meaning |
| --- | --- |
| `--splits test,dev` | score only these splits (default: every split with a reflex export) |
| `--gap N` | fix the anchor merge gap instead of auto-selecting it |
| `--gaps 0,5,10,15,20,30` | sweep candidates; best **overlap micro-F1** wins |
| `--compare NAME=GLOB` | repeatable; scores another system's `*_final.html` on the same documents |
| `--context-chars 200` | context window of the EXACT match (Table-1 metric) |
| `--min-ratio 0.85` | content-token similarity required to accept a fuzzy id match |
| `--out DIR` | output folder (default `scripts/reflex/report/`) |

Run time for the full 100 documents is a few minutes (text alignment dominates).

## Files

* `common.py` - parsing (`parse_reflex`, `parse_gold`), normalised-text offset walker,
  anchor merging, `Alignment` (token-diff of the two renderings), document pairing,
  `build_spans` -> `evaluation_l1_util.Span`s.
* `evaluate_reflex.py` - CLI, gap sweep, per-split / per-label / per-doc tables,
  false-negative taxonomy, `--compare` systems, report builder.
* `explore.py` - read-only inventory: href classes, id pairing, text-diff diagnostics.
* `report/reflex_extraction_report.md` + `report/reflex_extraction_results.json` - generated
  deliverables (regenerate; do not edit by hand).

## How a Reflex prediction is built

1. **Anchor classification by href**: `/laws/` -> legislation; `/.../doc/` -> decision;
   `/jnl/ /book/ /comm/` -> secondary sources (in practice Reflex exports contain **no**
   secondary-source links at all); external hosts (westlaw/lexis/FJA) -> decision.
   Internal navigation (`reflex-parag`), footnote markers and bare-URL links to
   unclassifiable sites are dropped.
2. **Merging**: consecutive anchors resolving to the same base href within the merge gap
   (normalised characters) become one mention, so `Federal Court Act` + `s. 18` score as
   the single gold mention `Federal Court Act, s. 18`. The gap is swept and the value with
   the best overlap micro-F1 is used (plateau at 10-30; selected: **15**).
3. **Alignment**: the reflex rendering is token-diffed onto the gold rendering of the same
   document (similarity ~0.96-1.00). Reflex *injects* neutral citations (`1982 CanLII 19
   (SCC)`) absent from the gold text; those tokens map to their insertion point, i.e.
   inside the gold decision mention that names the same case. Every reflex prediction is
   therefore expressed on gold coordinates before scoring.
4. **Document pairing**: exact id match first, then fuzzy (normalised id / digit key)
   gated by content-token Jaccard >= 0.85 (same document ~0.96-1.00, different ~0.10);
   100/100 exports pair cleanly, 10 via the fuzzy route (court-suffix differences).

## Metrics

* **EXACT** = label + boundary text + 200-char context -- the Table-1 metric. Structurally
  harsh for a citator: Reflex links the *citation* part of a case reference while the gold
  parent span covers the whole reference, and pre-1999 citations it links do not exist in
  the gold text at all.
* **OVERLAP** = label + character overlap -- the lenient metric `src/evaluation/evaluate.py`
  already prints. This is the fair *detection* metric for Reflex: precision stays ~96-99%,
  recall is bounded by what a citator links.

## Headline numbers (test split, 38 documents, gap = 15)

| metric | P | R | F1 |
| --- | --- | --- | --- |
| EXACT (Table-1) | 35.6 | 13.1 | 19.2 |
| OVERLAP | 96.4 | 35.5 | 51.9 |

Per mention type (test, OVERLAP): decision 96.6/36.1/52.6, legislation 96.1/39.4/55.9,
secondary sources 0 (out of scope for Reflex). On the same documents and scorer:
rules exact 31.6 / overlap 65.0; GPT-5.2 exact 81.3 / overlap 89.5; Gemma-4 exact 82.0 /
overlap 89.7; Qwen3.8-27B exact 72.9 / overlap 80.3 (LLMs on the 4-doc subset shared with
the Reflex run; their full-test Table-1 numbers are in `latex/main_results.tex`).

## Caveats

* Recall is capped twice: by the number of Reflex predictions (2919 vs 7917 gold mentions
  on test) and by gold mentions a citator never links - short forms (`Newport`,
  `idem, s. 3`), fragments and **all** secondary sources.
* Silver-split gold is auto-annotated; treat `split silver` rows as indicative only.
* Reflex was scored as exported; no post-processing beyond merging was tuned per split
  (the single merge gap is selected globally by sweep).
* `report/` files are generated - rerun the command above after changing code.
