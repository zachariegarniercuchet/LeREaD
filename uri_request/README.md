# URI request — no-gold existence study (SPLADE)

Mission: decide, for each of the 1551 coref profiles in `uri_request.json`
(55 no-gold documents, gemma coref registry), whether its URI **already
exists** in `data/candidate_pool.csv`, using the SPLADE index
`cache/candidate_pool_index_splade_v1/`.

## Contents

- `uri_request.json` — copy of `data/uri_request.json` (profiles to resolve).
- `export_no_gold_profiles.py` — copy of `scripts/export_no_gold_profiles.py`
  (how the registry was produced; kept here for traceability).
- `resolve_uri_request_splade.py` — resolution + confidence study script.
- `output_splade/` — generated on run (gitignored-style working output).

## Workflow

1. **Study (no threshold):**
   `python uri_request/resolve_uri_request_splade.py`
   → `splade_topk.{json,csv}` + `score_analysis.json` + printed sweep table.
2. **Decide:** score alone cannot give high precision (see
   `CALIBRATION_REPORT.md`), so the final rule is
   `EXISTS <=> score >= T AND verify(hit)` (citation/title/alias check,
   rank-1 else rank-2/3 fallback):
   `python uri_request/apply_decision.py --threshold 10.0`
   → `output_decision/uri_decision.{json,csv}` (EXISTS/MISSING per profile,
   `needs_review` flag) + `missing_by_document.json` (per source document:
   what is NOT in the pool) + `decision_summary.json`.
3. **Calibrate (how T and the verifier were chosen):**
   `sample_for_judge.py` (72 stratified examples) →
   `calibrate_thresholds.py` (LLM-judge precision/recall tables).
   Full story in `CALIBRATION_REPORT.md`.
4. **Missing-only request for the company:**
   `python uri_request/build_missing_request.py`
   → `uri_request_missing.json` (same shape as `uri_request.json`,
   1344 MISSING profiles, `uri` empty to fill).
5. **Silver split:** `python uri_request/build_silver_split.py`
   → `data/annotated/no-gold/<DOC>.html` (55 flat files like the gold
   splits; verified EXISTS profiles injected as `uri` on the
   `auto_label`/`manual_label` tags via `src/main_resolution.py`'s
   injector; MISSING profiles left without `uri` for the company
   round-trip to complete).
6. **Cross-document grouping (one group per source):**
   `crossdoc_merge.py` → `crossdoc_splade.py` (SPLADE pair scores +
   60-pair judge sample) → judge → `crossdoc_finalize.py`
   → `uri_request_merged.json` (all groups: VERIFIED/UNVERIFIED/CONFLICT)
   + `uri_request_merged_missing.json` (company send-out, one entry per
   group needing a `uri`). Full story in `CROSSDOC_REPORT.md`.

Retrieval mirrors `src/main_resolution.py` exactly (same index loader,
same doctype bucketing, same batched `retriever.search`). SPLADE scores are
dot products, not probabilities — the threshold is heuristic and must be
chosen from the observed distribution, not assumed.
