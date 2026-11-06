# SPLADE threshold calibration — per-doctype decision rule

## Question

For each of the 1551 no-gold profiles, does its URI already exist in
`data/candidate_pool.csv` (SPLADE index `cache/candidate_pool_index_splade_v1`)?
Goal: near-certain precision on accepted URIs, maximum recall under that
constraint. One rule per doc type (decision / legislation / secondary sources).

## Method (all scripts in `uri_request/`)

1. `resolve_uri_request_splade.py` — full SPLADE top-5 for all 1551 profiles
   (`output_splade/`). Medians already hinted per-type regimes:
   decision 19.6 / legislation 15.4 / secondary 7.4.
2. `sample_for_judge.py` — stratified sample: 6 profiles × 4 per-type score
   bands × 3 types = 72 examples (`judge/judge_sample.json`), enriched with
   candidate `metadata_text`.
3. LLM-as-judge: read all 72 (query vs top-3 metadata). Verdicts in
   `calibrate_thresholds.py::JUDGE` — **8 CORRECT / 60 WRONG / 4 UNCERTAIN**.
   Rule for CORRECT: top1 names the SAME work (citation matches); same-title
   wrong-jurisdiction counts as WRONG (a URI points at one exact document).
4. `calibrate_thresholds.py` — precision/recall per type over a T grid,
   score-only vs score+verifier.
5. `apply_decision.py` — final rule applied to all 1551 (`output_decision/`).

## Key finding: score alone cannot work

Presence audit (grep over `candidates.json`) showed almost every WRONG top1 is
a **similar-but-different work whose true URI is absent from the pool** —
some scoring 25–32. Score-only precision on judged samples:

| type | T=15 | T=20 | T=25 | T=35 |
|---|---|---|---|---|
| decision (4 correct) | prec 0.21 / rec 1.00 | 0.36 / 1.00 | 0.67 / 1.00 | 1.00 / 0.50 |
| legislation (4 correct) | 0.33 / 1.00 | 0.50 / 0.50 | 1.00 / 0.25 | – |
| secondary (0 correct) | – (nothing accepted ≥15) | – | – | – |

No threshold gives high precision with usable recall. The pool simply lacks
most cited works (only 64 DOCTRINE entries vs 185 secondary-source profiles;
many cited decisions/acts never indexed).

## Recommended rule: SPLADE shortlist + citation/title verifier

`EXISTS <=> score >= T AND verify(hit)` (rank 1, else first verified rank ≤ 3):

- **citation**: query's neutral cite / CanLII id / statute ref appears in the
  candidate metadata; if both sides carry citations they must overlap
  (rejects same-title/wrong-jurisdiction like #67, same-parties/wrong-instance
  like #47, rescues the right rank-2).
- **title-unique**: exact normalized title match AND that title occurs once in
  the pool (rejects generic "Workers' Compensation Act" → manual).
- **alias**: Canadian Charter → Constitution Act, 1982.
- **T = 10 for all three types** (junk-guard only: T∈{8,10,12} give 207/207/206
  EXISTS — the verifier, not T, decides).

Validation on judged samples at T=10: **TP=8, FP=0, FN=0** (+1 correct rank-2
rescue). Verifier alone passes 0/60 WRONG.

## Final result (T=10, all 1551 profiles; refreshed after §fixes)

- **EXISTS: 206** (decision 131 / legislation 74 / secondary 1),
  of which auto-safe + **64 `needs_review`** (profile title differs from
  candidate, or citeless title-unique legislation assignment — e.g. bare
  "Workers' Compensation Act" in an NB doc taking Alberta's act: best guess,
  company confirms).
- **MISSING: 1345**, spanning **all 55 source documents**
  (`missing_by_document.json`).
- Verification mix: citation ~110, unique-title ~88, alias 6; 3 accepted at
  rank 2–4. Jurisdiction-mismatch + citeless flags removed/flagged the
  PEI-vs-Alberta Workers' Comp overclaim found during cross-doc audit.

## What this means

- **Secondary sources: effectively not in the pool** (1/185). Send all 184 to
  the company for external matching — expected, the pool has ~no books/articles.
- **Decisions/legislation: ~1 in 5 already in the pool** with a verified URI;
  the rest are genuinely absent (pool = 5862 decisions + 593 acts, but the
  no-gold docs cite far beyond it).
- Suggested send-out: `uri_decision.csv` (EXISTS rows + MISSING rows) with the
  `needs_review` column highlighted for a 33-row human glance.

## Reproduce

    python uri_request/resolve_uri_request_splade.py            # study
    python uri_request/sample_for_judge.py                       # 72-sample
    python uri_request/calibrate_thresholds.py                   # judge tables
    python uri_request/apply_decision.py --threshold 10.0        # final
