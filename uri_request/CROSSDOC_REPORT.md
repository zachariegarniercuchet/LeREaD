# Cross-document grouping: one group per real-world source

## Problem

`uri_request.json` holds 1551 profiles scoped per document (`DOC::docid`).
The same source cited in two documents (e.g. the Canadian Charter in 11
documents) appears as several profiles. The company wants ONE group per
source, with one `uri` per group.

## Method (scripts in `uri_request/`, precision-first)

1. `crossdoc_merge.py` (phase 1, deterministic):
   R1 same verified URI + R2 same title + shared strong citation,
   with hard constraint: never merge distinct verified URIs.
   Result: 30 multi-doc groups; 151 undecided same-title pairs.
2. `crossdoc_splade.py` (phase 2): blocked pair universe (same title,
   shared top-3 candidate, shared cite → 5479 pairs), SPLADE dot(queryA,
   queryB) for all, 60 sampled across score bands → `judge/judge_pairs.json`.
3. LLM-as-judge on all 60 (`judge/pair_labels.json`):
   **2 SAME / 57 DIFFERENT / 1 UNCERTAIN — evenly spread across ALL score
   bands, including 31.5 (high)**. Conclusion: SPLADE similarity measures
   relatedness, not identity, and is *rejected* as a merge criterion. It was
   used only for candidate-pair blocking.
4. `crossdoc_finalize.py` (phase 3): rule merges + union-find closure +
   post-closure eviction + outputs.

## Final rules

- R1 same accepted URI · R2 same title + shared strong cite ·
  R3 same title/type + same pool-verified candidate ·
  R4 charter-family alias (`charter`/`canadian charter`/`charter of rights…`,
  Quebec-marked excluded) · R5 identical distinctive (≥4-token) full case
  name + cite-compatible · R6 same title + overlapping authors ·
  R7 same specific (≥12-char) title, unambiguous in pool AND corpus,
  cite-compatible + jurisdiction-compatible.
- Post-closure split of VERIFIED groups: a legislation member citing another
  jurisdiction's enactment, or titled differently with no shared statute
  cite, is evicted to a singleton (catches NB-vs-Alberta Workers' Comp,
  BC-vs-Alberta Labour Relations Code, amendments merged with base acts).
- Decisions are exempt from eviction (candidate metadata omits cites too
  often — e.g. Kerr v. Baranow 2011 SCC 10): instance-level pairs
  (RJR QCA vs SCC) stay flagged `needs_review` for the company glance.

## Validation

- Judge pairs: 2/2 SAME merged, 0/57 DIFFERENT merged.
- URI calibration re-run after the citation-coverage fixes: 8/8 correct,
  0/60 wrong (`calibrate_thresholds.py`, now delegating to the shared
  `apply_decision.Verifier`).
- 0 CONFLICT groups (no group mixes two verified URIs).

## Result (`uri_request_merged.json`, 1483 groups from 1551 profiles)

Each group carries an **`extra_metadata`** pool (union of members'
`alternative_titles`, `citations`, `fragments_mentioned`, authors,
`query_texts`, total mentions), and each member embeds its **full original
registry profile** (`main_title`, `alternative_titles`, `citations`,
`fragments_mentioned`, `authors`, `mention_ids/texts`, `query_text`…) so the
company can match against their database with the same evidence as
`uri_request.json` — nothing was dropped in the merge.

- **170 VERIFIED** (uri known, incl. Charter×11 docs, Dunsmuir×4,
  Hryniak/Bhasin×3, Oakes/Whiten/Housen/Vavilov/Baker…), 38 flagged
  `needs_review` (typos, reversed parties, short forms, 2–3 upstream coref
  mixes, 2 instance-level decision pairs).
- **1313 UNVERIFIED → `uri_request_merged_missing.json`**: one entry per
  group (1338 profiles) for the company to fill — replaces the per-profile
  `uri_request_missing.json` at group granularity.
- **45 multi-document groups** (75→… profiles linked across docs); the rest
  are singletons (source cited in one doc only).

## Fixes applied along the way (agentic findings)

- `STATUTE_RX`/`JURIS_RX`: full statute-code coverage (PEI, NWT, RSS, SNS…)
  and chapters without `c.` (`W-7.1`); trailing-`\b` bug on normalized cites.
- `Verifier`: jurisdiction-mismatch rejection on title-unique
  (PEI-cited query no longer takes Alberta's act); citeless title-unique
  legislation flagged `needs_review`.
- Cascade re-run everywhere: `output_decision` (206 EXISTS),
  `uri_request_missing.json` (1345), `data/annotated/no-gold` silver (55
  files, 1929 tags with `uri`).

## Reproduce

    python uri_request/crossdoc_merge.py
    python uri_request/crossdoc_splade.py --sample 60
    # judge judge/judge_pairs.json -> judge/pair_labels.json
    python uri_request/crossdoc_finalize.py
