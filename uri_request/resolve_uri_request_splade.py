"""SPLADE resolution + confidence study for the no-gold URI request.

Mission folder: ``uri_request/`` (self-contained).
Input profiles: ``uri_request/uri_request.json`` — 55 no-gold documents,
1551 coref profiles produced by ``scripts/export_no_gold_profiles.py``
(coref run: gemma4 / DEC / greedy / paragraph).

What this does (mirrors ``src/main_resolution.py`` stage 3 exactly):
  1. Loads the SPLADE candidate-pool index built by
     ``python -m src.precompute_index --retriever splade``
     (``cache/candidate_pool_index_splade_v1/`` <- ``data/candidate_pool.csv``).
  2. Uses each profile's ``query_text`` as the query (the same string stage 3
     builds with ``build_profile_text``), groups profiles by doctype bucket,
     and calls ``retriever.search(texts, k=topk, subset_idx=...)`` in batches.
  3. Records the full top-k + scores per profile so you can STUDY the
     confidence (score) distribution BEFORE choosing a threshold.
  4. With ``--threshold T``: classifies each profile as
     EXISTS (top1 score >= T -> uri assigned from the pool) or
     MISSING (top1 score < T -> uri not currently in the candidate pool),
     and reports which source documents have missing profiles.

Why a threshold: SPLADE scores are raw dot products (see
``src/encoders/splade.py``: ``similarity = "dot"``). They are NOT calibrated
probabilities and they grow with query length / term weights, so there is no
universal "0.9 = sure" cutoff. The right workflow is:

  Step 1 (no threshold): run this script WITHOUT --threshold.
  Step 2: read the printed percentile table / threshold sweep and the
          ``score_analysis.json`` histogram; pick T where the top1-score
          distribution separates (and where the top1-top2 margin is large).
  Step 3: re-run WITH --threshold T to get ``uri_decision.*`` +
          ``missing_by_document.json`` = the answer to
          "which URIs exist in candidate_pool.csv and which source
          documents are NOT in the pool".

Usage (from repo root)::

    # 1. Confidence study (no decision yet) — top-5 per profile
    python uri_request/resolve_uri_request_splade.py

    # 2. Quick smoke test on 3 profiles (checks the SPLADE model loads)
    python uri_request/resolve_uri_request_splade.py --limit 3

    # 3. Apply a threshold once chosen from the sweep table
    python uri_request/resolve_uri_request_splade.py --threshold 12.0

    # 4. Exact stage-3 parity flags (defaults already match main_resolution)
    python uri_request/resolve_uri_request_splade.py --topk 5 --device cpu --batch-size 16
    python uri_request/resolve_uri_request_splade.py --no-filter-by-doctype  # full-pool search

Outputs (under ``--out-dir``, default ``uri_request/output_splade/``):
    splade_topk.json        per-profile top-k hits {uri, score, candidate_title}
    splade_topk.csv         flat table for spreadsheet / pandas analysis
    score_analysis.json     percentiles, histogram, margin stats, threshold sweep
    uri_decision.json/csv   ONLY with --threshold: EXISTS/MISSING per profile
    missing_by_document.json ONLY with --threshold: source_document -> missing profiles
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main_resolution import (  # noqa: E402 — same logic as stage 3
    _normalize_doctype,
    _subset_for_doctype,
    build_doctype_index,
    load_candidate_pool,
)
from src.precompute_index import default_cache_dir  # noqa: E402

MISSION_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = MISSION_DIR / "uri_request.json"
DEFAULT_OUT_DIR = MISSION_DIR / "output_splade"

# Thresholds evaluated for the sweep when no --threshold is given.
# SPLADE dot scores are unbounded; this grid covers the plausible range and
# the sweep table shows, for each T, how many profiles would count as EXISTS.
SWEEP_THRESHOLDS = [1.0, 2.0, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0, 25.0, 30.0]


def load_profiles(input_path: Path, limit: int | None = None) -> list[dict]:
    """Flatten uri_request.json into [{source_document, profile_key, ...}]."""
    with input_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    flat: list[dict] = []
    for doc in payload.get("documents", []):
        for prof in doc.get("profiles", []):
            flat.append(
                {
                    "source_document": doc.get("source_document"),
                    "profile_key": prof.get("profile_key"),
                    "docid": prof.get("docid"),
                    "doc_type": prof.get("doc_type"),
                    "query_text": prof.get("query_text") or "",
                    "n_mentions": prof.get("n_mentions"),
                }
            )
            if limit is not None and len(flat) >= limit:
                return flat
    return flat


def run_search(
    profiles: list[dict],
    retriever,
    candidates: list[dict],
    doctype_index: dict,
    filter_by_doctype: bool,
    topk: int,
) -> tuple[list[dict], int]:
    """Batched SPLADE search, grouped by doctype exactly like stage 3.

    Returns (results, n_empty_query).
    """
    groups: dict[str | None, list[int]] = defaultdict(list)
    n_empty = 0
    for i, prof in enumerate(profiles):
        if not prof["query_text"].strip():
            n_empty += 1
            continue
        subset = _subset_for_doctype(prof["doc_type"], doctype_index, filter_by_doctype)
        key = None if subset is None else _normalize_doctype(prof["doc_type"])
        groups[key].append(i)

    results: list[dict] = [None] * len(profiles)  # type: ignore[list-item]
    for group_key, idxs in groups.items():
        subset_idx = None if group_key is None else doctype_index[group_key]
        texts = [profiles[i]["query_text"] for i in idxs]
        top_idx, top_scores = retriever.search(texts, k=topk, subset_idx=subset_idx)
        for pos, i in enumerate(idxs):
            row_idx, row_scores = top_idx[pos], top_scores[pos]
            hits = [
                {
                    "rank": r + 1,
                    "uri": candidates[int(ci)]["original_url"]
                    if "original_url" in candidates[int(ci)]
                    else candidates[int(ci)].get("uri"),
                    "score": float(s),
                    "candidate_title": candidates[int(ci)].get("docTitle"),
                    "candidate_doctype": candidates[int(ci)].get("docType"),
                }
                for r, (ci, s) in enumerate(zip(row_idx, row_scores))
            ]
            top1 = float(row_scores[0])
            top2 = float(row_scores[1]) if len(row_scores) > 1 else None
            results[i] = {
                **profiles[i],
                "search_subset": group_key or "FULL_POOL",
                "top1_score": top1,
                "top1_uri": hits[0]["uri"],
                "top1_title": hits[0]["candidate_title"],
                "margin_top1_top2": (top1 - top2) if top2 is not None else None,
                "hits": hits,
            }

    # Empty-query profiles: no retrieval possible -> MISSING by construction.
    for i, prof in enumerate(profiles):
        if results[i] is None:
            results[i] = {
                **prof,
                "search_subset": None,
                "top1_score": None,
                "top1_uri": None,
                "top1_title": None,
                "margin_top1_top2": None,
                "hits": [],
            }
    return results, n_empty


def score_analysis(results: list[dict], sweep: list[float]) -> dict:
    scored = np.array([r["top1_score"] for r in results if r["top1_score"] is not None])
    margins = np.array(
        [r["margin_top1_top2"] for r in results if r["margin_top1_top2"] is not None]
    )
    analysis: dict = {
        "n_profiles": len(results),
        "n_scored": int(scored.size),
        "n_empty_query": int(len(results) - scored.size),
    }
    if scored.size:
        analysis["top1_score"] = {
            "min": float(scored.min()),
            "max": float(scored.max()),
            "mean": float(scored.mean()),
            "std": float(scored.std()),
            "percentiles": {
                f"p{p}": float(v)
                for p, v in zip(
                    [1, 5, 10, 25, 50, 75, 90, 95, 99],
                    np.percentile(scored, [1, 5, 10, 25, 50, 75, 90, 95, 99]),
                )
            },
            "hist_counts_20bins": [int(c) for c in np.histogram(scored, bins=20)[0]],
            "hist_bin_edges_20bins": [float(e) for e in np.histogram(scored, bins=20)[1]],
        }
    if margins.size:
        analysis["margin_top1_top2"] = {
            "mean": float(margins.mean()),
            "percentiles": {
                f"p{p}": float(v)
                for p, v in zip(
                    [10, 25, 50, 75, 90],
                    np.percentile(margins, [10, 25, 50, 75, 90]),
                )
            },
        }
    # Threshold sweep: for each T, how many profiles would be EXISTS vs MISSING.
    sweep_table = []
    for t in sweep:
        n_exists = int(sum(1 for r in results if (r["top1_score"] or -1) >= t))
        sweep_table.append(
            {
                "threshold": t,
                "n_exists": n_exists,
                "n_missing": len(results) - n_exists,
                "exists_rate": round(n_exists / len(results), 4) if results else 0.0,
            }
        )
    analysis["threshold_sweep"] = sweep_table
    # Per-doc-type median top1: thresholds may need to differ by type.
    by_type: dict[str, list[float]] = defaultdict(list)
    for r in results:
        if r["top1_score"] is not None:
            by_type[str(r["doc_type"])].append(r["top1_score"])
    analysis["top1_median_by_doctype"] = {
        k: float(np.median(v)) for k, v in sorted(by_type.items())
    }
    return analysis


def apply_threshold(results: list[dict], threshold: float) -> tuple[list[dict], dict]:
    decisions = []
    missing_by_doc: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        exists = r["top1_score"] is not None and r["top1_score"] >= threshold
        decisions.append(
            {
                "profile_key": r["profile_key"],
                "source_document": r["source_document"],
                "docid": r["docid"],
                "doc_type": r["doc_type"],
                "decision": "EXISTS" if exists else "MISSING",
                "top1_score": r["top1_score"],
                "top1_uri": r["top1_uri"],
                "top1_title": r["top1_title"],
                "margin_top1_top2": r["margin_top1_top2"],
            }
        )
        if not exists:
            missing_by_doc[r["source_document"]].append(
                {
                    "profile_key": r["profile_key"],
                    "docid": r["docid"],
                    "top1_score": r["top1_score"],
                    "top1_uri": r["top1_uri"],
                }
            )
    return decisions, dict(missing_by_doc)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SPLADE resolution + confidence study for uri_request profiles."
    )
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--retriever", default="splade")
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="If given, classify EXISTS (>= T) vs MISSING (< T). Omit for study-only run.",
    )
    p.add_argument("--no-filter-by-doctype", action="store_true")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="Smoke test: first N profiles only.")
    p.add_argument("--overwrite", action="store_true")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    cache_dir = args.cache_dir or default_cache_dir(args.retriever)

    print(f"[Setup] Profiles  : {args.input}")
    profiles = load_profiles(args.input, limit=args.limit)
    print(f"[Setup] {len(profiles)} profile(s){' (LIMIT smoke test)' if args.limit else ''}")
    print(f"[Setup] Index     : {cache_dir} (retriever={args.retriever})")

    kwargs = {}
    if args.device is not None:
        kwargs["device"] = args.device
    if args.batch_size is not None:
        kwargs["batch_size"] = args.batch_size
    retriever, candidates = load_candidate_pool(cache_dir, args.retriever, **kwargs)
    print(f"[Setup] {len(candidates)} candidates | filter_by_doctype={not args.no_filter_by_doctype}")
    doctype_index = build_doctype_index(candidates)

    results, n_empty = run_search(
        profiles, retriever, candidates, doctype_index,
        filter_by_doctype=not args.no_filter_by_doctype, topk=args.topk,
    )
    print(f"[Search] scored={len(results) - n_empty}, empty_query={n_empty}")

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    with (out_dir / "splade_topk.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    with (out_dir / "splade_topk.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_document", "profile_key", "docid", "doc_type",
                    "n_mentions", "search_subset", "top1_uri", "top1_title",
                    "top1_score", "margin_top1_top2", "query_text"])
        for r in results:
            w.writerow([r["source_document"], r["profile_key"], r["docid"],
                        r["doc_type"], r["n_mentions"], r["search_subset"],
                        r["top1_uri"], r["top1_title"], r["top1_score"],
                        r["margin_top1_top2"],
                        (r["query_text"] or "").replace("\n", " ")[:500]])

    analysis = score_analysis(results, SWEEP_THRESHOLDS)
    with (out_dir / "score_analysis.json").open("w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    print("\n--- top1-score percentiles (SPLADE dot) ---")
    for k, v in analysis.get("top1_score", {}).get("percentiles", {}).items():
        print(f"  {k}: {v:.3f}")
    print("median top1 by doc_type:", analysis.get("top1_median_by_doctype"))
    print("\n--- threshold sweep: T -> EXISTS (uri in pool) vs MISSING ---")
    for row in analysis["threshold_sweep"]:
        print(f"  T={row['threshold']:>5}: EXISTS {row['n_exists']:>5} "
              f"({row['exists_rate']:.1%}) | MISSING {row['n_missing']:>5}")

    if args.threshold is not None:
        decisions, missing_by_doc = apply_threshold(results, args.threshold)
        with (out_dir / "uri_decision.json").open("w", encoding="utf-8") as f:
            json.dump(decisions, f, indent=2, ensure_ascii=False)
        with (out_dir / "uri_decision.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["profile_key", "source_document", "docid", "decision",
                        "top1_score", "top1_uri", "top1_title"])
            for d in decisions:
                w.writerow([d["profile_key"], d["source_document"], d["docid"],
                            d["decision"], d["top1_score"], d["top1_uri"], d["top1_title"]])
        with (out_dir / "missing_by_document.json").open("w", encoding="utf-8") as f:
            json.dump(missing_by_doc, f, indent=2, ensure_ascii=False)
        n_exists = sum(1 for d in decisions if d["decision"] == "EXISTS")
        print(f"\n[Decision T={args.threshold}] EXISTS={n_exists} "
              f"MISSING={len(decisions) - n_exists} "
              f"across {len(missing_by_doc)} affected source document(s)")
        print(f"[Decision] 'what is not in the pool' -> missing_by_document.json")
    else:
        print("\n[Decision] no --threshold given: study-only run. "
              "Pick T from the sweep above, then re-run with --threshold T.")

    print(f"\n[DONE] -> {out_dir}")


if __name__ == "__main__":
    main()
