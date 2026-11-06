"""Stratified sampling of SPLADE top-k results for LLM-as-judge calibration.

Reads ``output_splade/splade_topk.json`` (1551 profiles with top-k hits) and
draws a fixed-seed stratified sample: per doc_type x per-type score band, so
the judge sees low / mid / high / very-high confidence examples of every
type. Each sampled record is enriched with the candidate's full
``metadata_text`` (from the SPLADE ``candidates.json``) so correctness can be
judged without re-running retrieval.

Usage (from repo root)::

    python uri_request/sample_for_judge.py
    python uri_request/sample_for_judge.py --per-cell 8 --seed 7

Output (default ``uri_request/judge/``):
    judge_sample.json   full records: profile query + top-3 hits + metadata
    judge_sample.csv    compact table for spreadsheet review
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

from src.precompute_index import default_cache_dir  # noqa: E402

MISSION_DIR = Path(__file__).resolve().parent
DEFAULT_TOPK = MISSION_DIR / "output_splade" / "splade_topk.json"
DEFAULT_JUDGE_DIR = MISSION_DIR / "judge"


def load_candidates(cache_dir: Path) -> dict[str, dict]:
    with (cache_dir / "candidates.json").open("r", encoding="utf-8") as f:
        candidates = json.load(f)
    by_uri: dict[str, dict] = {}
    for c in candidates:
        for key in ("original_url", "uri"):
            if c.get(key):
                by_uri.setdefault(str(c[key]), c)
    return by_uri


def band_of(score: float, edges: list[float], labels: list[str]) -> str:
    for edge, label in zip(edges, labels):
        if score < edge:
            return label
    return labels[-1]


def main() -> None:
    p = argparse.ArgumentParser(description="Sample SPLADE hits for LLM judging.")
    p.add_argument("--topk-path", type=Path, default=DEFAULT_TOPK)
    p.add_argument("--judge-dir", type=Path, default=DEFAULT_JUDGE_DIR)
    p.add_argument("--cache-dir", type=Path, default=None)
    p.add_argument("--retriever", default="splade")
    p.add_argument("--per-cell", type=int, default=6)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    cache_dir = args.cache_dir or default_cache_dir(args.retriever)
    by_uri = load_candidates(cache_dir)

    with args.topk_path.open("r", encoding="utf-8") as f:
        results = json.load(f)

    # Per-type quartile edges -> 4 bands: low / mid-low / mid-high / high.
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        if r["top1_score"] is not None:
            by_type[str(r["doc_type"])].append(r)
    labels = ["low", "mid-low", "mid-high", "high"]
    edges = {t: list(np.percentile([x["top1_score"] for x in v], [25, 50, 75]))
             for t, v in by_type.items()}
    print("[Bands] per-type quartile edges:")
    for t, e in edges.items():
        print(f"  {t}: " + ", ".join(f"{l}>={v:.2f}" if i else f"{l}<{v:.2f}"
              for i, (l, v) in enumerate(zip(labels, [e[0]] * 1 + e))))

    rng = np.random.default_rng(args.seed)
    sample: list[dict] = []
    unmatched = 0
    for dtype, rows in sorted(by_type.items()):
        cells: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            cells[band_of(r["top1_score"], edges[dtype], labels)].append(r)
        for label in labels:
            cell = cells[label]
            take = rng.choice(cell, size=min(args.per_cell, len(cell)), replace=False)
            for r in take:
                hits = []
                for h in r["hits"][:3]:
                    meta = by_uri.get(str(h["uri"]), {})
                    if not meta:
                        unmatched += 1
                    hits.append({**h, "metadata_text": meta.get("metadata_text")})
                sample.append({
                    "profile_key": r["profile_key"],
                    "source_document": r["source_document"],
                    "docid": r["docid"],
                    "doc_type": r["doc_type"],
                    "band": label,
                    "query_text": r["query_text"],
                    "top1_score": r["top1_score"],
                    "margin_top1_top2": r["margin_top1_top2"],
                    "top3": hits,
                    "judge": "",  # filled by the LLM judge: CORRECT / WRONG / UNCERTAIN
                })
    # Chronological-ish shuffle so bands/types interleave for the judge.
    rng.shuffle(sample)

    args.judge_dir.mkdir(parents=True, exist_ok=True)
    with (args.judge_dir / "judge_sample.json").open("w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    with (args.judge_dir / "judge_sample.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["doc_type", "band", "top1_score", "margin", "profile_key",
                    "docid", "top1_title", "top1_uri", "query_text"])
        for s in sample:
            w.writerow([s["doc_type"], s["band"], round(s["top1_score"], 3),
                        round(s["margin_top1_top2"], 3) if s["margin_top1_top2"] is not None else "",
                        s["profile_key"], s["docid"], s["top3"][0]["candidate_title"] if s["top3"] else "",
                        s["top3"][0]["uri"] if s["top3"] else "",
                        (s["query_text"] or "").replace("\n", " ")[:400]])
    print(f"\n[DONE] {len(sample)} sampled | unmatched candidate joins: {unmatched} "
          f"-> {args.judge_dir}/judge_sample.json")


if __name__ == "__main__":
    main()
