"""Phase 2: SPLADE merge scores for undecided cross-document pairs.

Pair universe (blocked, precision-oriented):
  B1: same normalized main title, different docs, no shared strong cite
      (phase-1 leftovers in ``crossdoc_phase1.json``).
  B2: shared candidate URI in SPLADE top-3 with different titles
      (spelling variants like Trade-marks/Trademarks, Charter/Canadian Charter).
  B3: shared strong citation with different titles.

For each pair: SPLADE dot(queryA, queryB) (embeddings cached to
``crossdoc_emb.npy``) + interpretable features. High band -> auto-merge
candidate, middle band -> ``judge_pairs.json`` for LLM-as-judge calibration,
low band -> no merge. Thresholds are set AFTER judging (see
``crossdoc_finalize.py``), never before.

Usage: ``python uri_request/crossdoc_splade.py [--sample N]``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apply_decision import extract_cites, query_main_title, title_key  # noqa: E402
from crossdoc_merge import UnionFind, strong_cites  # noqa: E402
from src.encoders import create_encoder  # noqa: E402

MISSION_DIR = Path(__file__).resolve().parent
FULL_PATH = MISSION_DIR / "uri_request.json"
PHASE1_PATH = MISSION_DIR / "crossdoc_phase1.json"
TOPK_PATH = MISSION_DIR / "output_splade" / "splade_topk.json"
EMB_PATH = MISSION_DIR / "crossdoc_emb.npy"
IDS_PATH = MISSION_DIR / "crossdoc_emb_ids.json"
OUT_PAIRS = MISSION_DIR / "crossdoc_pairs.json"
OUT_JUDGE = MISSION_DIR / "judge" / "judge_pairs.json"


def load_profiles() -> dict[str, dict]:
    full = json.loads(FULL_PATH.read_text(encoding="utf-8"))
    topk = {r["profile_key"]: r
            for r in json.loads(TOPK_PATH.read_text(encoding="utf-8"))}
    out = {}
    for doc in full["documents"]:
        for p in doc["profiles"]:
            q = p.get("query_text") or ""
            out[p["profile_key"]] = {
                "key": p["profile_key"], "doc": doc["source_document"],
                "docid": p.get("docid"), "doc_type": p.get("doc_type"),
                "query": q, "title": title_key(query_main_title(q)),
                "cites": sorted(strong_cites(" | ".join(p.get("citations", [])))),
                "top3": [h["uri"] for h in topk.get(p["profile_key"], {}).get("hits", [])[:3]],
                "top1_title": (topk.get(p["profile_key"], {}).get("hits", [{}])[0]
                               .get("candidate_title")),
            }
    return out


def build_universe(profiles: dict[str, dict], phase1: dict
                   ) -> dict[tuple[str, str], set[str]]:
    pairs: dict[tuple[str, str], set[str]] = defaultdict(set)
    protonorm = defaultdict(list)
    for p in profiles.values():
        protonorm[p["title"]].append(p["key"])
    for title, keys in protonorm.items():
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = profiles[keys[i]], profiles[keys[j]]
                if a["doc"] != b["doc"]:
                    pairs[(keys[i], keys[j])].add("same-title")
    by_cand: dict[str, list[str]] = defaultdict(list)
    for p in profiles.values():
        for u in p["top3"]:
            by_cand[u].append(p["key"])
    for u, keys in by_cand.items():
        uniq = sorted(set(keys))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                a, b = profiles[uniq[i]], profiles[uniq[j]]
                if a["doc"] != b["doc"] and a["title"] != b["title"]:
                    pairs[(uniq[i], uniq[j])].add("shared-top3")
    by_cite: dict[str, list[str]] = defaultdict(list)
    for p in profiles.values():
        for c in p["cites"]:
            by_cite[c].append(p["key"])
    for c, keys in by_cite.items():
        uniq = sorted(set(keys))
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                a, b = profiles[uniq[i]], profiles[uniq[j]]
                if a["doc"] != b["doc"] and a["title"] != b["title"]:
                    pairs[(uniq[i], uniq[j])].add("shared-cite")
    # Drop pairs already merged in phase 1 (same component): need components.
    merged = phase1.get("merged_multi_doc", {})
    comp = {}
    for root, keys in merged.items():
        for k in keys:
            comp[k] = root
    pairs = {pair: reasons for pair, reasons in pairs.items()
             if comp.get(pair[0], pair[0]) != comp.get(pair[1], pair[1])}
    return pairs


def get_embeddings(keys: list[str], queries: list[str]) -> np.ndarray:
    if EMB_PATH.exists() and IDS_PATH.exists():
        ids = json.loads(IDS_PATH.read_text(encoding="utf-8"))
        if ids == keys:
            print("[Emb] cache hit")
            return np.load(EMB_PATH)
    print("[Emb] encoding", len(keys), "queries with SPLADE ...")
    enc = create_encoder("splade")
    mat = enc.encode(queries).astype(np.float32)
    np.save(EMB_PATH, mat)
    IDS_PATH.write_text(json.dumps(keys), encoding="utf-8")
    return mat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    profiles = load_profiles()
    phase1 = json.loads(PHASE1_PATH.read_text(encoding="utf-8"))
    pairs = build_universe(profiles, phase1)
    print(f"[Pairs] universe={len(pairs)}")

    uniq_keys = sorted({k for pair in pairs for k in pair})
    index = {k: i for i, k in enumerate(uniq_keys)}
    mat = get_embeddings(uniq_keys, [profiles[k]["query"] for k in uniq_keys])

    scored = []
    for (ka, kb), reasons in pairs.items():
        va, vb = mat[index[ka]], mat[index[kb]]
        a, b = profiles[ka], profiles[kb]
        shared_cites = sorted(set(a["cites"]) & set(b["cites"]))
        scored.append({
            "a": ka, "b": kb, "blocks": sorted(reasons),
            "splade": float(va @ vb),
            "title_equal": a["title"] == b["title"],
            "shared_cites": shared_cites,
            "shared_top3": sorted(set(a["top3"]) & set(b["top3"])),
        })
    scored.sort(key=lambda r: -r["splade"])
    OUT_PAIRS.write_text(json.dumps(scored, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    scores = np.array([r["splade"] for r in scored])
    print(f"[Scores] n={len(scores)} min={scores.min():.2f} "
          f"p25={np.percentile(scores, 25):.2f} p50={np.percentile(scores, 50):.2f} "
          f"p75={np.percentile(scores, 75):.2f} max={scores.max():.2f}")

    rng = np.random.default_rng(args.seed)
    bands = {"low": [r for r in scored if r["splade"] < np.percentile(scores, 33)],
             "mid": [r for r in scored
                     if np.percentile(scores, 33) <= r["splade"] < np.percentile(scores, 75)],
             "high": [r for r in scored if r["splade"] >= np.percentile(scores, 75)]}
    sample = []
    for band, rows in bands.items():
        take = rng.choice(rows, size=min(args.sample // 3, len(rows)), replace=False)
        for r in take:
            a, b = profiles[r["a"]], profiles[r["b"]]
            sample.append({"band": band, **r,
                           "a_query": a["query"], "b_query": b["query"],
                           "a_docid": a["docid"], "b_docid": b["docid"],
                           "judge": ""})
    rng.shuffle(sample)
    OUT_JUDGE.parent.mkdir(parents=True, exist_ok=True)
    OUT_JUDGE.write_text(json.dumps(sample, indent=2, ensure_ascii=False),
                         encoding="utf-8")
    print(f"[DONE] pairs -> {OUT_PAIRS} | sample {len(sample)} -> {OUT_JUDGE}")


if __name__ == "__main__":
    main()
