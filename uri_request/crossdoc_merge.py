"""Cross-document profile merging: one global group per real-world source.

Within-document coref (stage 2) groups mentions per document, but the same
source cited in two documents yields two profiles (``DOC::docid``). The
company wants ONE group per source across all 55 no-gold documents.

Pipeline (precision-first; a wrong merge corrupts the company's URI table):

  Phase 1 (this script, deterministic, no judge):
    R1: same verified URI (both EXISTS, ``output_decision``) -> merge.
    R2: same normalized main title + shared STRONG citation -> merge.
    HARD CONSTRAINT: never merge two profiles with distinct verified URIs.
  Phase 2 (``crossdoc_splade.py``): SPLADE dot(queryA, queryB) for blocked
    pairs left undecided; auto-merge high band, LLM-judge middle band.
  Phase 3 (``crossdoc_finalize.py``): union-find closure -> merged registry.

Usage: ``python uri_request/crossdoc_merge.py`` (from repo root).
Output: ``uri_request/crossdoc_phase1.json`` (proposed merges + stats).
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apply_decision import extract_cites, norm, query_main_title, title_key  # noqa: E402

MISSION_DIR = Path(__file__).resolve().parent
FULL_PATH = MISSION_DIR / "uri_request.json"
DECISION_PATH = MISSION_DIR / "output_decision" / "uri_decision.json"
TOPK_PATH = MISSION_DIR / "output_splade" / "splade_topk.json"
OUT_PATH = MISSION_DIR / "crossdoc_phase1.json"


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def strong_cites(text: str) -> set[str]:
    """Citation signatures identifying one work (drop year-only fragments)."""
    return {c for c in extract_cites(text or "") if len(c) >= 6}


def main() -> None:
    full = json.loads(FULL_PATH.read_text(encoding="utf-8"))
    decisions = {d["profile_key"]: d
                 for d in json.loads(DECISION_PATH.read_text(encoding="utf-8"))}
    topk = {r["profile_key"]: r
            for r in json.loads(TOPK_PATH.read_text(encoding="utf-8"))}

    profiles = []
    for doc in full["documents"]:
        for p in doc["profiles"]:
            key = p["profile_key"]
            d = decisions.get(key, {})
            profiles.append({
                "key": key,
                "doc": doc["source_document"],
                "docid": p.get("docid"),
                "title": title_key(query_main_title(p.get("query_text") or "")),
                "cites": strong_cites(" | ".join(p.get("citations", []))),
                "verified_uri": d.get("uri") if d.get("decision") == "EXISTS" else None,
                "top_uris": [h["uri"] for h in topk.get(key, {}).get("hits", [])[:5]],
            })
    print(f"[Setup] {len(profiles)} profiles")

    uf = UnionFind()
    r1 = r2 = 0
    by_uri: dict[str, list[str]] = defaultdict(list)
    for p in profiles:
        if p["verified_uri"]:
            by_uri[p["verified_uri"]].append(p["key"])
    for uri, keys in by_uri.items():
        for k in keys[1:]:
            uf.union(keys[0], k)
            r1 += 1

    by_title: dict[str, list[dict]] = defaultdict(list)
    for p in profiles:
        by_title[p["title"]].append(p)
    for title, group in by_title.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a["doc"] == b["doc"]:
                    continue
                if a["verified_uri"] and b["verified_uri"] \
                        and a["verified_uri"] != b["verified_uri"]:
                    continue  # HARD CONSTRAINT
                if a["cites"] and (a["cites"] & b["cites"]):
                    uf.union(a["key"], b["key"])
                    r2 += 1

    groups: dict[str, list[str]] = defaultdict(list)
    for p in profiles:
        groups[uf.find(p["key"])].append(p["key"])
    multi = {k: v for k, v in groups.items() if len({x.split("::")[0] for x in v}) > 1}
    print(f"[Phase1] R1 same-uri merges={r1} | R2 title+cite merges={r2}")
    print(f"[Phase1] {len(groups)} components | {len(multi)} span >1 doc "
          f"({sum(len(v) for v in multi.values())} profiles)")

    # Undecided blocks for phase 2: same title but NO shared cite (or no
    # cites at all), different docs, no conflicting verified URIs.
    undecided: list[list[str]] = []
    for title, group in by_title.items():
        docs = defaultdict(list)
        for p in group:
            docs[p["doc"]].append(p)
        if len(docs) < 2:
            continue
        pairs = []
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                if a["doc"] == b["doc"]:
                    continue
                if uf.find(a["key"]) == uf.find(b["key"]):
                    continue
                if a["verified_uri"] and b["verified_uri"] \
                        and a["verified_uri"] != b["verified_uri"]:
                    continue
                if a["cites"] and (a["cites"] & b["cites"]):
                    continue  # already merged by R2
                pairs.append((a["key"], b["key"]))
        if pairs:
            undecided.append({"title": title, "pairs": pairs})
    n_pairs = sum(len(b["pairs"]) for b in undecided)
    print(f"[Phase1] undecided blocks={len(undecided)} pairs={n_pairs}")

    OUT_PATH.write_text(json.dumps({
        "r1_merges": r1, "r2_merges": r2,
        "merged_multi_doc": {k: sorted(v) for k, v in sorted(multi.items())},
        "undecided_blocks": undecided,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] -> {OUT_PATH}")


if __name__ == "__main__":
    main()
