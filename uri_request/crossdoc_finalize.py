"""Phase 3: finalize cross-document groups (rules + closure + outputs).

Merge rules (all precision-first; SPLADE dot is NOT a merge criterion —
judge calibration showed 57/60 sampled pairs DIFFERENT across ALL score
bands, incl. 31.5 high: similarity != identity):

  R1: same accepted URI in ``uri_decision.json`` (phase 1).
  R2: same normalized main title + shared STRONG citation (phase 1).
  R3: same title + both queries verify to the SAME pool candidate
      (``Verifier`` on top-3, any score; pool confirms identity, e.g. CCRA
      with cite on one side only).
  R4: charter alias: {charter} <-> {canadiancharterofrightsandfreedoms},
      provided neither query carries Quebec markers.
  R5: decisions: same full "A v B" case name on both sides + cite-compatible
      (shared cite, or both citeless). Bare single-party names never merge.
  R6: secondary: same title + overlapping author set (both non-empty).

HARD CONSTRAINTS (checked at every union AND post-closure):
  H1: never merge two profiles with distinct verified (EXISTS) URIs.
  H2: post-closure groups containing >=2 distinct verified URIs are
      quarantined as CONFLICT (no uri assigned).

Outputs (``uri_request/``):
  ``uri_request_merged.json``          all groups (VERIFIED/UNVERIFIED/CONFLICT)
  ``uri_request_merged_missing.json``  company send-out: one entry per group
                                       still needing a uri (members listed)

Usage: ``python uri_request/crossdoc_finalize.py``.
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.path.insert(0, str(Path(__file__).resolve().parent))
from apply_decision import Verifier, extract_cites, norm, query_main_title, title_key  # noqa: E402
from apply_decision import statute_jurisdictions  # noqa: E402
from crossdoc_merge import UnionFind, strong_cites  # noqa: E402

MISSION_DIR = Path(__file__).resolve().parent
FULL_PATH = MISSION_DIR / "uri_request.json"
DECISION_PATH = MISSION_DIR / "output_decision" / "uri_decision.json"
TOPK_PATH = MISSION_DIR / "output_splade" / "splade_topk.json"
CAND_PATH = Path("cache/candidate_pool_index_splade_v1/candidates.json")
PHASE1_PATH = MISSION_DIR / "crossdoc_phase1.json"
OUT_MERGED = MISSION_DIR / "uri_request_merged.json"
OUT_MISSING = MISSION_DIR / "uri_request_merged_missing.json"

QUEBEC_RX = re.compile(r"qu[eé]bec|\bloi\b|cqlr|lrq|rsq|rlrq", re.I)
CHARTER_ALIAS = {"charter", "canadiancharterofrightsandfreedoms"}
# Titles that can only denote the Canadian Charter (Quebec's charter is
# titled "Charter of human rights and freedoms" — deliberately excluded).
CHARTER_TITLES = {"charter", "canadiancharterofrightsandfreedoms",
                  "charterofrightsandfreedoms", "charterofrights"}


def load() -> tuple[dict, dict, Verifier]:
    full = json.loads(FULL_PATH.read_text(encoding="utf-8"))
    decisions = {d["profile_key"]: d
                 for d in json.loads(DECISION_PATH.read_text(encoding="utf-8"))}
    topk = {r["profile_key"]: r
            for r in json.loads(TOPK_PATH.read_text(encoding="utf-8"))}
    cands = json.loads(CAND_PATH.read_text(encoding="utf-8"))
    verifier = Verifier(cands)
    profiles = {}
    for doc in full["documents"]:
        for p in doc["profiles"]:
            q = p.get("query_text") or ""
            key = p["profile_key"]
            d = decisions.get(key, {})
            authors = {norm(a) for a in p.get("authors", []) if norm(a)}
            hits = topk.get(key, {}).get("hits", [])[:3]
            verified = set()
            for h in hits:
                ok, _ = verifier.check(h, q)
                if ok:
                    meta = verifier.meta_by_uri.get(str(h.get("uri")), {})
                    uri = meta.get("original_url") or meta.get("uri") or h.get("uri")
                    verified.add(str(uri))
            profiles[key] = {
                "key": key, "doc": doc["source_document"], "docid": p.get("docid"),
                "doc_type": p.get("doc_type"), "query": q,
                "title": title_key(query_main_title(q)),
                "spaced": re.sub(r"\s+", " ",
                                 re.sub(r"[^a-z0-9 ]", " ", (p.get("docid") or "").lower())).strip(), "authors": sorted(authors),
                "cites": strong_cites(" | ".join(p.get("citations", []))),
                "uri": d.get("uri") if d.get("decision") == "EXISTS" else None,
                "top1": topk.get(key, {}).get("top1_uri"),
                "top1_score": topk.get(key, {}).get("top1_score"),
                "verified_cands": sorted(verified),
                # Full original registry evidence for the company match
                # (same fields as uri_request.json profiles).
                "main_title": p.get("main_title"),
                "alternative_titles": p.get("alternative_titles", []),
                "citations": p.get("citations", []),
                "fragments_mentioned": p.get("fragments_mentioned", []),
                "authors_raw": p.get("authors", []),
                "first_seen_id": p.get("first_seen_id"),
                "n_mentions": p.get("n_mentions"),
                "mention_ids": p.get("mention_ids", []),
                "mention_texts": p.get("mention_texts", []),
            }
    return profiles, json.loads(PHASE1_PATH.read_text(encoding="utf-8")), verifier


def quebec_marked(q: str) -> bool:
    return bool(QUEBEC_RX.search(q or ""))


def full_case_name(spaced_title: str) -> bool:
    return " v " in f" {spaced_title} "


def try_union(uf: UnionFind, profiles: dict, a: str, b: str, rule: str,
              counts: dict) -> bool:
    pa, pb = profiles[a], profiles[b]
    if pa["uri"] and pb["uri"] and pa["uri"] != pb["uri"]:
        return False  # H1
    if uf.find(a) == uf.find(b):
        return False
    # H1 across components: joining must not mix distinct verified URIs.
    uris: set[str] = set()
    for k, p in profiles.items():
        if p["uri"] and (uf.find(k) == uf.find(a) or uf.find(k) == uf.find(b)):
            uris.add(p["uri"])
    if len(uris) > 1:
        return False
    uf.union(a, b)
    counts[rule] += 1
    return True


def main() -> None:
    profiles, phase1, verifier = load()
    uf = UnionFind()
    counts: dict[str, int] = defaultdict(int)

    # Replay phase-1 merges through the guarded union (H1 enforced).
    for keys in phase1.get("merged_multi_doc", {}).values():
        for k in keys[1:]:
            try_union(uf, profiles, keys[0], k, "R1/R2", counts)

    keys = sorted(profiles)
    by_title: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        by_title[profiles[k]["title"]].append(k)

    # Ambiguity oracles for R7: a title demonstrably naming several works
    # must never merge on titles alone.
    pool_title_counts = verifier.title_counts
    corpus_title_uris: dict[str, set[str]] = defaultdict(set)
    for k in keys:
        if profiles[k]["uri"]:
            corpus_title_uris[profiles[k]["title"]].add(profiles[k]["uri"])

    def r7_safe(title: str) -> bool:
        return (len(title) >= 12
                and pool_title_counts.get(title, 0) <= 1
                and len(corpus_title_uris.get(title, set())) <= 1)

    for title, group in by_title.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = profiles[group[i]], profiles[group[j]]
                if a["doc"] == b["doc"]:
                    continue
                ka, kb = a["key"], b["key"]
                # R3: pool confirms the same candidate on both sides.
                if a["doc_type"] == b["doc_type"]:
                    # R3: pool confirms the same candidate on both sides.
                    if set(a["verified_cands"]) & set(b["verified_cands"]):
                        try_union(uf, profiles, ka, kb, "R3", counts)
                        continue
                    # R7: same specific title, cite-compatible (shared, or
                    # citeless on at least one side), title unambiguous in
                    # pool and corpus. Catches pool-absent works like the CCRA.
                    # Jurisdiction guard: two different provinces' acts sharing
                    # a title never merge (e.g. NB vs Alberta Workers' Comp).
                    if r7_safe(a["title"]) and (
                            (a["cites"] & b["cites"])
                            or not a["cites"] or not b["cites"]):
                        ja = statute_jurisdictions(" ".join(a["cites"]))
                        jb = statute_jurisdictions(" ".join(b["cites"]))
                        if ja and jb and not (ja & jb):
                            continue
                        try_union(uf, profiles, ka, kb, "R7", counts)
                        continue
                # R5: decisions with identical full case names + cite-compatible.
                # R5: decisions with identical distinctive full case names +
                # cite-compatible (shared cite, or citeless on at least one
                # side). Short/generic names never merge on titles alone.
                if a["doc_type"] == "decision" and b["doc_type"] == "decision" \
                        and full_case_name(a["spaced"]) and full_case_name(b["spaced"]) \
                        and len(a["spaced"].split()) >= 4 \
                        and len(b["spaced"].split()) >= 4 \
                        and a["title"] == b["title"]:
                    if not a["cites"] or not b["cites"] or (a["cites"] & b["cites"]):
                        try_union(uf, profiles, ka, kb, "R5", counts)
                        continue
                # R6: secondary with overlapping authors.
                if a["doc_type"] == "secondary sources" \
                        and b["doc_type"] == "secondary sources":
                    if set(a["authors"]) & set(b["authors"]):
                        try_union(uf, profiles, ka, kb, "R6", counts)

    # R4: charter alias pairs (titles differ: charter <-> canadian charter).
    charter_keys = [k for k in keys
                    if profiles[k]["title"] in CHARTER_TITLES
                    and not quebec_marked(profiles[k]["query"])]
    for i in range(len(charter_keys)):
        for j in range(i + 1, len(charter_keys)):
            a, b = profiles[charter_keys[i]], profiles[charter_keys[j]]
            if a["doc"] == b["doc"] or a["title"] == b["title"]:
                continue
            try_union(uf, profiles, a["key"], b["key"], "R4", counts)

    components: dict[str, list[str]] = defaultdict(list)
    for k in keys:
        components[uf.find(k)].append(k)

    cand_title_by_uri = {}
    cand_meta_by_uri = {}
    for c in json.loads(CAND_PATH.read_text(encoding="utf-8")):
        for kk in ("original_url", "uri"):
            if c.get(kk):
                cand_title_by_uri.setdefault(str(c[kk]), c.get("docTitle") or "")
                cand_meta_by_uri.setdefault(str(c[kk]), c.get("metadata_text") or "")

    def evict_member(p: dict, uri: str) -> str | None:
        """Why a legislation member must leave a VERIFIED group (None = keep).

        A verified group URI must be valid for EVERY member: a member citing
        another jurisdiction's enactment (NB vs Alberta Workers' Comp), or a
        differently-titled instrument with no shared statute cite (an
        amendment merged with its base act), is a different source.
        Decisions are exempt (candidate metadata omits cites too often).
        """
        if p["doc_type"] != "legislation":
            return None
        meta = cand_meta_by_uri.get(uri, "")
        jm = statute_jurisdictions(" ".join(p["cites"]))
        jc = statute_jurisdictions(meta)
        if jm and jc and not (jm & jc):
            return "jurisdiction-mismatch"
        if title_key(query_main_title(p["query"])) == title_key(
                cand_title_by_uri.get(uri, "")):
            return None
        if {c for c in p["cites"] if any(ch.isdigit() for ch in c)} & {
                c for c in extract_cites(meta) if len(c) >= 6}:
            return None
        # Charter alias: Canadian-Charter-family titles belong to the
        # Constitution Act group unless Quebec-marked or citing elsewhere.
        if title_key(cand_title_by_uri.get(uri, "")) == "constitutionact1982" \
                and title_key(query_main_title(p["query"])) in CHARTER_TITLES \
                and not quebec_marked(p["query"]):
            qj = statute_jurisdictions(" ".join(p["cites"]))
            cj = statute_jurisdictions(meta)
            if not (qj and cj and not (qj & cj)):
                return None
        return "title-without-shared-cite"

    # Post-closure split: evicted members become UNVERIFIED singletons.
    final_components: dict[str, list[str]] = {}
    for root, members in components.items():
        uris = {profiles[k]["uri"] for k in members if profiles[k]["uri"]}
        if len(uris) == 1:
            uri = next(iter(uris))
            kept = [k for k in members if evict_member(profiles[k], uri) is None]
            for k in members:
                if k not in kept:
                    final_components[f"evicted:{k}"] = [k]
            if kept:
                final_components[root] = kept
        else:
            final_components[root] = members
    components = final_components

    groups = []
    for gid, (root, members) in enumerate(sorted(components.items())):
        members = sorted(members)
        docs = sorted({profiles[k]["doc"] for k in members})
        uris = {profiles[k]["uri"] for k in members if profiles[k]["uri"]}
        titles = sorted({profiles[k]["docid"] for k in members})
        if len(uris) > 1:
            status, uri = "CONFLICT", ""  # H2 quarantine
        elif len(uris) == 1:
            status, uri = "VERIFIED", next(iter(uris))
        else:
            status, uri = "UNVERIFIED", ""
        # Canonical title: most frequent docid, tie -> longest.
        freq: dict[str, int] = defaultdict(int)
        for k in members:
            freq[profiles[k]["docid"]] += 1
        canon = sorted(freq, key=lambda t: (-freq[t], -len(t)))[0]
        needs_review = bool(
            uri and title_key(canon) != title_key(cand_title_by_uri.get(uri, uri)))
        # Group-level evidence pool for the company match: union of every
        # member's registry fields (same vocabulary as uri_request.json).
        extra_metadata: dict[str, list] = {}
        for field in ("alternative_titles", "citations", "fragments_mentioned",
                      "authors_raw"):
            seen: dict[str, None] = {}
            for k in members:
                for v in profiles[k].get(field) or []:
                    seen.setdefault(str(v))
            extra_metadata[field] = sorted(seen)
        extra_metadata["query_texts"] = sorted(
            {profiles[k]["query"] for k in members})
        extra_metadata["n_mentions_total"] = sum(
            profiles[k].get("n_mentions") or 0 for k in members)
        groups.append({
            "group_id": f"G{gid:04d}",
            "canonical_title": canon,
            "needs_review": needs_review,
            "doc_type": max({profiles[k]["doc_type"] for k in members},
                            key=lambda t: sum(1 for k in members
                                              if profiles[k]["doc_type"] == t)),
            "status": status, "uri": uri,
            "n_members": len(members), "n_docs": len(docs),
            "titles": titles, "documents": docs,
            "extra_metadata": extra_metadata,
            "members": [{
                "profile_key": k, "source_document": profiles[k]["doc"],
                "docid": profiles[k]["docid"],
                "doc_type": profiles[k]["doc_type"],
                "main_title": profiles[k]["main_title"],
                "alternative_titles": profiles[k]["alternative_titles"],
                "citations": profiles[k]["citations"],
                "fragments_mentioned": profiles[k]["fragments_mentioned"],
                "authors": profiles[k]["authors_raw"],
                "first_seen_id": profiles[k]["first_seen_id"],
                "n_mentions": profiles[k]["n_mentions"],
                "mention_ids": profiles[k]["mention_ids"],
                "mention_texts": profiles[k]["mention_texts"],
                "query_text": profiles[k]["query"],
                "top1_uri": profiles[k]["top1"],
                "top1_score": profiles[k]["top1_score"],
            } for k in members],
        })

    groups.sort(key=lambda g: (g["status"] != "CONFLICT", g["group_id"]))
    OUT_MERGED.write_text(json.dumps({"meta": {
        "producer": "uri_request/crossdoc_finalize.py",
        "rules": ["R1 same accepted uri", "R2 same title + shared strong cite",
                  "R3 same title/type + same pool-verified candidate",
                  "R4 charter alias (quebec excluded)",
                  "R5 identical distinctive full case name + cite-compatible",
                  "R6 same title + overlapping authors",
                  "R7 same specific unambiguous title + cite-compatible + "
                  "jurisdiction-compatible",
                  "EVICT legislation member cite-jurisdiction conflicts with "
                  "group uri, or title differs without shared statute cite"],
        "rule_merges": dict(counts),
        "n_groups": len(groups),
        "n_multidoc": sum(1 for g in groups if g["n_docs"] > 1),
    }, "groups": groups}, indent=2, ensure_ascii=False), encoding="utf-8")

    missing = [g for g in groups if g["status"] != "VERIFIED"]
    OUT_MISSING.write_text(json.dumps({"meta": {
        "producer": "uri_request/crossdoc_finalize.py",
        "note": "One entry per cross-document source group still needing a "
                "uri (company fills one uri per group_id). CONFLICT groups "
                "need human split first.",
        "n_groups": len(missing),
        "n_profiles": sum(g["n_members"] for g in missing),
    }, "groups": missing}, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[Merges]", dict(counts))
    print(f"[Groups] total={len(groups)} "
          f"VERIFIED={sum(1 for g in groups if g['status']=='VERIFIED')} "
          f"UNVERIFIED={sum(1 for g in groups if g['status']=='UNVERIFIED')} "
          f"CONFLICT={sum(1 for g in groups if g['status']=='CONFLICT')} "
          f"multidoc={sum(1 for g in groups if g['n_docs']>1)}")
    print(f"[DONE] -> {OUT_MERGED} + {OUT_MISSING}")


if __name__ == "__main__":
    main()
