"""Build synthetic gold candidate-pool entries for gold `uri="none"` mentions.

Mirror of the silver mission (uri_request/build_full_candidate_pool.py), but for
gold truth (data/annotated/{train,dev,test_manual,test_assisted}):

Reads:
  - data/annotated/<split>/*.html (gold files; parent mentions with docid)
  - data/gold_candidate_pool.csv (for the no-doubles check: a none-group whose
    titles/citations already match a pool entry reuses that pool URI)

Does:
  - groups every parent mention whose uri is missing/"none" by normalized docid
    (global across the 4 gold splits: one group per source, no cross-doc doubles;
    grouping key is casefolded, every raw variant is kept for injection)
  - for each none-group, looks for strong evidence (shared citation len>=6 or
    shared official title) that it is the SAME source as:
      (a) a real-URI variant of the same docid elsewhere in gold -> reuse it
      (b) an existing gold-pool entry -> reuse that pool URI
    exactly one such URI  -> reuse (no synthetic row)
    zero / ambiguous      -> create one synthetic candidate (needs_review=True
                             when ambiguous)
  - candidate rows use the EXACT gold uri,metadata format (outer metadata
    json.dumps(indent=2), inner autoCompletions compact), same vocabulary as
    build_full_candidate_pool.py, but under https://synthetic.leread.ca/gold/

Writes:
  - data/synthetic_metadata_from_gold.csv (synthetic rows only)
  - appends those rows to data/gold_candidate_pool.csv and
    data/full_candidate_pool.csv (backups: .bak_pre_gold_synth.csv)
  - uri_request/gold_synthetic_mapping.json (norm docid variant -> uri, kind,
    evidence; kinds: synthetic | reused_gold | reused_pool)

Usage (from repo root):
    py uri_request/build_gold_synthetic_pool.py --dry-run
    py uri_request/build_gold_synthetic_pool.py
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

MISSION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MISSION_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bs4 import BeautifulSoup  # noqa: E402

GOLD_SPLITS = ["train", "dev", "test_manual", "test_assisted"]
PARENT_LABELS = ("decision", "legislation", "secondary sources")
TAG_NAMES = ("manual_label", "auto_label")

SYNTHETIC_URI_PREFIX = "https://synthetic.leread.ca/gold/"

DOC_TYPE_MAP = {
    "decision": "DECISION",
    "legislation": "LEGISLATION",
    "secondary sources": "DOCTRINE",
}

DEFAULT_GOLD_POOL = PROJECT_ROOT / "data" / "gold_candidate_pool.csv"
DEFAULT_FULL_POOL = PROJECT_ROOT / "data" / "full_candidate_pool.csv"
DEFAULT_SYNTH_OUT = PROJECT_ROOT / "data" / "synthetic_metadata_from_gold.csv"
DEFAULT_MAPPING = MISSION_DIR / "gold_synthetic_mapping.json"


def norm_docid(docid: str) -> str:
    return html.unescape(docid).strip()


def norm_text(s: object) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s).replace("\r", " ").replace("\n", " ")).strip()


def key_text(s: str) -> str:
    return norm_text(s).casefold()


def clean_text(s: object) -> str:
    return norm_text(s)


def ordered_unique(items) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for it in items:
        t = clean_text(it)
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def is_none_uri(tag) -> bool:
    uri = tag.get("uri")
    return uri is None or str(uri).strip().lower() == "none"


def build_autocompletions(doc_type_raw, canonical, alt_titles, citations, authors):
    ac = [{"type": "MAIN_TITLE", "text": canonical}]
    n = (doc_type_raw or "").strip().lower()
    if n == "decision":
        for t in alt_titles:
            if t != canonical:
                ac.append({"type": "DECISION_ALTERNATE_TITLE", "text": t})
        for c in citations:
            ac.append({"type": "DECISION_OTHER_REFERENCES", "text": c})
    elif n == "legislation":
        for t in alt_titles:
            if t != canonical:
                ac.append({"type": "LEGISLATION_ALTERNATE_TITLE", "text": t})
        for i, c in enumerate(citations):
            typ = "LEGISLATION_REFERENCE" if i == 0 else "LEGISLATION_ALTERNATE_REFERENCE"
            ac.append({"type": typ, "text": c})
    else:  # secondary sources -> DOCTRINE
        for t in alt_titles:
            if t != canonical:
                ac.append({"type": "MAIN_TITLE", "text": t})
        for a in authors:
            ac.append({"type": "DOCTRINE_AUTHOR", "text": a})
    return ac


def strong_cite(t: str) -> bool:
    return len(t) >= 6


def main() -> None:
    p = argparse.ArgumentParser(description="Build synthetic gold pool entries.")
    p.add_argument("--gold-pool", type=Path, default=DEFAULT_GOLD_POOL)
    p.add_argument("--full-pool", type=Path, default=DEFAULT_FULL_POOL)
    p.add_argument("--synth-out", type=Path, default=DEFAULT_SYNTH_OUT)
    p.add_argument("--mapping-out", type=Path, default=DEFAULT_MAPPING)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    csv.field_size_limit(sys.maxsize)

    # ---- 1. scan gold files: none-groups + real variants ----
    # gkey (casefolded norm docid) -> aggregate
    groups: dict[str, dict] = {}
    real_by_docid: dict[str, dict[str, dict]] = defaultdict(dict)  # gkey -> uri -> evidence

    def group(gkey):
        if gkey not in groups:
            groups[gkey] = {
                "raw_variants": Counter(), "labels": Counter(),
                "official_titles": Counter(), "other_titles": Counter(),
                "citations": Counter(), "authors": Counter(),
                "n_mentions": 0, "members": [], "docs": set(),
            }
        return groups[gkey]

    n_files = 0
    for split in GOLD_SPLITS:
        sdir = PROJECT_ROOT / "data" / "annotated" / split
        for f in sorted(sdir.glob("*.html")):
            n_files += 1
            soup = BeautifulSoup(f.read_text(encoding="utf-8", errors="ignore"), "html.parser")
            for tag in soup.find_all(list(TAG_NAMES)):
                lab = (tag.get("labelname") or "").strip().lower()
                if lab not in PARENT_LABELS:
                    continue
                if tag.get("docid") is None:
                    continue
                raw = (tag.get("docid") or "").strip()
                if not raw:
                    continue
                gkey = key_text(norm_docid(raw))
                # child evidence
                off, oth, cit, aut = [], [], [], []
                for c in tag.find_all(list(TAG_NAMES)):
                    cl = (c.get("labelname") or "").strip().lower()
                    t = norm_text(c.get_text())
                    if not t:
                        continue
                    if cl == "title":
                        (off if (c.get("titletype") or "").strip().lower() == "official" else oth).append(t)
                    elif cl == "citation":
                        cit.append(t)
                    elif cl == "authors":
                        aut.append(t)
                uri = tag.get("uri")
                if uri is None or str(uri).strip().lower() == "none":
                    g = group(gkey)
                    g["raw_variants"][norm_docid(raw)] += 1
                    g["labels"][lab] += 1
                    for t in off:
                        g["official_titles"][t] += 1
                    for t in oth:
                        g["other_titles"][t] += 1
                    for t in cit:
                        g["citations"][t] += 1
                    for t in aut:
                        g["authors"][t] += 1
                    g["n_mentions"] += 1
                    g["docs"].add(f"{split}/{f.name}")
                    mid = tag.get("id")
                    g["members"].append({
                        "source_document": f.stem, "split": split,
                        "mention_id": str(mid) if mid is not None else None,
                    })
                else:
                    ev = real_by_docid[gkey].setdefault(str(uri).strip(), {
                        "titles": Counter(), "citations": Counter(), "n": 0})
                    ev["n"] += 1
                    for t in off + oth:
                        ev["titles"][t] += 1
                    for t in cit:
                        ev["citations"][t] += 1

    print(f"[Setup] {n_files} gold files | {len(groups)} none-groups")
    tot_none = sum(g["n_mentions"] for g in groups.values())
    print(f"[Setup] {tot_none} none-mentions "
          f"({100*tot_none/sum(g['n_mentions'] for g in groups.values()):.1f}% of none sample)")

    # ---- 2. pool index (title/citation text -> uris) ----
    with args.gold_pool.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert set(reader.fieldnames or []) == {"uri", "metadata"}, reader.fieldnames
        gold_rows = list(reader)
    pool_texts: dict[str, set[str]] = defaultdict(set)  # key_text -> uris
    for r in gold_rows:
        try:
            meta = json.loads(r["metadata"])
        except Exception:
            continue
        if not meta:
            continue
        cand = meta[0]
        texts = [cand.get("docTitle", "")]
        try:
            acs = json.loads(cand.get("autoCompletions", "[]"))
            texts += [a.get("text", "") for a in acs if isinstance(a, dict)]
        except Exception:
            pass
        for t in texts:
            if norm_text(t):
                pool_texts[key_text(t)].add(r["uri"])
    print(f"[Setup] gold pool rows: {len(gold_rows)} | indexed texts: {len(pool_texts)}")

    # ---- 3. decide reuse vs synthetic per group ----
    def strong_keys(g):
        ck, tk = set(), set()
        for t in g["citations"]:
            if strong_cite(norm_text(t)):
                ck.add(key_text(t))
        for t in g["official_titles"]:
            if len(norm_text(t)) >= 8:
                tk.add(key_text(t))
        return ck, tk

    mapping = []      # per raw-variant mapping entries (for injection)
    synth_rows = []   # new candidate rows
    stats = Counter()
    review = []
    seq = 0
    for gkey in sorted(groups):
        g = groups[gkey]
        ckeys, tkeys = strong_keys(g)
        skeys = ckeys | tkeys
        cands: dict[str, list[str]] = defaultdict(list)  # uri -> evidence
        cite_hit: dict[str, bool] = defaultdict(bool)
        for uri, ev in real_by_docid.get(gkey, {}).items():
            ev_keys = {key_text(t) for t in list(ev["titles"]) + list(ev["citations"])}
            shared = [k for k in skeys if k in ev_keys]
            if shared:
                cands[uri].append(f"gold-real shared: {len(shared)}")
                ev_ckeys = {key_text(t) for t in ev["citations"]}
                if any(k in ev_ckeys for k in ckeys):
                    cite_hit[uri] = True
        for k in skeys:
            for uri in pool_texts.get(k, ()):
                cands[uri].append(f"pool shared: {k[:60]}")
                if k in ckeys:
                    cite_hit[uri] = True
        doc_type_raw = g["labels"].most_common(1)[0][0]
        base = {
            "docid_variants": sorted(g["raw_variants"]),
            "doc_type": doc_type_raw,
            "n_mentions": g["n_mentions"], "n_docs": len(g["docs"]),
        }
        if len(cands) == 1:
            uri = next(iter(cands))
            # Title-only match while the group HAS citations that match nothing:
            # too weak (e.g. generic "Patent Act" title, version-specific
            # citations) -> synthetic + review instead of a risky merge.
            if ckeys and not cite_hit[uri]:
                cands[uri].append("WEAK: title-only, citations unmatched")
            else:
                kind = "reused_gold" if uri in real_by_docid.get(gkey, {}) else "reused_pool"
                stats[kind] += g["n_mentions"]
                for raw in g["raw_variants"]:
                    mapping.append({**base, "docid": raw, "uri": uri, "kind": kind,
                                    "evidence": cands[uri][:4]})
                continue
        # synthetic (zero or ambiguous/weak candidates)
        gid = f"G{seq:04d}"
        seq += 1
        uri = f"{SYNTHETIC_URI_PREFIX}{gid}"
        official = g["official_titles"].most_common(1)
        any_title = (g["official_titles"] + g["other_titles"]).most_common(1)
        canonical = (official[0][0] if official
                     else (any_title[0][0] if any_title
                           else g["raw_variants"].most_common(1)[0][0]))
        alt = ordered_unique([t for t, _ in (g["official_titles"] + g["other_titles"]).most_common()])
        cit = ordered_unique([t for t, _ in g["citations"].most_common()])
        aut = ordered_unique([t for t, _ in g["authors"].most_common()])
        doc_type = DOC_TYPE_MAP.get(doc_type_raw, "DECISION")
        ac = build_autocompletions(doc_type_raw, canonical, alt, cit, aut)
        candidate = {
            "id": f"gold-{gid}", "docType": doc_type,
            "path": f"/synthetic/gold/{gid}", "docTitle": canonical,
            "autoCompletions": json.dumps(ac, ensure_ascii=False, separators=(",", ":")),
            "jurisdictionId": "unspecified", "citedCount": g["n_mentions"],
        }
        metadata_str = json.dumps([candidate], ensure_ascii=False, indent=2)
        synth_rows.append({"uri": uri, "metadata": metadata_str})
        weak_single = (len(cands) == 1
                       and any("WEAK" in e for ev in cands.values() for e in ev))
        needs_review = len(cands) > 1 or weak_single
        stats["synthetic"] += 1
        stats["synthetic_mentions"] += g["n_mentions"]
        if needs_review:
            review.append({"docid_variants": sorted(g["raw_variants"]),
                           "uri": uri, "candidates": dict(cands)})
            stats["synthetic_needs_review"] += 1
        for raw in g["raw_variants"]:
            mapping.append({**base, "docid": raw, "uri": uri, "kind": "synthetic",
                            "group_id": gid, "needs_review": needs_review})

    print(f"[Decide] none-groups={len(groups)} | "
          f"reused_gold mentions={stats['reused_gold']} | "
          f"reused_pool mentions={stats['reused_pool']} | "
          f"synthetic groups={stats['synthetic']} "
          f"({stats['synthetic_mentions']} mentions, {stats['synthetic_needs_review']} ambiguous)")

    if args.dry_run:
        print("[DRY-RUN] nothing written. Ambiguous groups needing review:")
        for r in review[:20]:
            print("  ", r["docid_variants"], "->", r["uri"])
            for u, ev in list(r["candidates"].items())[:4]:
                print("      cand:", u, ev[:2])
        print(f"[DRY-RUN] would write {len(synth_rows)} synthetic rows + "
              f"{len(mapping)} mapping entries")
        return

    # ---- 4. collision guard + write outputs ----
    pool_uris = {r["uri"] for r in gold_rows}
    with args.full_pool.open(newline="", encoding="utf-8") as f:
        full_rows = list(csv.DictReader(f))
    pool_uris |= {r["uri"] for r in full_rows}
    for r in synth_rows:
        if r["uri"] in pool_uris:
            sys.exit(f"URI collision: {r['uri']}")
    print(f"[Setup] full pool rows: {len(full_rows)}")

    args.synth_out.parent.mkdir(parents=True, exist_ok=True)
    with args.synth_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["uri", "metadata"])
        w.writeheader()
        w.writerows(synth_rows)
    for pool_path, rows in ((args.gold_pool, gold_rows), (args.full_pool, full_rows)):
        bak = pool_path.with_name(pool_path.name + ".bak_pre_gold_synth.csv")
        if not bak.exists():
            shutil.copy2(pool_path, bak)
            print(f"  backup -> {bak.name}")
        with pool_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["uri", "metadata"])
            w.writeheader()
            w.writerows(rows)
            w.writerows(synth_rows)
    args.mapping_out.write_text(json.dumps(mapping, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[DONE] synthetic={len(synth_rows)} -> {args.synth_out}")
    print(f"  gold pool: {len(gold_rows)} -> {len(gold_rows)+len(synth_rows)}")
    print(f"  full pool: {len(full_rows)} -> {len(full_rows)+len(synth_rows)}")
    print(f"  mapping: {len(mapping)} entries -> {args.mapping_out}")
    if review:
        (MISSION_DIR / "gold_synthetic_review.json").write_text(
            json.dumps(review, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  review: {len(review)} ambiguous groups -> gold_synthetic_review.json")
    print("  NOTE: candidate-pool indexes (cache/candidate_pool_index_*) are now stale; "
          "rebuild with: python -m src.precompute_index --retriever <name>")


if __name__ == "__main__":
    main()
