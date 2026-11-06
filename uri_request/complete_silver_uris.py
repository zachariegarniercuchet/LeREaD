"""Complete the silver metadata: inject synthetic URIs into annotated silver docs.

Reads:
  - uri_request/uri_request_merged_missing_deduped.json (1294 cross-doc groups,
    each member = one coref profile with source_document / docid / mention_ids)
  - uri_request/silver_synthetic_mapping.json (group_id -> synthetic uri,
    written by build_full_candidate_pool.py)
  - uri_request/uri_request_merged.json (all cross-doc groups; VERIFIED ones
    carry the real CanLII uri that was never injected into silver either)

Does:
  - for every group member, sets ``uri="..."`` on the matching
    parent ``manual_label``/``auto_label`` tags of
    ``data/annotated/silver/<SOURCE_DOCUMENT>.html``: the real CanLII uri
    for VERIFIED groups, the synthetic
    ``https://synthetic.leread.ca/silver/<GROUP_ID>`` uri for the MISSING
    groups (already added to the full candidate pool by
    build_full_candidate_pool.py).
    (``data/original/silver`` holds raw HTML with no spans, so there is
    nothing to complete there — the spans and their metadata live in
    ``data/annotated/silver``.)
  - injection reuses ``src/main_resolution.apply_uris_to_html``, i.e. it is
    byte-identical in behaviour to the stage-3 resolution output and to
    ``build_silver_split.py``.

Safety:
  - matching is by (normalized) docid, exactly like the pipeline injector;
    ``mention_ids`` are used as a pre-flight coverage check (reported, the
    run continues on mismatch).
  - tags that already carry a real uri (not "none") are NEVER overwritten:
    their docid is excluded from the injection mapping and reported as
    already-set.
  - ``--dry-run`` prints what would change without writing anything.

Writes:
  - data/annotated/silver/<DOC>.html (in place, unless --dry-run)
  - uri_request/silver_completion_report.json (per-document stats)

Usage (from repo root)::

    python uri_request/complete_silver_uris.py --dry-run
    python uri_request/complete_silver_uris.py
    python uri_request/complete_silver_uris.py --backup-dir uri_request/silver_backup_pre_synth
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

MISSION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MISSION_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bs4 import BeautifulSoup  # noqa: E402

from src.main_resolution import apply_uris_to_html  # noqa: E402 pipeline convention
from src.rpr import normalize_docid  # noqa: E402

DEFAULT_DEDUPED = MISSION_DIR / "uri_request_merged_missing_deduped.json"
DEFAULT_MERGED = MISSION_DIR / "uri_request_merged.json"
DEFAULT_MAPPING = MISSION_DIR / "silver_synthetic_mapping.json"
DEFAULT_SILVER_DIR = PROJECT_ROOT / "data" / "annotated" / "silver"
DEFAULT_REPORT = MISSION_DIR / "silver_completion_report.json"

TAG_NAMES = ("manual_label", "auto_label")
SYNTHETIC_URI_PREFIX = "https://synthetic.leread.ca/silver/"


def is_none_uri(tag) -> bool:
    uri = tag.get("uri")
    return uri is None or str(uri).strip().lower() == "none"


def count_tags(html: str) -> tuple[int, int]:
    """(n_parent_tags, n_parent_tags_with_real_uri)."""
    soup = BeautifulSoup(html, "html.parser")
    n_par = n_uri = 0
    for tag in soup.find_all(list(TAG_NAMES)):
        if tag.get("docid") is None:
            continue
        n_par += 1
        if not is_none_uri(tag):
            n_uri += 1
    return n_par, n_uri


def main() -> None:
    p = argparse.ArgumentParser(description="Inject synthetic URIs into silver docs.")
    p.add_argument("--deduped", type=Path, default=DEFAULT_DEDUPED)
    p.add_argument("--merged", type=Path, default=DEFAULT_MERGED,
                   help="Full cross-doc groups file (for VERIFIED real uris).")
    p.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    p.add_argument("--silver-dir", type=Path, default=DEFAULT_SILVER_DIR)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--backup-dir", type=Path, default=None,
                   help="If given, copy originals there before writing.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.deduped.exists():
        sys.exit(f"Missing input: {args.deduped}")
    if not args.mapping.exists():
        sys.exit(f"Missing input: {args.mapping} "
                 f"(run uri_request/build_full_candidate_pool.py first)")
    if not args.silver_dir.exists():
        sys.exit(f"Missing input: {args.silver_dir}")

    # --- group_id -> synthetic uri (mapping file is source of truth) ---
    mapping_entries = json.loads(args.mapping.read_text(encoding="utf-8"))
    group_uri = {e["group_id"]: e["uri"] for e in mapping_entries}

    payload = json.loads(args.deduped.read_text(encoding="utf-8"))
    groups = payload.get("groups", [])
    print(f"[Setup] {len(groups)} group(s), {len(group_uri)} mapped uri(s)")

    # --- per-document injection mapping: norm_docid -> {"uri": ...} ---
    # Records keep (source_document, member, uri, kind) for the coverage check.
    by_doc: dict[str, dict[str, dict]] = defaultdict(dict)
    records: list[tuple[str, str, dict, str, str]] = []
    member_count = 0
    missing_uri_groups: list[str] = []
    for g in groups:
        uri = group_uri.get(g["group_id"])
        if not uri:
            # Fallback: deterministic uri scheme from build_full_candidate_pool.py
            uri = f"{SYNTHETIC_URI_PREFIX}{g['group_id']}"
            missing_uri_groups.append(g["group_id"])
        for mm in g["members"]:
            docid = (mm.get("docid") or "").strip()
            if not docid:
                print(f"  [warn] {g['group_id']}: member with empty docid "
                      f"in {mm.get('source_document')}, skipped")
                continue
            by_doc[mm["source_document"]][normalize_docid(docid)] = {
                "uri": uri, "group_id": g["group_id"], "docid": docid,
                "kind": "synthetic",
            }
            records.append((mm["source_document"], g["group_id"], mm, uri, "synthetic"))
            member_count += 1
    if missing_uri_groups:
        print(f"  [warn] {len(missing_uri_groups)} group(s) absent from mapping file, "
              f"rebuilt uris from prefix (e.g. {missing_uri_groups[:3]})")
    print(f"[Setup] {member_count} missing-profile(s) across {len(by_doc)} document(s)")

    # --- overlay VERIFIED real uris (they win over synthetic on overlap) ---
    n_real = n_override = 0
    if args.merged.exists():
        merged_groups = json.loads(args.merged.read_text(encoding="utf-8")).get("groups", [])
        for g in merged_groups:
            if g.get("status") != "VERIFIED" or not (g.get("uri") or "").strip():
                continue
            for mm in g["members"]:
                docid = (mm.get("docid") or "").strip()
                if not docid:
                    continue
                key = normalize_docid(docid)
                prev = by_doc[mm["source_document"]].get(key)
                if prev is not None and prev["uri"] != g["uri"]:
                    n_override += 1
                    print(f"  [warn] {mm['source_document']} :: {docid}: "
                          f"synthetic {prev['uri']} overridden by real {g['uri']}")
                by_doc[mm["source_document"]][key] = {
                    "uri": g["uri"], "group_id": g["group_id"], "docid": docid,
                    "kind": "verified",
                }
                records.append((mm["source_document"], g["group_id"], mm, g["uri"], "verified"))
                n_real += 1
        print(f"[Setup] {n_real} verified-profile(s) with real uris "
              f"({n_override} override(s) of synthetic)")
    else:
        print(f"  [warn] merged file not found: {args.merged} "
              f"(verified real uris will NOT be injected)")

    if args.backup_dir and not args.dry_run:
        args.backup_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {"docs": {}, "already_set": [], "unmatched_mention_ids": []}
    tot_tags = tot_new = tot_set = 0

    for src, doc_mapping in sorted(by_doc.items()):
        f = args.silver_dir / f"{src}.html"
        if not f.exists():
            print(f"  [warn] {src}: file not found, skipped")
            report["docs"][src] = {"status": "file-not-found"}
            continue
        html = f.read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "html.parser")

        # Pre-flight 1: tags already carrying a real uri are left untouched
        # (excluded from the mapping).
        already = sorted({
            normalize_docid(t.get("docid"))
            for t in soup.find_all(list(TAG_NAMES), attrs={"docid": True})
            if normalize_docid(t.get("docid")) in doc_mapping and not is_none_uri(t)
        })
        safe_mapping = {k: v for k, v in doc_mapping.items() if k not in already}
        for c in already:
            report["already_set"].append({
                "document": src, "docid": doc_mapping[c]["docid"],
                "kept_uri": "existing", "injected_uri": doc_mapping[c]["uri"],
                "group_id": doc_mapping[c]["group_id"],
            })

        # Pre-flight 2: mention_id coverage — every member mention id should
        # exist as a tag id with the same docid.
        tags_by_id = {}
        for t in soup.find_all(list(TAG_NAMES)):
            if t.get("id") is not None:
                tags_by_id[str(t.get("id"))] = t
        n_mid = n_mid_ok = 0
        for rec_src, rec_gid, mm, _uri, kind in records:
            if rec_src != src:
                continue
            for mid in mm.get("mention_ids", []):
                n_mid += 1
                t = tags_by_id.get(str(mid))
                if t is not None and normalize_docid(t.get("docid")) == \
                        normalize_docid(mm.get("docid") or ""):
                    n_mid_ok += 1
                else:
                    report["unmatched_mention_ids"].append({
                        "document": src, "mention_id": str(mid),
                        "docid": mm.get("docid"),
                        "group_id": rec_gid,
                        "kind": kind,
                    })

        # Inject (same function as stage-3 resolution output).
        n_before_par, n_before_uri = count_tags(html)
        out_html = apply_uris_to_html(html, safe_mapping)
        n_after_par, n_after_uri = count_tags(out_html)
        n_new = n_after_uri - n_before_uri

        if not args.dry_run:
            if args.backup_dir:
                shutil.copy2(f, args.backup_dir / f.name)
            f.write_text(out_html, encoding="utf-8")

        tot_tags += n_before_par
        tot_new += n_new
        tot_set += len(already)
        report["docs"][src] = {
            "status": "dry-run" if args.dry_run else "completed",
            "parent_tags": n_before_par,
            "uris_before": n_before_uri,
            "uris_after": n_after_uri,
            "new_uris": n_new,
            "profiles": len(doc_mapping),
            "already_set": len(already),
            "mention_ids_matched": f"{n_mid_ok}/{n_mid}",
        }
        print(f"  [{src}] profiles={len(doc_mapping)} parents={n_before_par} "
              f"uris {n_before_uri}->{n_after_uri} (+{n_new}) "
              f"mention_ids={n_mid_ok}/{n_mid} already-set={len(already)}")

    report["summary"] = {
        "dry_run": args.dry_run,
        "docs": len(by_doc),
        "parent_tags": tot_tags,
        "new_uris": tot_new,
        "already_set": tot_set,
        "unmatched_mention_ids": len(report["unmatched_mention_ids"]),
    }
    if not args.dry_run:
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                               encoding="utf-8")
    print(f"\n[DONE]{' (dry-run, nothing written)' if args.dry_run else ''} "
          f"{len(by_doc)} doc(s), {tot_tags} parent tags, +{tot_new} new uris, "
          f"{tot_set} already-set, "
          f"{len(report['unmatched_mention_ids'])} unmatched mention id(s)")
    if not args.dry_run:
        print(f"  report -> {args.report}")


if __name__ == "__main__":
    main()
