"""Inject gold synthetic/reused URIs into annotated gold docs (in place).

Mirror of uri_request/complete_silver_uris.py, but for gold truth:

Reads:
  - uri_request/gold_synthetic_mapping.json (docid variant -> uri, written by
    uri_request/build_gold_synthetic_pool.py; kinds: synthetic | reused_gold |
    reused_pool)

Does:
  - for every mapping entry, sets ``uri="..."`` on the matching parent
    ``manual_label``/``auto_label`` tags of
    ``data/annotated/{train,dev,test_manual,test_assisted}/<DOC>.html``,
    reusing ``src/main_resolution.apply_uris_to_html`` (byte-identical behaviour
    to stage-3 resolution output and to the silver completion).

Safety:
  - matching is by (normalized) docid, exactly like the pipeline injector.
  - tags that already carry a real uri DIFFERENT from the mapped uri are NEVER
    overwritten (reported as conflicts); tags whose uri already EQUALS the
    mapped uri are left as-is (reported as already-set).
  - ``--dry-run`` prints what would change without writing anything.

Writes:
  - data/annotated/<split>/<DOC>.html (in place, unless --dry-run)
  - uri_request/gold_completion_report.json (per-document stats)

Usage (from repo root)::

    py uri_request/complete_gold_uris.py --dry-run
    py uri_request/complete_gold_uris.py
    py uri_request/complete_gold_uris.py --backup-dir uri_request/gold_backup_pre_completion
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

DEFAULT_MAPPING = MISSION_DIR / "gold_synthetic_mapping.json"
DEFAULT_REPORT = MISSION_DIR / "gold_completion_report.json"

GOLD_SPLITS = ("train", "dev", "test_manual", "test_assisted")
TAG_NAMES = ("manual_label", "auto_label")


def is_none_uri(tag) -> bool:
    uri = tag.get("uri")
    return uri is None or str(uri).strip().lower() == "none"


def count_tags(html: str) -> tuple[int, int, int]:
    """(n_parent_tags, n_with_real_uri, n_none)."""
    soup = BeautifulSoup(html, "html.parser")
    n_par = n_uri = n_none = 0
    for tag in soup.find_all(list(TAG_NAMES)):
        if tag.get("docid") is None:
            continue
        n_par += 1
        if is_none_uri(tag):
            n_none += 1
        else:
            n_uri += 1
    return n_par, n_uri, n_none


def find_gold_file(doc: str) -> Path | None:
    for split in GOLD_SPLITS:
        f = PROJECT_ROOT / "data" / "annotated" / split / f"{doc}.html"
        if f.exists():
            return f
    return None


def main() -> None:
    p = argparse.ArgumentParser(description="Inject gold URIs into gold docs.")
    p.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    p.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--backup-dir", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.mapping.exists():
        sys.exit(f"Missing input: {args.mapping} "
                 f"(run uri_request/build_gold_synthetic_pool.py first)")

    entries = json.loads(args.mapping.read_text(encoding="utf-8"))
    print(f"[Setup] {len(entries)} mapping entrie(s)")

    # Per-file injection mapping is derived by scanning gold files for none
    # tags whose norm docid has a mapping entry (exact, no guessing).
    lookup = {normalize_docid(e["docid"]): e for e in entries}
    print(f"[Setup] lookup covers {len(lookup)} norm docid(s)")

    # group target files by scanning for none tags with a mapped docid
    targets: dict[str, dict[str, dict]] = defaultdict(dict)
    for split in GOLD_SPLITS:
        sdir = PROJECT_ROOT / "data" / "annotated" / split
        for f in sorted(sdir.glob("*.html")):
            soup = BeautifulSoup(f.read_text(encoding="utf-8", errors="ignore"), "html.parser")
            for tag in soup.find_all(list(TAG_NAMES)):
                if tag.get("docid") is None or not is_none_uri(tag):
                    continue
                key = normalize_docid(tag.get("docid"))
                e = lookup.get(key)
                if e is not None:
                    targets[f][key] = {"uri": e["uri"]}
    print(f"[Setup] {len(targets)} file(s) with mappable none tags")

    if args.backup_dir and not args.dry_run:
        args.backup_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {"docs": {}, "conflicts": [], "unmapped_none": []}
    tot_par = tot_new = tot_skip = 0
    for f in sorted(targets):
        html = f.read_text(encoding="utf-8")
        soup = BeautifulSoup(html, "html.parser")
        # conflicts: real uri present and different from mapped uri
        conflicts = []
        conflict_keys = set()
        for t in soup.find_all(list(TAG_NAMES), attrs={"docid": True}):
            key = normalize_docid(t.get("docid"))
            if key in targets[f] and not is_none_uri(t):
                if str(t.get("uri")).strip() != targets[f][key]["uri"]:
                    conflicts.append({"document": f.name,
                                      "docid": t.get("docid"),
                                      "kept_uri": t.get("uri"),
                                      "mapped_uri": targets[f][key]["uri"]})
                    conflict_keys.add(key)
        safe = {k: v for k, v in targets[f].items() if k not in conflict_keys}
        for c in conflicts:
            report["conflicts"].append(c)

        n_before = count_tags(html)
        out_html = apply_uris_to_html(html, safe)
        n_after = count_tags(out_html)
        n_new = n_after[1] - n_before[1]

        if not args.dry_run:
            if args.backup_dir:
                dest = args.backup_dir / f.parent.name / f.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dest)
            f.write_text(out_html, encoding="utf-8")

        tot_par += n_before[0]
        tot_new += n_new
        tot_skip += len(conflicts)
        rel = f.relative_to(PROJECT_ROOT).as_posix()
        report["docs"][rel] = {
            "status": "dry-run" if args.dry_run else "completed",
            "parent_tags": n_before[0],
            "uris_before": n_before[1], "none_before": n_before[2],
            "uris_after": n_after[1], "none_after": n_after[2],
            "new_uris": n_new, "conflicts": len(conflicts),
        }
        print(f"  [{rel}] uris {n_before[1]}->{n_after[1]} (+{n_new}) "
              f"none left={n_after[2]} conflicts={len(conflicts)}")

    # residual: mappable-label none tags with no mapping entry (should be none;
    # unable-to-classify / docid-less tags are out of scope by design)
    for split in GOLD_SPLITS:
        sdir = PROJECT_ROOT / "data" / "annotated" / split
        for f in sorted(sdir.glob("*.html")):
            soup = BeautifulSoup(f.read_text(encoding="utf-8", errors="ignore"), "html.parser")
            for tag in soup.find_all(list(TAG_NAMES)):
                if tag.get("docid") is None or not is_none_uri(tag):
                    continue
                lab = (tag.get("labelname") or "").strip().lower()
                if lab not in ("decision", "legislation", "secondary sources"):
                    continue
                if normalize_docid(tag.get("docid")) in lookup:
                    continue
                report["unmapped_none"].append({
                    "file": f.relative_to(PROJECT_ROOT).as_posix(),
                    "docid": tag.get("docid"), "id": str(tag.get("id"))})

    report["summary"] = {
        "dry_run": args.dry_run,
        "files": len(targets),
        "parent_tags": tot_par,
        "new_uris": tot_new,
        "conflicts": tot_skip,
        "unmapped_none": len(report["unmapped_none"]),
    }
    if not args.dry_run:
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[DONE]{' (dry-run)' if args.dry_run else ''} {len(targets)} file(s), "
          f"{tot_par} parent tags, +{tot_new} new uris, {tot_skip} conflict(s), "
          f"{len(report['unmapped_none'])} unmapped none(s)")
    if not args.dry_run:
        print(f"  report -> {args.report}")


if __name__ == "__main__":
    main()
