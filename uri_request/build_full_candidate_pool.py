"""Build the full candidate pool: gold pool + silver synthetic sources.

Reads:
  - data/gold_candidate_pool.csv (6437 rows: uri,metadata)
  - uri_request/uri_request_merged_missing_deduped.json (1294 cross-doc groups)

Writes (all in the exact gold uri,metadata format: metadata is
json.dumps(..., indent=2), i.e. multi-line pretty-printed inside the
quoted CSV field, like cited_metadata.csv / extra_metadata.csv):
  - data/synthetic_metadata.csv (1294 synthetic rows only)
  - data/full_candidate_pool.csv (7731 rows: gold + synthetic)
  - uri_request/silver_synthetic_mapping.json (group_id -> synthetic uri)

Each silver group becomes one candidate whose titles / citations / authors are
aggregated from the cluster's mentions, with a synthetic uri/id/path. The
autoCompletion *types* mirror the gold pool vocabulary; the retrieval index
(src/precompute_index.py) only encodes the *texts*, so the types are for
format consistency only.

Usage (from repo root)::

    python uri_request/build_full_candidate_pool.py
    python uri_request/build_full_candidate_pool.py --overwrite
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

MISSION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MISSION_DIR.parent
DEFAULT_GOLD = PROJECT_ROOT / "data" / "gold_candidate_pool.csv"
DEFAULT_DEDUPED = MISSION_DIR / "uri_request_merged_missing_deduped.json"
DEFAULT_OUT = PROJECT_ROOT / "data" / "full_candidate_pool.csv"
DEFAULT_SYNTH = PROJECT_ROOT / "data" / "synthetic_metadata.csv"
DEFAULT_MAPPING = MISSION_DIR / "silver_synthetic_mapping.json"

SYNTHETIC_URI_PREFIX = "https://synthetic.leread.ca/silver/"

DOC_TYPE_MAP = {
    "decision": "DECISION",
    "legislation": "LEGISLATION",
    "secondary sources": "DOCTRINE",
}


def clean_text(s: object) -> str:
    """Collapse whitespace/newlines like gold titles (no raw \\n)."""
    if s is None:
        return ""
    t = str(s).replace("\r", " ").replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t


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


def build_autocompletions(doc_type_raw: str, canonical: str,
                           alt_titles: list[str], citations: list[str],
                           authors: list[str]) -> list[dict]:
    ac: list[dict] = [{"type": "MAIN_TITLE", "text": canonical}]
    norm = (doc_type_raw or "").strip().lower()
    if norm == "decision":
        for t in alt_titles:
            if t != canonical:
                ac.append({"type": "DECISION_ALTERNATE_TITLE", "text": t})
        for c in citations:
            ac.append({"type": "DECISION_OTHER_REFERENCES", "text": c})
    elif norm == "legislation":
        for t in alt_titles:
            if t != canonical:
                ac.append({"type": "LEGISLATION_ALTERNATE_TITLE", "text": t})
        for i, c in enumerate(citations):
            typ = "LEGISLATION_REFERENCE" if i == 0 else "LEGISLATION_ALTERNATE_REFERENCE"
            ac.append({"type": typ, "text": c})
    else:  # secondary sources -> DOCTRINE (gold has no ALTERNATE_TITLE type)
        for t in alt_titles:
            if t != canonical:
                ac.append({"type": "MAIN_TITLE", "text": t})
        for a in authors:
            ac.append({"type": "DOCTRINE_AUTHOR", "text": a})
    return ac


def group_to_candidate(group: dict) -> tuple[str, dict, dict]:
    group_id = group["group_id"]
    doc_type_raw = group.get("doc_type", "")
    doc_type = DOC_TYPE_MAP.get((doc_type_raw or "").strip().lower(), "DECISION")
    canonical = clean_text(group.get("canonical_title", group_id))
    extra = group.get("extra_metadata", {})
    alt_titles = ordered_unique(extra.get("alternative_titles", []))
    citations = ordered_unique(extra.get("citations", []))
    authors = ordered_unique(extra.get("authors_raw", []))
    n_mentions = int(extra.get("n_mentions_total", group.get("n_members", 1)))

    uri = f"{SYNTHETIC_URI_PREFIX}{group_id}"
    synth_id = f"silver-{group_id}"
    path = f"/synthetic/silver/{group_id}"
    ac = build_autocompletions(doc_type_raw, canonical, alt_titles, citations, authors)
    candidate = {
        "id": synth_id,
        "docType": doc_type,
        "path": path,
        "docTitle": canonical,
        # Gold stores autoCompletions as a JSON-encoded *string* inside the
        # metadata JSON, and the outer metadata list is pretty-printed with
        # indent=2 (multi-line quoted CSV field). Reproduce both exactly.
        # Inner list is compact (separators without spaces), like gold.
        "autoCompletions": json.dumps(ac, ensure_ascii=False, separators=(",", ":")),
        "jurisdictionId": "unspecified",
        "citedCount": n_mentions,
    }
    # Outer metadata encoding identical to gold: indent=2.
    metadata_str = json.dumps([candidate], ensure_ascii=False, indent=2)
    map_entry = {
        "group_id": group_id,
        "uri": uri,
        "id": synth_id,
        "doc_type": doc_type_raw,
        "candidate_docType": doc_type,
        "docTitle": canonical,
        "n_members": group.get("n_members", 0),
        "n_docs": group.get("n_docs", 0),
        "n_mentions_total": n_mentions,
    }
    return uri, candidate, metadata_str, map_entry


def main() -> None:
    p = argparse.ArgumentParser(description="Build full candidate pool.")
    p.add_argument("--gold-pool", type=Path, default=DEFAULT_GOLD)
    p.add_argument("--deduped", type=Path, default=DEFAULT_DEDUPED)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--synth-out", type=Path, default=DEFAULT_SYNTH)
    p.add_argument("--mapping-out", type=Path, default=DEFAULT_MAPPING)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if not args.gold_pool.exists():
        sys.exit(f"Missing input: {args.gold_pool}")
    if not args.deduped.exists():
        sys.exit(f"Missing input: {args.deduped}")
    if args.out.exists() and not args.overwrite:
        sys.exit(f"Output exists: {args.out} (use --overwrite)")
    if args.synth_out.exists() and not args.overwrite:
        sys.exit(f"Output exists: {args.synth_out} (use --overwrite)")

    # --- gold rows (kept verbatim) ---
    with args.gold_pool.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if set(reader.fieldnames or []) != {"uri", "metadata"}:
            sys.exit(f"Unexpected gold columns: {reader.fieldnames}")
        gold_rows = list(reader)
    gold_uris = {r["uri"] for r in gold_rows}
    print(f"[Setup] gold rows: {len(gold_rows)} | unique uris: {len(gold_uris)}")

    # --- silver groups ---
    payload = json.loads(args.deduped.read_text(encoding="utf-8"))
    groups = payload.get("groups", [])
    print(f"[Setup] silver groups: {len(groups)}")

    synth_rows: list[dict] = []
    mapping: list[dict] = []
    for g in groups:
        uri, cand, metadata_str, entry = group_to_candidate(g)
        if uri in gold_uris:
            sys.exit(f"URI collision with gold pool: {uri}")
        synth_rows.append({"uri": uri, "metadata": metadata_str})
        mapping.append(entry)
    synth_uris = [r["uri"] for r in synth_rows]
    assert len(set(synth_uris)) == len(synth_rows), "duplicate synthetic uris"

    # --- write synthetic-only + full pool (gold rows kept verbatim) ---
    args.synth_out.parent.mkdir(parents=True, exist_ok=True)
    with args.synth_out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["uri", "metadata"])
        w.writeheader()
        w.writerows(synth_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["uri", "metadata"])
        w.writeheader()
        w.writerows(gold_rows)
        w.writerows(synth_rows)

    args.mapping_out.write_text(json.dumps(mapping, indent=2, ensure_ascii=False),
                                encoding="utf-8")

    # --- validate output ---
    with args.out.open(newline="", encoding="utf-8") as f:
        out_rows = list(csv.DictReader(f))
    bad = 0
    for r in out_rows[len(gold_rows):]:
        try:
            meta = json.loads(r["metadata"])
            ac = json.loads(meta[0]["autoCompletions"])
            assert isinstance(ac, list) and ac and meta[0]["docTitle"]
        except Exception:
            bad += 1
    from collections import Counter
    doctypes = Counter(json.loads(r["metadata"])[0].get("docType", "?")
                       if json.loads(r["metadata"]) else "EMPTY"
                       for r in out_rows if True)
    # (EMPTY row in gold pool has metadata '[]')
    print(f"[DONE] gold={len(gold_rows)} + synthetic={len(synth_rows)} "
          f"-> {len(out_rows)} rows -> {args.out}")
    print(f"  synthetic-only: {len(synth_rows)} rows -> {args.synth_out}")
    print(f"  synthetic validation failures: {bad}")
    print(f"  full-pool docTypes: {dict(doctypes)}")
    print(f"  mapping: {len(mapping)} entries -> {args.mapping_out}")


if __name__ == "__main__":
    main()
