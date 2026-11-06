"""Build the no-gold silver split: inject verified URIs into coref HTML.

Takes the stage-2 coref files (``*_coref.html`` with ``<auto_label docid=...>``
tags, no ``uri``) and adds ``uri="..."`` to every tag whose profile was
verified EXISTS by ``apply_decision.py`` — reusing
``src/main_resolution.apply_uris_to_html`` so the injection is identical to
stage-3 resolution output. Profiles decided MISSING keep no ``uri``.

Output layout mirrors the gold splits (flat files, one per document)::

    data/annotated/no-gold/<SOURCE_DOCUMENT>.html   # e.g. 1982CanLII3042NBCA.html

like ``data/annotated/test/<doc>.html`` (same ``manual_label``/``auto_label``
tag style, now carrying ``uri`` attributes) — a first silver split the
company can complete with the returned ``uri_request_missing.json`` URIs.

Usage (from repo root)::

    python uri_request/build_silver_split.py
    python uri_request/build_silver_split.py --overwrite

Validation: re-parses every written file and reports tags-with-uri counts vs
the expected number of EXISTS profiles per document.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.main_resolution import apply_uris_to_html  # noqa: E402 pipeline convention
from src.rpr import normalize_docid  # noqa: E402

MISSION_DIR = Path(__file__).resolve().parent
DEFAULT_COREF_DIR = Path(
    "output_coref/no-gold_gemma4_coref_fs6_random_from-no-gold_gemma4_DEC_fs6_greedy_paragraph_long-final"
)
DEFAULT_DECISION = MISSION_DIR / "output_decision" / "uri_decision.json"
DEFAULT_OUT_DIR = Path("data/annotated/no-gold")

TAG_NAMES = ("manual_label", "auto_label")


def count_uri_tags(html: str) -> tuple[int, int]:
    """(n_tags_with_docid, n_tags_with_uri) via the same parser as the pipeline."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    tags = soup.find_all(list(TAG_NAMES), attrs={"docid": True})
    return len(tags), sum(1 for t in tags if t.get("uri"))


def main() -> None:
    p = argparse.ArgumentParser(description="Build no-gold silver split.")
    p.add_argument("--coref-dir", type=Path, default=DEFAULT_COREF_DIR)
    p.add_argument("--decision", type=Path, default=DEFAULT_DECISION)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    decisions = json.loads(args.decision.read_text(encoding="utf-8"))
    by_doc: dict[str, dict[str, dict]] = {}
    for d in decisions:
        if d["decision"] != "EXISTS" or not d.get("uri"):
            continue
        by_doc.setdefault(d["source_document"], {})[
            normalize_docid(d["docid"])] = {"uri": d["uri"]}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    doc_dirs = sorted(x for x in args.coref_dir.iterdir() if x.is_dir())
    print(f"[Setup] {len(doc_dirs)} coref doc(s), "
          f"{sum(len(m) for m in by_doc.values())} verified profile(s)")

    n_files = 0
    skipped: list[str] = []
    for doc_dir in doc_dirs:
        coref_files = sorted(doc_dir.glob("*_coref.html"))
        if not coref_files:
            skipped.append(doc_dir.name)
            continue
        src = doc_dir.name  # source document id
        out_path = args.out_dir / f"{src}.html"
        if out_path.exists() and not args.overwrite:
            print(f"  [skip] {src}: exists (use --overwrite)")
            continue
        html = coref_files[0].read_text(encoding="utf-8")
        mapping = by_doc.get(src, {})
        out_html = apply_uris_to_html(html, mapping)
        out_path.write_text(out_html, encoding="utf-8")
        _, n_uri = count_uri_tags(out_html)
        n_files += 1
        print(f"  [{src}] profiles={len(mapping)} tags-with-uri={n_uri}")

    # Global validation: re-parse outputs, sum tags-with-uri, compare to
    # total verified profiles (a profile's docid may cover several tags).
    total_uri_tags = 0
    for f in sorted(args.out_dir.glob("*.html")):
        _, n_uri = count_uri_tags(f.read_text(encoding="utf-8"))
        total_uri_tags += n_uri
    n_verified = sum(len(m) for m in by_doc.values())
    print(f"\n[DONE] {n_files} silver file(s) -> {args.out_dir}")
    print(f"  verified profiles: {n_verified} | tags carrying uri: {total_uri_tags}")
    if skipped:
        print(f"  skipped (no coref file): {skipped}")


if __name__ == "__main__":
    main()
