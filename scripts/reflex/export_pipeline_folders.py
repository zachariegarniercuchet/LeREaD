#!/usr/bin/env python
"""
Export the Reflex (CanLII citator) predictions as *standard run folders*, so
that Reflex can be evaluated and compared exactly like the LLM / rules runs.

Two folders are produced (test split by default):

    output/test_reflex/<doc>/<doc>_final.html
        Extraction output.  Each predicted mention is an `auto_label` tag
        (parent-level only: decision / legislation / secondary sources) placed
        on the **gold text**, at the gold coordinates obtained by aligning the
        Reflex rendering onto the gold rendering.

    output_coref/test_reflex/<doc>/<doc>_coref.html
    output_coref/test_reflex/<doc>/<doc>_coref_clusters.json
    output_coref/test_reflex/<doc>/<doc>_coref_mapping.json
        Coreference output.  Same spans, plus `id` and `docid` attributes.
        `docid` is the Reflex `concatenated-id` when the export has one
        (legislation); decision anchors carry none, so each decision mention
        becomes its own singleton cluster - exactly the limitation documented
        in scripts/reflex/report/reflex_coref_report.md.

Why the files are written on the gold text
------------------------------------------
Reflex renders the same document differently from gold (it injects neutral
citations such as `1982 CanLII 19 (SCC)`), so a raw copy of the reflex HTML
cannot be scored by `evaluate.py` / `evaluate_coref.py`.  Writing the
predictions on gold coordinates (the same thing the dedicated Reflex scorer
does internally) makes the folders drop-in compatible with the standard
evaluators and with the per-type report script.

Usage (repo root):
    python scripts/reflex/export_pipeline_folders.py
    python scripts/reflex/export_pipeline_folders.py --splits test --overwrite
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts" / "reflex")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common as C  # noqa: E402
import evaluate_reflex_coref_resolution as CR  # noqa: E402

DEFAULT_GAP = 15
EXTRACTION_DIR = PROJECT_ROOT / "output"
COREF_DIR = PROJECT_ROOT / "output_coref"


# ---------------------------------------------------------------------------
# Span selection / rendering
# ---------------------------------------------------------------------------

def _mapped_spans(
    mentions: List[CR.Mention],
    alignment: C.Alignment,
) -> List[Tuple[int, int, CR.Mention]]:
    """Map every reflex mention to gold coordinates.

    Positions are clamped the same way as the dedicated reflex scorer
    (`_build_aligned_mentions`), so the document count / system mention count
    stay comparable.  On this corpus the mapped spans are either disjoint or
    properly nested (an injected citation collapses onto a point inside the
    gold mention that names the same case), which HTML can represent.
    """
    len_g = alignment.len_g
    if len_g <= 0:
        return []
    mapped: List[Tuple[int, int, CR.Mention]] = []
    for m in mentions:
        gs, ge = alignment.map_span(m.start, m.end)
        gs = min(max(0, gs), len_g - 1)
        ge = max(gs + 1, min(ge, len_g))
        mapped.append((gs, ge, m))
    # outer before inner; left before right
    mapped.sort(key=lambda t: (t[0], -t[1]))
    return mapped


def _render_html(text: str, spans, attrs_of) -> str:
    """Build a minimal HTML document whose normalised text is exactly `text`.

    Spans are non-crossing, so they are rendered as a properly nested tree.
    """
    def build(start: int, end: int, k: int):
        parts: List[str] = []
        pos = start
        while k < len(spans) and spans[k][0] < end:
            gs, ge, mention = spans[k]
            if ge > end:          # defensive: clip an unexpected crossing span
                ge = end
            if gs < pos:          # defensive: skip a span already consumed
                k += 1
                continue
            parts.append(html.escape(text[pos:gs], quote=True))
            inner, k = build(gs, ge, k + 1)
            attrs = " ".join(f'{key}="{html.escape(str(value), quote=True)}"'
                             for key, value in attrs_of(mention).items())
            parts.append(f"<auto_label {attrs}>{inner}</auto_label>")
            pos = ge
        parts.append(html.escape(text[pos:end], quote=True))
        return "".join(parts), k

    body, _ = build(0, len(text), 0)
    return ("<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\"></head>"
            "<body><p>" + body + "</p></body></html>\n")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def _clean_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} already exists (use --overwrite)")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def export_documents(
    pairs: List[C.Pair],
    gap: int,
    extraction_out: Path,
    coref_out: Path,
    overwrite: bool,
    verbose: bool = True,
) -> Dict[str, int]:
    _clean_dir(extraction_out, overwrite)
    _clean_dir(coref_out, overwrite)

    n_docs = 0
    n_spans = 0
    for pair in pairs:
        reflex_text, mentions = CR.build_reflex_mentions(pair.reflex_path, gap)
        gold = C.parse_gold(pair.gold_path)
        alignment = C.Alignment(reflex_text, gold.text)
        spans = _mapped_spans(mentions, alignment)

        doc_id = pair.doc_id
        n_docs += 1
        n_spans += len(spans)

        # ---- extraction output -------------------------------------------
        ext_dir = extraction_out / doc_id
        ext_dir.mkdir(parents=True, exist_ok=True)

        def ext_attrs(m: CR.Mention):
            return {"labelname": m.label, "parent": "", "docid": "", "uri": ""}

        (ext_dir / f"{doc_id}_final.html").write_text(
            _render_html(gold.text, spans, ext_attrs), encoding="utf-8")

        # ---- coref output ------------------------------------------------
        coref_dir = coref_out / doc_id
        coref_dir.mkdir(parents=True, exist_ok=True)

        mapping: Dict[str, str] = {}
        clusters: Dict[str, List[str]] = defaultdict(list)
        for i, (_gs, _ge, m) in enumerate(spans):
            mention_id = str(i)
            cluster = m.docid.strip() or f"__singleton_{doc_id}_{i}"
            mapping[mention_id] = cluster
            clusters[cluster].append(mention_id)

        coref_ids = {id(m): str(i) for i, (_gs, _ge, m) in enumerate(spans)}

        def coref_attrs(m: CR.Mention):
            mention_id = coref_ids[id(m)]
            return {"labelname": m.label, "parent": "", "id": mention_id,
                    "docid": mapping[mention_id]}

        (coref_dir / f"{doc_id}_coref.html").write_text(
            _render_html(gold.text, spans, coref_attrs), encoding="utf-8")

        (coref_dir / f"{doc_id}_coref_clusters.json").write_text(
            json.dumps(list(clusters.values()), indent=2, ensure_ascii=False),
            encoding="utf-8")
        (coref_dir / f"{doc_id}_coref_mapping.json").write_text(
            json.dumps(mapping, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8")

        if verbose:
            print(f"  {doc_id} [{pair.split}] -> {len(spans)} mention(s)")

    return {"docs": n_docs, "spans": n_spans}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default="test",
                    help="comma-separated splits to export (default: test)")
    ap.add_argument("--gap", type=int, default=DEFAULT_GAP,
                    help=f"anchor merge gap (default: {DEFAULT_GAP})")
    ap.add_argument("--split-tag", default=None,
                    help="folder tag (default: the split list, 'test')")
    ap.add_argument("--output-dir", default=str(EXTRACTION_DIR))
    ap.add_argument("--coref-dir", default=str(COREF_DIR))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    wanted = {s.strip() for s in args.splits.split(",") if s.strip()}
    all_pairs = C.pair_documents()
    pairs = [p for p in all_pairs if p.split in wanted and p.how in ("exact", "fuzzy")]
    if not pairs:
        raise SystemExit(f"No reflex documents paired for split(s): {sorted(wanted)}")

    tag = args.split_tag or ("test" if wanted == {"test"} else "-".join(sorted(wanted)))
    extraction_out = Path(args.output_dir) / f"{tag}_reflex"
    coref_out = Path(args.coref_dir) / f"{tag}_reflex"

    print(f"Exporting {len(pairs)} document(s), gap={args.gap}")
    stats = export_documents(pairs, args.gap, extraction_out, coref_out,
                             overwrite=args.overwrite, verbose=not args.quiet)
    print(f"\nExtraction folder : {extraction_out}  ({stats['docs']} docs, {stats['spans']} mentions)")
    print(f"Coref folder      : {coref_out}")


if __name__ == "__main__":
    main()
