#!/usr/bin/env python
"""
Overall (no per-type breakdown) evaluation of the Reflex / CanLII citator at
the three LeREaD stages:

    extraction  -> the standard `src/evaluation/evaluate.py` machinery
                   (`evaluation_l1_util.evaluate_batch` + `..._overlap`)
    coreference -> the standard `src/evaluation/evaluate_coref.py` span-mode
                   machinery (exact + overlap alignment, MUC/B3/CEAF_e/LEA)
    linking     -> the standard `src/evaluation/evaluate_resolution.py`
                   machinery (`compute_uri_metrics` + `..._overlap`)

Unlike `evaluate_reflex.py` / `evaluate_reflex_coref_resolution.py`, this
script does **not** split results by mention type (decision / legislation /
secondary).  Extraction is reported as parent / children / total, using the
same `PARENT vs CHILDREN vs TOTAL` convention as `evaluate_batch`.  Coreference
and linking are reported overall (their natural scope: only parent-level
mentions carry a `docid`, and only parent-level gold mentions carry a `uri`).

Reflex predictions are materialised as standard run folders by
`export_pipeline_folders.py` (extraction `*_final.html`, coref `*_coref.html`,
resolution `*_resolution.html`), all expressed in gold coordinates, so the
stock evaluators can read them unchanged.

Usage (from repo root)::

    python scripts/reflex/evaluate_reflex_overall.py
    python scripts/reflex/evaluate_reflex_overall.py --candidate-pool gold
    python scripts/reflex/evaluate_reflex_overall.py --no-export   # reuse folders

Outputs (default `scripts/reflex/report/`):
    reflex_overall_report.md
    reflex_overall_results.json
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import sys
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
for _p in (str(SCRIPT_DIR), str(PROJECT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import common as C  # noqa: E402
import export_pipeline_folders as EXP  # noqa: E402
import evaluate_reflex_coref_resolution as CR  # noqa: E402


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


L1 = _load("evaluation_l1_util_overall", PROJECT_ROOT / "src" / "evaluation" / "evaluation_l1_util.py")
COREF = _load("evaluate_coref_overall", PROJECT_ROOT / "src" / "evaluation" / "evaluate_coref.py")
RES = _load("evaluate_resolution_overall", PROJECT_ROOT / "src" / "evaluation" / "evaluate_resolution.py")

DATA_DIR = PROJECT_ROOT / "data"
ANNOTATED_DIR = DATA_DIR / "annotated"
REPORT_DIR = SCRIPT_DIR / "report"


# ---------------------------------------------------------------------------
# true-offset cross-check
# ---------------------------------------------------------------------------

def _faithful_extract_spans(html_path: str, context_chars: int = 200) -> list:
    """Offset-faithful drop-in for ``evaluation_l1_util.extract_spans``.

    The stock extractor recovers a tag's character position by searching for
    the nth occurrence of its surface text (and it processes all
    ``manual_label`` tags before all ``auto_label`` tags).  On nested / highly
    repetitive legal documents that is approximate: a span can land on a
    different occurrence of the same string.  This version reads the real DOM
    offsets via ``common.normalised_text_with_tags`` and returns the same
    ``Span`` objects, so the stock matching/metrics code is unchanged.

    Used only for the clearly-labelled cross-check columns; the headline
    numbers still come from the unmodified stock extractor.
    """
    soup = BeautifulSoup(Path(html_path).read_text(encoding="utf-8"), "html.parser")
    body = soup.find("body")
    if body is None:
        return []

    def label_of(tag):
        label = (tag.get("labelname") or "").strip().lower()
        return label or None

    text, tags = C.normalised_text_with_tags(
        body, ("manual_label", "auto_label"), label_of)
    spans = []
    for idx, (label, s, e, attrs) in sorted(tags.items()):
        if e <= s:
            continue
        spans.append(L1.Span(
            text[s:e], label,
            text[max(0, s - context_chars):min(len(text), e + context_chars)],
            attributes=dict(attrs), start_pos=s, end_pos=e,
        ))
    return spans


@contextmanager
def _patched_extractors():
    """Temporarily swap the stock offset extractor for the faithful one on the
    already-loaded evaluator modules (offset only; matching and metrics stay
    stock).  Restores the originals afterwards."""
    l1_orig, res_orig = L1.extract_spans, getattr(RES, "extract_spans", None)
    L1.extract_spans = _faithful_extract_spans
    if res_orig is not None:
        RES.extract_spans = _faithful_extract_spans
    try:
        yield
    finally:
        L1.extract_spans = l1_orig
        if res_orig is not None:
            RES.extract_spans = res_orig


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------

def export_reflex(pairs: List[C.Pair], gap: int, tag: str,
                  candidate_pool: str, quiet: bool = False) -> Tuple[Path, Path, dict]:
    """Write extraction + coref + resolution run folders on gold coordinates."""
    extraction_out = EXP.EXTRACTION_DIR / f"{tag}_reflex"
    coref_out = EXP.COREF_DIR / f"{tag}_reflex"
    pool_path = EXP.GOLD_POOL if candidate_pool == "gold" else EXP.FULL_POOL
    id_to_uri = CR.load_candidate_pool(pool_path)

    stats = EXP.export_documents(
        pairs, gap, extraction_out, coref_out, overwrite=True,
        verbose=not quiet, id_to_uri=id_to_uri,
    )
    return extraction_out, coref_out, stats


# ---------------------------------------------------------------------------
# stage 1: extraction (stock evaluate.py)
# ---------------------------------------------------------------------------

def evaluate_extraction(pairs: List[C.Pair], extraction_out: Path) -> dict:
    file_pairs = [
        (str(p.gold_path), str(extraction_out / p.doc_id / f"{p.doc_id}_final.html"))
        for p in pairs
    ]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exact = L1.evaluate_batch(file_pairs, verbose_per_file=False)
        overlap = L1.evaluate_batch_overlap(file_pairs, verbose_per_file=False)
        # cross-check: same stock machinery, but with DOM-faithful offsets
        with _patched_extractors():
            true_exact = L1.evaluate_batch(file_pairs, verbose_per_file=False)
            true_overlap = L1.evaluate_batch_overlap(file_pairs, verbose_per_file=False)
    return {
        "exact": exact,
        "overlap": overlap,
        "true_exact": true_exact,
        "true_overlap": true_overlap,
        "n_documents": len(file_pairs),
    }


# ---------------------------------------------------------------------------
# stage 2: coreference (stock evaluate_coref.py, span mode)
# ---------------------------------------------------------------------------

def evaluate_coref(pairs: List[C.Pair], coref_out: Path, split: str,
                   unmatched: str = "singleton") -> dict:
    file_pairs = [
        (p.gold_path, coref_out / p.doc_id / f"{p.doc_id}_coref.html")
        for p in pairs
    ]
    config = COREF.CorefEvaluationConfig(
        mode="batch", split=split, batch_folder=str(coref_out),
        data_dir=DATA_DIR, match_mode="span", unmatched=unmatched,
    )

    def run(loader):
        raws, total_stats = [], defaultdict(int)
        for gold_path, system_path in file_pairs:
            gold_clusters, system_clusters, stats = loader(gold_path, system_path, config)
            raws.append(COREF.evaluate_coref_raw(gold_clusters, system_clusters))
            if stats:
                for key, value in stats.items():
                    total_stats[key] += value
        scores = COREF.finalize_raw(COREF.aggregate_raw(raws))
        return scores, dict(total_stats)

    exact, stats_exact = run(COREF.load_cluster_pair)
    overlap, stats_overlap = run(COREF.load_cluster_pair_overlap)
    return {
        "exact": exact,
        "overlap": overlap,
        "stats": stats_exact,
        "stats_overlap": stats_overlap,
        "n_documents": len(file_pairs),
    }


# ---------------------------------------------------------------------------
# stage 3: URI linking (stock evaluate_resolution.py)
# ---------------------------------------------------------------------------

def evaluate_linking(pairs: List[C.Pair], coref_out: Path) -> dict:
    file_pairs = [
        (str(p.gold_path), str(coref_out / p.doc_id / f"{p.doc_id}_resolution.html"))
        for p in pairs
    ]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exact = RES.evaluate_resolution_batch(file_pairs, verbose_per_file=False)
        overlap = RES.evaluate_resolution_batch_overlap(file_pairs, verbose_per_file=False)
        with _patched_extractors():
            true_exact = RES.evaluate_resolution_batch(file_pairs, verbose_per_file=False)
            true_overlap = RES.evaluate_resolution_batch_overlap(file_pairs, verbose_per_file=False)

    # label composition of real-uri mentions (to explain precision/recall)
    gold_by_label: Dict[str, int] = defaultdict(int)
    sys_by_label: Dict[str, int] = defaultdict(int)
    for gold_path, system_path in file_pairs:
        for span in RES.extract_spans(gold_path):
            if RES._is_real_uri(RES._get_uri(span)):
                gold_by_label[span.labelname] += 1
        for span in RES.extract_spans(system_path):
            if RES._is_real_uri(RES._get_uri(span)):
                sys_by_label[span.labelname] += 1

    return {
        "exact": exact[0],
        "overlap": overlap[0],
        "per_label_exact": exact[1],
        "per_label_overlap": overlap[1],
        "true_exact": true_exact[0],
        "true_overlap": true_overlap[0],
        "gold_uri_by_label": dict(gold_by_label),
        "system_uri_by_label": dict(sys_by_label),
        "n_documents": len(file_pairs),
    }


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _pct(value) -> str:
    return "-" if value is None else f"{value * 100:.2f}"


def _prf_row(level: str, m: dict) -> List[str]:
    return [level, m["n_gold"], m["n_system"], m["tp"],
            _pct(m["P"]), _pct(m["R"]), _pct(m["F1"])]


def md_table(header: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def build_report(args, pairs, extraction, coref, linking, export_stats,
                 extraction_out, coref_out, candidate_pool) -> str:
    lines: List[str] = []
    A = lines.append

    by_split = Counter(p.split for p in pairs)
    A("# Reflex (CanLII citator) — overall evaluation (no per-type breakdown)")
    A("")
    A(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} — "
      f"`python scripts/reflex/evaluate_reflex_overall.py`*")
    A("")
    A("**Scope.** All results are pooled over mention types; there is no "
      "decision / legislation / secondary breakdown. Extraction is reported "
      "for parent mentions, child components and the total "
      "(parent + children), exactly as `evaluate_batch` does. Coreference and "
      "URI linking are reported overall.")
    A("")
    A("Headline numbers come from the unmodified stock evaluators. Where the "
      "stock code reconstructs span offsets heuristically (extraction and "
      "linking), a clearly-labelled *DOM-faithful offset* cross-check is "
      "given alongside; coreference computes real offsets, so it needs none.")
    A("")
    A(f"* Documents: {len(pairs)} paired Reflex export(s) "
      f"({', '.join(f'{v} {k}' for k, v in sorted(by_split.items()))}).")
    A(f"* Merge gap: `{args.gap}` normalised characters.")
    A(f"* Candidate pool for URI lookup: `{candidate_pool}`.")
    A(f"* Exported run folders: `{extraction_out}` and `{coref_out}`.")
    A("")
    A("The same scripts the paper uses score Reflex here: "
      "`evaluation_l1_util.evaluate_batch` / `..._overlap` for extraction, "
      "`evaluate_coref.py` span mode for coreference, and "
      "`evaluate_resolution_batch` / `..._overlap` for linking. Reflex spans "
      "are aligned into gold coordinates first, so no scorer is re-implemented.")
    A("")

    # ---- extraction ------------------------------------------------------
    A("## 1. Extraction (standard `evaluate.py` machinery)")
    A("")
    A("**EXACT** — label + boundary text + 200-char context (the Table-1 metric)")
    A("")
    rows = []
    for level, key in (("Parent", "parent"), ("Children", "children")):
        rows.append(_prf_row(level, extraction["exact"]["parent_child_breakdown"][key]))
    rows.append(_prf_row("**Total (parent + children)**", extraction["exact"]["overall"]))
    A(md_table(["mentions", "gold", "system", "TP", "P %", "R %", "F1 %"], rows))
    A("")
    A("**OVERLAP** — label + character overlap (lenient comparison)")
    A("")
    rows = []
    for level, key in (("Parent", "parent"), ("Children", "children")):
        rows.append(_prf_row(level, extraction["overlap"]["parent_child_breakdown"][key]))
    rows.append(_prf_row("**Total (parent + children)**", extraction["overlap"]["overall"]))
    A(md_table(["mentions", "gold", "system", "TP", "P %", "R %", "F1 %"], rows))
    A("")
    A("*Reflex produces no child components (`title`, `citation`, `fragment`, "
      "`authors`, `source`), so the children row is 0/0 by construction and "
      "the total is the parent score diluted by every gold child. The parent "
      "row is the fair, citator-scope number; the total row is the number "
      "comparable with a full pipeline that annotates both levels.*")
    A("")
    A("**Cross-check — DOM-faithful offsets.** The stock `extract_spans` "
      "does not read a tag's real position: it looks up the *n*-th occurrence "
      "of the tag's surface text (processing `manual_label` tags before "
      "`auto_label` ones). On nested, highly repetitive legal text a span can "
      "land on a different occurrence of the same string — and Reflex's "
      "alignment puts many anchors on 1-character gold insertion points whose "
      "text repeats everywhere. Re-running the **same stock matching and "
      "metrics** with true DOM offsets isolates that scorer effect:")
    A("")
    rows = []
    for name, level, key in (
        ("parent", "Parent", "parent"),
        ("total", "Total", "overall"),
    ):
        for mode, label in (("exact", "EXACT"), ("overlap", "OVERLAP")):
            true_mode = f"true_{mode}"
            if key == "parent":
                stock_m = extraction[mode]["parent_child_breakdown"][key]
                true_m = extraction[true_mode]["parent_child_breakdown"][key]
            else:
                stock_m = extraction[mode]["overall"]
                true_m = extraction[true_mode]["overall"]
            rows.append([
                f"{level} {label}",
                stock_m["tp"], true_m["tp"],
                _pct(stock_m["F1"]), _pct(true_m["F1"]),
                f"{true_m['F1'] * 100 - stock_m['F1'] * 100:+.2f}",
            ])
    A(md_table(["metric", "TP stock", "TP true", "F1 stock %",
                "F1 true %", "delta"], rows))
    A("")

    # ---- coref -----------------------------------------------------------
    A("## 2. Coreference (standard `evaluate_coref.py`, span mode)")
    A("")
    A(f"Unmatched policy: `singleton` (end-to-end / twinless). "
      f"{coref['n_documents']} document(s).")
    A("")
    rows = []
    for metric in ("MUC", "B3", "CEAF_e", "LEA", "CoNLL"):
        e, o = coref["exact"][metric], coref["overlap"][metric]
        rows.append([
            metric,
            _pct(e.get("precision")), _pct(e.get("recall")), _pct(e["f1"]),
            _pct(o.get("precision")), _pct(o.get("recall")), _pct(o["f1"]),
        ])
    A(md_table(["metric", "exact P %", "exact R %", "exact F1 %",
                "overlap P %", "overlap R %", "overlap F1 %"], rows))
    A("")
    se, so = coref["stats"], coref["stats_overlap"]
    A(f"Alignment (exact): gold mentions {se.get('gold', '-')} | "
      f"system mentions {se.get('system', '-')} | matched {se.get('matched', '-')}.")
    A(f"Alignment (overlap): gold mentions {so.get('gold', '-')} | "
      f"system mentions {so.get('system', '-')} | matched {so.get('matched', '-')}.")
    A("")
    A("*Only parent-level mentions carry a gold `docid`; child components are "
      "not coreference instances. Reflex exposes a ``concatenated-id`` only on "
      "legislation anchors, so decision predictions are singleton clusters. "
      "Offsets here are read from the DOM, so no cross-check is needed.*")
    A("")

    # ---- linking ---------------------------------------------------------
    A("## 3. URI linking (standard `evaluate_resolution.py` machinery)")
    A("")
    A("Only gold mentions with a real `uri` are counted; child components carry "
      "no `uri`, so linking is effectively parent-level.")
    A("")
    rows = []
    for mode in ("exact", "overlap"):
        m = linking[mode]
        rows.append([mode, _pct(m["precision"]), _pct(m["recall"]), _pct(m["f1"])])
    A(md_table(["match", "P %", "R %", "F1 %"], rows))
    A("")
    gold_comp = linking.get("gold_uri_by_label", {})
    sys_comp = linking.get("system_uri_by_label", {})
    n_gold_uri = sum(gold_comp.values())
    n_sys_uri = sum(sys_comp.values())
    A(f"Counts behind these numbers: Reflex emits **{n_sys_uri}** URI(s) "
      f"({', '.join(f'{v} {k}' for k, v in sorted(sys_comp.items())) or 'none'}), "
      f"while **{n_gold_uri}** gold mentions need one "
      f"({', '.join(f'{v} {k}' for k, v in sorted(gold_comp.items()))}). "
      "Precision is over Reflex's emitted URIs; recall is over all gold "
      "real-uri mentions, including the decision/secondary mentions that "
      "Reflex cannot link at all. That asymmetry is why the overall F1 is "
      "low despite reasonable precision.")
    A("")
    A(f"Spans with a Reflex-derived uri: {export_stats.get('uri_spans', 0)}. "
      "Anchors without a ``concatenated-id`` (all decisions) produce no uri "
      "and therefore count as `FN — no uri produced` when the gold span needs "
      "one.")
    A("")
    A("**Cross-check — DOM-faithful offsets** (same stock resolution "
      "matching, true positions):")
    A("")
    rows = []
    for mode, label in (("exact", "EXACT"), ("overlap", "OVERLAP")):
        s, t = linking[mode], linking[f"true_{mode}"]
        rows.append([label,
                     _pct(s["precision"]), _pct(s["recall"]), _pct(s["f1"]),
                     _pct(t["precision"]), _pct(t["recall"]), _pct(t["f1"])])
    A(md_table(["match", "P stock %", "R stock %", "F1 stock %",
                "P true %", "R true %", "F1 true %"], rows))
    A("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Overall (non-per-type) Reflex evaluation at extraction, "
                    "coreference and URI linking, using the stock evaluators.")
    ap.add_argument("--split", default="test",
                    help="gold split to evaluate (default: test)")
    ap.add_argument("--gap", type=int, default=EXP.DEFAULT_GAP,
                    help=f"anchor merge gap (default: {EXP.DEFAULT_GAP})")
    ap.add_argument("--candidate-pool", choices=["gold", "full"], default="gold",
                    help="candidate pool for URI lookup (default: gold)")
    ap.add_argument("--unmatched", choices=["singleton", "drop"], default="singleton",
                    help="coref unmatched policy (default: singleton, end-to-end)")
    ap.add_argument("--no-export", action="store_true",
                    help="reuse the already-exported run folders")
    ap.add_argument("--out", default=str(REPORT_DIR),
                    help=f"report folder (default: {REPORT_DIR})")
    args = ap.parse_args()

    all_pairs = C.pair_documents(min_ratio=0.85, verbose=False)
    pairs = [p for p in all_pairs if p.split == args.split and p.how in ("exact", "fuzzy")]
    if not pairs:
        raise SystemExit(f"No Reflex documents paired for split {args.split!r}.")

    tag = args.split
    extraction_out = EXP.EXTRACTION_DIR / f"{tag}_reflex"
    coref_out = EXP.COREF_DIR / f"{tag}_reflex"

    print("=" * 74)
    print(f"REFLEX OVERALL EVALUATION — split={args.split}, gap={args.gap}")
    print("=" * 74)
    print(f"Paired {len(pairs)} Reflex document(s) with gold.")

    if args.no_export:
        export_stats = {"uri_spans": -1}
    else:
        print("\nExporting Reflex predictions as standard run folders ...")
        extraction_out, coref_out, export_stats = export_reflex(
            pairs, args.gap, tag, args.candidate_pool, quiet=True)
        print(f"  extraction: {extraction_out}")
        print(f"  coref     : {coref_out}")

    print("\n[1/3] Extraction (standard evaluate.py) ...")
    extraction = evaluate_extraction(pairs, extraction_out)
    p = extraction["exact"]["parent_child_breakdown"]["parent"]
    t = extraction["exact"]["overall"]
    print(f"  exact parent F1={p['F1']*100:.2f}  total F1={t['F1']*100:.2f}")

    print("\n[2/3] Coreference (standard evaluate_coref.py, span mode) ...")
    coref = evaluate_coref(pairs, coref_out, args.split, args.unmatched)
    print(f"  exact CoNLL F1={coref['exact']['CoNLL']['f1']*100:.2f}  "
          f"overlap CoNLL F1={coref['overlap']['CoNLL']['f1']*100:.2f}")

    print("\n[3/3] URI linking (standard evaluate_resolution.py) ...")
    linking = evaluate_linking(pairs, coref_out)
    print(f"  exact F1={linking['exact']['f1']*100:.2f}  "
          f"overlap F1={linking['overlap']['f1']*100:.2f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args, pairs, extraction, coref, linking, export_stats,
                          extraction_out, coref_out, args.candidate_pool)
    report_path = out_dir / "reflex_overall_report.md"
    report_path.write_text(report, encoding="utf-8")

    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "split": args.split,
        "merge_gap": args.gap,
        "candidate_pool": args.candidate_pool,
        "unmatched": args.unmatched,
        "n_documents": len(pairs),
        "export": export_stats,
        "pairs": [{"reflex": p.reflex_path.name, "doc_id": p.doc_id,
                   "split": p.split, "match": p.how} for p in pairs],
        "extraction": extraction,
        "coref": coref,
        "linking": linking,
    }
    json_path = out_dir / "reflex_overall_results.json"
    json_path.write_text(json.dumps(payload, indent=2, default=dict), encoding="utf-8")

    print(f"\nReport: {report_path}")
    print(f"JSON  : {json_path}")


if __name__ == "__main__":
    main()
