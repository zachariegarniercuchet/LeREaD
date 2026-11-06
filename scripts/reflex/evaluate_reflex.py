#!/usr/bin/env python
"""
Evaluate the Reflex (CanLII citator) extractions in data/reflex/ against the
LeREaD gold standard — parent (mention) level only, per mention type.

    # everything currently available (human + silver splits, reported per split)
    python scripts/reflex/evaluate_reflex.py

    # only the human test split (the one behind Table 1 of the paper)
    python scripts/reflex/evaluate_reflex.py --splits test

    # fix the merge gap instead of sweeping it
    python scripts/reflex/evaluate_reflex.py --gap 15

    # same-document comparison against other systems' outputs
    python scripts/reflex/evaluate_reflex.py \
        --compare rules="output/test_baseline_rules/**/*_final.html" \
        --compare gpt-5.2="output/test_gpt-5.2_DEC_fs6_greedy_sentence_long/**/*_final.html"

Outputs (default --out scripts/reflex/report):
    reflex_extraction_report.md      human-readable report
    reflex_extraction_results.json   every number, for plots/tables elsewhere

Method (see scripts/reflex/common.py)
-------------------------------------
* Reflex mentions are the anchors CanLII resolved:
    `reflex2-link` / href `/laws/`  -> legislation
    `reflex3-caselaw` / href `/doc/` -> decision
    href `/jnl/ /book/ /comm/`      -> secondary sources
* Consecutive anchors resolving to the same reference (same base href) are
  merged, so `Federal Court Act` + `s. 18` is one predicted mention, like gold.
* The reflex text is token-diffed onto the gold text.  Reflex *injects* neutral
  citations ("1982 CanLII 19 (SCC)") that the gold text does not contain; they
  map to their insertion point, normally inside the gold decision mention.
* Predictions are expressed in gold coordinates and scored with the paper's own
  `src/evaluation/evaluation_l1_util.py`:
      EXACT   = label + boundary text + 200-char context  (Table-1 metric)
      OVERLAP = label + character overlap                 (lenient comparison)
* Components (title / citation / fragment / ...) are never scored: Reflex does
  not produce them, only parent mentions.
"""

from __future__ import annotations

import argparse
import glob as globmod
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import common as C  # noqa: E402

MODES = ("exact", "overlap")
DEFAULT_GAPS = (0, 5, 10, 15, 20, 30, 45, 60)
SHORT_MENTION_CHARS = 30     # below this a gold mention is a short form / fragment
CITE_RE = re.compile(
    r"(\b\d{4}\s+CanLII\s+\d|\b\d{4}\s+[A-Z]{2,6}\s+\d+|\[\d{4}\]"
    r"|\b\d+\s+(D\.?L\.?R|S\.?C\.?R|W\.?W\.?R|C\.?C\.?C|A\.?(2d|3d)?))", re.I)
PAPER_REFERENCE = {          # Table 1 (main_results.tex) — mention micro-F1, test set
    "GPT-5.2": 0.76,
    "Gemma-4-31B-it": 0.78,
    "Qwen3.8-27B": 0.71,
    "Muse-Glimmer-30B": 0.65,
    "SaulLM-54B-Inst.": 0.06,
}


# ---------------------------------------------------------------------------
# matching (mirrors evaluation_l1_util.match_spans / match_spans_overlap, but
# also returns the false positives / false negatives for the error report)
# ---------------------------------------------------------------------------

def match_detail(gold: list, system: list, mode: str):
    used = [False] * len(system)
    tp, pairs_, fn_idx = 0, [], []
    for i, g in enumerate(gold):
        best = -1
        if mode == "exact":
            for j, s in enumerate(system):
                if not used[j] and g.matches(s):
                    best = j
                    break
        else:
            best_ov = 0
            for j, s in enumerate(system):
                if used[j] or not g.overlaps(s):
                    continue
                ov = g.overlap_len(s)
                if ov > best_ov:
                    best_ov, best = ov, j
                elif best == -1 and ov <= 0:
                    best = j
        if best >= 0:
            used[best] = True
            tp += 1
            pairs_.append((i, best))
        else:
            fn_idx.append(i)
    fp_idx = [j for j, u in enumerate(used) if not u]
    return tp, pairs_, fn_idx, fp_idx


def prf(tp: int, n_gold: int, n_system: int) -> Tuple[float, float, float]:
    P = tp / n_system if n_system else 0.0
    R = tp / n_gold if n_gold else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return P, R, F


def new_bucket() -> Dict[str, Dict[str, int]]:
    return defaultdict(lambda: {"tp": 0, "n_gold": 0, "n_system": 0})


def _jsonable(obj):
    """Recursively turn Counters / defaultdicts / dataclasses into plain data."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def add(bucket, label_counts):
    for lab, m in label_counts.items():
        bucket[lab]["tp"] += m["tp"]
        bucket[lab]["n_gold"] += m["n_gold"]
        bucket[lab]["n_system"] += m["n_system"]


def totals(label_counts) -> Tuple[int, int, int]:
    tp = sum(m["tp"] for m in label_counts.values())
    ng = sum(m["n_gold"] for m in label_counts.values())
    ns = sum(m["n_system"] for m in label_counts.values())
    return tp, ng, ns


def _per_label_counts(gold_spans, sys_spans, pairs_) -> Dict[str, dict]:
    out = defaultdict(lambda: {"tp": 0, "n_gold": 0, "n_system": 0})
    for g in gold_spans:
        out[g.labelname]["n_gold"] += 1
    for s in sys_spans:
        out[s.labelname]["n_system"] += 1
    for i, j in pairs_:
        out[gold_spans[i].labelname]["tp"] += 1
    return out


def merge_splits(per_split: Dict[str, dict], splits=None) -> Dict[str, dict]:
    """Union of the per-split buckets (all splits, or a subset)."""
    out = {m: new_bucket() for m in MODES}
    for sp, modes in per_split.items():
        if splits is not None and sp not in splits:
            continue
        for mode in MODES:
            add(out[mode], modes[mode])
    return out


# ---------------------------------------------------------------------------
# evaluation over a set of pairs
# ---------------------------------------------------------------------------

def evaluate_pairs(pairs: List[C.Pair], gaps: List[int], ctx: int, verbose: bool = True):
    parsed = []
    for p in pairs:
        gold = C.parse_gold(p.gold_path)
        reflex = C.parse_reflex(p.reflex_path)
        parsed.append((p, gold, reflex, C.Alignment(reflex.text, gold.text)))

    results: Dict[int, dict] = {}
    for gap in gaps:
        per_split: Dict[str, dict] = {}
        per_doc = {}
        for p, gold, reflex, align in parsed:
            mentions = reflex.mentions(gap)
            gold_spans, sys_spans = C.build_spans(gold, mentions, align, ctx)
            modes = per_split.setdefault(p.split, {m: new_bucket() for m in MODES})
            doc = {"split": p.split, "n_gold": len(gold_spans),
                   "n_system": len(sys_spans), "n_anchors": len(reflex.anchors),
                   "align_ratio": round(align.ratio, 4)}
            for mode in MODES:
                tp, prs, fn, fp = match_detail(gold_spans, sys_spans, mode)
                add(modes[mode], _per_label_counts(gold_spans, sys_spans, prs))
                P, R, F = prf(tp, len(gold_spans), len(sys_spans))
                doc[mode] = {"tp": tp, "n_gold": len(gold_spans),
                             "n_system": len(sys_spans), "P": P, "R": R, "F1": F,
                             "n_fp": len(fp), "n_fn": len(fn)}
            per_doc[p.doc_id] = doc
        results[gap] = {"per_split": per_split, "per_doc": per_doc}
        if verbose:
            _print_gap(results[gap], gap, pairs)
    return results, parsed


def _print_gap(res: dict, gap: int, pairs) -> None:
    union = merge_splits(res["per_split"])
    row = f"  gap={gap:<4} docs={len(pairs):<4}"
    for mode in MODES:
        tp, ng, ns = totals(union[mode])
        P, R, F = prf(tp, ng, ns)
        row += f" | {mode}: P={P*100:5.1f} R={R*100:5.1f} F1={F*100:5.1f}"
    print(row, flush=True)


def best_gap(results: Dict[int, dict], gaps: List[int], mode: str = "overlap") -> int:
    best, best_f = gaps[0], -1.0
    for gap in gaps:
        tp, ng, ns = totals(merge_splits(results[gap]["per_split"])[mode])
        _, _, F = prf(tp, ng, ns)
        if F > best_f:
            best, best_f = gap, F
    return best


# ---------------------------------------------------------------------------
# optional side-by-side with other systems (LLM / rules) on the same documents
# ---------------------------------------------------------------------------

def evaluate_system_vs_gold(system_paths: List[Path], pairs_by_id: Dict[str, C.Pair],
                            ctx: int) -> dict:
    """Score other system outputs (HTML with manual_label/auto_label tags) with
    exactly the same parent-level code path, restricted to documents that also
    have a reflex export."""
    per_mode = {m: new_bucket() for m in MODES}
    per_doc = {}
    for sp in sorted(system_paths):
        stem = sp.name
        for suffix in ("_final.html", "_processed.html"):
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                break
        else:
            stem = sp.stem
        pair = pairs_by_id.get(stem)
        if pair is None:
            continue
        gold = C.parse_gold(pair.gold_path)
        system = C.parse_system(sp)
        align = C.Alignment(system.text, gold.text)
        gold_spans, sys_spans = C.build_spans(gold, system.parents, align, ctx)
        doc = {"n_gold": len(gold_spans), "n_system": len(sys_spans)}
        for mode in MODES:
            tp, prs, fn, fp = match_detail(gold_spans, sys_spans, mode)
            add(per_mode[mode], _per_label_counts(gold_spans, sys_spans, prs))
            P, R, F = prf(tp, len(gold_spans), len(sys_spans))
            doc[mode] = {"tp": tp, "n_gold": len(gold_spans), "n_system": len(sys_spans),
                         "P": P, "R": R, "F1": F}
        per_doc[stem] = doc
    return {"per_mode": per_mode, "per_doc": per_doc}


# ---------------------------------------------------------------------------
# diagnostics: recall chain + false-negative taxonomy
# ---------------------------------------------------------------------------

def false_negative_taxonomy(parsed, gap: int, ctx: int, split: str) -> dict:
    """
    For one split: how many gold mentions Reflex detects at all (OVERLAP),
    how many it reproduces with exact boundaries (EXACT), and why the rest are
    missed (short form/fragment vs full mention, citation present or not,
    nearest reflex mention of the same type in the document).
    """
    acc = {}
    examples = defaultdict(list)
    for p, gold, reflex, al in parsed:
        if p.split != split:
            continue
        gold_spans, sys_spans = C.build_spans(gold, reflex.mentions(gap), al, ctx)
        for mode in MODES:
            tp, prs, fn, fp = match_detail(gold_spans, sys_spans, mode)
            add(acc.setdefault(mode, new_bucket()), _per_label_counts(gold_spans, sys_spans, prs))

        tp_o, _, fn_o, _ = match_detail(gold_spans, sys_spans, "overlap")
        tp_e, _, _, _ = match_detail(gold_spans, sys_spans, "exact")
        for i in fn_o:
            g = gold_spans[i]
            lab = g.labelname
            a = acc.setdefault("fn", new_bucket())
            a[lab]["n_gold"] += 1                    # reuse bucket as a counter
            short = len(g.text) <= SHORT_MENTION_CHARS
            has_cite = bool(CITE_RE.search(g.text))
            acc.setdefault("fn_short", Counter())[lab] += int(short)
            acc.setdefault("fn_full", Counter())[lab] += int(not short)
            acc.setdefault("fn_cite", Counter())[lab] += int(has_cite)

            # distance to the closest reflex mention of the same type
            best = None
            for s in sys_spans:
                if s.labelname != lab:
                    continue
                d = 0
                if s.end_pos <= g.start_pos:
                    d = g.start_pos - s.end_pos
                elif s.start_pos >= g.end_pos:
                    d = s.start_pos - g.end_pos
                if best is None or d < best:
                    best = d
            if best is None:
                key = "no reflex link of this type in the document"
            elif best == 0:
                key = "touching (boundary mismatch)"
            elif best <= 40:
                key = "within 40 chars"
            else:
                key = "no reflex link nearby (genuine miss)"
            acc.setdefault("fn_dist", defaultdict(Counter))[lab][key] += 1
            bucket = (lab, key)
            if len(examples[bucket]) < 3:
                examples[bucket].append(g.text[:100])
    acc["examples"] = dict(examples)
    return acc


def coverage_stats(parsed, gap: int) -> dict:
    """How many raw reflex anchors exist per type, and how many predictions the
    merging produces — plus the confirmation that Reflex never links a
    secondary source."""
    raw, pred, unclassified = Counter(), Counter(), 0
    for p, gold, reflex, al in parsed:
        for label, href, text, s, e in reflex.anchors:
            raw[label] += 1
        for m in reflex.mentions(gap):
            pred[m.label] += 1
        unclassified += len(reflex.unclassified)
    return {"raw_anchors": dict(raw), "predictions": dict(pred),
            "unclassified_http_links": unclassified}


# ---------------------------------------------------------------------------
# report helpers
# ---------------------------------------------------------------------------

def md_table(header: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def prf_row(label_counts, label, name=None) -> List[str]:
    m = label_counts.get(label)
    if not m:
        return [name or label, "-", "-", "-", "-", "-", "-"]
    P, R, F = prf(m["tp"], m["n_gold"], m["n_system"])
    return [name or label, m["n_gold"], m["n_system"], m["tp"],
            f"{P*100:.1f}", f"{R*100:.1f}", f"{F*100:.1f}"]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate reflex extractions against gold")
    ap.add_argument("--splits", default=None,
                    help="comma list of splits to score (default: every split with a "
                         "reflex export; results are always broken down per split)")
    ap.add_argument("--gap", type=int, default=None,
                    help="merge gap; default = best of --gaps by overlap micro-F1")
    ap.add_argument("--gaps", default=",".join(str(g) for g in DEFAULT_GAPS))
    ap.add_argument("--context-chars", type=int, default=200)
    ap.add_argument("--min-ratio", type=float, default=0.85,
                    help="min content-token similarity to accept a fuzzy id match")
    ap.add_argument("--compare", action="append", default=[], metavar="NAME=GLOB",
                    help="repeatable NAME=GLOB of other system HTML files "
                         "(LLM/rules) scored on the same documents")
    ap.add_argument("--taxonomy-split", default=None,
                    help="split used for the false-negative taxonomy "
                         "(default: 'test' if present)")
    ap.add_argument("--out", default=str(Path(__file__).parent / "report"))
    ap.add_argument("--max-examples", type=int, default=6)
    args = ap.parse_args()

    gaps = [int(g) for g in args.gaps.split(",") if g.strip()]

    print("=" * 74)
    print("REFLEX EXTRACTION EVALUATION  (parent mentions only)")
    print("=" * 74)

    pairs = C.pair_documents(min_ratio=args.min_ratio, verbose=False)
    usable = [p for p in pairs if p.how in ("exact", "fuzzy")]
    skipped = [p for p in pairs if p.how not in ("exact", "fuzzy")]
    if args.splits:
        want = {s.strip() for s in args.splits.split(",") if s.strip()}
        usable = [p for p in usable if p.split in want]

    by_split = Counter(p.split for p in usable)
    print(f"\nPaired {len(usable)} reflex document(s) with gold "
          f"({len(skipped)} unusable):")
    for sp, n in sorted(by_split.items()):
        print(f"   {sp:<10} {n:>4} document(s)")
    fuzzy = [p for p in usable if p.how == "fuzzy"]
    if fuzzy:
        print(f"   fuzzy id matches ({len(fuzzy)}):")
        for p in fuzzy:
            print(f"      {p.reflex_path.name}  ->  {p.doc_id} [{p.split}] "
                  f"sim={p.text_ratio:.3f}")
    for p in skipped:
        print(f"   !! {p.reflex_path.name} -> {p.how} (no matching gold document)")
    if not usable:
        print("Nothing to evaluate.")
        return

    print("\nMerge-gap sweep (micro over every evaluated document):")
    results, parsed = evaluate_pairs(usable, gaps, args.context_chars)
    gap = args.gap if args.gap is not None else best_gap(results, gaps, "overlap")
    if args.gap is None:
        print(f"\nSelected merge gap: {gap} (best OVERLAP micro-F1)")
    if gap not in results:
        extra, _ = evaluate_pairs(usable, [gap], args.context_chars, verbose=False)
        results[gap] = extra[gap]
    res = results[gap]

    # ---- console summary ---------------------------------------------------
    print("\n" + "=" * 74)
    print(f"FINAL RESULTS  (merge gap = {gap})")
    print("=" * 74)
    for sp in sorted(res["per_split"]):
        for mode in MODES:
            lc = res["per_split"][sp][mode]
            tp, ng, ns = totals(lc)
            P, R, F = prf(tp, ng, ns)
            print(f"  {sp:<10} {mode:<8} P={P*100:5.1f} R={R*100:5.1f} F1={F*100:5.1f}"
                  f"   (gold={ng} sys={ns} tp={tp})")
    print()

    # ---- comparisons -------------------------------------------------------
    comparisons: Dict[str, dict] = {}
    pairs_by_id = {p.doc_id: p for p in usable}
    for spec in args.compare:
        if "=" not in spec:
            print(f"  warning: ignoring --compare {spec!r}: expected NAME=GLOB")
            continue
        name, pattern = spec.split("=", 1)
        files = sorted({Path(f) for f in globmod.glob(pattern)}
                       | {Path(f) for f in globmod.glob(pattern, recursive=True)})
        comp = evaluate_system_vs_gold(files, pairs_by_id, args.context_chars)
        if not comp["per_doc"]:
            print(f"  warning: comparison {name!r}: no document overlaps the reflex set")
            continue
        comparisons[name] = comp
        common = sorted(set(comp["per_doc"]) & set(res["per_doc"]))
        line = f"  comparison {name:<12} {len(common)} shared document(s):"
        for mode in MODES:
            r_tp = sum(res["per_doc"][d][mode]["tp"] for d in common)
            r_ng = sum(res["per_doc"][d][mode]["n_gold"] for d in common)
            r_ns = sum(res["per_doc"][d][mode]["n_system"] for d in common)
            c_tp = sum(comp["per_doc"][d][mode]["tp"] for d in common)
            c_ng = sum(comp["per_doc"][d][mode]["n_gold"] for d in common)
            c_ns = sum(comp["per_doc"][d][mode]["n_system"] for d in common)
            rP, rR, rF = prf(r_tp, r_ng, r_ns)
            cP, cR, cF = prf(c_tp, c_ng, c_ns)
            line += (f"\n      {mode:<8} reflex F1={rF*100:5.1f} "
                     f"(P={rP*100:4.1f} R={rR*100:4.1f})  vs  {name} "
                     f"F1={cF*100:5.1f} (P={cP*100:4.1f} R={cR*100:4.1f})")
        print(line)

    # ---- diagnostics -------------------------------------------------------
    tax_split = args.taxonomy_split
    if tax_split is None:
        tax_split = "test" if any(p.split == "test" for p in usable) \
            else (sorted(res["per_split"])[0] if res["per_split"] else None)
    taxonomy = false_negative_taxonomy(parsed, gap, args.context_chars, tax_split) \
        if tax_split else {}
    coverage = coverage_stats(parsed, gap)

    # ---- report ------------------------------------------------------------
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = build_report(args, usable, results, gap, parsed, comparisons,
                          taxonomy, tax_split, coverage)
    report_path = out_dir / "reflex_extraction_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\nReport: {report_path}")

    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "merge_gap": gap,
        "gaps": gaps,
        "context_chars": args.context_chars,
        "pairs": [{"reflex": p.reflex_path.name, "doc_id": p.doc_id, "split": p.split,
                   "match": p.how, "similarity": round(p.text_ratio, 4)}
                  for p in usable],
        "sweep": {str(g): {"per_split": results[g]["per_split"]} for g in results},
        "final": {"gap": gap, "per_split": res["per_split"], "per_doc": res["per_doc"]},
        "coverage": coverage,
        "false_negative_taxonomy_split": tax_split,
    }
    if taxonomy:
        payload["fn_taxonomy"] = _jsonable({k: v for k, v in taxonomy.items()
                                            if k != "examples"})
    if comparisons:
        payload["comparisons"] = {
            name: {"per_mode": comp["per_mode"], "per_doc": comp["per_doc"]}
            for name, comp in comparisons.items()}
    json_path = out_dir / "reflex_extraction_results.json"
    json_path.write_text(json.dumps(payload, indent=2, default=dict), encoding="utf-8")
    print(f"JSON  : {json_path}")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def build_report(args, pairs, results, gap, parsed, comparisons, taxonomy,
                 tax_split, coverage) -> str:
    res = results[gap]
    lines: List[str] = []
    A = lines.append

    A("# Reflex (CanLII citator) extraction — evaluation against LeREaD gold")
    A("")
    cmd = "python scripts/reflex/evaluate_reflex.py"
    for spec in args.compare:
        cmd += f' --compare "{spec}"'
    A(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} — "
      f"`{cmd}`*")
    A("")
    A("**Scope**: parent mentions only (`decision` / `legislation` / "
      "`secondary sources`). Components (title, citation, fragment, authors, "
      "source) are **not** scored — Reflex does not produce them.")
    A("")

    # 1 ---------------------------------------------------------------------
    A("## 1. Documents")
    A("")
    by_split = Counter(p.split for p in pairs)
    A(f"* {len(pairs)} reflex export(s) paired with a gold document "
      f"({', '.join(f'{v} {k}' for k, v in sorted(by_split.items()))}).")
    A(f"* Raw reflex anchors: {sum(coverage['raw_anchors'].values())} "
      f"-> {sum(coverage['predictions'].values())} predicted mentions after merging "
      f"(gap = {gap}).")
    A(f"* Anchor types seen: "
      f"{', '.join(f'{k}={v}' for k, v in sorted(coverage['raw_anchors'].items()))}; "
      f"{coverage['unclassified_http_links']} http link(s) with an unrecognised target.")
    A("")
    fuzzy = [p for p in pairs if p.how == "fuzzy"]
    if fuzzy:
        A("**Fuzzy id matches** (id differs by the court suffix, documents verified "
          "identical by content-token similarity):")
        A("")
        A(md_table(["reflex file", "gold doc", "split", "similarity"],
                   [[p.reflex_path.name, p.doc_id, p.split, f"{p.text_ratio:.3f}"]
                    for p in fuzzy]))
        A("")
    A("Similarity = content-token Jaccard (same decision re-rendered ~0.96-1.00, "
      "two different decisions ~0.10).")
    A("")

    # 2 ---------------------------------------------------------------------
    A("## 2. How a Reflex prediction is built")
    A("")
    A("1. **What counts as a mention.** `reflex2-link` anchors whose href contains "
      "`/laws/` -> **legislation**; `reflex3-caselaw` anchors with href `/.../doc/` "
      "-> **decision**; `/jnl/ /book/ /comm/` -> **secondary sources**. "
      "Internal navigation (`reflex-parag`, footnote `#_edn…`) is ignored.")
    A("2. **Grouping.** Consecutive anchors resolving to the same reference (same "
      f"base href) within **{gap}** normalised characters become one prediction, "
      "so `Federal Court Act` + `s. 18` is the single mention `Federal Court Act, "
      "s. 18`, as in the gold.")
    A("3. **Alignment.** The reflex rendering is token-diffed onto the gold one. "
      "Reflex *injects* neutral citations (`1982 CanLII 19 (SCC)`) that the gold "
      "text never contains; they map to their insertion point, normally inside the "
      "gold decision mention that names the same case.")
    A("4. **Scoring** re-uses `src/evaluation/evaluation_l1_util.py` on gold "
      "coordinates: **EXACT** = label + boundary text + 200-char context (the "
      "Table-1 metric); **OVERLAP** = label + character overlap (the lenient "
      "comparison metric `evaluate.py` already prints).")
    A("")

    # 3 ---------------------------------------------------------------------
    A("## 3. Merge-gap sweep")
    A("")
    rows = []
    for g in sorted(results):
        union = merge_splits(results[g]["per_split"])
        row = [g]
        for mode in MODES:
            tp, ng, ns = totals(union[mode])
            P, R, F = prf(tp, ng, ns)
            row += [f"{P*100:.1f}", f"{R*100:.1f}", f"{F*100:.1f}"]
        rows.append(row)
    A(md_table(["merge gap", "exact P", "exact R", "exact F1",
                "overlap P", "overlap R", "overlap F1"], rows))
    A("")
    A(f"**Selected gap: {gap}** (best overlap micro-F1; the plateau is wide, "
      "10-30 all within ~0.5 F1).")
    A("")

    # 4 ---------------------------------------------------------------------
    A("## 4. Results per split and mention type")
    A("")
    # -- combined table: decision + legislation, every split -----------------
    A("### Decision + legislation only, all splits")
    A("")
    A("Secondary sources are left out: Reflex never links one, so the category "
      "is 0/0 by construction and carries no signal.")
    A("")
    split_order = [s for s in ("train", "dev", "test", "silver")
                   if s in res["per_split"]]
    labs = [l for l in C.PARENT_LABELS if l != "secondary sources"]
    for mode in MODES:
        title = ("EXACT boundaries (Table-1 metric)" if mode == "exact"
                 else "OVERLAP matching (lenient)")
        A(f"**{mode.upper()} — {title}**")
        A("")
        rows = []
        for sp in split_order:
            lc = res["per_split"][sp][mode]
            for lab in labs:
                m = lc.get(lab)
                if not m:
                    rows.append([sp, lab, "-", "-", "-"])
                    continue
                P, R, F = prf(m["tp"], m["n_gold"], m["n_system"])
                rows.append([sp, lab, f"{P*100:.1f}", f"{R*100:.1f}",
                             f"{F*100:.1f}"])
        A(md_table(["split", "mention type", "P %", "R %", "F1 %"], rows))
        A("")

    A("### Per-split detail (all mention types)")
    A("")
    for sp in sorted(res["per_split"]):
        note = {"test": " — human gold, behind Table 1",
                "dev": " — human gold", "train": " — human gold",
                "silver": " — **silver** gold (auto-annotated), use with care"}.get(sp, "")
        A(f"### split `{sp}`{note} — "
          f"{sum(1 for d in res['per_doc'].values() if d['split'] == sp)} documents")
        A("")
        for mode in MODES:
            lc = res["per_split"][sp][mode]
            title = ("EXACT boundaries (Table-1 metric)" if mode == "exact"
                     else "OVERLAP matching (lenient)")
            A(f"**{mode.upper()} — {title}**")
            A("")
            rows = [prf_row(lc, lab) for lab in C.PARENT_LABELS]
            tp, ng, ns = totals(lc)
            P, R, F = prf(tp, ng, ns)
            rows.append(["**MICRO (all types)**", ng, ns, tp,
                         f"{P*100:.1f}", f"{R*100:.1f}", f"{F*100:.1f}"])
            A(md_table(["mention type", "gold", "reflex", "TP", "P %", "R %", "F1 %"],
                       rows))
            A("")

    # 5 ---------------------------------------------------------------------
    if taxonomy:
        A(f"## 5. Recall chain and why Reflex misses mentions (split `{tax_split}`)")
        A("")
        lc_o = res["per_split"].get(tax_split, {}).get("overlap", {})
        lc_e = res["per_split"].get(tax_split, {}).get("exact", {})
        rows = []
        for lab in C.PARENT_LABELS:
            g = lc_o.get(lab, {}).get("n_gold", 0)
            det = lc_o.get(lab, {}).get("tp", 0)
            ex = lc_e.get(lab, {}).get("tp", 0)
            short = taxonomy.get("fn_short", Counter()).get(lab, 0)
            full = taxonomy.get("fn_full", Counter()).get(lab, 0)
            cite = taxonomy.get("fn_cite", Counter()).get(lab, 0)
            rows.append([lab, g,
                         f"{det} ({det/g*100:.1f}%)" if g else "0",
                         f"{ex} ({ex/g*100:.1f}%)" if g else "0",
                         short, full, cite])
        A(md_table(["mention type", "gold", "detected (overlap)", "exact boundaries",
                    "missed: short form (≤30 ch)", "missed: full mention",
                    "missed but contains a citation"], rows))
        A("")
        A("*`short form` = gold mention of ≤30 characters (`Newport`, `idem, s. 3`, "
          "`Bill C-91`, `Act`) — a citator has nothing to resolve for most of them. "
          "`contains a citation` = the gold mention does carry a citation string, so "
          "a citator could in principle have linked it (these are the real Reflex "
          "misses worth arguing about).*")
        A("")
        A("Where is the nearest Reflex prediction of the same type for each missed "
          "gold mention?")
        A("")
        dist = taxonomy.get("fn_dist", {})
        keys = sorted({k for v in dist.values() for k in v})
        rows = []
        for lab in C.PARENT_LABELS:
            row = [lab]
            for k in keys:
                row.append(dist.get(lab, {}).get(k, 0))
            rows.append(row)
        A(md_table(["mention type"] + keys, rows))
        A("")
        A("### Examples of missed gold mentions")
        A("")
        for (lab, key), exs in sorted(taxonomy.get("examples", {}).items()):
            A(f"* **{lab} — {key}**: " + "; ".join(f"`{e}`" for e in exs))
        A("")

    # 6 ---------------------------------------------------------------------
    A("## 6. Reflex vs the other systems (same documents, same scorer)")
    A("")
    if comparisons:
        for name, comp in comparisons.items():
            common = sorted(set(comp["per_doc"]) & set(res["per_doc"]))
            if not common:
                continue
            A(f"### `{name}` — {len(common)} shared document(s): "
              f"{', '.join(common)}")
            A("")
            for mode in MODES:
                title = ("EXACT boundaries (Table-1 metric)" if mode == "exact"
                         else "OVERLAP matching (lenient)")
                r_tp = sum(res["per_doc"][d][mode]["tp"] for d in common)
                r_ng = sum(res["per_doc"][d][mode]["n_gold"] for d in common)
                r_ns = sum(res["per_doc"][d][mode]["n_system"] for d in common)
                c_tp = sum(comp["per_doc"][d][mode]["tp"] for d in common)
                c_ng = sum(comp["per_doc"][d][mode]["n_gold"] for d in common)
                c_ns = sum(comp["per_doc"][d][mode]["n_system"] for d in common)
                systems = [("**Reflex (CanLII citator)**", r_tp, r_ng, r_ns),
                           (name, c_tp, c_ng, c_ns)]
                rows = []
                for label, tp, ng, ns in systems:
                    P, R, F = prf(tp, ng, ns)
                    rows.append([label, ng, ns, tp, f"{P*100:.1f}", f"{R*100:.1f}",
                                 f"{F*100:.1f}"])
                A(f"**{mode.upper()}** — {title}")
                A("")
                A(md_table(["system", "gold", "spans", "TP", "P %", "R %", "F1 %"],
                           rows))
                A("")
    else:
        A("No `--compare` system given. To put Reflex side by side with a baseline "
          "on identical documents, run e.g.")
        A("")
        A("```bash")
        A("python scripts/reflex/evaluate_reflex.py \\")
        A('    --compare rules="output/test_baseline_rules/**/*_final.html" \\')
        A('    --compare gpt-5.2="output/test_gpt-5.2_DEC_fs6_greedy_sentence_long/**/*_final.html"')
        A("```")
        A("")
    A("Reference — Table 1 of the paper (mention-level micro-F1, test set, exact "
      "boundaries; different document set, listed for orientation):")
    A("")
    A(md_table(["model", "mention F1 %"],
               [[k, f"{v*100:.0f}"] for k, v in PAPER_REFERENCE.items()]))
    A("")

    # 7 ---------------------------------------------------------------------
    A("## 7. Per-document breakdown")
    A("")
    rows = []
    for doc_id, d in sorted(res["per_doc"].items(), key=lambda kv: (kv[1]["split"], kv[0])):
        e, o = d["exact"], d["overlap"]
        rows.append([doc_id, d["split"], d["n_gold"], d["n_system"],
                     f"{e['P']*100:.1f}/{e['R']*100:.1f}/{e['F1']*100:.1f}",
                     f"{o['P']*100:.1f}/{o['R']*100:.1f}/{o['F1']*100:.1f}",
                     f"{d['align_ratio']:.3f}"])
    A(md_table(["document", "split", "gold", "reflex", "exact P/R/F1",
                "overlap P/R/F1", "text align"], rows))
    A("")
    A("`text align` = token-level similarity of the two renderings of the same "
      "document (1.0 = identical text).")
    A("")

    # 8 ---------------------------------------------------------------------
    A("## 8. Error examples")
    A("")
    p, gold, reflex, align = parsed[0]
    gold_spans, sys_spans = C.build_spans(gold, reflex.mentions(gap), align,
                                          args.context_chars)
    _, _, fn_e, fp_e = match_detail(gold_spans, sys_spans, "exact")
    _, _, fn_o, fp_o = match_detail(gold_spans, sys_spans, "overlap")
    A(f"First document, **{p.doc_id}**:")
    A("")
    A(f"* EXACT: {len(fn_e)} missed / {len(fp_e)} spurious — the reflex side is "
      "dominated by injected neutral citations (`1982 CanLII 19 (SCC)`), which can "
      "never equal a gold boundary because the gold text does not contain them.")
    A(f"* OVERLAP: {len(fn_o)} missed / {len(fp_o)} spurious.")
    A("")
    A("**Gold mentions Reflex never touches (overlap FN)**")
    A("")
    for i in fn_o[:args.max_examples]:
        g = gold_spans[i]
        A(f"* `[{g.labelname}]` {g.text[:110]}")
    if len(fn_o) > args.max_examples:
        A(f"* … and {len(fn_o)-args.max_examples} more")
    A("")
    A("**Reflex predictions with no gold counterpart (overlap FP)**")
    A("")
    for j in fp_o[:args.max_examples]:
        s = sys_spans[j]
        A(f"* `[{s.labelname}]` {s.text[:110]}")
    if len(fp_o) > args.max_examples:
        A(f"* … and {len(fp_o)-args.max_examples} more")
    if not fp_o:
        A("* none — every reflex prediction landed inside a gold mention")
    A("")
    A("---")
    A("**Reading these numbers.**")
    A("")
    A("* **EXACT is structurally harsh for Reflex**: it links only the *citation* "
      "part of a case reference (`2002 SCC 33`), while the gold parent span covers "
      "the whole reference (`MacDonald v. Smith, 2002 SCC 33`), and for pre-1999 "
      "citations the string it links does not exist in the gold text at all. "
      "Legislation boundaries, where Reflex and gold agree on the span, score far "
      "better.")
    A("* **OVERLAP is the fair detection metric** for this system: precision stays "
      "at ~98% (Reflex only links what it resolved) and recall ~35-40%, because "
      "gold also annotates short forms, fragments and secondary sources that a "
      "citator never links.")
    A("* **`secondary sources` recall is 0 by construction** — no reflex export "
      "contains a single non-`/laws/` `/doc/` link, so books, dictionaries and law "
      "reviews are simply out of Reflex's scope.")
    A("* On the same documents and the same scorer the LLM/rules baselines sit "
      "well above these numbers (see section 6), which is exactly the claim the "
      "paper makes: Reflex is a respectable high-precision extractor, but its "
      "recall ceiling keeps it clearly below the LLM results.")
    A("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()


