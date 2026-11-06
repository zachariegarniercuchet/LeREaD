#!/usr/bin/env python
"""
Extraction error analysis (Level 1 spans) per model.

Produces, for each extraction run folder:
  1. Error by mention type (parent: decision / legislation / secondary sources)
     and by component type (child: title / citation / fragment / source / authors),
     both overall (child label) and as (parent -> child) pairs.
     Miss rate = FN / gold = 1 - Recall  ("X% of decisions are missed").
  2. Boundary-vs-type analysis (parent level only): when a system parent span
     covers the exact same location (same text + same context) as a gold parent
     span, did the model also pick the right mention type?  Confusion matrix
     gold_type x system_type over boundary-matched pairs + type accuracy.
  3. Mention-pattern analysis (parent level only): each gold parent mention is
     described by its structural pattern, i.e. the ordered tuple of its child
     sublabels (e.g. legislation -> fragment -> title).  Miss rate per pattern.

Span matching reuses src/evaluation/evaluation_l1_util.py exactly
(same label + same normalised text + same normalised context, greedy).

Usage (from repo root):
    python Evaluation_analysis/analyze_extraction.py
    python Evaluation_analysis/analyze_extraction.py --gold-dir private/dataset/Annotated --out Evaluation_analysis/results
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ann_extractor import extract_parent_level_annotations  # noqa: E402
from src.htmlLabel import ReferenceMention  # noqa: E402
from src.evaluation.evaluation_l1_util import (  # noqa: E402
    CONTEXT_CHARS,
    PARENT_LABELS,
    Span,
    _prf,
)
from bs4 import BeautifulSoup  # noqa: E402

RUNS_INCOMING = {
    "gemma4": "output/incoming_gemma4_DEC_fs6_greedy_sentence_long",
    "gpt-5.2": "output/incoming_gpt-5.2_DEC_fs6_greedy_sentence_long",
    "muse": "output/incoming_muse_DEC_fs6_greedy_sentence_long",
    "qwen38-27b": "output/incoming_qwen38-27b_DEC_fs6_greedy_sentence_long",
    "saul-54b": "output/incoming_saul-54b_DEC_fs6_greedy_sentence_long",
}

RUNS_AAFINAL = {
    "gemma4": "output/AA_final/gemma4",
    "gpt5.2": "output/AA_final/gpt5.2",
    "muse": "output/AA_final/muse",
    "qwen38": "output/AA_final/qwen38",
    "saullm": "output/AA_final/saullm",
}

RUNS_BASELINE = {
    "baseline-rules": "output/test_baseline_rules",
}

SUITES = {
    "incoming": (RUNS_INCOMING, "private/dataset/Annotated",
                 "Evaluation_analysis/results"),
    "aafinal": (RUNS_AAFINAL, "data/annotated/test",
                "Evaluation_analysis/results_aafinal"),
    "baseline": (RUNS_BASELINE, "data/annotated/test",
                 "Evaluation_analysis/results_baseline"),
}

PARENT_ORDER = ["decision", "legislation", "secondary sources"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_pairs(run_dir: Path, gold_dir: Path):
    """Pair each <doc>/*final.html with gold <doc>.html (exact stem match)."""
    pairs = []
    missing = []
    for sub in sorted(d for d in run_dir.iterdir() if d.is_dir()):
        finals = list(sub.glob("*final.html"))
        if len(finals) != 1:
            missing.append((sub.name, f"{len(finals)} *final.html files"))
            continue
        gold = gold_dir / f"{sub.name}.html"
        if not gold.exists():
            missing.append((sub.name, "no gold file"))
            continue
        pairs.append((sub.name, str(gold), str(finals[0])))
    return pairs, missing


def read_html(path: str) -> str:
    p = Path(path)
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return p.read_text(encoding="latin-1")


def extract_spans_fast(html: str, context_chars: int = CONTEXT_CHARS):
    """Same spans/matching semantics as evaluation_l1_util.extract_spans.

    Optimised: the normalised plain text is built once and each unique span
    text is located in a single str.find scan (the reference implementation
    re-normalises the whole document and re-scans it for every span).
    Occurrence handling replicates _get_context exactly: the counter is per
    (label, text) but occurrences are counted over the whole plain text.
    """
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("body")
    if body is None:
        return []
    plain = " ".join(body.get_text().split())
    # Single find_all keeps document order (the reference implementation
    # concatenates manual + auto lists, which breaks mention alignment when
    # a file mixes both; TP counts are unaffected since matching is per-key).
    tags = body.find_all(["manual_label", "auto_label"])

    items = []
    for tag in tags:
        labelname = (tag.get("labelname") or "").strip().lower()
        if not labelname:
            continue
        nt = " ".join(tag.get_text().split())
        if not nt:
            continue
        items.append((labelname, nt, tag.get_text(), dict(tag.attrs)))

    # all occurrence positions per unique text, in one scan each
    positions = {}
    for key in {nt for _, nt, _, _ in items}:
        lst = []
        start = 0
        while True:
            idx = plain.find(key, start)
            if idx == -1:
                break
            lst.append(idx)
            start = idx + len(key)
        positions[key] = lst

    spans = []
    seen: dict = {}
    plen = len(plain)
    for labelname, nt, raw_text, attrs in items:
        key = (labelname, nt)
        n = seen.get(key, 0)
        seen[key] = n + 1
        lst = positions[nt]
        start = lst[n] if n < len(lst) else -1  # same -1 fallback as _get_context
        end = start + len(nt)
        ctx = plain[max(0, start - context_chars):min(plen, end + context_chars)]
        sp = Span(raw_text, labelname, ctx, attributes=attrs,
                  start_pos=start)
        # Normalised plain-text offsets for overlap matching. Unlocatable
        # spans (-1 fallback) get an empty interval so they never overlap.
        # Context computation above is intentionally unchanged (reference
        # behaviour preserved bit-for-bit).
        sp.end_pos = end if start >= 0 else -1
        spans.append(sp)
    return spans


def greedy_match_fast(gold, system, key):
    """Greedy 1-1 matching identical to match_spans(), in O(G+S).

    match_spans() pairs each gold span (doc order) with the first still-unused
    system span (doc order) whose key is equal. Grouping system indices by key
    and popping from the left yields exactly that pairing.
    """
    sys_by_key = defaultdict(deque)
    for j, s in enumerate(system):
        sys_by_key[key(s)].append(j)
    pairs = []
    unmatched_g = []
    for i, g in enumerate(gold):
        dq = sys_by_key.get(key(g))
        if dq:
            pairs.append((i, dq.popleft()))
        else:
            unmatched_g.append(i)
    used = set(j for _, j in pairs)
    unmatched_s = [j for j in range(len(system)) if j not in used]
    return pairs, unmatched_g, unmatched_s


SPAN_KEY = lambda s: (s.labelname, s.text, s.context_text)  # noqa: E731
BOUNDARY_KEY = lambda s: (s.text, s.context_text)  # noqa: E731


def parent_attr_of(span: Span) -> str:
    return (span.attributes.get("parent") or "").strip().lower()


def norm_text(t: str) -> str:
    return " ".join(t.split())


def pattern_str(sublabels) -> str:
    subs = [s.strip().lower() for s in sublabels if s and s.strip()]
    return " -> ".join(subs) if subs else "(no child)"


# ---------------------------------------------------------------------------
# Per-model analysis
# ---------------------------------------------------------------------------

def analyze_model(model: str, run_dir: Path, gold_dir: Path, only_docs=None):
    pairs, missing = find_pairs(run_dir, gold_dir)
    if only_docs is not None:
        pairs = [p for p in pairs if p[0] in only_docs]

    # Analysis 1 accumulators: bucket key -> [tp, n_gold, n_system]
    # bucket = ("parent", mention_label, "-") or ("child", parent_attr, child_label)
    buckets = defaultdict(lambda: [0, 0, 0])
    label_acc = defaultdict(lambda: [0, 0, 0])  # label -> [tp, gold, sys] (cross-check)

    # Analysis 2: boundary matching on parent spans (label ignored)
    n_gold_parent = 0
    n_boundary_matched = 0
    n_correct_type = 0
    confusion = Counter()  # (gold_label, sys_label) over boundary-matched pairs
    missed_boundary = Counter()  # gold_label -> no boundary match at all
    spurious_boundary = Counter()  # sys_label -> no boundary match at all

    # Analysis 3: pattern stats: (parent_label, pattern) -> [n_gold, n_matched]
    patterns = defaultdict(lambda: [0, 0])
    align_mismatch = 0
    align_total = 0

    n_docs = 0
    for di, (doc, gold_path, sys_path) in enumerate(pairs, 1):
        if di % 10 == 0 or di == len(pairs):
            print(f"  [{di}/{len(pairs)}] {doc}", flush=True)
        gold_html = read_html(gold_path)
        sys_html = read_html(sys_path)
        gold_spans = extract_spans_fast(gold_html)
        sys_spans = extract_spans_fast(sys_html)

        # ---- Analysis 1: full greedy match, bucketed ----
        mpairs, un_g, un_s = greedy_match_fast(gold_spans, sys_spans, SPAN_KEY)
        matched_g = set(i for i, _ in mpairs)
        for i, j in mpairs:
            g = gold_spans[i]
            s = sys_spans[j]
            if g.labelname in PARENT_LABELS:
                b = buckets[("parent", g.labelname, "-")]
            else:
                b = buckets[("child", parent_attr_of(g) or "?", g.labelname)]
            b[0] += 1  # TP
            b[1] += 1  # gold side
            # system side goes to the system's own bucket (parent attr may differ)
            if s.labelname in PARENT_LABELS:
                buckets[("parent", s.labelname, "-")][2] += 1
            else:
                buckets[("child", parent_attr_of(s) or "?", s.labelname)][2] += 1
        for i in un_g:
            g = gold_spans[i]
            if g.labelname in PARENT_LABELS:
                buckets[("parent", g.labelname, "-")][1] += 1
            else:
                buckets[("child", parent_attr_of(g) or "?", g.labelname)][1] += 1
        for j in un_s:
            s = sys_spans[j]
            if s.labelname in PARENT_LABELS:
                buckets[("parent", s.labelname, "-")][2] += 1
            else:
                buckets[("child", parent_attr_of(s) or "?", s.labelname)][2] += 1

        # per-label accumulation derived from the same matching (== compute_metrics)
        for i, j in mpairs:
            a = label_acc[gold_spans[i].labelname]
            a[0] += 1  # TP
            a[1] += 1  # gold side
            a[2] += 1  # system side (matched pair shares the label by construction)
        for i in un_g:
            label_acc[gold_spans[i].labelname][1] += 1
        for j in un_s:
            label_acc[sys_spans[j].labelname][2] += 1

        # ---- Analysis 2: boundary match ignoring label (parents only) ----
        g_par_idx = [i for i, s in enumerate(gold_spans) if s.labelname in PARENT_LABELS]
        s_par_idx = [j for j, s in enumerate(sys_spans) if s.labelname in PARENT_LABELS]
        g_par = [gold_spans[i] for i in g_par_idx]
        s_par = [sys_spans[j] for j in s_par_idx]
        n_gold_parent += len(g_par)
        bpairs, bun_g, bun_s = greedy_match_fast(g_par, s_par, BOUNDARY_KEY)
        for gi, sj in bpairs:
            g = g_par[gi]
            s = s_par[sj]
            n_boundary_matched += 1
            confusion[(g.labelname, s.labelname)] += 1
            if g.labelname == s.labelname:
                n_correct_type += 1
        for gi in bun_g:
            missed_boundary[g_par[gi].labelname] += 1
        for sj in bun_s:
            spurious_boundary[s_par[sj].labelname] += 1

        # ---- Analysis 3: gold mention patterns vs match status ----
        # Raw parent tags in document order (same selector as
        # extract_parent_level_annotations). Text is taken from the RAW tag
        # with the same whitespace normalisation as the span extractor;
        # ReferenceMention is used only for the child-sublabel pattern, since
        # its .text is lossy (HTML reserialisation drops inter-tag spaces).
        gsoup = BeautifulSoup(gold_html, "html.parser")
        raw_parents = gsoup.find_all(
            ["manual_label", "auto_label"], attrs={"parent": "", "docid": True}
        )
        spans_by_label = defaultdict(list)  # label -> list of gold-span idx (doc order)
        for idx, sp in enumerate(gold_spans):
            if sp.labelname in PARENT_LABELS:
                spans_by_label[sp.labelname].append(idx)
        use_count = Counter()
        n_empty_parents = 0
        for raw in raw_parents:
            lab = (raw.get("labelname") or "").strip().lower()
            if lab not in PARENT_LABELS:
                continue
            raw_text = " ".join(raw.get_text().split())
            if not raw_text:
                # the span evaluator skips empty tags too — unevaluable
                n_empty_parents += 1
                continue
            order = use_count[lab]
            use_count[lab] += 1
            align_total += 1
            try:
                pat = pattern_str(ReferenceMention(str(raw)).sublabels)
            except ValueError:
                pat = "(unparseable)"
            patterns[(lab, pat)][0] += 1
            lst = spans_by_label.get(lab, [])
            matched = False
            if order < len(lst):
                if raw_text != gold_spans[lst[order]].text:
                    align_mismatch += 1
                else:
                    matched = lst[order] in matched_g
            if matched:
                patterns[(lab, pat)][1] += 1

        n_docs += 1

    return {
        "model": model,
        "n_docs": n_docs,
        "n_missing": len(missing),
        "missing": missing,
        "buckets": dict(buckets),
        "label_acc": dict(label_acc),
        "boundary": {
            "n_gold_parent": n_gold_parent,
            "n_boundary_matched": n_boundary_matched,
            "n_correct_type": n_correct_type,
            "confusion": dict(confusion),
            "missed": dict(missed_boundary),
            "spurious": dict(spurious_boundary),
        },
        "patterns": dict(patterns),
        "align_mismatch": align_mismatch,
        "align_total": align_total,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def validate_model(res):
    """Sanity: TP can never exceed gold or system counts."""
    bad = []
    for lab, (tp, gold, sys) in res["label_acc"].items():
        if not (tp <= gold and tp <= sys):
            bad.append((lab, tp, gold, sys))
    if bad:
        raise ValueError(f"[{res['model']}] inconsistent counts (TP>gold/sys): {bad}")
    bt = sum(v[0] for v in res["buckets"].values())
    bg = sum(v[1] for v in res["buckets"].values())
    bs = sum(v[2] for v in res["buckets"].values())
    lt = sum(v[0] for v in res["label_acc"].values())
    lg = sum(v[1] for v in res["label_acc"].values())
    ls = sum(v[2] for v in res["label_acc"].values())
    if not (bt == lt and bg == lg and bs == ls):
        raise ValueError(f"[{res['model']}] bucket/label totals differ: "
                         f"buckets=({bt},{bg},{bs}) labels=({lt},{lg},{ls})")


def row_metrics(tp, gold, sys):
    p = tp / sys if sys else 0.0
    r = tp / gold if gold else 0.0
    miss = 1.0 - r if gold else 0.0
    fdr = 1.0 - p if sys else 0.0  # false discovery rate (spurious share)
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, miss, fdr, f1


def write_label_csv(res, out_path: Path):
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "level", "gold", "system", "TP", "FN", "FP",
                    "miss_rate_FN_over_gold", "false_discovery_FP_over_sys",
                    "precision", "recall", "F1"])
        for lab in sorted(res["label_acc"]):
            tp, gold, sys = res["label_acc"][lab]
            p, r, miss, fdr, f1 = row_metrics(tp, gold, sys)
            level = "parent(mention)" if lab in PARENT_LABELS else "child(component)"
            w.writerow([lab, level, gold, sys, tp, gold - tp, sys - tp,
                        f"{miss:.4f}", f"{fdr:.4f}", f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}"])


def write_component_csv(res, out_path: Path):
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["level", "mention_parent", "component", "gold", "system", "TP",
                    "FN", "FP", "miss_rate_FN_over_gold",
                    "false_discovery_FP_over_sys", "precision", "recall", "F1"])
        for (level, parent, comp) in sorted(res["buckets"]):
            tp, gold, sys = res["buckets"][(level, parent, comp)]
            p, r, miss, fdr, f1 = row_metrics(tp, gold, sys)
            w.writerow([level, parent, comp, gold, sys, tp, gold - tp, sys - tp,
                        f"{miss:.4f}", f"{fdr:.4f}", f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}"])


def write_confusion_csv(res, out_path: Path):
    b = res["boundary"]
    labels = PARENT_ORDER + sorted(
        {l for pair in b["confusion"] for l in pair} - set(PARENT_ORDER))
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gold \\ system"] + labels + ["NO_BOUNDARY_MATCH"])
        for g in labels:
            row = [g] + [b["confusion"].get((g, s), 0) for s in labels] + [b["missed"].get(g, 0)]
            w.writerow(row)
        w.writerow(["NO_BOUNDARY_MATCH"] + [b["spurious"].get(s, 0) for s in labels] + ["-"])


def write_pattern_csv(res, out_path: Path):
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mention", "pattern", "gold", "matched", "missed", "miss_rate"])
        for (lab, pat) in sorted(res["patterns"], key=lambda k: (-res["patterns"][k][0], k[0], k[1])):
            n, m = res["patterns"][(lab, pat)]
            w.writerow([lab, pat, n, m, n - m, f"{(n - m) / n:.4f}" if n else "0.0000"])


def summary_lines(res) -> list:
    L = []
    L.append(f"## {res['model']} — {res['n_docs']} docs evaluated ({res['n_missing']} skipped)")
    L.append("")
    L.append("### 1. Miss rate by mention type (parent) and component type (child)")
    L.append("")
    L.append("| label | level | gold | sys | TP | miss rate (FN/gold) | FDR (FP/sys) | R | F1 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for lab in sorted(res["label_acc"]):
        tp, gold, sys = res["label_acc"][lab]
        p, r, miss, fdr, f1 = row_metrics(tp, gold, sys)
        level = "parent" if lab in PARENT_LABELS else "child"
        L.append(f"| {lab} | {level} | {gold} | {sys} | {tp} | {miss*100:.1f}% | {fdr*100:.1f}% | {r*100:.1f}% | {f1*100:.1f}% |")
    L.append("")
    L.append("### 2. Boundary match -> correct mention type? (parents only)")
    b = res["boundary"]
    n, matched, correct = b["n_gold_parent"], b["n_boundary_matched"], b["n_correct_type"]
    L.append(f"- Gold parent mentions: {n}; with a system span on the exact same boundary: "
             f"{matched} ({matched/n*100:.1f}% of gold)" if n else "- no gold parents")
    if matched:
        L.append(f"- Of those, correct mention type: {correct} ({correct/matched*100:.2f}%) — "
                 f"wrong type: {matched-correct} ({(matched-correct)/matched*100:.2f}%)")
    L.append("- Confusion (gold x system, boundary-matched pairs):")
    L.append("")
    labels = PARENT_ORDER
    L.append("| gold \\ sys | " + " | ".join(labels) + " | missed (no boundary) |")
    L.append("|---|---|---|---|---|")
    for g in labels:
        cells = [str(b["confusion"].get((g, s), 0)) for s in labels]
        L.append(f"| {g} | " + " | ".join(cells) + f" | {b['missed'].get(g, 0)} |")
    L.append(f"| spurious (no boundary) | " + " | ".join(str(b["spurious"].get(s, 0)) for s in labels) + " | - |")
    L.append("")
    L.append("### 3. Miss rate by mention pattern (parent level)")
    L.append(f"_Alignment mismatches mention->span: {res['align_mismatch']}/{res['align_total']}_")
    L.append("")
    L.append("| mention | pattern | gold | miss rate |")
    L.append("|---|---|---|---|")
    pats = sorted(res["patterns"].items(), key=lambda kv: (-kv[1][0], kv[0][0], kv[0][1]))[:20]
    for (lab, pat), (n, m) in pats:
        L.append(f"| {lab} | {pat} | {n} | {(n-m)/n*100:.1f}% |")
    L.append("")
    return L


def write_comparison(all_res, out_dir: Path):
    # Miss-rate comparison across models, per label
    labels = sorted({l for r in all_res for l in r["label_acc"]})
    with (out_dir / "comparison_miss_rate_by_label.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label"] + [r["model"] for r in all_res])
        for lab in labels:
            row = [lab]
            for r in all_res:
                tp, gold, sys = r["label_acc"].get(lab, [0, 0, 0])
                _, _, miss, _, _ = row_metrics(tp, gold, sys)
                row.append(f"{miss:.4f}")
            w.writerow(row)
    # Boundary/type accuracy comparison
    with (out_dir / "comparison_boundary_type.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "docs", "gold_parents", "boundary_matched",
                    "boundary_recall", "correct_type", "type_accuracy_on_matched",
                    "wrong_type"])
        for r in all_res:
            b = r["boundary"]
            n, m, c = b["n_gold_parent"], b["n_boundary_matched"], b["n_correct_type"]
            w.writerow([r["model"], r["n_docs"], n, m,
                        f"{m/n:.4f}" if n else "0",
                        c, f"{c/m:.4f}" if m else "0", m - c])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=sorted(SUITES), default="incoming",
                    help="Run suite preset (folders + gold + output dir)")
    ap.add_argument("--gold-dir", default=None,
                    help="Override gold HTML folder (default from suite)")
    ap.add_argument("--out", default=None,
                    help="Override output folder (default from suite)")
    args = ap.parse_args()

    runs, suite_gold, suite_out = SUITES[args.suite]
    gold_dir = (PROJECT_ROOT / (args.gold_dir or suite_gold))
    out_dir = (PROJECT_ROOT / (args.out or suite_out))
    out_dir.mkdir(parents=True, exist_ok=True)

    all_res = []
    run_dirs = {}
    report = ["# Extraction error analysis", "",
              f"Suite: `{args.suite}`; gold: `{gold_dir.relative_to(PROJECT_ROOT)}`; "
              "matching = same label + same text + same context (eval L1).",
              ""]
    for model, run in runs.items():
        run_dir = PROJECT_ROOT / run
        print(f"=== {model} ({run}) ===", flush=True)
        if not run_dir.exists():
            print(f"  SKIP: run folder not found: {run_dir}")
            continue
        run_dirs[model] = run_dir
    # Common-doc intersection (gemma4 lacks 2019QCCA1492): needed so that
    # cross-model comparisons are apples-to-apples.
    doc_sets = {m: {p[0] for p in find_pairs(d, gold_dir)[0]}
                for m, d in run_dirs.items()}
    common_docs = set.intersection(*doc_sets.values()) if doc_sets else set()
    print(f"Common docs across {len(doc_sets)} runs: {len(common_docs)}", flush=True)

    for model, run_dir in run_dirs.items():
        res = analyze_model(model, run_dir, gold_dir)
        validate_model(res)
        print(f"  docs={res['n_docs']} skipped={res['n_missing']} "
              f"align_mismatch={res['align_mismatch']}/{res['align_total']}")
        write_label_csv(res, out_dir / f"{model}_label_errors.csv")
        write_component_csv(res, out_dir / f"{model}_component_errors.csv")
        write_confusion_csv(res, out_dir / f"{model}_boundary_confusion.csv")
        write_pattern_csv(res, out_dir / f"{model}_pattern_errors.csv")
        all_res.append(res)
        report.extend(summary_lines(res))

    if all_res:
        write_comparison(all_res, out_dir)
        # Common-docs pass: same metrics on the doc intersection only.
        common_res = []
        for model, run_dir in run_dirs.items():
            print(f"=== {model} (common {len(common_docs)} docs) ===", flush=True)
            res = analyze_model(f"{model}", run_dir, gold_dir, only_docs=common_docs)
            validate_model(res)
            common_res.append(res)
            write_pattern_csv(res, out_dir / f"{model}_common{len(common_docs)}_pattern_errors.csv")
        with (out_dir / f"comparison_common{len(common_docs)}_miss_rate_by_label.csv").open(
                "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["label"] + [r["model"] for r in common_res])
            labels = sorted({l for r in common_res for l in r["label_acc"]})
            for lab in labels:
                row = [lab]
                for r in common_res:
                    tp, gold, sys = r["label_acc"].get(lab, [0, 0, 0])
                    _, _, miss, _, _ = row_metrics(tp, gold, sys)
                    row.append(f"{miss:.4f}")
                w.writerow(row)
        with (out_dir / f"comparison_common{len(common_docs)}_boundary_type.csv").open(
                "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "docs", "gold_parents", "boundary_matched",
                        "boundary_recall", "correct_type",
                        "type_accuracy_on_matched", "wrong_type"])
            for r in common_res:
                b = r["boundary"]
                n, m, c = b["n_gold_parent"], b["n_boundary_matched"], b["n_correct_type"]
                w.writerow([r["model"], r["n_docs"], n, m,
                            f"{m/n:.4f}" if n else "0",
                            c, f"{c/m:.4f}" if m else "0", m - c])
        # Pattern comparison on common docs: union of each model's top-12
        # patterns by gold frequency.
        top_pats = set()
        for r in common_res:
            top = sorted(r["patterns"].items(), key=lambda kv: -kv[1][0])[:12]
            top_pats.update(k for k, _ in top)
        with (out_dir / f"comparison_common{len(common_docs)}_pattern_miss_rate.csv").open(
                "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["mention", "pattern", "gold_common"] +
                       [r["model"] for r in common_res])
            for (lab, pat) in sorted(top_pats,
                                     key=lambda k: (-sum(r["patterns"].get(k, [0, 0])[0]
                                                         for r in common_res), k)):
                gold_n = sum(r["patterns"].get((lab, pat), [0, 0])[0]
                             for r in common_res) // max(len(common_res), 1)
                row = [lab, pat, gold_n]
                for r in common_res:
                    n, m = r["patterns"].get((lab, pat), [0, 0])
                    row.append(f"{(n-m)/n:.4f}" if n else "-")
                w.writerow(row)
        report.append("## Cross-model comparison")
        report.append("")
        report.append("See `comparison_miss_rate_by_label.csv` and `comparison_boundary_type.csv`.")
        report.append("")
        for r in all_res:
            b = r["boundary"]
            n, m, c = b["n_gold_parent"], b["n_boundary_matched"], b["n_correct_type"]
            report.append(f"- {r['model']}: boundary recall {m/n*100:.1f}% (N={n}), "
                          f"type accuracy on matched {c/m*100:.2f}%"
                          if n and m else f"- {r['model']}: no data")
    (out_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
