#!/usr/bin/env python
"""
Strict end-to-end pattern evaluation.

A pattern instance = one parent mention + ALL of its nested component spans
(e.g. decision + fragment + title = 3 spans, pattern "decision -> fragment
-> title", grouped i.e. consecutive duplicates collapsed).

A gold pattern instance (real uri only; uri="none"/absent ignored) counts as
correct (TP) iff ALL of the following hold:
    1. the parent span is Level-1 matched,
    2. EVERY nested child span is Level-1 matched,
    3. the system uri on the matched parent agrees with the gold uri.
If any span is unmatched -> end-to-end error. If all match but uri is wrong
-> end-to-end error.

Level-1 matching (--match):
    exact    same label + same normalised text + same context (the official
             evaluator's definition; mention boundaries must coincide exactly)
    overlap  same label + intersecting plain-text character offsets (boundary
             errors forgiven as long as both spans cover the same passage)

Precision side: a system parent mention (real uri, grouped pattern P) counts
as correct iff its parent span is matched AND all its nested children are
matched AND its uri agrees with the matched gold parent's uri.

    Recall_P    = TP_gold(P) / gold_instances(P)
    Precision_P = TP_sys(P)  / sys_instances(P)
    F1_P        = harmonic mean

Ranking is worst -> best by F1, filtered to patterns with >= 100 gold
instances. Aggregation micro-averages counts over the working models on
their common docs (saullm excluded from the aggregate).

Usage (from repo root):
    python Evaluation_analysis/analyze_e2e_strict.py [--match exact|overlap]
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Evaluation_analysis"))

from analyze_e2e_patterns import (  # noqa: E402
    RUNS_E2E,
    find_pairs,
    get_uri,
    is_real_uri,
    uri_agree,
)
from analyze_extraction import (  # noqa: E402
    PARENT_LABELS,
    SPAN_KEY,
    extract_spans_fast,
    greedy_match_fast,
    read_html,
)
from bs4 import BeautifulSoup  # noqa: E402

MIN_GOLD = 100
AGG_MODELS = ("gemma4", "gpt5.2", "muse", "qwen38")


def group_pattern(sublabels) -> str:
    out = []
    for s in sublabels:
        s = (s or "").strip().lower()
        if not s:
            continue
        if not out or out[-1] != s:
            out.append(s)
    return " -> ".join(out) if out else "(no child)"


def overlaps(a_start, a_end, b_start, b_end) -> bool:
    """True iff two plain-text intervals share at least one character."""
    if a_start < 0 or b_start < 0:
        return False  # unlocatable spans never overlap
    return a_start < b_end and b_start < a_end


def greedy_match_overlap(gold, system):
    """Greedy 1-1 matching on (same label + intersecting offsets).

    Mirrors match_spans() traversal: each gold span (doc order) takes the
    first still-unused system span (doc order) with the same label whose
    character interval intersects its own. Returns
    (pairs, unmatched_gold, unmatched_sys) like greedy_match_fast.
    """
    sys_by_label = defaultdict(list)
    for j, s in enumerate(system):
        sys_by_label[s.labelname].append(j)
    used = [False] * len(system)
    pairs = []
    unmatched_g = []
    for i, g in enumerate(gold):
        hit = None
        for j in sys_by_label.get(g.labelname, []):
            if used[j]:
                continue
            s = system[j]
            if overlaps(g.start_pos, g.end_pos, s.start_pos, s.end_pos):
                hit = j
                break
        if hit is None:
            unmatched_g.append(i)
        else:
            used[hit] = True
            pairs.append((i, hit))
    unmatched_s = [j for j, u in enumerate(used) if not u]
    return pairs, unmatched_g, unmatched_s


def index_document(html: str, spans):
    """Align raw tags to spans 1:1.

    extract_spans_fast iterates body.find_all(["manual_label","auto_label"])
    in document order, skipping tags with empty labelname/empty text. Replay
    the exact same iteration here so raw_tags[i] <-> spans[i].
    Returns (parents, children_of) where parents = list of
    {"tag", "span_idx", "label"} for top-level parent mentions (no ancestor
    label tag) and children_of maps parent order -> list of child span idxs.
    """
    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("body")
    if body is None:
        return [], {}
    tags = body.find_all(["manual_label", "auto_label"])
    raw_tags = []
    for tag in tags:
        labelname = (tag.get("labelname") or "").strip().lower()
        if not labelname:
            continue
        if not " ".join(tag.get_text().split()):
            continue
        raw_tags.append(tag)
    assert len(raw_tags) == len(spans), (len(raw_tags), len(spans))

    # span index of each tag, then children per top-level parent
    tag_span_idx = {id(t): i for i, t in enumerate(raw_tags)}
    parents = []
    for tag in raw_tags:
        lab = (tag.get("labelname") or "").strip().lower()
        if lab not in PARENT_LABELS:
            continue
        if tag.find_parent(["manual_label", "auto_label"]) is not None:
            continue
        child_idx = []
        for d in tag.find_all(["manual_label", "auto_label"]):
            if id(d) in tag_span_idx:
                child_idx.append(tag_span_idx[id(d)])
        pat = group_pattern([(d.get("labelname") or "") for d in
                             tag.find_all(["manual_label", "auto_label"])])
        parents.append({"span_idx": tag_span_idx[id(tag)], "label": lab,
                        "pattern": pat, "children": child_idx})
    return parents, None


def analyze_model(model: str, run_dir: Path, gold_dir: Path, only_docs=None,
                  match: str = "exact"):
    pairs, missing = find_pairs(run_dir, gold_dir)
    if only_docs is not None:
        pairs = [p for p in pairs if p[0] in only_docs]

    gold_stats = defaultdict(lambda: {"n": 0, "tp": 0})  # (lab,pat) -> gold/TP
    sys_stats = defaultdict(lambda: {"n": 0, "tp": 0})   # (lab,pat) -> sys/TP
    n_docs = 0
    for di, (doc, gold_path, sys_path) in enumerate(pairs, 1):
        if di % 10 == 0 or di == len(pairs):
            print(f"  [{di}/{len(pairs)}] {doc}", flush=True)
        gold_html = read_html(gold_path)
        sys_html = read_html(sys_path)
        gold_spans = extract_spans_fast(gold_html)
        sys_spans = extract_spans_fast(sys_html)
        if match == "overlap":
            mpairs, _, _ = greedy_match_overlap(gold_spans, sys_spans)
        else:
            mpairs, _, _ = greedy_match_fast(gold_spans, sys_spans, SPAN_KEY)
        sys_of_gold = {i: j for i, j in mpairs}
        gold_of_sys = {j: i for i, j in mpairs}
        matched_g = set(sys_of_gold)
        matched_s = set(gold_of_sys)

        gold_parents, _ = index_document(gold_html, gold_spans)
        sys_parents, _ = index_document(sys_html, sys_spans)

        for m in gold_parents:
            u_g = get_uri(gold_spans[m["span_idx"]])
            if not is_real_uri(u_g):
                continue
            key = (m["label"], m["pattern"])
            gold_stats[key]["n"] += 1
            gi = m["span_idx"]
            if gi not in sys_of_gold:
                continue
            if any(c not in matched_g for c in m["children"]):
                continue
            u_s = get_uri(sys_spans[sys_of_gold[gi]])
            if uri_agree(u_g, u_s):
                gold_stats[key]["tp"] += 1

        for m in sys_parents:
            u_s = get_uri(sys_spans[m["span_idx"]])
            if not is_real_uri(u_s):
                continue
            key = (m["label"], m["pattern"])
            sys_stats[key]["n"] += 1
            sj = m["span_idx"]
            if sj not in gold_of_sys:
                continue
            if any(c not in matched_s for c in m["children"]):
                continue
            u_g = get_uri(gold_spans[gold_of_sys[sj]])
            if uri_agree(u_g, u_s):
                sys_stats[key]["tp"] += 1

        n_docs += 1
    return {"model": model, "n_docs": n_docs, "missing": missing,
            "gold": dict(gold_stats), "sys": dict(sys_stats)}


def prf(tp_g, n_g, tp_s, n_s):
    r = tp_g / n_g if n_g else 0.0
    p = tp_s / n_s if n_s else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def write_model_csv(res, path: Path):
    keys = sorted(set(res["gold"]) | set(res["sys"]),
                  key=lambda k: (-res["gold"].get(k, {"n": 0})["n"], k))
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mention", "pattern_grouped", "n_gold", "TP_gold",
                    "n_sys", "TP_sys", "recall", "precision", "F1"])
        for lab, pattern in keys:
            g = res["gold"].get((lab, pattern), {"n": 0, "tp": 0})
            s = res["sys"].get((lab, pattern), {"n": 0, "tp": 0})
            p, r, f1 = prf(g["tp"], g["n"], s["tp"], s["n"])
            w.writerow([lab, pattern, g["n"], g["tp"], s["n"], s["tp"],
                        f"{r:.4f}", f"{p:.4f}", f"{f1:.4f}"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-dir", default="data/annotated/test")
    ap.add_argument("--out", default=None,
                    help="Output folder (default: results_e2e_strict[_overlap])")
    ap.add_argument("--min-gold", type=int, default=MIN_GOLD)
    ap.add_argument("--match", choices=("exact", "overlap"), default="exact",
                    help="Level-1 span matching: exact boundaries or overlap")
    args = ap.parse_args()

    match = args.match
    infix = "" if match == "exact" else "_overlap"
    gold_dir = PROJECT_ROOT / args.gold_dir
    out_dir = PROJECT_ROOT / (args.out or f"Evaluation_analysis/results_e2e_strict{infix}")
    out_dir.mkdir(parents=True, exist_ok=True)
    match_desc = ("same label + same text + same context (exact boundaries)"
                  if match == "exact" else
                  "same label + intersecting character offsets (overlap)")

    run_dirs = {}
    for model, run in RUNS_E2E.items():
        d = PROJECT_ROOT / run
        if d.exists():
            run_dirs[model] = d
        else:
            print(f"SKIP {model}: not found", flush=True)
    doc_sets = {m: {p[0] for p in find_pairs(d, gold_dir)[0]}
                for m, d in run_dirs.items()}
    common_all = set.intersection(*doc_sets.values()) if doc_sets else set()
    agg_sets = [doc_sets[m] for m in AGG_MODELS if m in doc_sets]
    common_agg = set.intersection(*agg_sets) if agg_sets else set()
    print(f"Common docs (all runs): {len(common_all)}; "
          f"(aggregate models): {len(common_agg)}", flush=True)

    # per-model full-coverage tables
    all_res = {}
    for model, d in run_dirs.items():
        print(f"=== {model} [{match}] ===", flush=True)
        res = analyze_model(model, d, gold_dir, match=match)
        all_res[model] = res
        write_model_csv(res, out_dir / f"{model}_strict{infix}_pattern_prf.csv")

    # aggregate micro-averaged counts on common docs of working models
    common_res = {}
    for model in AGG_MODELS:
        if model not in run_dirs:
            continue
        print(f"=== {model} (common {len(common_agg)}) [{match}] ===",
              flush=True)
        common_res[model] = analyze_model(model, run_dirs[model], gold_dir,
                                          only_docs=common_agg, match=match)
    agg_g = Counter()
    agg_tp_g = Counter()
    agg_s = Counter()
    agg_tp_s = Counter()
    for res in common_res.values():
        for k, v in res["gold"].items():
            agg_g[k] += v["n"]
            agg_tp_g[k] += v["tp"]
        for k, v in res["sys"].items():
            agg_s[k] += v["n"]
            agg_tp_s[k] += v["tp"]

    rows = []
    for key, n_g in agg_g.items():
        if n_g < args.min_gold:
            continue
        p, r, f1 = prf(agg_tp_g[key], n_g, agg_tp_s[key], agg_s[key])
        rows.append((key[0], key[1], n_g, agg_tp_g[key], agg_s[key],
                     agg_tp_s[key], r, p, f1))
    rows.sort(key=lambda x: (x[8], -x[2]))  # worst F1 first

    with (out_dir / f"ranking_strict{infix}_pattern_f1.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rank", "mention", "pattern_grouped",
                    "n_gold_agg", "TP_gold_agg", "n_sys_agg", "TP_sys_agg",
                    "recall", "precision", "F1"])
        for i, (lab, pattern, n_g, tp_g, n_s, tp_s, r, p, f1) in enumerate(
                rows, 1):
            w.writerow([i, lab, pattern, n_g, tp_g, n_s, tp_s,
                        f"{r:.4f}", f"{p:.4f}", f"{f1:.4f}"])

    report = [f"# Strict end-to-end pattern ranking ({match} match, worst -> best F1)",
              "",
              f"Aggregate: {', '.join(AGG_MODELS)} on {len(common_agg)} "
              f"common docs; gold instances >= {args.min_gold}.",
              f"Level-1 matching: {match_desc}.",
              "TP = every span of the mention matched AND uri correct.",
              "",
              "| rank | mention | pattern | gold | TP | sys | TP_sys | R | P | F1 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, (lab, pattern, n_g, tp_g, n_s, tp_s, r, p, f1) in enumerate(
            rows, 1):
        report.append(f"| {i} | {lab} | {pattern} | {n_g} | {tp_g} | {n_s} | "
                      f"{tp_s} | {r*100:.1f}% | {p*100:.1f}% | {f1*100:.1f}% |")
    text = "\n".join(report)
    (out_dir / f"REPORT_STRICT{infix.upper()}.md").write_text(text,
                                                             encoding="utf-8")
    # Also save the report at Evaluation_analysis/ level, as requested.
    (PROJECT_ROOT / "Evaluation_analysis" /
     f"e2e_strict_{match}_report.md").write_text(text, encoding="utf-8")
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
