#!/usr/bin/env python
"""
End-to-end (extraction -> coref -> resolution) pattern failure analysis.

For every *full-pipeline* run folder (output_coref/1-final/*_pip), each GOLD
parent mention that needs a URI (real uri: present and != "none", exactly the
counted instances of src/evaluation/evaluate_resolution.py) gets one outcome:

    TP               span extracted AND system uri agrees with gold uri
    FN-extraction    gold span never extracted (no Level-1 match)
    FN-no-uri        span extracted but the pipeline produced no uri at all
    FN-wrong-uri     span extracted but the pipeline produced the WRONG uri

Gold mentions are grouped by (mention type, GROUPED structural pattern):
consecutive duplicate sublabels are collapsed, e.g.
decision -> title -> title  becomes  decision -> title
(the --group convention of src/precompute_pattern_dict.py).

Outputs per model + cross-model comparison, including the overall URI P/R/F1
(same definitions as evaluate_resolution.py) so pattern failures link
directly to the URI F1, with the decomposition

    end-to-end recall = extraction recall x resolver accuracy-on-matched

Usage (from repo root):
    python Evaluation_analysis/analyze_e2e_patterns.py
"""

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "Evaluation_analysis"))

from analyze_extraction import (  # noqa: E402
    PARENT_LABELS,
    SPAN_KEY,
    extract_spans_fast,
    greedy_match_fast,
    read_html,
)
from bs4 import BeautifulSoup  # noqa: E402
from src.htmlLabel import ReferenceMention  # noqa: E402

RUNS_E2E = {
    "gemma4": "output_coref/1-final/gemma4_pip",
    "gpt5.2": "output_coref/1-final/gpt5.2_pip",
    "muse": "output_coref/1-final/muse_pip",
    "qwen38": "output_coref/1-final/qwen38_pip",
    "saullm": "output_coref/1-final/saullm_pip",
}

NO_ANSWER = {"none"}


# ---------------------------------------------------------------------------
# URI helpers (same semantics as src/evaluation/evaluate_resolution.py)
# ---------------------------------------------------------------------------

def get_uri(span) -> str | None:
    v = span.attributes.get("uri")
    if v is None:
        return None
    v = v.strip()
    return v if v else None


def is_real_uri(v) -> bool:
    return v is not None and v.strip().lower() not in NO_ANSWER


def uri_agree(a, b) -> bool:
    if a is None or b is None:
        return False
    return a.lower().strip() == b.lower().strip()


def group_pattern(sublabels) -> str:
    """Collapse consecutive duplicate sublabels: title->title => title."""
    out = []
    for s in sublabels:
        s = (s or "").strip().lower()
        if not s:
            continue
        if not out or out[-1] != s:
            out.append(s)
    return " -> ".join(out) if out else "(no child)"


# ---------------------------------------------------------------------------
# Mention <-> span alignment (document order within label)
# ---------------------------------------------------------------------------

def align_parent_mentions(html: str, spans):
    """Return list of (span_idx, label, grouped_pattern, text).

    raw parent tags (parent="" + docid, document order) aligned to parent
    spans (document order) by order-within-label; texts must agree.
    Returns (aligned, n_empty, n_mismatch).
    """
    soup = BeautifulSoup(html, "html.parser")
    # Scope to <body>: the span evaluator ignores anything outside it, and
    # pipeline outputs occasionally contain stray tags elsewhere (e.g. head).
    body = soup.find("body")
    if body is None:
        return [], 0, 0
    raw_parents = body.find_all(
        ["manual_label", "auto_label"], attrs={"parent": "", "docid": True}
    )
    by_label = defaultdict(list)
    for idx, sp in enumerate(spans):
        if sp.labelname in PARENT_LABELS:
            by_label[sp.labelname].append(idx)
    aligned = []
    use = Counter()
    n_empty = 0
    n_mismatch = 0
    for raw in raw_parents:
        lab = (raw.get("labelname") or "").strip().lower()
        if lab not in PARENT_LABELS:
            continue
        text = " ".join(raw.get_text().split())
        if not text:
            n_empty += 1
            continue
        try:
            pat = group_pattern(ReferenceMention(str(raw)).sublabels)
        except ValueError:
            pat = "(unparseable)"
        order = use[lab]
        use[lab] += 1
        lst = by_label.get(lab, [])
        if order < len(lst) and text == spans[lst[order]].text:
            aligned.append((lst[order], lab, pat, text))
        else:
            n_mismatch += 1
    return aligned, n_empty, n_mismatch


def find_pairs(run_dir: Path, gold_dir: Path):
    pairs, missing = [], []
    for sub in sorted(d for d in run_dir.iterdir() if d.is_dir()):
        res = list(sub.glob("*_resolution.html"))
        if len(res) != 1:
            missing.append((sub.name, f"{len(res)} *_resolution.html"))
            continue
        gold = gold_dir / f"{sub.name}.html"
        if not gold.exists():
            missing.append((sub.name, "no gold file"))
            continue
        pairs.append((sub.name, str(gold), str(res[0])))
    return pairs, missing


# ---------------------------------------------------------------------------
# Per-model analysis
# ---------------------------------------------------------------------------

def analyze_model(model: str, run_dir: Path, gold_dir: Path, only_docs=None):
    pairs, missing = find_pairs(run_dir, gold_dir)
    if only_docs is not None:
        pairs = [p for p in pairs if p[0] in only_docs]

    # (mention, grouped pattern) -> counts
    pat = defaultdict(lambda: {"n": 0, "tp": 0, "fn_ext": 0,
                               "fn_nouri": 0, "fn_wrong": 0})
    # mention -> overall uri counts (mirrors evaluate_resolution)
    overall = defaultdict(lambda: {"tp": 0, "fn_ext": 0, "fn_nouri": 0,
                                   "fn_wrong": 0, "fp_spurious": 0,
                                   "excluded": 0})
    # system-side spurious attribution: (mention, grouped pattern) -> count
    spurious_pat = Counter()

    n_docs = 0
    align_mismatch = 0
    for di, (doc, gold_path, sys_path) in enumerate(pairs, 1):
        if di % 10 == 0 or di == len(pairs):
            print(f"  [{di}/{len(pairs)}] {doc}", flush=True)
        gold_html = read_html(gold_path)
        sys_html = read_html(sys_path)
        gold_spans = extract_spans_fast(gold_html)
        sys_spans = extract_spans_fast(sys_html)

        mpairs, un_g, un_s = greedy_match_fast(gold_spans, sys_spans, SPAN_KEY)
        sys_of_gold = {i: j for i, j in mpairs}
        matched_s = set(j for _, j in mpairs)

        gold_aligned, _, mm = align_parent_mentions(gold_html, gold_spans)
        align_mismatch += mm
        for gi, lab, pattern, _text in gold_aligned:
            u_g = get_uri(gold_spans[gi])
            if not is_real_uri(u_g):
                overall[lab]["excluded"] += 1
                continue
            c = pat[(lab, pattern)]
            o = overall[lab]
            c["n"] += 1
            if gi not in sys_of_gold:
                c["fn_ext"] += 1
                o["fn_ext"] += 1
                continue
            u_s = get_uri(sys_spans[sys_of_gold[gi]])
            if u_s is None:
                c["fn_nouri"] += 1
                o["fn_nouri"] += 1
            elif uri_agree(u_g, u_s):
                c["tp"] += 1
                o["tp"] += 1
            else:
                c["fn_wrong"] += 1
                o["fn_wrong"] += 1

        # system-side: spurious uri assertions (no gold counterpart)
        sys_aligned, _, mm2 = align_parent_mentions(sys_html, sys_spans)
        align_mismatch += mm2
        sys_pat_of_span = {gi: (lab, pattern)
                           for gi, lab, pattern, _ in sys_aligned}
        for j in un_s:
            s = sys_spans[j]
            if s.labelname not in PARENT_LABELS:
                continue
            if not is_real_uri(get_uri(s)):
                continue
            lab = s.labelname
            overall[lab]["fp_spurious"] += 1
            if j in sys_pat_of_span:
                _, pattern = sys_pat_of_span[j]
                spurious_pat[(lab, pattern)] += 1
            else:
                spurious_pat[(lab, "(unaligned)")] += 1

        n_docs += 1

    return {"model": model, "n_docs": n_docs, "missing": missing,
            "patterns": dict(pat), "overall": dict(overall),
            "spurious_pat": dict(spurious_pat),
            "align_mismatch": align_mismatch}


def uri_prf(tp, fn, fp):
    r = tp / (tp + fn) if (tp + fn) else 0.0
    p = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_overall_csv(res, path: Path):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mention", "n_gold_real_uri", "TP", "FN_extraction",
                    "FN_no_uri", "FN_wrong_uri", "FP_spurious",
                    "excluded_no_uri", "precision", "recall", "F1",
                    "extraction_recall", "resolver_accuracy_on_matched"])
        for lab in sorted(res["overall"]):
            o = res["overall"][lab]
            n = o["tp"] + o["fn_ext"] + o["fn_nouri"] + o["fn_wrong"]
            fn = o["fn_ext"] + o["fn_nouri"] + o["fn_wrong"]
            fp = o["fn_wrong"] + o["fp_spurious"]
            p, r, f1 = uri_prf(o["tp"], fn, fp)
            matched = o["tp"] + o["fn_wrong"] + o["fn_nouri"]
            w.writerow([lab, n, o["tp"], o["fn_ext"], o["fn_nouri"],
                        o["fn_wrong"], o["fp_spurious"], o["excluded"],
                        f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}",
                        f"{matched/n:.4f}" if n else "0",
                        f"{o['tp']/matched:.4f}" if matched else "0"])


def write_pattern_csv(res, path: Path):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mention", "pattern_grouped", "n_gold", "TP",
                    "FN_extraction", "FN_no_uri", "FN_wrong_uri",
                    "failures", "fail_rate", "extraction_recall",
                    "resolver_accuracy_on_matched"])
        items = sorted(res["patterns"].items(),
                       key=lambda kv: (-(kv[1]["n"] - kv[1]["tp"]),
                                       -kv[1]["n"], kv[0]))
        for (lab, pat), c in items:
            fail = c["fn_ext"] + c["fn_nouri"] + c["fn_wrong"]
            matched = c["tp"] + c["fn_wrong"] + c["fn_nouri"]
            w.writerow([lab, pat, c["n"], c["tp"], c["fn_ext"],
                        c["fn_nouri"], c["fn_wrong"], fail,
                        f"{fail/c['n']:.4f}" if c["n"] else "0",
                        f"{matched/c['n']:.4f}" if c["n"] else "0",
                        f"{c['tp']/matched:.4f}" if matched else "0"])


def summary_lines(res) -> list:
    L = [f"## {res['model']} — {res['n_docs']} docs "
         f"(align mismatches: {res['align_mismatch']})", ""]
    L.append("### URI F1 overall (end-to-end, real-uri instances only)")
    L.append("")
    L.append("| mention | gold | TP | FN_ext | FN_nouri | FN_wrong | FP_spur | P | R | F1 | extr_R | resolv_acc |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    tot = {"tp": 0, "fn": 0, "fp": 0}
    for lab in sorted(res["overall"]):
        o = res["overall"][lab]
        n = o["tp"] + o["fn_ext"] + o["fn_nouri"] + o["fn_wrong"]
        fn = o["fn_ext"] + o["fn_nouri"] + o["fn_wrong"]
        fp = o["fn_wrong"] + o["fp_spurious"]
        p, r, f1 = uri_prf(o["tp"], fn, fp)
        matched = o["tp"] + o["fn_wrong"] + o["fn_nouri"]
        tot["tp"] += o["tp"]
        tot["fn"] += fn
        tot["fp"] += fp
        L.append(f"| {lab} | {n} | {o['tp']} | {o['fn_ext']} | {o['fn_nouri']} | "
                 f"{o['fn_wrong']} | {o['fp_spurious']} | {p*100:.1f}% | {r*100:.1f}% | "
                 f"{f1*100:.1f}% | {matched/n*100:.1f}% | {o['tp']/matched*100:.1f}% |")
    p, r, f1 = uri_prf(tot["tp"], tot["fn"], tot["fp"])
    L.append(f"| ALL | - | {tot['tp']} | - | - | - | - | {p*100:.1f}% | {r*100:.1f}% | {f1*100:.1f}% | - | - |")
    L.append("")
    L.append("### Top failure patterns (by failure count)")
    L.append("")
    L.append("| mention | pattern | gold | failures | fail rate | extr_R | resolv_acc |")
    L.append("|---|---|---|---|---|---|---|")
    items = sorted(res["patterns"].items(),
                   key=lambda kv: (-(kv[1]["n"] - kv[1]["tp"]), -kv[1]["n"]))[:15]
    for (lab, pattern), c in items:
        fail = c["fn_ext"] + c["fn_nouri"] + c["fn_wrong"]
        matched = c["tp"] + c["fn_wrong"] + c["fn_nouri"]
        racc = f"{c['tp']/matched*100:.1f}%" if matched else "-"
        L.append(f"| {lab} | {pattern} | {c['n']} | {fail} | {fail/c['n']*100:.1f}% | "
                 f"{matched/c['n']*100:.1f}% | {racc} |")
    L.append("")
    return L, (p, r, f1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold-dir", default="data/annotated/test")
    ap.add_argument("--out", default="Evaluation_analysis/results_e2e")
    args = ap.parse_args()

    gold_dir = PROJECT_ROOT / args.gold_dir
    out_dir = PROJECT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    all_res = []
    report = ["# End-to-end URI resolution failure analysis (by grouped pattern)",
              "",
              f"Runs: `output_coref/1-final/*_pip`; gold: `{args.gold_dir}`.",
              "Pattern grouping collapses consecutive duplicates "
              "(decision -> title -> title = decision -> title).",
              ""]
    f1line = []
    run_dirs = {}
    for model, run in RUNS_E2E.items():
        run_dir = PROJECT_ROOT / run
        if not run_dir.exists():
            print(f"=== {model} ({run}) ===\n  SKIP: not found", flush=True)
            continue
        run_dirs[model] = run_dir
    doc_sets = {m: {p[0] for p in find_pairs(d, gold_dir)[0]}
                for m, d in run_dirs.items()}
    common_docs = set.intersection(*doc_sets.values()) if doc_sets else set()
    print(f"Common docs across {len(doc_sets)} runs: {len(common_docs)}",
          flush=True)
    for m, s in sorted(doc_sets.items()):
        print(f"  {m}: {len(s)} docs", flush=True)

    for model, run_dir in run_dirs.items():
        print(f"=== {model} ===", flush=True)
        res = analyze_model(model, run_dir, gold_dir)
        print(f"  docs={res['n_docs']} skipped={len(res['missing'])} "
              f"align_mismatch={res['align_mismatch']}", flush=True)
        write_overall_csv(res, out_dir / f"{model}_e2e_uri_by_mention.csv")
        write_pattern_csv(res, out_dir / f"{model}_e2e_pattern_failures.csv")
        with (out_dir / f"{model}_e2e_spurious_by_pattern.csv").open(
                "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["mention", "pattern_grouped", "fp_spurious"])
            for (lab, pattern), n in sorted(res["spurious_pat"].items(),
                                            key=lambda kv: (-kv[1], kv[0])):
                w.writerow([lab, pattern, n])
        all_res.append(res)
        lines, (p, r, f1) = summary_lines(res)
        report.extend(lines)
        f1line.append((model, p, r, f1))

    if all_res:
        # comparison: URI F1 per model
        with (out_dir / "comparison_e2e_uri_f1.csv").open(
                "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "precision", "recall", "F1"])
            for model, p, r, f1 in f1line:
                w.writerow([model, f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}"])
        # comparison: fail rate per (mention, pattern); union of top-12 per model
        top = set()
        for res in all_res:
            best = sorted(res["patterns"].items(),
                          key=lambda kv: -kv[1]["n"])[:12]
            top.update(k for k, _ in best)
        with (out_dir / "comparison_e2e_pattern_fail_rate.csv").open(
                "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["mention", "pattern_grouped", "gold_n"] +
                       [r["model"] for r in all_res])
            for key in sorted(top, key=lambda k: (-sum(
                    r["patterns"].get(k, {"n": 0})["n"] for r in all_res), k)):
                n = sum(r["patterns"].get(key, {"n": 0})["n"]
                        for r in all_res) // max(len(all_res), 1)
                row = [key[0], key[1], n]
                for r in all_res:
                    c = r["patterns"].get(key, {"n": 0, "tp": 0, "fn_ext": 0,
                                                "fn_nouri": 0, "fn_wrong": 0})
                    fail = c["fn_ext"] + c["fn_nouri"] + c["fn_wrong"]
                    row.append(f"{fail/c['n']:.4f}" if c["n"] else "-")
                w.writerow(row)
        # most failures by mention type, aggregated over models
        agg = defaultdict(Counter)  # mention -> Counter(pattern -> failures)
        for res in all_res:
            if res["model"] == "saullm":
                continue
            for (lab, pattern), c in res["patterns"].items():
                agg[lab][pattern] += c["fn_ext"] + c["fn_nouri"] + c["fn_wrong"]
        report.append("## Most failures by mention type (4 models aggregated, excl. saullm)")
        report.append("")
        for lab in ["decision", "legislation", "secondary sources"]:
            report.append(f"### {lab}")
            report.append("")
            report.append("| pattern | failures (all models) |")
            report.append("|---|---|")
            for pattern, n in agg[lab].most_common(10):
                report.append(f"| {pattern} | {n} |")
            report.append("")

    if len(run_dirs) > 1 and common_docs:
        # Common-docs pass so cross-model comparison is apples-to-apples
        # (pipeline coverage differs: e.g. gpt5.2 has fewer resolved docs).
        common_res = []
        for model, run_dir in run_dirs.items():
            print(f"=== {model} (common {len(common_docs)} docs) ===",
                  flush=True)
            res = analyze_model(model, run_dir, gold_dir,
                                only_docs=common_docs)
            common_res.append(res)
        tag = f"common{len(common_docs)}"
        with (out_dir / f"comparison_{tag}_e2e_uri_f1.csv").open(
                "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["model", "docs", "precision", "recall", "F1"])
            for res in common_res:
                t = sum(v["tp"] for v in res["overall"].values())
                fn = sum(v["fn_ext"] + v["fn_nouri"] + v["fn_wrong"]
                         for v in res["overall"].values())
                fp = sum(v["fn_wrong"] + v["fp_spurious"]
                         for v in res["overall"].values())
                p, r, f1 = uri_prf(t, fn, fp)
                w.writerow([res["model"], res["n_docs"],
                            f"{p:.4f}", f"{r:.4f}", f"{f1:.4f}"])
        top = set()
        for res in common_res:
            best = sorted(res["patterns"].items(),
                          key=lambda kv: -kv[1]["n"])[:12]
            top.update(k for k, _ in best)
        with (out_dir / f"comparison_{tag}_e2e_pattern_fail_rate.csv").open(
                "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["mention", "pattern_grouped", "gold_n"] +
                       [r["model"] for r in common_res])
            for key in sorted(top, key=lambda k: (-sum(
                    r["patterns"].get(k, {"n": 0})["n"]
                    for r in common_res), k)):
                n = sum(r["patterns"].get(key, {"n": 0})["n"]
                        for r in common_res) // max(len(common_res), 1)
                row = [key[0], key[1], n]
                for r in common_res:
                    c = r["patterns"].get(key, {"n": 0, "tp": 0, "fn_ext": 0,
                                                "fn_nouri": 0, "fn_wrong": 0})
                    fail = c["fn_ext"] + c["fn_nouri"] + c["fn_wrong"]
                    row.append(f"{fail/c['n']:.4f}" if c["n"] else "-")
                w.writerow(row)
        report.append(f"## Common-docs comparison ({len(common_docs)} docs "
                      f"present in every pipeline run)")
        report.append("")
        report.append(f"See `comparison_{tag}_e2e_uri_f1.csv` and "
                      f"`comparison_{tag}_e2e_pattern_fail_rate.csv`.")
    (out_dir / "REPORT_E2E.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote results to {out_dir}")


if __name__ == "__main__":
    main()
