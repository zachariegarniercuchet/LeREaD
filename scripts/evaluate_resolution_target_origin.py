#!/usr/bin/env python
"""Stratified stage-3 (URI linking) evaluation by candidate-pool origin.

Motivation (reviewer W1)
------------------------
Part of the candidate pool is *synthetic*: for gold mentions whose truth was
``uri="none"``, ``uri_request/build_gold_synthetic_pool.py`` created one
candidate row per cited source, using the titles/citations/authors of the very
mentions that will later be resolved.  The reviewer asks whether those
synthetic targets are easier to link than real CanLII targets, because the
linking query may be a near-exact string match against the synthetic row.

This script answers that with *existing* stage-3 output only (no GPU, no
re-indexing).  It is a thin stratified layer on top of
``src/evaluation/evaluate_resolution.py``: identical span extraction, identical
Level-1 exact matching, identical TP/FN/FP semantics -- the only change is that
gold spans are bucketed by the origin of their gold URI:

    synthetic   gold uri starts with https://synthetic.leread.ca/gold/
    canlii      any other real gold uri (in practice canlii.ca / canlii.org)

Reported per bucket:

  * mention-level micro precision / recall / F1 (the paper metric),
  * resolver-only accuracy (Level-1 matched spans only),
  * per-label (decision / legislation / secondary sources),
  * per-unique-target macro accuracy (each cited document weighted equally),
  * cluster-bootstrap 95% CIs and a paired bootstrap for synthetic - canlii,
  * a "near-exact shortcut" diagnostic built from the run's
    ``*_resolution_mapping.json`` (query_text vs. candidate metadata), and
  * top-1 / top-k retrieval hit rates.

Usage (from repo root)::

    python scripts/evaluate_resolution_target_origin.py \
        --batch-folder output_coref/test_gold_resolution_qwen3-8b --split test

    python scripts/evaluate_resolution_target_origin.py \
        --batch-folder output_coref/2-resolution_gold/test_gold_resolution_rule \
        --split test --out scripts/report/test_rule_target_origin.md
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bs4 import BeautifulSoup  # noqa: E402

# Reuse the official evaluator (path loading avoids importing the whole package).
_spec = importlib.util.spec_from_file_location(
    "evaluate_resolution_target_origin",
    PROJECT_ROOT / "src" / "evaluation" / "evaluate_resolution.py",
)
ER = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ER)

Span = ER.Span
extract_spans = ER.extract_spans
collect_pairs_from_batch_folder = ER.collect_pairs_from_batch_folder
extract_split_from_folder_name = ER.extract_split_from_folder_name

SYNTHETIC_PREFIX = "https://synthetic.leread.ca/gold/"
TARGET_CLASSES = ("synthetic", "canlii")
LABELS = ("decision", "legislation", "secondary sources")
LABEL_DISPLAY = {
    "decision": "Decision",
    "legislation": "Legislation",
    "secondary sources": "Secondary sources",
}
LABEL_SHORT = {
    "decision": "Decision",
    "legislation": "Legislation",
    "secondary sources": "Secondary",
}

# One target's contribution to the confusion counts.
_EMPTY_TARGET = {
    "tp": 0,
    "fn_wrong": 0,
    "fn_missing_sys_uri": 0,
    "fn_missed_extraction": 0,
    "fp_spurious": 0,
    "n": 0,
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def norm_ws(text: object) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("\r", " ").replace("\n", " ")).strip()


def folding(text: object) -> str:
    return norm_ws(html.unescape(str(text or ""))).casefold()


def token_set(text: object) -> set:
    return set(re.findall(r"[0-9a-z]+", folding(text)))


def target_class(uri: Optional[str]) -> Optional[str]:
    """synthetic / canlii for a real uri; None for absent or 'none'."""
    if uri is None:
        return None
    u = uri.strip()
    if not u or u.lower() == "none":
        return None
    return "synthetic" if u.startswith(SYNTHETIC_PREFIX) else "canlii"


def get_uri(span: Span) -> Optional[str]:
    return ER._get_uri(span)


def span_docid(span: Span) -> str:
    return norm_ws(html.unescape(str(span.attributes.get("docid", ""))))


def prf1(tp: int, fn: int, fp: int) -> Tuple[float, float, float]:
    return ER._prf1(tp, fn, fp)


def _sum_targets(targets: Iterable[dict]) -> Dict[str, int]:
    out = dict(_EMPTY_TARGET)
    for t in targets:
        for k in out:
            out[k] += t.get(k, 0)
    return out


def counts_to_metrics(c: Dict[str, int]) -> Dict[str, float]:
    fn = c["fn_wrong"] + c["fn_missing_sys_uri"] + c["fn_missed_extraction"]
    fp = c["fn_wrong"] + c["fp_spurious"]
    p, r, f1 = prf1(c["tp"], fn, fp)
    matched = c["tp"] + c["fn_wrong"] + c["fn_missing_sys_uri"]
    return {
        "tp": c["tp"],
        "fn": fn,
        "fp": fp,
        "fn_wrong": c["fn_wrong"],
        "fn_missing_sys_uri": c["fn_missing_sys_uri"],
        "fn_missed_extraction": c["fn_missed_extraction"],
        "fp_spurious": c["fp_spurious"],
        "precision": p,
        "recall": r,
        "f1": f1,
        "accuracy": c["tp"] / matched if matched else 0.0,
        "matched_total": matched,
        "n": c["n"],
    }


# ---------------------------------------------------------------------------
# Candidate metadata (for the shortcut diagnostic)
# ---------------------------------------------------------------------------


def load_pool_texts(pool_path: Path) -> Dict[str, dict]:
    """uri -> {title, texts, tokens, tokens_title_only}."""
    import csv

    csv.field_size_limit(10**8)
    pool: Dict[str, dict] = {}
    with pool_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            uri = (row.get("uri") or "").strip()
            if not uri:
                continue
            try:
                meta = json.loads(row.get("metadata") or "[]")
            except Exception:
                meta = []
            cand = meta[0] if meta else {}
            title = norm_ws(cand.get("docTitle", ""))
            texts = [title] if title else []
            try:
                acs = json.loads(cand.get("autoCompletions") or "[]")
            except Exception:
                acs = []
            for a in acs:
                if isinstance(a, dict):
                    t = norm_ws(a.get("text", ""))
                    if t:
                        texts.append(t)
            texts = list(dict.fromkeys(texts))
            pool[uri] = {
                "title": title,
                "texts": texts,
                "tokens": token_set(" ".join(texts)),
                "title_tokens": token_set(title),
            }
    return pool


def load_mapping(mapping_path: Path) -> Dict[str, dict]:
    if not mapping_path.exists():
        return {}
    try:
        data = json.load(mapping_path.open(encoding="utf-8"))
    except Exception:
        return {}
    return {norm_ws(k): v for k, v in data.items()}


def shortcut_features(query_text: str, target: dict) -> Dict[str, float]:
    """Lexical overlap between the run's query text and the gold candidate row."""
    q_norm = folding(query_text)
    q_tokens = token_set(query_text)
    t = target
    t_tokens = t["tokens"]
    title_norm = folding(t["title"])

    # Exact normalised substring of a meaningful metadata string.
    exact_any = any(
        len(x) >= 8 and folding(x) in q_norm
        for x in t["texts"]
    )
    exact_title = bool(title_norm) and len(title_norm) >= 8 and title_norm in q_norm
    containment = (len(q_tokens & t_tokens) / len(t_tokens)) if t_tokens else 0.0
    title_containment = (
        len(q_tokens & t["title_tokens"]) / len(t["title_tokens"])
        if t["title_tokens"] else 0.0
    )
    return {
        "exact_any": 1.0 if exact_any else 0.0,
        "exact_title": 1.0 if exact_title else 0.0,
        "token_containment": containment,
        "title_containment": title_containment,
    }


# ---------------------------------------------------------------------------
# Core stratified evaluation
# ---------------------------------------------------------------------------


def evaluate_folder(
    batch_folder: Path,
    data_dir: Path,
    split: str,
    pool_path: Path,
    context_chars: int,
) -> dict:
    pairs = collect_pairs_from_batch_folder(
        batch_folder, data_dir, split, suffix=ER.RESOLUTION_SUFFIX, verbose=False
    )

    pool = load_pool_texts(pool_path)

    # target uri -> counters (for mention-level micro and per-target macro)
    targets: Dict[str, dict] = {}
    target_meta: Dict[str, dict] = {}
    # (class, label) -> counters
    by_class_label: Dict[Tuple[str, str], Dict[str, int]] = defaultdict(
        lambda: dict(_EMPTY_TARGET)
    )
    unmatched_system_by_class: Counter = Counter()
    spurious_by_class_label: Counter = Counter()
    doc_class_counts: List[dict] = []
    docs = []
    span_records: List[dict] = []
    missing_mappings = 0

    for gold_path, system_path in pairs:
        gold_path = Path(gold_path)
        system_path = Path(system_path)
        gold_spans = extract_spans(gold_path, context_chars)
        system_spans = extract_spans(system_path, context_chars)
        matched = ER._match_spans_l1(gold_spans, system_spans)
        matched_gold = {id(g) for g, _ in matched}
        matched_sys = {id(s) for _, s in matched}

        # The resolution mapping is keyed by normalised docid and carries the
        # exact query_text the resolver used.
        mapping = load_mapping(system_path.with_name(
            system_path.name.replace("_resolution.html", "_resolution_mapping.json")
        ))
        if not mapping:
            missing_mappings += 1

        doc_counts = defaultdict(int)
        doc_by_class = {cls: dict(_EMPTY_TARGET) for cls in TARGET_CLASSES}

        for g, s in matched:
            cls = target_class(get_uri(g))
            if cls is None:
                continue
            u_g = get_uri(g)
            label = (g.labelname or "").strip().lower()
            if label not in LABELS:
                label = "secondary sources" if "secondary" in label else label
            entry = targets.setdefault(u_g, dict(_EMPTY_TARGET))
            entry["n"] += 1
            doc_by_class[cls]["n"] += 1
            doc_counts["n"] += 1

            u_s = get_uri(s)
            if u_s is not None and ER._uri_agree(u_g, u_s):
                key = "tp"
                correct = True
            elif u_s is not None:
                key = "fn_wrong"
                correct = False
            else:
                key = "fn_missing_sys_uri"
                correct = False

            entry[key] += 1
            by_class_label[(cls, label)]["n"] += 1
            by_class_label[(cls, label)][key] += 1
            doc_by_class[cls][key] += 1
            doc_counts[key] += 1

            # shortcut features (only meaningful when mapping + pool row exist)
            mid = span_docid(g)
            m = mapping.get(mid)
            rec = {
                "doc": Path(gold_path).stem,
                "target": u_g,
                "class": cls,
                "label": label,
                "correct": correct,
            }
            if m and u_g in pool:
                feats = shortcut_features(m.get("query_text", ""), pool[u_g])
                rec.update(feats)
                rec["top1"] = int(bool(m.get("topk")) and m["topk"][0].get("uri") == u_g)
                rec["topk_hit"] = int(any(t.get("uri") == u_g for t in (m.get("topk") or [])))
                rec["topk_len"] = len(m.get("topk") or [])
                span_records.append(rec)
            else:
                rec["no_features"] = True
                span_records.append(rec)

        # extraction misses on real-uri gold
        for g in gold_spans:
            if id(g) in matched_gold:
                continue
            cls = target_class(get_uri(g))
            if cls is None:
                continue
            u_g = get_uri(g)
            label = (g.labelname or "").strip().lower()
            entry = targets.setdefault(u_g, dict(_EMPTY_TARGET))
            entry["n"] += 1
            entry["fn_missed_extraction"] += 1
            by_class_label[(cls, label)]["n"] += 1
            by_class_label[(cls, label)]["fn_missed_extraction"] += 1
            doc_by_class[cls]["n"] += 1
            doc_by_class[cls]["fn_missed_extraction"] += 1
            doc_counts["n"] += 1
            doc_counts["fn_missed_extraction"] += 1

        # spurious system assertions (no gold counterpart)
        for s in system_spans:
            if id(s) in matched_sys:
                continue
            cls = target_class(get_uri(s))
            if cls is None:
                continue
            label = (s.labelname or "").strip().lower()
            unmatched_system_by_class[cls] += 1
            spurious_by_class_label[(cls, label)] += 1
            doc_by_class[cls]["fp_spurious"] += 1

        doc_class_counts.append({
            "document": Path(gold_path).stem,
            "by_class": doc_by_class,
        })
        docs.append({
            "document": Path(gold_path).stem,
            "n": doc_counts["n"],
            "tp": doc_counts["tp"],
            "fn_wrong": doc_counts["fn_wrong"],
            "fn_missing_sys_uri": doc_counts["fn_missing_sys_uri"],
            "fn_missed_extraction": doc_counts["fn_missed_extraction"],
            "n_profiles": len(mapping),
            "has_mapping": bool(mapping),
        })

    # ---- aggregate ----
    class_targets: Dict[str, List[dict]] = defaultdict(list)
    for uri, t in targets.items():
        class_targets[target_class(uri)].append(t)

    per_class = {}
    for cls in TARGET_CLASSES:
        c = _sum_targets(class_targets[cls])
        c["fp_spurious"] = unmatched_system_by_class.get(cls, 0)
        c["n"] = sum(t["n"] for t in class_targets[cls])
        per_class[cls] = counts_to_metrics(c)

    overall_c = _sum_targets(targets.values())
    overall_c["fp_spurious"] = sum(unmatched_system_by_class.values())
    overall_c["n"] = sum(t["n"] for t in targets.values())
    overall = counts_to_metrics(overall_c)

    per_label = {}
    for cls in TARGET_CLASSES:
        per_label[cls] = {}
        for label in LABELS:
            c = dict(by_class_label.get((cls, label), _EMPTY_TARGET))
            c["fp_spurious"] += spurious_by_class_label.get((cls, label), 0)
            per_label[cls][label] = counts_to_metrics(c)

    # ---- per-unique-target macro (each cited document weighted equally) ----
    per_target_macro = {}
    for cls in TARGET_CLASSES:
        accs = []
        for t in class_targets[cls]:
            fn = t["fn_wrong"] + t["fn_missing_sys_uri"] + t["fn_missed_extraction"]
            denom = t["tp"] + fn
            if denom:
                accs.append(t["tp"] / denom)
        per_target_macro[cls] = {
            "n_targets": len(accs),
            "mean_accuracy": sum(accs) / len(accs) if accs else 0.0,
            "median_accuracy": _median(accs),
            "pct_targets_100": (sum(1 for a in accs if a == 1.0) / len(accs)) if accs else 0.0,
            "pct_targets_0": (sum(1 for a in accs if a == 0.0) / len(accs)) if accs else 0.0,
        }

    # ---- bootstrap CIs (resample unique targets, cluster by target uri) ----
    boot = {}
    for cls in TARGET_CLASSES:
        boot[cls] = bootstrap_class(class_targets[cls], unmatched_system_by_class.get(cls, 0))
    boot["difference_f1"] = paired_bootstrap_diff(
        class_targets["synthetic"], class_targets["canlii"],
        unmatched_system_by_class.get("synthetic", 0),
        unmatched_system_by_class.get("canlii", 0),
    )
    boot["difference_accuracy"] = paired_bootstrap_diff(
        class_targets["synthetic"], class_targets["canlii"],
        unmatched_system_by_class.get("synthetic", 0),
        unmatched_system_by_class.get("canlii", 0),
        metric="accuracy",
    )
    # Primary inference: resample *documents* (targets within a document are
    # correlated, and the same target recurs across documents).
    boot["document_cluster"] = bootstrap_documents(doc_class_counts)
    boot["document_cluster_difference_f1"] = paired_bootstrap_documents(
        doc_class_counts, metric="f1")
    boot["document_cluster_difference_accuracy"] = paired_bootstrap_documents(
        doc_class_counts, metric="accuracy")

    # ---- headline impact of adding synthetic targets ----
    impact = {}
    for name, c in (("all", overall_c), ("canlii_only", {
        k: _sum_targets(class_targets["canlii"]).get(k, 0) for k in _EMPTY_TARGET
    }), ("synthetic_only", {
        k: _sum_targets(class_targets["synthetic"]).get(k, 0) for k in _EMPTY_TARGET
    })):
        cc = dict(c)
        if name == "canlii_only":
            cc["fp_spurious"] = unmatched_system_by_class.get("canlii", 0)
        elif name == "synthetic_only":
            cc["fp_spurious"] = unmatched_system_by_class.get("synthetic", 0)
        impact[name] = counts_to_metrics(cc)
    # Bound: what the headline would be if every synthetic mention were wrong
    # (all synthetic TP -> FN, no new FP).
    worst = dict(overall_c)
    syn_tp = per_class["synthetic"]["tp"]
    worst["tp"] -= syn_tp
    worst["fn_missing_sys_uri"] += syn_tp
    impact["synthetic_all_wrong_bound"] = counts_to_metrics(worst)

    # ---- shortcut diagnostic ----
    shortcut = shortcut_summary(span_records)

    return {
        "batch_folder": str(batch_folder),
        "split": split,
        "pool": str(pool_path),
        "documents": len(pairs),
        "documents_without_mapping": missing_mappings,
        "overall": overall,
        "per_class": per_class,
        "per_label": per_label,
        "per_target_macro": per_target_macro,
        "bootstrap": boot,
        "impact": impact,
        "shortcut": shortcut,
        "per_document": docs,
        "n_unique_targets": {cls: len(class_targets[cls]) for cls in TARGET_CLASSES},
        "n_mentions": {cls: sum(t["n"] for t in class_targets[cls]) for cls in TARGET_CLASSES},
        "spurious_system": dict(unmatched_system_by_class),
    }


def _median(xs: List[float]) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2


def _metrics_from_targets(targets: List[dict], fp_spurious: int = 0,
                           metric: str = "f1") -> Dict[str, float]:
    c = _sum_targets(targets)
    c["fp_spurious"] += fp_spurious
    m = counts_to_metrics(c)
    return m


def bootstrap_class(targets: List[dict], fp_spurious: int,
                    n_boot: int = 2000, seed: int = 20261124) -> dict:
    """Percentile CI over unique targets (cluster bootstrap)."""
    if not targets:
        return {"f1_ci95": [0.0, 0.0], "accuracy_ci95": [0.0, 0.0]}
    rng = random.Random(seed)
    n = len(targets)
    f1s, accs = [], []
    for _ in range(n_boot):
        sample = [targets[rng.randrange(n)] for _ in range(n)]
        m = _metrics_from_targets(sample, fp_spurious)
        f1s.append(m["f1"])
        accs.append(m["accuracy"])
    f1s.sort()
    accs.sort()
    lo, hi = int(0.025 * n_boot), int(0.975 * n_boot) - 1
    return {
        "f1_ci95": [f1s[lo], f1s[hi]],
        "accuracy_ci95": [accs[lo], accs[hi]],
        "n_targets": n,
    }


def paired_bootstrap_diff(a: List[dict], b: List[dict], fp_a: int, fp_b: int,
                          n_boot: int = 2000, seed: int = 20261124,
                          metric: str = "f1") -> dict:
    """Independent cluster bootstrap of metric(synthetic) - metric(canlii)."""
    if not a or not b:
        return {"diff": 0.0, "ci95": [0.0, 0.0]}
    rng = random.Random(seed)
    na, nb = len(a), len(b)
    diffs = []
    base_a = _metrics_from_targets(a, fp_a)[metric]
    base_b = _metrics_from_targets(b, fp_b)[metric]
    for _ in range(n_boot):
        sa = [a[rng.randrange(na)] for _ in range(na)]
        sb = [b[rng.randrange(nb)] for _ in range(nb)]
        ma = _metrics_from_targets(sa, fp_a)[metric]
        mb = _metrics_from_targets(sb, fp_b)[metric]
        diffs.append(ma - mb)
    diffs.sort()
    lo, hi = int(0.025 * n_boot), int(0.975 * n_boot) - 1
    return {
        "diff": base_a - base_b,
        "ci95": [diffs[lo], diffs[hi]],
        "metric": metric,
    }


def _metrics_from_docs(docs: List[dict], cls: str, metric: str = "f1") -> float:
    c = dict(_EMPTY_TARGET)
    for d in docs:
        dc = d["by_class"][cls]
        for k in c:
            c[k] += dc.get(k, 0)
    return counts_to_metrics(c)[metric]


def bootstrap_documents(doc_counts: List[dict], n_boot: int = 2000,
                        seed: int = 20261124) -> dict:
    """Percentile CI resampling whole documents (all target classes together)."""
    if not doc_counts:
        return {}
    rng = random.Random(seed)
    n = len(doc_counts)
    samples = {cls: {"f1": [], "accuracy": []} for cls in TARGET_CLASSES}
    for _ in range(n_boot):
        sample = [doc_counts[rng.randrange(n)] for _ in range(n)]
        for cls in TARGET_CLASSES:
            m = counts_to_metrics({
                k: sum(d["by_class"][cls].get(k, 0) for d in sample)
                for k in _EMPTY_TARGET
            })
            samples[cls]["f1"].append(m["f1"])
            samples[cls]["accuracy"].append(m["accuracy"])
    lo, hi = int(0.025 * n_boot), int(0.975 * n_boot) - 1
    out = {}
    for cls in TARGET_CLASSES:
        for metric in ("f1", "accuracy"):
            samples[cls][metric].sort()
        out[cls] = {
            "f1_ci95": [samples[cls]["f1"][lo], samples[cls]["f1"][hi]],
            "accuracy_ci95": [samples[cls]["accuracy"][lo],
                              samples[cls]["accuracy"][hi]],
        }
    return out


def paired_bootstrap_documents(doc_counts: List[dict], n_boot: int = 2000,
                               seed: int = 20261124,
                               metric: str = "f1") -> dict:
    """Paired document bootstrap of metric(synthetic) - metric(canlii)."""
    if not doc_counts:
        return {"diff": 0.0, "ci95": [0.0, 0.0], "metric": metric}
    rng = random.Random(seed)
    n = len(doc_counts)
    diffs = []
    for _ in range(n_boot):
        sample = [doc_counts[rng.randrange(n)] for _ in range(n)]
        diffs.append(_metrics_from_docs(sample, "synthetic", metric)
                     - _metrics_from_docs(sample, "canlii", metric))
    diffs.sort()
    lo, hi = int(0.025 * n_boot), int(0.975 * n_boot) - 1
    return {
        "diff": (_metrics_from_docs(doc_counts, "synthetic", metric)
                 - _metrics_from_docs(doc_counts, "canlii", metric)),
        "ci95": [diffs[lo], diffs[hi]],
        "metric": metric,
    }


def shortcut_summary(records: List[dict]) -> dict:
    """Accuracy/containment stratified by near-exact string match."""
    out = {}
    for cls in TARGET_CLASSES:
        rs = [r for r in records if r["class"] == cls and "no_features" not in r]
        if not rs:
            out[cls] = {"n": 0}
            continue

        def acc(sub):
            return sum(1 for r in sub if r["correct"]) / len(sub) if sub else None

        exact = [r for r in rs if r["exact_any"]]
        non_exact = [r for r in rs if not r["exact_any"]]
        title_exact = [r for r in rs if r["exact_title"]]
        hard = [r for r in rs if r["token_containment"] < 0.8 and not r["exact_any"]]
        topk = [r for r in rs if r.get("topk_len")]
        out[cls] = {
            "n": len(rs),
            "mean_token_containment": sum(r["token_containment"] for r in rs) / len(rs),
            "pct_exact_any": sum(1 for r in rs if r["exact_any"]) / len(rs),
            "pct_exact_title": sum(1 for r in rs if r["exact_title"]) / len(rs),
            "accuracy_exact_any": acc(exact),
            "n_exact_any": len(exact),
            "accuracy_non_exact": acc(non_exact),
            "n_non_exact": len(non_exact),
            "accuracy_exact_title": acc(title_exact),
            "n_exact_title": len(title_exact),
            "accuracy_hard": acc(hard),
            "n_hard": len(hard),
            "top1_hit": (sum(r["top1"] for r in topk) / len(topk)) if topk else None,
            "topk_hit": (sum(r["topk_hit"] for r in topk) / len(topk)) if topk else None,
            "n_topk": len(topk),
        }
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "-"
    return f"{x * 100:.2f}"


def md_table(header: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def build_report(result: dict) -> str:
    L: List[str] = []
    A = L.append
    name = Path(result["batch_folder"]).name
    o = result["overall"]
    oracle = o["fn_missed_extraction"] == 0 and o["fp_spurious"] == 0
    setting = ("gold-span (oracle) resolution" if oracle
               else "end-to-end pipeline (extraction + coref + resolution)")
    A(f"# Target-origin stratified linking evaluation — `{name}`")
    A("")
    A(f"*Split: `{result['split']}` — exact span matching — {setting}*")
    A("")
    A(f"Documents evaluated: {result['documents']} | "
      f"documents without a resolution mapping: {result['documents_without_mapping']}")
    A("")
    A("Gold targets: "
      f"**{result['n_unique_targets']['synthetic']} synthetic** "
      f"({result['n_mentions']['synthetic']} mentions) vs "
      f"**{result['n_unique_targets']['canlii']} CanLII** "
      f"({result['n_mentions']['canlii']} mentions).")
    A("")
    A("Gold URIs are classified by the gold `uri` attribute: "
      f"`{SYNTHETIC_PREFIX}*` = synthetic, everything else = CanLII. "
      "TP/FN/FP are charged to the *gold* target's class (spurious spans "
      "without a gold counterpart are charged to the predicted URI's class).")
    A("")

    A("## Overall (mention-level micro)")
    A("")
    A(f"TP {o['tp']} | FN {o['fn']} | FP {o['fp']} | "
      f"**P {pct(o['precision'])} / R {pct(o['recall'])} / F1 {pct(o['f1'])}** | "
      f"resolver-only accuracy {pct(o['accuracy'])} ({o['matched_total']} spans)")
    A("")

    A("## By candidate origin (mention-level micro)")
    A("")
    rows = []
    for cls in TARGET_CLASSES:
        m = result["per_class"][cls]
        rows.append([
            cls, result["n_unique_targets"][cls], result["n_mentions"][cls],
            m["tp"], m["fn"], m["fp"],
            pct(m["precision"]), pct(m["recall"]), pct(m["f1"]),
            pct(m["accuracy"]),
        ])
    A(md_table(["origin", "unique targets", "mentions", "TP", "FN", "FP",
                "Precision %", "Recall %", "F1 %", "Resolver acc %"], rows))
    A("")
    d = result["bootstrap"]["document_cluster_difference_f1"]
    A(f"**Synthetic - CanLII F1 difference: {pct(d['diff'])} points** "
      f"(95% CI {pct(d['ci95'][0])} to {pct(d['ci95'][1])}; "
      f"paired cluster bootstrap over documents).")
    da = result["bootstrap"]["document_cluster_difference_accuracy"]
    A(f"Resolver-accuracy difference: {pct(da['diff'])} points "
      f"(95% CI {pct(da['ci95'][0])} to {pct(da['ci95'][1])}).")
    A("")
    A("Document-cluster bootstrap 95% CIs (primary; targets within a document "
      "are correlated):")
    A("")
    rows = []
    for cls in TARGET_CLASSES:
        b = result["bootstrap"]["document_cluster"][cls]
        rows.append([cls, f"{pct(b['f1_ci95'][0])} - {pct(b['f1_ci95'][1])}",
                     f"{pct(b['accuracy_ci95'][0])} - {pct(b['accuracy_ci95'][1])}"])
    A(md_table(["origin", "F1 95% CI %", "resolver accuracy 95% CI %"], rows))
    A("")

    imp = result["impact"]
    A("## Impact on the headline number")
    A("")
    A("Synthetic targets are "
      f"{pct(result['n_mentions']['synthetic'] / max(1, o['n']))}% of real-uri "
      f"mentions and {pct(result['n_unique_targets']['synthetic'] / max(1, result['n_unique_targets']['synthetic'] + result['n_unique_targets']['canlii']))}% "
      "of unique cited targets in this split.")
    A("")
    rows = []
    for name, label in (("all", "All targets (headline)"),
                        ("canlii_only", "CanLII only (synthetic removed)"),
                        ("synthetic_only", "Synthetic only"),
                        ("synthetic_all_wrong_bound", "Bound: all synthetic wrong")):
        m = imp[name]
        rows.append([label, m["tp"], m["fn"], m["fp"],
                     pct(m["precision"]), pct(m["recall"]), pct(m["f1"])])
    A(md_table(["setting", "TP", "FN", "FP", "Precision %", "Recall %", "F1 %"], rows))
    A("")
    A(f"Headline minus CanLII-only F1: **{pct(o['f1'] - imp['canlii_only']['f1'])} points**.")
    A("")

    A("## By label × candidate origin (mention-level micro)")
    A("")
    rows = []
    for label in LABELS:
        for cls in TARGET_CLASSES:
            m = result["per_label"][cls][label]
            if m["n"] == 0:
                continue
            rows.append([
                LABEL_DISPLAY[label], cls, m["n"], m["tp"], m["fn"], m["fp"],
                pct(m["precision"]), pct(m["recall"]), pct(m["f1"]), pct(m["accuracy"]),
            ])
    A(md_table(["label", "origin", "mentions", "TP", "FN", "FP",
                "Precision %", "Recall %", "F1 %", "Resolver acc %"], rows))
    A("")

    A("## Per unique cited target (each target weighted equally)")
    A("")
    rows = []
    for cls in TARGET_CLASSES:
        m = result["per_target_macro"][cls]
        rows.append([
            cls, m["n_targets"], pct(m["mean_accuracy"]),
            pct(m["median_accuracy"]), pct(m["pct_targets_100"]), pct(m["pct_targets_0"]),
        ])
    A(md_table(["origin", "targets", "Mean accuracy %", "Median accuracy %",
                "% targets fully correct", "% targets fully wrong"], rows))
    A("")

    A("## Near-exact string-match diagnostic")
    A("")
    A("Built from each run's `*_resolution_mapping.json` (`query_text`) compared against "
      "the gold candidate row's `docTitle` + `autoCompletions`.")
    A("")
    rows = []
    for cls in TARGET_CLASSES:
        s = result["shortcut"][cls]
        if not s.get("n"):
            rows.append([cls, 0, "-", "-", "-", "-", "-", "-"])
            continue
        rows.append([
            cls, s["n"], pct(s["mean_token_containment"]),
            pct(s["pct_exact_any"]), pct(s["pct_exact_title"]),
            f"{pct(s['accuracy_exact_any'])} ({s['n_exact_any']})",
            f"{pct(s['accuracy_non_exact'])} ({s['n_non_exact']})",
            f"{pct(s['accuracy_hard'])} ({s['n_hard']})",
        ])
    A(md_table(["origin", "spans with features", "mean target-token containment %",
                "% with exact metadata match", "% with exact title match",
                "accuracy | exact match %", "accuracy | no exact match %",
                "accuracy | hard (no exact, containment<0.8) %"], rows))
    A("")
    A("Retrieval of the gold target (from the saved top-k lists):")
    A("")
    rows = []
    for cls in TARGET_CLASSES:
        s = result["shortcut"][cls]
        rows.append([cls, s.get("n_topk", 0), pct(s.get("top1_hit")), pct(s.get("topk_hit"))])
    A(md_table(["origin", "spans with top-k", "Top-1 hit %", "Top-k hit %"], rows))
    A("")
    A("_Note: top-k lists were saved by the original run (k = debug_topk), so "
      "`Top-k hit %` is a lower bound when k < 10._")
    A("")

    sp = result["spurious_system"]
    if sp:
        A("Spurious system assertions (no gold counterpart, class of asserted URI): "
          + ", ".join(f"{k}={v}" for k, v in sorted(sp.items())))
        A("")

    return "\n".join(L)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stratify stage-3 URI-linking metrics by gold target origin "
                    "(synthetic vs CanLII).")
    ap.add_argument("--batch-folder", required=True,
                    help="Stage-3 batch folder (subfolder per document).")
    ap.add_argument("--split", default=None,
                    help="Gold split (auto-detected from the folder name if omitted).")
    ap.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    ap.add_argument("--pool", default=str(PROJECT_ROOT / "data" / "gold_candidate_pool.csv"),
                    help="Candidate pool CSV used by the run (for the shortcut diagnostic).")
    ap.add_argument("--context-chars", type=int, default=ER.CONTEXT_CHARS)
    ap.add_argument("--out", default=None, help="Optional markdown report path.")
    ap.add_argument("--json-out", default=None, help="Optional JSON dump path.")
    args = ap.parse_args()

    batch = Path(args.batch_folder)
    if not batch.exists():
        raise FileNotFoundError(f"Batch folder not found: {batch}")

    split = args.split
    if not split:
        split = extract_split_from_folder_name(batch.name)
        if not split:
            raise SystemExit("Could not auto-detect split; pass --split explicitly.")
        print(f"Auto-detected split: {split}")

    result = evaluate_folder(
        batch_folder=batch,
        data_dir=Path(args.data_dir),
        split=split,
        pool_path=Path(args.pool),
        context_chars=args.context_chars,
    )

    report = build_report(result)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(report)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report + "\n", encoding="utf-8")
        print(f"\nSaved report: {out}")
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved JSON: {out}")


if __name__ == "__main__":
    main()
