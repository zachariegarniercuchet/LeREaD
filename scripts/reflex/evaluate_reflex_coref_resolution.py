#!/usr/bin/env python
"""
Evaluate the Reflex (CanLII citator) exports in data/reflex/ at the
coreference-resolution and URI-linking (resolution) stages, using the
evaluation primitives in src/evaluation/.

Reports (markdown + JSON) are written to scripts/reflex/report/.

Coreference
-----------
Reflex anchors carry a `concatenated-id` attribute which acts as the docid /
cluster key. Anchors sharing the same concatenated-id form one coref cluster.
Spans are aligned to gold (exact match + overlap match) through the token-diff
Alignment from scripts/reflex/common.py, then scored with
src/evaluation/evaluation_coref.py (MUC / B3 / CEAFe / LEA + CoNLL average).

Only the *grouping* by docid matters, never the docid string itself — gold
mentions use their annotated docid, reflex mentions use concatenated-id.

Resolution
----------
Each concatenated-id is mapped to a URI via the candidate pool
(data/full_candidate_pool.csv by default — use --candidate-pool gold for the
gold pool). The resulting URI is attached to the reflex span (mapped to gold
coordinates) and URI Precision / Recall / F1 is computed against gold with
src/evaluation/evaluate_resolution.py (exact + overlap span matching).

Only gold spans carrying a real uri value are counted as instances.

Usage (from repo root):

    # everything available, reports -> scripts/reflex/report/
    python scripts/reflex/evaluate_reflex_coref_resolution.py

    # only the human test split, fix merge gap at 15
    python scripts/reflex/evaluate_reflex_coref_resolution.py --splits test --gap 15

    # use the gold candidate pool instead of the full pool
    python scripts/reflex/evaluate_reflex_coref_resolution.py --candidate-pool gold

    # skip one of the two stages
    python scripts/reflex/evaluate_reflex_coref_resolution.py --skip-coref
    python scripts/reflex/evaluate_reflex_coref_resolution.py --skip-resolution

"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Project-root path setup (same convention as scripts/reflex/evaluate_reflex.py)
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for p in (str(SCRIPT_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from bs4 import BeautifulSoup, NavigableString, Tag  # noqa: E402

import common as C  # noqa: E402  (scripts/reflex/common.py)

# ---------------------------------------------------------------------------
# Load evaluation_coref + evaluation_resolution by path (same pattern used by
# src/evaluation/evaluate_coref.py / evaluate_resolution.py)
# ---------------------------------------------------------------------------

import importlib.util

_COREF_PATH = PROJECT_ROOT / "src" / "evaluation" / "evaluation_coref.py"
_RES_PATH = PROJECT_ROOT / "src" / "evaluation" / "evaluate_resolution.py"
_L1_PATH = PROJECT_ROOT / "src" / "evaluation" / "evaluation_l1_util.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_eval_coref = _load_module("evaluation_coref_util_reflex", _COREF_PATH)
_evaluate_resolution = _load_module("evaluate_resolution_util_reflex", _RES_PATH)
_evaluation_l1 = _load_module("evaluation_l1_util_reflex", _L1_PATH)

evaluate_coref_raw = _eval_coref.evaluate_coref_raw
aggregate_raw = _eval_coref.aggregate_raw
finalize_raw = _eval_coref.finalize_raw
Mention = _eval_coref.Mention
align_and_build_clusters = _eval_coref.align_and_build_clusters
align_and_build_clusters_overlap = _eval_coref.align_and_build_clusters_overlap

Span = _evaluation_l1.Span

compute_uri_metrics = _evaluate_resolution.compute_uri_metrics
compute_uri_metrics_overlap = _evaluate_resolution.compute_uri_metrics_overlap

CONTEXT_CHARS = 200
DEFAULT_GAP = 15
PARENT_LABELS = ("decision", "legislation", "secondary sources")

REFLEX_DIR = PROJECT_ROOT / "data" / "reflex"
ANNOTATED_DIR = PROJECT_ROOT / "data" / "annotated"
FULL_POOL = PROJECT_ROOT / "data" / "full_candidate_pool.csv"
GOLD_POOL = PROJECT_ROOT / "data" / "gold_candidate_pool.csv"
REPORT_DIR = SCRIPT_DIR / "report"

# ---------------------------------------------------------------------------
# Candidate pool: id -> uri
# ---------------------------------------------------------------------------


def load_candidate_pool(path: Path) -> Dict[str, str]:
    """Build a flat {candidate_pool_id: uri} mapping from the pool CSV.

    Each CSV row is (uri, metadata-json-array). Every entry in the metadata
    array carries an ``id`` that uniquely identifies the document, so we map
    id -> uri for every entry. First occurrence wins on duplicates.
    """
    id_to_uri: Dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uri = row.get("uri", "").strip()
            if not uri:
                continue
            try:
                metadata = json.loads(row.get("metadata", "[]"))
            except json.JSONDecodeError:
                continue
            if not isinstance(metadata, list):
                continue
            for entry in metadata:
                eid = (entry or {}).get("id", "")
                if eid and eid not in id_to_uri:
                    id_to_uri[eid] = uri
    return id_to_uri


_NUM_RE = re.compile(r"^(\d+)")


def lookup_uri(concatenated_id: str, id_to_uri: Dict[str, str]) -> Optional[str]:
    """Map a reflex concatenated-id to a candidate-pool URI.

    Tries, in order:
      1. the concatenated-id as-is (decision-style ids like 1997csc-scc33)
      2. l<num>!<concatenated-id> (legislation-style ids like l36028!36028-current-1)
      3. l<concatenated-id>
    """
    if concatenated_id in id_to_uri:
        return id_to_uri[concatenated_id]
    m = _NUM_RE.match(concatenated_id)
    if m:
        num = m.group(1)
        for key in (f"l{num}!{concatenated_id}", f"l{concatenated_id}"):
            if key in id_to_uri:
                return id_to_uri[key]
    return None


# ---------------------------------------------------------------------------
# Gold parsing with docid extraction (parent-level mentions only)
# ---------------------------------------------------------------------------


@dataclass
class GoldMention:
    label: str
    text: str
    start: int
    end: int
    docid: str
    uri: Optional[str]


def parse_gold_with_docid(path: Path) -> Tuple[str, List[GoldMention]]:
    """Parse a gold annotated HTML file into (normalised_text, [GoldMention]).

    Only parent-level mentions (labelname in {decision, legislation,
    secondary sources} with empty/missing parent) are returned.  Each mention
    carries its docid (cluster key for coref) and uri (for resolution).
    """
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    body = soup.find("body")
    if body is None:
        return "", []

    chunks: List[str] = []
    tags_meta: List[Tuple[str, int, int, dict]] = []
    pos = 0

    def walk(node) -> None:
        nonlocal pos
        for child in node.children:
            if isinstance(child, NavigableString):
                if type(child) is NavigableString:
                    chunks.append(str(child))
                    pos += len(child)
            elif isinstance(child, Tag):
                start = pos
                walk(child)
                if child.name in ("manual_label", "auto_label"):
                    lab = (child.get("labelname") or "").strip().lower()
                    parent = (child.get("parent") or "").strip()
                    if lab in PARENT_LABELS and not parent:
                        end = pos
                        attrs = dict(child.attrs)
                        tags_meta.append((lab, start, end, attrs))

    walk(body)
    norm = " ".join(chunks)
    norm = " ".join(norm.split())

    mentions: List[GoldMention] = []
    for label, s, e, attrs in tags_meta:
        text = norm[s:e].strip()
        if not text:
            continue
        mentions.append(
            GoldMention(
                label=label,
                text=text,
                start=s,
                end=e,
                docid=(attrs.get("docid") or "").strip(),
                uri=(attrs.get("uri") or "").strip() or None,
            )
        )
    return norm, mentions


# ---------------------------------------------------------------------------
# Per-pair coref evaluation
# ---------------------------------------------------------------------------


@dataclass
class CorefPairResult:
    doc_id: str
    split: str
    exact: dict
    overlap: dict
    stats: dict
    stats_overlap: dict
    n_gold_mentions: int
    n_sys_mentions: int
    n_matched: int
    exact_raw: dict
    overlap_raw: dict


def evaluate_coref_pair(
    gold_mentions: List[GoldMention],
    reflex_mentions: List[Mention],
    alignment: C.Alignment,
    unmatched: str = "drop",
) -> CorefPairResult:
    """Evaluate coreference for one (gold, reflex) document pair.

    Reflex spans are mapped into gold coordinates via the Alignment.  Both
    sides are re-expressed as Mention objects with keys in gold coordinates,
    then aligned (exact + overlap) and clustered by docid.  Clusters are
    scored with MUC / B3 / CEAFe / LEA.
    """
    len_g = alignment.len_g

    # Gold side: key = label|start|end in gold coordinates
    gm: List[Mention] = []
    for g in gold_mentions:
        gm.append(
            Mention(
                key=f"{g.label}|{g.start}|{g.end}",
                docid=g.docid,
                label=g.label,
                start=g.start,
                end=g.end,
                text=g.text,
            )
        )

    # Reflex side: map anchor spans into gold coordinates
    sm: List[Mention] = []
    for m in reflex_mentions:
        gs, ge = alignment.map_span(m.start, m.end)
        gs = min(max(0, gs), max(0, len_g - 1))
        ge = max(gs + 1, min(ge, len_g))
        sm.append(
            Mention(
                key=f"{m.label}|{gs}|{ge}",
                docid=m.docid,
                label=m.label,
                start=gs,
                end=ge,
                text=m.text,
            )
        )

    # ---- exact match ----
    g_clusters, s_clusters, stats = align_and_build_clusters(gm, sm, unmatched=unmatched)
    raw = evaluate_coref_raw(g_clusters, s_clusters)
    exact = finalize_raw(raw)

    # ---- overlap match ----
    g_clusters_ov, s_clusters_ov, stats_ov = align_and_build_clusters_overlap(
        gm, sm, unmatched=unmatched
    )
    raw_ov = evaluate_coref_raw(g_clusters_ov, s_clusters_ov)
    overlap = finalize_raw(raw_ov)

    return CorefPairResult(
        doc_id="",
        split="",
        exact=exact,
        overlap=overlap,
        stats=stats,
        stats_overlap=stats_ov,
        n_gold_mentions=len(gm),
        n_sys_mentions=len(sm),
        n_matched=stats["matched"],
        exact_raw=raw,
        overlap_raw=raw_ov,
    )


# ---------------------------------------------------------------------------
# Per-pair resolution evaluation
# ---------------------------------------------------------------------------


@dataclass
class ResolutionPairResult:
    doc_id: str
    split: str
    exact: dict
    overlap: dict
    n_gold_real_uri: int
    n_sys_uri: int
    uri_lookup_mapped: int
    uri_lookup_unmapped: int
    uri_lookup_total: int


def evaluate_resolution_pair(
    gold_mentions: List[GoldMention],
    reflex_mentions: List[Mention],
    alignment: C.Alignment,
    id_to_uri: Dict[str, str],
    gold_text: str,
) -> ResolutionPairResult:
    """Evaluate URI linking for one (gold, reflex) document pair.

    Each reflex anchor's concatenated-id is mapped to a URI via the candidate
    pool.  System spans are built in gold coordinates with the uri attribute,
    then compared against gold spans (built from GoldMention objects) using
    compute_uri_metrics (exact) and compute_uri_metrics_overlap (overlap).
    """
    len_g = alignment.len_g

    # System spans in gold coordinates, with uri from candidate pool
    sys_spans: List[Span] = []
    lookup_mapped = 0
    lookup_unmapped = 0
    lookup_total = 0
    for m in reflex_mentions:
        lookup_total += 1
        uri = lookup_uri(m.docid, id_to_uri)
        if uri:
            lookup_mapped += 1
        else:
            lookup_unmapped += 1
        gs, ge = alignment.map_span(m.start, m.end)
        gs = min(max(0, gs), max(0, len_g - 1))
        ge = max(gs + 1, min(ge, len_g))
        ctx = C._context(gold_text, gs, ge, CONTEXT_CHARS)
        sys_spans.append(
            Span(
                text=m.text,
                labelname=m.label,
                context_text=ctx,
                attributes={"uri": uri} if uri else {},
                start_pos=gs,
                end_pos=ge,
            )
        )

    # Gold spans (built from GoldMention objects, carry uri attribute)
    gold_spans: List[Span] = []
    for g in gold_mentions:
        ctx = C._context(gold_text, g.start, g.end, CONTEXT_CHARS)
        gold_spans.append(
            Span(
                text=g.text,
                labelname=g.label,
                context_text=ctx,
                attributes={"uri": g.uri} if g.uri else {},
                start_pos=g.start,
                end_pos=g.end,
            )
        )

    exact = compute_uri_metrics(gold_spans, sys_spans)
    overlap = compute_uri_metrics_overlap(gold_spans, sys_spans)

    n_gold_real_uri = exact.get("matched_total", 0) + exact.get("fn_missed_extraction", 0)

    return ResolutionPairResult(
        doc_id="",
        split="",
        exact=exact,
        overlap=overlap,
        n_gold_real_uri=n_gold_real_uri,
        n_sys_uri=lookup_mapped,
        uri_lookup_mapped=lookup_mapped,
        uri_lookup_unmapped=lookup_unmapped,
        uri_lookup_total=lookup_total,
    )


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def _macro_aggregate(scores_list: List[dict]) -> dict:
    """Macro-average finalize_raw output across documents."""
    metrics = ["MUC", "B3", "CEAF_e", "LEA"]
    agg = {}
    for m in metrics:
        ps = [s[m]["precision"] for s in scores_list]
        rs = [s[m]["recall"] for s in scores_list]
        fs = [s[m]["f1"] for s in scores_list]
        agg[m] = {
            "precision": sum(ps) / len(ps) if ps else 0.0,
            "recall": sum(rs) / len(rs) if rs else 0.0,
            "f1": sum(fs) / len(fs) if fs else 0.0,
        }
    conll = [s.get("CoNLL", {}).get("f1", 0.0) for s in scores_list]
    agg["CoNLL"] = {"f1": sum(conll) / len(conll) if conll else 0.0}
    return agg


def aggregate_coref(pair_results: List[CorefPairResult]) -> dict:
    """Aggregate coref pair results across documents.

    Proper micro-aggregation sums raw counts before dividing.  Macro-F1 is
    also reported for reference.  Per-document scores are retained.
    """
    exact_raws = [pr.exact_raw for pr in pair_results]
    overlap_raws = [pr.overlap_raw for pr in pair_results]

    exact_agg_raw = aggregate_raw(exact_raws) if exact_raws else {}
    overlap_agg_raw = aggregate_raw(overlap_raws) if overlap_raws else {}

    exact_agg = finalize_raw(exact_agg_raw) if exact_agg_raw else {}
    overlap_agg = finalize_raw(overlap_agg_raw) if overlap_agg_raw else {}

    total_stats: Dict[str, int] = defaultdict(int)
    total_stats_ov: Dict[str, int] = defaultdict(int)
    for pr in pair_results:
        for k, v in pr.stats.items():
            total_stats[k] += v
        for k, v in pr.stats_overlap.items():
            total_stats_ov[k] += v

    per_doc_exact = {}
    per_doc_overlap = {}
    for pr in pair_results:
        per_doc_exact[pr.doc_id] = {
            "split": pr.split,
            "n_gold_mentions": pr.n_gold_mentions,
            "n_sys_mentions": pr.n_sys_mentions,
            "n_matched": pr.n_matched,
            "scores": pr.exact,
        }
        per_doc_overlap[pr.doc_id] = {
            "split": pr.split,
            "n_gold_mentions": pr.n_gold_mentions,
            "n_sys_mentions": pr.n_sys_mentions,
            "n_matched": pr.n_matched,
            "scores": pr.overlap,
        }

    return {
        "exact": exact_agg,
        "overlap": overlap_agg,
        "exact_macro": _macro_aggregate([pr.exact for pr in pair_results]),
        "overlap_macro": _macro_aggregate([pr.overlap for pr in pair_results]),
        "stats": dict(total_stats),
        "stats_overlap": dict(total_stats_ov),
        "per_doc_exact": per_doc_exact,
        "per_doc_overlap": per_doc_overlap,
        "n_documents": len(pair_results),
    }


def _prf1(tp: int, fn: int, fp: int) -> Tuple[float, float, float]:
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return prec, rec, f1


def aggregate_resolution(pair_results: List[ResolutionPairResult]) -> dict:
    """Aggregate resolution pair results across documents (micro over counts)."""
    acc = defaultdict(int)
    per_label = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})
    per_cat = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})

    total_lookup_mapped = 0
    total_lookup_unmapped = 0
    total_lookup_total = 0

    per_doc = {}
    for pr in pair_results:
        ov = pr.exact["overall"]
        for k in ("tp", "fn_wrong", "fn_missing_sys_uri", "fn_missed_extraction",
                  "fp_spurious", "skipped_gold_none", "matched_spans",
                  "matched_total"):
            acc[k] += ov.get(k, 0)

        for label, m in ov.get("per_label", {}).items():
            per_label[label]["tp"] += m["tp"]
            per_label[label]["fn"] += m["fn"]
            per_label[label]["fp"] += m["fp"]

        for cat, m in ov.get("per_category", {}).items():
            per_cat[cat]["tp"] += m["tp"]
            per_cat[cat]["fn"] += m["fn"]
            per_cat[cat]["fp"] += m["fp"]

        total_lookup_mapped += pr.uri_lookup_mapped
        total_lookup_unmapped += pr.uri_lookup_unmapped
        total_lookup_total += pr.uri_lookup_total

        per_doc[pr.doc_id] = {
            "split": pr.split,
            "exact": pr.exact,
            "overlap": pr.overlap,
            "uri_lookup": {
                "mapped": pr.uri_lookup_mapped,
                "unmapped": pr.uri_lookup_unmapped,
                "total": pr.uri_lookup_total,
            },
        }

    fn_total = acc["fn_wrong"] + acc["fn_missing_sys_uri"] + acc["fn_missed_extraction"]
    fp_total = acc["fn_wrong"] + acc["fp_spurious"]
    precision, recall, f1 = _prf1(acc["tp"], fn_total, fp_total)
    matched_total = acc["tp"] + acc["fn_wrong"] + acc["fn_missing_sys_uri"]
    accuracy = acc["tp"] / matched_total if matched_total else 0.0

    overall = {
        "tp": acc["tp"],
        "fn_wrong": acc["fn_wrong"],
        "fn_missing_sys_uri": acc["fn_missing_sys_uri"],
        "fn_missed_extraction": acc["fn_missed_extraction"],
        "fp_spurious": acc["fp_spurious"],
        "skipped_gold_none": acc["skipped_gold_none"],
        "matched_spans": acc["matched_spans"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "matched_total": matched_total,
    }

    def _finalize(acc_dict):
        out = {}
        for key, d in acc_dict.items():
            p, r, f = _prf1(d["tp"], d["fn"], d["fp"])
            out[key] = {"tp": d["tp"], "fn": d["fn"], "fp": d["fp"],
                        "precision": p, "recall": r, "f1": f}
        return out

    return {
        "exact": {
            "overall": overall,
            "per_label": _finalize(per_label),
            "per_category": _finalize(per_cat),
        },
        "lookup": {
            "mapped": total_lookup_mapped,
            "unmapped": total_lookup_unmapped,
            "total": total_lookup_total,
        },
        "per_doc": per_doc,
        "n_documents": len(pair_results),
    }


# ---------------------------------------------------------------------------
# Per-split grouping
# ---------------------------------------------------------------------------


def group_by_split(pair_results):
    out = defaultdict(list)
    for pr in pair_results:
        out[pr.split].append(pr)
    return dict(out)


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------

def md_table(header: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return str(obj)


def build_coref_report(args, pair_results, aggregate, candidate_pool_name) -> str:
    lines: List[str] = []
    A = lines.append

    A("# Reflex (CanLII citator) — coreference resolution evaluation")
    A("")
    A(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} — "
      f"`python scripts/reflex/evaluate_reflex_coref_resolution.py`*")
    A("")
    A("**Scope**: coreference resolution only. Reflex anchors sharing the same "
      "``concatenated-id`` form one cluster; gold mentions sharing the same "
      "``docid`` form one cluster.  Spans are aligned into gold coordinates "
      "(token-diff Alignment) before clustering.  Only the grouping by docid "
      "is scored — the docid strings themselves are never compared.")
    A("")
    A(f"**Candidate pool**: `{candidate_pool_name}`")
    A(f"**Merge gap**: `{args.gap}` normalised characters")
    A(f"**Unmatched policy**: `drop` (mentions present on only one side are "
      f"excluded — isolates the coref step from extraction errors)")
    A("")
    A(f"**Documents**: {aggregate['n_documents']} reflex export(s) paired with gold.")
    A("")

    # Overall table
    A("## 1. Overall results (micro over documents)")
    A("")
    for mode in ("exact", "overlap"):
        title = ("EXACT span match (label + identical offsets)" if mode == "exact"
                 else "OVERLAP span match (label + intersecting offsets)")
        ag = aggregate[mode]
        A(f"### {mode.upper()} — {title}")
        A("")
        rows = []
        for mname in ("MUC", "B3", "CEAF_e", "LEA"):
            s = ag[mname]
            rows.append([mname,
                         f"{s['precision']*100:.2f}",
                         f"{s['recall']*100:.2f}",
                         f"{s['f1']*100:.2f}"])
        conll = ag["CoNLL"]["f1"]
        rows.append(["**CoNLL (MUC/B3/CEAF_e)**", "-", "-", f"{conll*100:.2f}"])
        A(md_table(["Metric", "Precision %", "Recall %", "F1 %"], rows))
        A("")
        st = aggregate["stats"] if mode == "exact" else aggregate["stats_overlap"]
        A(f"Alignment stats — gold mentions: {st.get('gold', '-')} | "
          f"system mentions: {st.get('system', '-')} | "
          f"matched: {st.get('matched', '-')} | "
          f"gold-only: {st.get('gold_only', '-')} | "
          f"system-only: {st.get('system_only', '-')}")
        A("")

    # Per split
    by_split = group_by_split(pair_results)
    split_order = [s for s in ("test", "dev", "train", "silver") if s in by_split]
    split_order += [s for s in sorted(by_split) if s not in split_order]

    A("## 2. Results per split")
    A("")
    for mode in ("exact", "overlap"):
        A(f"### {mode.upper()}")
        A("")
        rows = []
        for sp in split_order:
            docs = by_split[sp]
            scores = [d.exact if mode == "exact" else d.overlap for d in docs]
            ag = _macro_aggregate(scores)
            n_docs = len(docs)
            n_gold = sum(d.n_gold_mentions for d in docs)
            n_sys = sum(d.n_sys_mentions for d in docs)
            n_matched = sum(d.n_matched for d in docs)
            rows.append([sp, n_docs, n_gold, n_sys, n_matched,
                         f"{ag['MUC']['f1']*100:.2f}",
                         f"{ag['B3']['f1']*100:.2f}",
                         f"{ag['CEAF_e']['f1']*100:.2f}",
                         f"{ag['LEA']['f1']*100:.2f}",
                         f"{ag['CoNLL']['f1']*100:.2f}"])
        A(md_table(["split", "docs", "gold mentions", "sys mentions", "matched",
                    "MUC F1 %", "B3 F1 %", "CEAF_e F1 %", "LEA F1 %",
                    "CoNLL F1 %"], rows))
        A("")

    A("## 3. Per-document breakdown (exact match)")
    A("")
    rows = []
    for doc_id, d in sorted(aggregate["per_doc_exact"].items(),
                            key=lambda kv: (kv[1]["split"], kv[0])):
        s = d["scores"]
        conll = s.get("CoNLL", {}).get("f1", 0.0)
        rows.append([doc_id, d["split"], d["n_gold_mentions"], d["n_sys_mentions"],
                     d["n_matched"],
                     f"{s['MUC']['f1']*100:.2f}",
                     f"{s['B3']['f1']*100:.2f}",
                     f"{s['CEAF_e']['f1']*100:.2f}",
                     f"{s['LEA']['f1']*100:.2f}",
                     f"{conll*100:.2f}"])
    A(md_table(["document", "split", "gold", "system", "matched",
                "MUC F1 %", "B3 F1 %", "CEAF_e F1 %", "LEA F1 %", "CoNLL F1 %"], rows))
    A("")

    A("## 4. Notes")
    A("")
    A("* Reflex was scored as exported; the only post-processing is anchor "
      "merging (consecutive same-target anchors within the merge gap become one "
      "mention).")
    A("* The `drop` unmatched policy means mentions found by only one side are "
      "excluded, so numbers reflect the coreference step in isolation, not "
      "end-to-end extraction+coref.")
    A("* Reflex never links secondary sources, so those clusters are empty on "
      "the system side by construction.")
    A("")
    return "\n".join(lines)


def build_resolution_report(args, pair_results, aggregate, candidate_pool_name) -> str:
    lines: List[str] = []
    A = lines.append

    A("# Reflex (CanLII citator) — URI linking (resolution) evaluation")
    A("")
    A(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} — "
      f"`python scripts/reflex/evaluate_reflex_coref_resolution.py`*")
    A("")
    A("**Scope**: URI linking (stage 3). Each reflex anchor's "
      "``concatenated-id`` is mapped to a URI via the candidate pool, then the "
      "predicted URI is compared against the gold ``uri`` attribute.  Only "
      "gold spans carrying a real uri value (not ``uri=\"none\"``, not absent) "
      "are counted as instances.")
    A("")
    A(f"**Candidate pool**: `{candidate_pool_name}`")
    A(f"**Merge gap**: `{args.gap}` normalised characters")
    A("")
    A(f"**Documents**: {aggregate['n_documents']} reflex export(s) paired with gold.")
    A("")
    A(f"**URI lookup**: {aggregate['lookup']['mapped']} of "
      f"{aggregate['lookup']['total']} concatenated-ids mapped to a URI "
      f"({aggregate['lookup']['unmapped']} unmapped).")
    A("")

    # Overall
    A("## 1. Overall results (micro over documents)")
    A("")
    for mode in ("exact", "overlap"):
        title = ("EXACT span match" if mode == "exact" else "OVERLAP span match")
        ov = aggregate[mode]["overall"]
        A(f"### {mode.upper()} — {title}")
        A("")
        rows = [
            ["TP (correct uri)", ov["tp"]],
            ["FN — wrong uri (matched)", ov["fn_wrong"]],
            ["FN — no uri produced", ov["fn_missing_sys_uri"]],
            ["FN — mention never extracted", ov["fn_missed_extraction"]],
            ["FP — spurious span w/ uri", ov["fp_spurious"]],
            ["Skipped (gold uri=none)", ov["skipped_gold_none"]],
        ]
        A(md_table(["Measure", "Count"], rows))
        A("")
        rows = [
            ["Precision %", f"{ov['precision']*100:.2f}"],
            ["Recall %", f"{ov['recall']*100:.2f}"],
            ["F1 %", f"{ov['f1']*100:.2f}"],
            ["Resolver-only accuracy % (Level-1-matched spans only)",
             f"{ov['accuracy']*100:.2f}"],
            ["Matched real-uri instances", ov["matched_total"]],
        ]
        A(md_table(["Measure", "Value"], rows))
        A("")

    # Per split
    by_split = group_by_split(pair_results)
    split_order = [s for s in ("test", "dev", "train", "silver") if s in by_split]
    split_order += [s for s in sorted(by_split) if s not in split_order]

    A("## 2. Results per split (overlap match)")
    A("")
    rows = []
    for sp in split_order:
        docs = [d for d in pair_results if d.split == sp]
        acc = defaultdict(int)
        for d in docs:
            ov = d.overlap["overall"]
            for k in ("tp", "fn_wrong", "fn_missing_sys_uri", "fn_missed_extraction",
                      "fp_spurious", "skipped_gold_none", "matched_spans",
                      "matched_total"):
                acc[k] += ov.get(k, 0)

        fn_total = acc["fn_wrong"] + acc["fn_missing_sys_uri"] + acc["fn_missed_extraction"]
        fp_total = acc["fn_wrong"] + acc["fp_spurious"]
        p, r, f = _prf1(acc["tp"], fn_total, fp_total)
        rows.append([sp, len(docs), acc["tp"], fn_total, fp_total,
                     f"{p*100:.2f}", f"{r*100:.2f}", f"{f*100:.2f}"])
    A(md_table(["split", "docs", "TP", "FN", "FP", "Precision %", "Recall %", "F1 %"], rows))
    A("")

    # Per label
    A("## 3. Per-label breakdown (exact match, micro)")
    A("")
    per_label = aggregate["exact"]["per_label"]
    if per_label:
        rows = []
        for label in sorted(per_label):
            m = per_label[label]
            rows.append([label, m["tp"], m["fn"], m["fp"],
                         f"{m['precision']*100:.2f}",
                         f"{m['recall']*100:.2f}",
                         f"{m['f1']*100:.2f}"])
        A(md_table(["label", "TP", "FN", "FP", "Precision %", "Recall %", "F1 %"], rows))
        A("")

    # Per category
    per_cat = aggregate["exact"]["per_category"]
    if per_cat:
        A("### Per-category (exact match, micro)")
        A("")
        rows = []
        for cat in sorted(per_cat):
            m = per_cat[cat]
            rows.append([cat, m["tp"], m["fn"], m["fp"],
                         f"{m['precision']*100:.2f}",
                         f"{m['recall']*100:.2f}",
                         f"{m['f1']*100:.2f}"])
        A(md_table(["category", "TP", "FN", "FP", "Precision %", "Recall %", "F1 %"], rows))
        A("")

    # Per document
    A("## 4. Per-document breakdown (exact match)")
    A("")
    rows = []
    for doc_id, d in sorted(aggregate["per_doc"].items(),
                            key=lambda kv: (kv[1]["split"], kv[0])):
        ov = d["exact"]["overall"]
        rows.append([doc_id, d["split"],
                     f"{ov['precision']*100:.2f}",
                     f"{ov['recall']*100:.2f}",
                     f"{ov['f1']*100:.2f}",
                     f"{ov['accuracy']*100:.2f}",
                     ov["tp"], ov["fn_wrong"], ov["fn_missing_sys_uri"],
                     ov["fn_missed_extraction"], ov["fp_spurious"]])
    A(md_table(["document", "split", "Precision %", "Recall %", "F1 %",
                "Accuracy %", "TP", "FN-wrong", "FN-no-uri", "FN-missed",
                "FP-spurious"], rows))
    A("")

    A("## 5. URI lookup diagnostics")
    A("")
    A(f"* Total concatenated-ids looked up: {aggregate['lookup']['total']}")
    A(f"* Mapped to a URI: {aggregate['lookup']['mapped']}")
    A(f"* Unmapped (no candidate-pool entry): {aggregate['lookup']['unmapped']}")
    A("")
    A("* Unmapped concatenated-ids produce system spans with no uri, which "
      "count as false negatives (no uri produced) when the corresponding gold "
      "span has a real uri.")
    A("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def build_reflex_mentions(reflex_doc: C.ReflexDoc, gap: int) -> List[Mention]:
    """Turn merged reflex anchors into Mention objects (reflex coordinates)."""
    mentions = []
    for m in reflex_doc.mentions(gap):
        label = m.label
        if label is None:
            continue
        mentions.append(Mention(
            key="",  # filled after alignment in evaluate_coref_pair
            docid=m.href or "",   # concatenated-id is stored in href
            label=label,
            start=m.start,
            end=m.end,
            text=m.text,
        ))
    return mentions


def run(args) -> dict:
    id_to_uri = load_candidate_pool(
        GOLD_POOL if args.candidate_pool == "gold" else FULL_POOL
    )

    pairs = C.pair_documents(min_ratio=args.min_ratio, verbose=False)
    usable = [p for p in pairs if p.how in ("exact", "fuzzy")]
    if args.splits:
        want = {s.strip() for s in args.splits.split(",") if s.strip()}
        usable = [p for p in usable if p.split in want]

    if not usable:
        print("No usable reflex/gold pairs.")
        return {}

    print(f"Paired {len(usable)} reflex document(s) with gold")
    print(f"Candidate pool: {'gold_candidate_pool.csv' if args.candidate_pool == 'gold' else 'full_candidate_pool.csv'}")
    print(f"Merge gap: {args.gap}")

    coref_results: List[CorefPairResult] = []
    resolution_results: List[ResolutionPairResult] = []

    for i, p in enumerate(usable, 1):
        print(f"\n[{i}/{len(usable)}] {p.reflex_path.name} -> {p.doc_id} [{p.split}]")
        gold_text, gold_mentions = parse_gold_with_docid(p.gold_path)
        reflex_doc = C.parse_reflex(p.reflex_path)
        reflex_mentions = build_reflex_mentions(reflex_doc, args.gap)
        alignment = C.Alignment(reflex_doc.text, gold_text)

        if not args.skip_coref:
            cr = evaluate_coref_pair(gold_mentions, reflex_mentions, alignment,
                                     unmatched="drop")
            cr.doc_id = p.doc_id
            cr.split = p.split
            coref_results.append(cr)
            print(f"  coref exact CoNLL={cr.exact['CoNLL']['f1']*100:.2f}  "
                  f"overlap CoNLL={cr.overlap['CoNLL']['f1']*100:.2f}")

        if not args.skip_resolution:
            rr = evaluate_resolution_pair(gold_mentions, reflex_mentions, alignment,
                                          id_to_uri, gold_text)
            rr.doc_id = p.doc_id
            rr.split = p.split
            resolution_results.append(rr)
            print(f"  resolution exact F1={rr.exact['overall']['f1']*100:.2f}  "
                  f"overlap F1={rr.overlap['overall']['f1']*100:.2f}")

    # ---- aggregate ----
    coref_agg = aggregate_coref(coref_results) if not args.skip_coref else {}
    resolution_agg = aggregate_resolution(resolution_results) if not args.skip_resolution else {}

    # ---- reports ----
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "candidate_pool": args.candidate_pool,
        "merge_gap": args.gap,
        "splits": args.splits,
        "pairs": [{"reflex": p.reflex_path.name, "doc_id": p.doc_id, "split": p.split,
                   "match": p.how, "similarity": round(p.text_ratio, 4)}
                  for p in usable],
    }

    if not args.skip_coref:
        coref_report = build_coref_report(args, coref_results, coref_agg,
                                          args.candidate_pool)
        (out_dir / "reflex_coref_report.md").write_text(coref_report, encoding="utf-8")
        print(f"\nCoref report: {out_dir / 'reflex_coref_report.md'}")

        coref_payload = {
            "generated": payload["generated"],
            "candidate_pool": args.candidate_pool,
            "merge_gap": args.gap,
            "aggregate": coref_agg,
            "per_doc": {pr.doc_id: {
                "split": pr.split,
                "exact": pr.exact,
                "overlap": pr.overlap,
                "stats": pr.stats,
                "stats_overlap": pr.stats_overlap,
                "n_gold_mentions": pr.n_gold_mentions,
                "n_sys_mentions": pr.n_sys_mentions,
                "n_matched": pr.n_matched,
            } for pr in coref_results},
        }
        (out_dir / "reflex_coref_results.json").write_text(
            json.dumps(_jsonable(coref_payload), indent=2, default=dict), encoding="utf-8")
        print(f"Coref JSON  : {out_dir / 'reflex_coref_results.json'}")

    if not args.skip_resolution:
        res_report = build_resolution_report(args, resolution_results,
                                             resolution_agg, args.candidate_pool)
        (out_dir / "reflex_resolution_report.md").write_text(res_report, encoding="utf-8")
        print(f"\nResolution report: {out_dir / 'reflex_resolution_report.md'}")

        res_payload = {
            "generated": payload["generated"],
            "candidate_pool": args.candidate_pool,
            "merge_gap": args.gap,
            "aggregate": resolution_agg,
            "per_doc": {pr.doc_id: {
                "split": pr.split,
                "exact": pr.exact,
                "overlap": pr.overlap,
                "uri_lookup": {
                    "mapped": pr.uri_lookup_mapped,
                    "unmapped": pr.uri_lookup_unmapped,
                    "total": pr.uri_lookup_total,
                },
            } for pr in resolution_results},
        }
        (out_dir / "reflex_resolution_results.json").write_text(
            json.dumps(_jsonable(res_payload), indent=2, default=dict), encoding="utf-8")
        print(f"Resolution JSON: {out_dir / 'reflex_resolution_results.json'}")

    return payload


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate Reflex coreference resolution and URI linking "
                    "using src/evaluation primitives.")
    ap.add_argument("--splits", default=None,
                    help="comma list of splits to score (default: every split "
                         "with a reflex export)")
    ap.add_argument("--gap", type=int, default=DEFAULT_GAP,
                    help=f"anchor merge gap in normalised characters (default: {DEFAULT_GAP})")
    ap.add_argument("--candidate-pool", choices=["full", "gold"], default="full",
                    help="candidate pool for URI lookup (default: full = "
                         "full_candidate_pool.csv; recommended for silver data)")
    ap.add_argument("--min-ratio", type=float, default=0.85,
                    help="content-token similarity gate for fuzzy id pairing")
    ap.add_argument("--skip-coref", action="store_true",
                    help="skip coreference evaluation")
    ap.add_argument("--skip-resolution", action="store_true",
                    help="skip URI resolution evaluation")
    ap.add_argument("--out", default=str(REPORT_DIR),
                    help=f"output folder (default: {REPORT_DIR})")
    args = ap.parse_args()

    if args.skip_coref and args.skip_resolution:
        print("Nothing to evaluate (both stages skipped).")
        return

    print("=" * 74)
    print("REFLEX COREFERENCE + URI LINKING EVALUATION")
    print("=" * 74)
    run(args)


if __name__ == "__main__":
    main()
