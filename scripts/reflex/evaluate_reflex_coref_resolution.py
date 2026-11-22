#!/usr/bin/env python
"""
Evaluate the Reflex (CanLII citator) exports in data/reflex/ at the
coreference-resolution and URI-linking (resolution) stages, using the
evaluation primitives in src/evaluation/.

Reports (markdown + JSON) are written to scripts/reflex/report/.

Coreference
-----------
Reflex anchors sharing the same `concatenated-id` form one cluster; gold
mentions sharing the same `docid` form one cluster.  Spans are aligned into
gold coordinates (token-diff Alignment) before clustering.  Only the grouping
by docid is scored — the docid strings themselves are never compared.

Two unmatched policies:
  * singleton (default, end-to-end / twinless): mentions found by only one
    side become singletons on the other side, so extraction errors cost P/R.
    This matches how the LLM/pipeline systems are scored — directly comparable.
  * drop (coref step only): isolates coreference from extraction errors
    (upper bound, not comparable with pipeline systems).

Metrics (MUC / B3 / CEAF_e / LEA + CoNLL) are implemented inline, equivalent
to the `coval` library used by src/evaluation/evaluation_coref.py (coval
cannot be installed on this Python 3.15 interpreter — no pyyaml wheel).

Resolution
----------
Each concatenated-id is mapped to a URI via the candidate pool
(data/gold_candidate_pool.csv by default — recommended for fair comparison).
The predicted URI is compared against the gold `uri` attribute with
src/evaluation/evaluate_resolution.py (compute_uri_metrics exact + overlap).

Only gold spans carrying a real uri value are counted as instances.

Usage (from repo root):

    # test + dev + train, gold pool, end-to-end (singleton) — comparable
    python scripts/reflex/evaluate_reflex_coref_resolution.py \
        --splits test,dev,train --candidate-pool gold

    # coref step only (drop) — upper bound, not comparable with pipelines
    python scripts/reflex/evaluate_reflex_coref_resolution.py \
        --splits test,dev,train --candidate-pool gold --unmatched drop

    # skip one stage
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
# Project-root path setup
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
for p in (str(SCRIPT_DIR), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from bs4 import BeautifulSoup, Tag  # noqa: E402

import common as C  # noqa: E402  (scripts/reflex/common.py)

# ---------------------------------------------------------------------------
# Load evaluation_resolution + evaluation_l1_util by path (src/evaluation/)
# ---------------------------------------------------------------------------

import importlib.util

_RES_PATH = PROJECT_ROOT / "src" / "evaluation" / "evaluate_resolution.py"
_L1_PATH = PROJECT_ROOT / "src" / "evaluation" / "evaluation_l1_util.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_evaluate_resolution = _load_module("evaluate_resolution_util_reflex", _RES_PATH)
_evaluation_l1 = _load_module("evaluation_l1_util_reflex", _L1_PATH)

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

COREF_METRICS = ("MUC", "B3", "CEAF_e", "LEA")

# Per-type evaluation groups: a group pools one or more base labelnames into
# one cluster set (same semantics as src/evaluation/evaluate_coref_per_type.py).
TYPE_GROUPS: Dict[str, Tuple[str, ...]] = {
    "decision": ("decision",),
    "legislation": ("legislation",),
    "secondary sources": ("secondary sources",),
}
# Types shown in the markdown reports (secondary sources are reported in JSON
# only: Reflex never links them, so the row would be 0 by construction).
REPORT_TYPES = ("decision", "legislation")
TYPE_DISPLAY = {
    "decision": "Decision",
    "legislation": "Legislation",
    "secondary sources": "Secondary",
}

# ---------------------------------------------------------------------------
# Mention dataclass (mirrors evaluation_coref.py's Mention for span alignment)
# ---------------------------------------------------------------------------


@dataclass
class Mention:
    key: str      # shared key used as the mention id after alignment
    docid: str    # cluster label (concatenated-id for reflex)
    label: str    # labelname attribute (e.g. "decision")
    start: int    # offsets in whitespace-normalised plain text
    end: int
    text: str
    href: str = ""   # full href URL (used for base-href merging)


# ---------------------------------------------------------------------------
# Coreference metrics (inline, equivalent to coval used by evaluation_coref.py)
# ---------------------------------------------------------------------------


def _m2c_index(clusters: List[List[str]]) -> Dict[str, int]:
    m2c: Dict[str, int] = {}
    for i, cluster in enumerate(clusters):
        for m in cluster:
            m2c[m] = i
    return m2c


def muc_raw(pred_clusters: List[List[str]],
            gold_clusters: List[List[str]]) -> Tuple[int, int, int, int]:
    gold_m2g = _m2c_index(gold_clusters)
    pred_m2p = _m2c_index(pred_clusters)
    p_num = p_den = 0
    for cluster in pred_clusters:
        if len(cluster) <= 1:
            continue
        gold_inds = {gold_m2g[m] for m in cluster if m in gold_m2g}
        p_num += len(cluster) - len(gold_inds)
        p_den += len(cluster) - 1
    r_num = r_den = 0
    for cluster in gold_clusters:
        if len(cluster) <= 1:
            continue
        pred_inds = {pred_m2p[m] for m in cluster if m in pred_m2p}
        r_num += len(cluster) - len(pred_inds)
        r_den += len(cluster) - 1
    return p_num, p_den, r_num, r_den


def b_cubed_raw(pred_clusters: List[List[str]],
                gold_clusters: List[List[str]]) -> Tuple[float, float, float, float]:
    gold_m2g = _m2c_index(gold_clusters)
    pred_m2p = _m2c_index(pred_clusters)
    p_num = p_den = 0.0
    for cluster in pred_clusters:
        cluster_set = set(cluster)
        for m in cluster:
            if m in gold_m2g:
                gold_cluster = gold_clusters[gold_m2g[m]]
                intersection = len(cluster_set & set(gold_cluster))
                p_num += intersection / len(cluster)
                p_den += 1
    r_num = r_den = 0.0
    for cluster in gold_clusters:
        cluster_set = set(cluster)
        for m in cluster:
            if m in pred_m2p:
                pred_cluster = pred_clusters[pred_m2p[m]]
                intersection = len(cluster_set & set(pred_cluster))
                r_num += intersection / len(cluster)
                r_den += 1
    return p_num, p_den, r_num, r_den


def _hungarian_max(matrix: List[List[float]]) -> List[Tuple[int, int]]:
    n_rows = len(matrix)
    n_cols = len(matrix[0]) if n_rows else 0
    if n_rows == 0 or n_cols == 0:
        return []
    cost = [[-matrix[i][j] for j in range(n_cols)] for i in range(n_rows)]
    n = max(n_rows, n_cols)
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)
    way = [0] * (n + 1)
    c = [[0.0] * (n + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, n + 1):
            if i <= n_rows and j <= n_cols:
                c[i][j] = cost[i - 1][j - 1]
            else:
                c[i][j] = 0.0
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [float("inf")] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = float("inf")
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = c[i0][j] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    assignment: List[Tuple[int, int]] = []
    for j in range(1, n + 1):
        i = p[j]
        if i != 0 and i <= n_rows and j <= n_cols:
            assignment.append((i - 1, j - 1))
    return assignment


_SCIPY_LSA_CHECKED = False
_SCIPY_LSA_CACHE = None


def _scipy_lsa():
    """Return (numpy, linear_sum_assignment) once, or None if unavailable.

    The import attempt is cached: on some environments scipy/numpy fail to load
    (blocked DLLs), and retrying the failing import per document is very slow.
    """
    global _SCIPY_LSA_CHECKED, _SCIPY_LSA_CACHE
    if not _SCIPY_LSA_CHECKED:
        _SCIPY_LSA_CHECKED = True
        try:
            import numpy as np
            from scipy.optimize import linear_sum_assignment

            linear_sum_assignment(np.zeros((1, 1)))
            _SCIPY_LSA_CACHE = (np, linear_sum_assignment)
        except Exception:
            _SCIPY_LSA_CACHE = None
    return _SCIPY_LSA_CACHE


def ceafe_raw(pred_clusters: List[List[str]],
              gold_clusters: List[List[str]]) -> Tuple[float, float, float, float]:
    """CEAFe raw counts, exactly as `coval.eval.evaluator.ceafe`:
    phi4 similarity (2*|A n B| / (|A|+|B|)) on the full cluster grid, best
    one-to-one assignment (Hungarian), denominators = number of clusters.

    Unlike a mention-weighted CEAF variant, coval keeps singleton clusters in
    the matrix and divides by the cluster counts, which is what
    `src/evaluation/evaluate_coref.py` reports.
    """
    n_pred = len(pred_clusters)
    n_gold = len(gold_clusters)
    if n_pred == 0 or n_gold == 0:
        return 0.0, n_pred, 0.0, n_gold

    # Both sides partition the mention space, so every shared mention belongs
    # to exactly one (gold cluster, pred cluster) cell: collect the cells.
    gold_m2g = _m2c_index(gold_clusters)
    pred_m2p = _m2c_index(pred_clusters)
    overlap: Dict[Tuple[int, int], int] = {}
    for mention, gi in gold_m2g.items():
        pj = pred_m2p.get(mention)
        if pj is not None:
            overlap[(gi, pj)] = overlap.get((gi, pj), 0) + 1

    gold_len = [len(c) for c in gold_clusters]
    pred_len = [len(c) for c in pred_clusters]

    # Only clusters sharing at least one mention can score > 0, so the rows and
    # columns outside this support can never change the optimal sum (they are
    # all-zero).  Restricting to the support keeps the dense matrix small even
    # when thousands of singleton clusters are present.
    gold_rows = sorted({i for i, _ in overlap})
    pred_cols = sorted({j for _, j in overlap})
    n_support_gold = len(gold_rows)
    n_support_pred = len(pred_cols)
    row_pos = {i: k for k, i in enumerate(gold_rows)}
    col_pos = {j: k for k, j in enumerate(pred_cols)}

    lsa = _scipy_lsa()
    if lsa is not None:
        np, linear_sum_assignment = lsa
        scores = np.zeros((n_support_gold, n_support_pred), dtype=float)
        for (i, j), inter in overlap.items():
            scores[row_pos[i], col_pos[j]] = 2.0 * inter / (gold_len[i] + pred_len[j])
        row_ind, col_ind = linear_sum_assignment(-scores)
        similarity = float(scores[row_ind, col_ind].sum())
        return similarity, n_pred, similarity, n_gold

    # Pure-Python fallback.  Zero-weight cells are dropped, so the bipartite
    # graph splits into small connected components; the optimal assignment
    # decomposes over those components, which keeps the Hungarian cubic cost
    # per tiny block instead of one huge matrix.
    edges = [
        (row_pos[i], col_pos[j], 2.0 * inter / (gold_len[i] + pred_len[j]))
        for (i, j), inter in overlap.items()
    ]
    parent = list(range(n_support_gold + n_support_pred))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r, c, _w in edges:
        ra, rb = find(r), find(n_support_gold + c)
        if ra != rb:
            parent[rb] = ra

    blocks: Dict[int, Tuple[Dict[int, int], Dict[int, int], List[Tuple[int, int, float]]]] = {}
    for r, c, w in edges:
        root = find(r)
        if root not in blocks:
            blocks[root] = ({}, {}, [])
        rows_idx, cols_idx, block_edges = blocks[root]
        if r not in rows_idx:
            rows_idx[r] = len(rows_idx)
        if c not in cols_idx:
            cols_idx[c] = len(cols_idx)
        block_edges.append((r, c, w))

    similarity = 0.0
    for rows_idx, cols_idx, block_edges in blocks.values():
        matrix = [[0.0] * len(cols_idx) for _ in range(len(rows_idx))]
        for r, c, w in block_edges:
            matrix[rows_idx[r]][cols_idx[c]] = w
        similarity += sum(matrix[i][j] for i, j in _hungarian_max(matrix))
    return similarity, n_pred, similarity, n_gold


def _lea_one(input_clusters: List[List[str]],
             output_clusters: List[List[str]],
             mention_to_out: Dict[str, int]) -> Tuple[float, float]:
    """One direction of coval's LEA (Moosavi & Strube): link-based, with the
    singleton convention (all_links = common_links = 1 iff the single mention
    lands in a singleton output cluster)."""
    num = den = 0.0
    for cluster in input_clusters:
        if len(cluster) == 1:
            all_links = 1
            m = cluster[0]
            if m in mention_to_out and len(output_clusters[mention_to_out[m]]) == 1:
                common_links = 1
            else:
                common_links = 0
        else:
            all_links = len(cluster) * (len(cluster) - 1) / 2.0
            common_links = 0
            for i, m in enumerate(cluster):
                if m not in mention_to_out:
                    continue
                for m2 in cluster[i + 1:]:
                    if m2 in mention_to_out and mention_to_out[m] == mention_to_out[m2]:
                        common_links += 1
        num += len(cluster) * common_links / float(all_links)
        den += len(cluster)
    return num, den


def lea_raw(pred_clusters: List[List[str]],
            gold_clusters: List[List[str]]) -> Tuple[float, float, float, float]:
    """LEA raw counts, exactly as `coval.eval.evaluator.lea`."""
    gold_m2g = _m2c_index(gold_clusters)
    pred_m2p = _m2c_index(pred_clusters)
    p_num, p_den = _lea_one(pred_clusters, gold_clusters, gold_m2g)
    r_num, r_den = _lea_one(gold_clusters, pred_clusters, pred_m2p)
    return p_num, p_den, r_num, r_den


def _prf1_from_raw(p_num, p_den, r_num, r_den):
    precision = p_num / p_den if p_den else 0.0
    recall = r_num / r_den if r_den else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def evaluate_coref_metrics(pred_clusters: List[List[str]],
                           gold_clusters: List[List[str]]) -> dict:
    raw_muc = muc_raw(pred_clusters, gold_clusters)
    raw_b3 = b_cubed_raw(pred_clusters, gold_clusters)
    raw_ceafe = ceafe_raw(pred_clusters, gold_clusters)
    raw_lea = lea_raw(pred_clusters, gold_clusters)
    scores = {}
    for name, raw in (("MUC", raw_muc), ("B3", raw_b3),
                      ("CEAF_e", raw_ceafe), ("LEA", raw_lea)):
        p, r, f = _prf1_from_raw(*raw)
        scores[name] = {"precision": p, "recall": r, "f1": f}
    conll_f1 = (scores["MUC"]["f1"] + scores["B3"]["f1"] + scores["CEAF_e"]["f1"]) / 3
    scores["CoNLL"] = {"f1": conll_f1, "precision": None, "recall": None}
    return scores


# ---------------------------------------------------------------------------
# Span alignment (exact + overlap) — mirrors evaluation_coref.py
# ---------------------------------------------------------------------------


def _mention_overlap_len(a: Mention, b: Mention) -> int:
    if a.label != b.label:
        return 0
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def _group_by_docid(mentions: List[Mention]) -> List[List[str]]:
    groups: Dict[str, List[str]] = defaultdict(list)
    clusters: List[List[str]] = []
    for m in mentions:
        if m.docid:
            groups[m.docid].append(m.key)
        else:
            clusters.append([m.key])
    return list(groups.values()) + clusters


def align_and_build_clusters(
    gold_mentions: List[Mention],
    system_mentions: List[Mention],
    unmatched: str = "singleton",
) -> Tuple[List[List[str]], List[List[str]], Dict[str, int]]:
    """unmatched="singleton" (end-to-end, twinless) or "drop" (coref step only)."""
    gold_keys = {m.key for m in gold_mentions}
    sys_keys = {m.key for m in system_mentions}
    matched = gold_keys & sys_keys
    gold_only = gold_keys - sys_keys
    sys_only = sys_keys - gold_keys
    stats = {
        "gold": len(gold_keys),
        "system": len(sys_keys),
        "matched": len(matched),
        "gold_only": len(gold_only),
        "system_only": len(sys_only),
    }
    if unmatched == "drop":
        gold_clusters = _group_by_docid([m for m in gold_mentions if m.key in matched])
        system_clusters = _group_by_docid([m for m in system_mentions if m.key in matched])
    else:
        gold_clusters = _group_by_docid(gold_mentions) + [[k] for k in sorted(sys_only)]
        system_clusters = _group_by_docid(system_mentions) + [[k] for k in sorted(gold_only)]
    return gold_clusters, system_clusters, stats


def align_and_build_clusters_overlap(
    gold_mentions: List[Mention],
    system_mentions: List[Mention],
    unmatched: str = "singleton",
) -> Tuple[List[List[str]], List[List[str]], Dict[str, int]]:
    used = [False] * len(system_mentions)
    sys_to_shared: Dict[int, str] = {}
    matched_gold_keys = set()
    n_pairs = 0
    for g in gold_mentions:
        best_j = -1
        best_ov = 0
        for j, s in enumerate(system_mentions):
            if used[j]:
                continue
            ov = _mention_overlap_len(g, s)
            if ov > best_ov:
                best_ov = ov
                best_j = j
        if best_j != -1:
            used[best_j] = True
            sys_to_shared[best_j] = g.key
            matched_gold_keys.add(g.key)
            n_pairs += 1
    sys_remapped = [
        Mention(
            key=sys_to_shared.get(j, s.key),
            docid=s.docid,
            label=s.label,
            start=s.start,
            end=s.end,
            text=s.text,
            href=s.href,
        )
        for j, s in enumerate(system_mentions)
    ]
    gold_keys = {m.key for m in gold_mentions}
    sys_keys = {m.key for m in system_mentions}
    gold_only = {k for k in gold_keys if k not in matched_gold_keys}
    sys_only = {s.key for j, s in enumerate(system_mentions) if not used[j]}
    stats = {
        "gold": len(gold_keys),
        "system": len(sys_keys),
        "matched": n_pairs,
        "gold_only": len(gold_only),
        "system_only": len(sys_only),
    }
    if unmatched == "drop":
        gold_clusters = _group_by_docid([m for m in gold_mentions if m.key in matched_gold_keys])
        system_clusters = _group_by_docid([m for m in sys_remapped if m.key in matched_gold_keys])
    else:
        gold_clusters = _group_by_docid(gold_mentions) + [[k] for k in sorted(sys_only)]
        system_clusters = _group_by_docid(sys_remapped) + [[k] for k in sorted(gold_only)]
    return gold_clusters, system_clusters, stats


# ---------------------------------------------------------------------------
# Candidate pool: id -> uri
# ---------------------------------------------------------------------------


def load_candidate_pool(path: Path) -> Dict[str, str]:
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


def _gold_parent_label(tag: Tag) -> Optional[str]:
    """Parent-level gold mention label (`manual_label` / `auto_label` tags)."""
    if (tag.get("parent") or "").strip():
        return None
    label = (tag.get("labelname") or "").strip().lower()
    return label if label in PARENT_LABELS else None


def parse_gold_with_docid(path: Path) -> Tuple[str, List[GoldMention]]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    body = soup.find("body")
    if body is None:
        return "", []
    # Offsets must be in the same whitespace-normalised coordinates as the text
    # that `Alignment` produces on the reflexive side, otherwise every mapped
    # span is compared against the wrong gold offsets.  `normalised_text_with_tags`
    # is the single correct implementation of that mapping and is what
    # `parse_gold()` uses for the extraction scorer.
    text, tags = C.normalised_text_with_tags(
        body, ("manual_label", "auto_label"), _gold_parent_label)
    mentions: List[GoldMention] = []
    for idx in sorted(tags):
        label, s, e, attrs = tags[idx]
        mention_text = text[s:e].strip()
        if not mention_text:
            continue
        mentions.append(
            GoldMention(
                label=label,
                text=mention_text,
                start=s,
                end=e,
                docid=(attrs.get("docid") or "").strip(),
                uri=(attrs.get("uri") or "").strip() or None,
            )
        )
    return text, mentions


# ---------------------------------------------------------------------------
# Reflex parsing with concatenated-id extraction
# ---------------------------------------------------------------------------


def _reflex_label_of(tag: Tag) -> Optional[str]:
    if tag.name != "a":
        return None
    return C.classify_href(tag.get("href", ""))


def parse_reflex_with_cid(path: Path) -> Tuple[str, List[Tuple[str, str, str, str, int, int]]]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    body = soup.find("body")
    if body is None:
        return "", []
    text, tags = C.normalised_text_with_tags(body, ("a",), _reflex_label_of)
    anchors: List[Tuple[str, str, str, str, int, int]] = []
    for idx in sorted(tags):
        label, s, e, attrs = tags[idx]
        if not text[s:e].strip():
            continue
        concatenated_id = attrs.get("concatenated-id", "")
        href = attrs.get("href", "")
        anchors.append((label, concatenated_id, href, text[s:e], s, e))
    return text, anchors


def _base_href(href: str) -> str:
    return href.split("#", 1)[0]


def merge_anchors_with_cid(
    anchors: List[Tuple[str, str, str, str, int, int]],
    merge_gap: int,
    doc_text: str,
) -> List[Mention]:
    merged: List[Mention] = []
    for label, cid, href, atext, s, e in anchors:
        if merged:
            last = merged[-1]
            gap_chars = s - last.end
            if (last.label == label
                    and _base_href(last.href) == _base_href(href)
                    and 0 <= gap_chars <= merge_gap):
                last.end = e
                last.text = " ".join(doc_text[last.start:e].split())
                continue
        merged.append(Mention(
            key="",
            docid=cid,
            label=label,
            start=s,
            end=e,
            text=atext,
            href=href,
        ))
    return merged


def build_reflex_mentions(path: Path, gap: int) -> Tuple[str, List[Mention]]:
    text, anchors = parse_reflex_with_cid(path)
    mentions = merge_anchors_with_cid(anchors, gap, text)
    return text, mentions


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
    per_type: Dict[str, dict] = field(default_factory=dict)


def _build_aligned_mentions(
    gold_mentions: List[GoldMention],
    reflex_mentions: List[Mention],
    alignment: C.Alignment,
) -> Tuple[List[Mention], List[Mention]]:
    """Express gold and reflex mentions in gold coordinates (shared span keys)."""
    len_g = alignment.len_g
    gm: List[Mention] = []
    for g in gold_mentions:
        gm.append(Mention(
            key=f"{g.label}|{g.start}|{g.end}",
            docid=g.docid,
            label=g.label,
            start=g.start,
            end=g.end,
            text=g.text,
        ))
    sm: List[Mention] = []
    for m in reflex_mentions:
        gs, ge = alignment.map_span(m.start, m.end)
        gs = min(max(0, gs), max(0, len_g - 1))
        ge = max(gs + 1, min(ge, len_g))
        sm.append(Mention(
            key=f"{m.label}|{gs}|{ge}",
            docid=m.docid,
            label=m.label,
            start=gs,
            end=ge,
            text=m.text,
            href=m.href,
        ))
    return gm, sm


def _filter_type(mentions: List[Mention], labels: Tuple[str, ...]) -> List[Mention]:
    label_set = set(labels)
    return [m for m in mentions if m.label in label_set]


def _score_coref_mentions(
    gold_mentions: List[Mention],
    system_mentions: List[Mention],
    unmatched: str,
) -> dict:
    """Exact + overlap coref scores and alignment stats for one mention set."""
    g_clusters, s_clusters, stats = align_and_build_clusters(
        gold_mentions, system_mentions, unmatched=unmatched
    )
    exact = evaluate_coref_metrics(s_clusters, g_clusters)
    g_clusters_ov, s_clusters_ov, stats_ov = align_and_build_clusters_overlap(
        gold_mentions, system_mentions, unmatched=unmatched
    )
    overlap = evaluate_coref_metrics(s_clusters_ov, g_clusters_ov)
    return {
        "exact": exact,
        "overlap": overlap,
        "stats": stats,
        "stats_overlap": stats_ov,
        "n_gold": len(gold_mentions),
        "n_system": len(system_mentions),
        "n_matched": stats["matched"],
    }


def evaluate_coref_pair(
    gold_mentions: List[GoldMention],
    reflex_mentions: List[Mention],
    alignment: C.Alignment,
    unmatched: str = "singleton",
) -> CorefPairResult:
    gm, sm = _build_aligned_mentions(gold_mentions, reflex_mentions, alignment)

    overall = _score_coref_mentions(gm, sm, unmatched)
    per_type: Dict[str, dict] = {}
    for type_key, labels in TYPE_GROUPS.items():
        per_type[type_key] = _score_coref_mentions(
            _filter_type(gm, labels), _filter_type(sm, labels), unmatched
        )

    return CorefPairResult(
        doc_id="", split="", exact=overall["exact"], overlap=overall["overlap"],
        stats=overall["stats"], stats_overlap=overall["stats_overlap"],
        n_gold_mentions=len(gm), n_sys_mentions=len(sm),
        n_matched=overall["stats"]["matched"],
        per_type=per_type,
    )


# ---------------------------------------------------------------------------
# Per-pair resolution evaluation (uses src/evaluation/evaluate_resolution.py)
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
    uri_lookup_by_label: Dict[str, Dict[str, int]] = field(default_factory=dict)


def evaluate_resolution_pair(
    gold_mentions: List[GoldMention],
    reflex_mentions: List[Mention],
    alignment: C.Alignment,
    id_to_uri: Dict[str, str],
    gold_text: str,
) -> ResolutionPairResult:
    len_g = alignment.len_g
    sys_spans: List[Span] = []
    lookup_mapped = lookup_unmapped = lookup_total = 0
    lookup_by_label: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"mapped": 0, "unmapped": 0, "total": 0})
    for m in reflex_mentions:
        lookup_total += 1
        lookup_by_label[m.label]["total"] += 1
        uri = lookup_uri(m.docid, id_to_uri)
        if uri:
            lookup_mapped += 1
            lookup_by_label[m.label]["mapped"] += 1
        else:
            lookup_unmapped += 1
            lookup_by_label[m.label]["unmapped"] += 1
        gs, ge = alignment.map_span(m.start, m.end)
        gs = min(max(0, gs), max(0, len_g - 1))
        ge = max(gs + 1, min(ge, len_g))
        ctx = C._context(gold_text, gs, ge, CONTEXT_CHARS)
        sys_spans.append(Span(
            text=m.text, labelname=m.label, context_text=ctx,
            attributes={"uri": uri} if uri else {},
            start_pos=gs, end_pos=ge,
        ))
    gold_spans: List[Span] = []
    for g in gold_mentions:
        ctx = C._context(gold_text, g.start, g.end, CONTEXT_CHARS)
        gold_spans.append(Span(
            text=g.text, labelname=g.label, context_text=ctx,
            attributes={"uri": g.uri} if g.uri else {},
            start_pos=g.start, end_pos=g.end,
        ))
    exact_overall, exact_per_label, exact_per_cat = compute_uri_metrics(gold_spans, sys_spans)
    overlap_overall, overlap_per_label, overlap_per_cat = compute_uri_metrics_overlap(gold_spans, sys_spans)
    n_gold_real_uri = exact_overall.get("matched_total", 0) + exact_overall.get("fn_missed_extraction", 0)
    return ResolutionPairResult(
        doc_id="", split="",
        exact={"overall": exact_overall, "per_label": exact_per_label, "per_category": exact_per_cat},
        overlap={"overall": overlap_overall, "per_label": overlap_per_label, "per_category": overlap_per_cat},
        n_gold_real_uri=n_gold_real_uri,
        n_sys_uri=lookup_mapped,
        uri_lookup_mapped=lookup_mapped,
        uri_lookup_unmapped=lookup_unmapped,
        uri_lookup_total=lookup_total,
        uri_lookup_by_label={k: dict(v) for k, v in lookup_by_label.items()},
    )


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _macro_aggregate(scores_list: List[dict]) -> dict:
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
    total_stats: Dict[str, int] = defaultdict(int)
    total_stats_ov: Dict[str, int] = defaultdict(int)
    per_doc_exact = {}
    per_doc_overlap = {}
    for pr in pair_results:
        for k, v in pr.stats.items():
            total_stats[k] += v
        for k, v in pr.stats_overlap.items():
            total_stats_ov[k] += v
        per_doc_exact[pr.doc_id] = {
            "split": pr.split, "n_gold_mentions": pr.n_gold_mentions,
            "n_sys_mentions": pr.n_sys_mentions, "n_matched": pr.n_matched,
            "scores": pr.exact,
        }
        per_doc_overlap[pr.doc_id] = {
            "split": pr.split, "n_gold_mentions": pr.n_gold_mentions,
            "n_sys_mentions": pr.n_sys_mentions, "n_matched": pr.n_matched,
            "scores": pr.overlap,
        }
    return {
        "exact": _macro_aggregate([pr.exact for pr in pair_results]),
        "overlap": _macro_aggregate([pr.overlap for pr in pair_results]),
        "stats": dict(total_stats),
        "stats_overlap": dict(total_stats_ov),
        "per_doc_exact": per_doc_exact,
        "per_doc_overlap": per_doc_overlap,
        "n_documents": len(pair_results),
    }


def aggregate_coref_per_type(
    pair_results: List[CorefPairResult],
    types: Tuple[str, ...] = tuple(TYPE_GROUPS),
) -> Dict[str, dict]:
    """Macro-average coref scores per mention type (and pooled group)."""
    out: Dict[str, dict] = {}
    for type_key in types:
        rows = [pr.per_type[type_key] for pr in pair_results if type_key in pr.per_type]
        if not rows:
            continue
        total_stats: Dict[str, int] = defaultdict(int)
        for row in rows:
            for k, v in row["stats"].items():
                total_stats[k] += v
        out[type_key] = {
            "exact": _macro_aggregate([row["exact"] for row in rows]),
            "overlap": _macro_aggregate([row["overlap"] for row in rows]),
            "stats": dict(total_stats),
            "n_gold": sum(row["n_gold"] for row in rows),
            "n_system": sum(row["n_system"] for row in rows),
            "n_matched_exact": sum(row["stats"]["matched"] for row in rows),
            "n_matched_overlap": sum(row["stats_overlap"]["matched"] for row in rows),
            "n_documents": len(rows),
        }
    return out


def _prf1(tp: int, fn: int, fp: int) -> Tuple[float, float, float]:
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return prec, rec, f1


def aggregate_resolution(pair_results: List[ResolutionPairResult]) -> dict:
    acc_exact = defaultdict(int)
    acc_overlap = defaultdict(int)
    per_label_exact = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})
    per_label_overlap = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})
    per_cat_exact = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})
    per_cat_overlap = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})
    total_lookup_mapped = total_lookup_unmapped = total_lookup_total = 0
    lookup_by_label: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"mapped": 0, "unmapped": 0, "total": 0})
    per_doc = {}
    for pr in pair_results:
        ov_e = pr.exact["overall"]
        for k in ("tp", "fn_wrong", "fn_missing_sys_uri", "fn_missed_extraction",
                  "fp_spurious", "skipped_gold_none", "matched_spans", "matched_total"):
            acc_exact[k] += ov_e.get(k, 0)
        for label, m in pr.exact.get("per_label", {}).items():
            per_label_exact[label]["tp"] += m["tp"]
            per_label_exact[label]["fn"] += m["fn"]
            per_label_exact[label]["fp"] += m["fp"]
        for cat, m in pr.exact.get("per_category", {}).items():
            per_cat_exact[cat]["tp"] += m["tp"]
            per_cat_exact[cat]["fn"] += m["fn"]
            per_cat_exact[cat]["fp"] += m["fp"]
        ov_o = pr.overlap["overall"]
        for k in ("tp", "fn_wrong", "fn_missing_sys_uri", "fn_missed_extraction",
                  "fp_spurious", "skipped_gold_none", "matched_spans", "matched_total"):
            acc_overlap[k] += ov_o.get(k, 0)
        for label, m in pr.overlap.get("per_label", {}).items():
            per_label_overlap[label]["tp"] += m["tp"]
            per_label_overlap[label]["fn"] += m["fn"]
            per_label_overlap[label]["fp"] += m["fp"]
        for cat, m in pr.overlap.get("per_category", {}).items():
            per_cat_overlap[cat]["tp"] += m["tp"]
            per_cat_overlap[cat]["fn"] += m["fn"]
            per_cat_overlap[cat]["fp"] += m["fp"]
        total_lookup_mapped += pr.uri_lookup_mapped
        total_lookup_unmapped += pr.uri_lookup_unmapped
        total_lookup_total += pr.uri_lookup_total
        for label, d in pr.uri_lookup_by_label.items():
            for k in ("mapped", "unmapped", "total"):
                lookup_by_label[label][k] += d.get(k, 0)
        per_doc[pr.doc_id] = {
            "split": pr.split, "exact": pr.exact, "overlap": pr.overlap,
            "uri_lookup": {"mapped": pr.uri_lookup_mapped,
                           "unmapped": pr.uri_lookup_unmapped,
                           "total": pr.uri_lookup_total},
        }

    def _build_overall(acc):
        fn_total = acc["fn_wrong"] + acc["fn_missing_sys_uri"] + acc["fn_missed_extraction"]
        fp_total = acc["fn_wrong"] + acc["fp_spurious"]
        precision, recall, f1 = _prf1(acc["tp"], fn_total, fp_total)
        matched_total = acc["tp"] + acc["fn_wrong"] + acc["fn_missing_sys_uri"]
        accuracy = acc["tp"] / matched_total if matched_total else 0.0
        return {
            "tp": acc["tp"], "fn_wrong": acc["fn_wrong"],
            "fn_missing_sys_uri": acc["fn_missing_sys_uri"],
            "fn_missed_extraction": acc["fn_missed_extraction"],
            "fp_spurious": acc["fp_spurious"],
            "skipped_gold_none": acc["skipped_gold_none"],
            "matched_spans": acc["matched_spans"],
            "precision": precision, "recall": recall, "f1": f1,
            "accuracy": accuracy, "matched_total": matched_total,
        }

    def _finalize(acc_dict):
        out = {}
        for key, d in acc_dict.items():
            p, r, f = _prf1(d["tp"], d["fn"], d["fp"])
            out[key] = {"tp": d["tp"], "fn": d["fn"], "fp": d["fp"],
                        "precision": p, "recall": r, "f1": f}
        return out

    def _group_types(per_label_final):
        """Map per-label counts onto the reported evaluation groups."""
        grouped = {}
        for type_key, labels in TYPE_GROUPS.items():
            tp = fn = fp = 0
            for label in labels:
                m = per_label_final.get(label)
                if m is None:
                    continue
                tp += m["tp"]
                fn += m["fn"]
                fp += m["fp"]
            p, r, f = _prf1(tp, fn, fp)
            grouped[type_key] = {"tp": tp, "fn": fn, "fp": fp,
                                 "precision": p, "recall": r, "f1": f}
        return grouped

    final_per_label_exact = _finalize(per_label_exact)
    final_per_label_overlap = _finalize(per_label_overlap)

    return {
        "exact": {
            "overall": _build_overall(acc_exact),
            "per_label": final_per_label_exact,
            "per_type": _group_types(final_per_label_exact),
            "per_category": _finalize(per_cat_exact),
        },
        "overlap": {
            "overall": _build_overall(acc_overlap),
            "per_label": final_per_label_overlap,
            "per_type": _group_types(final_per_label_overlap),
            "per_category": _finalize(per_cat_overlap),
        },
        "lookup": {"mapped": total_lookup_mapped,
                   "unmapped": total_lookup_unmapped,
                   "total": total_lookup_total},
        "lookup_by_label": {k: dict(v) for k, v in lookup_by_label.items()},
        "per_doc": per_doc,
        "n_documents": len(pair_results),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def group_by_split(pair_results):
    out = defaultdict(list)
    for pr in pair_results:
        out[pr.split].append(pr)
    return dict(out)


def _unmatched_desc(unmatched: str) -> str:
    if unmatched == "singleton":
        return "end-to-end / twinless — comparable with pipeline systems"
    return "coref step only — upper bound, not comparable with pipelines"


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


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------


def build_coref_report(args, pair_results, aggregate, candidate_pool_name,
                       unmatched: str) -> str:
    lines: List[str] = []
    A = lines.append
    A(f"# Reflex (CanLII citator) — coreference resolution evaluation")
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
    A(f"**Unmatched policy**: `{unmatched}` "
      f"({_unmatched_desc(unmatched)})")
    A("")
    A(f"**Documents**: {aggregate['n_documents']} reflex export(s) paired with gold.")
    A("")

    for mode in ("exact", "overlap"):
        title = ("EXACT span match (label + identical offsets)" if mode == "exact"
                 else "OVERLAP span match (label + intersecting offsets)")
        ag = aggregate[mode]
        A(f"## {mode.upper()} — {title}")
        A("")
        rows = []
        for mname in COREF_METRICS:
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

    by_split = group_by_split(pair_results)
    split_order = [s for s in ("test", "dev", "train", "silver") if s in by_split]
    split_order += [s for s in sorted(by_split) if s not in split_order]

    A("## Results per split")
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
            n_matched = sum(d.n_matched if mode == "exact" else d.stats_overlap["matched"]
                            for d in docs)
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

    A("## Per mention type")
    A("")
    A("Scores are macro-averaged across documents, separately for decisions "
      "and legislation. `Secondary` is excluded: Reflex never links secondary "
      "sources, so that row would be a trivial all-singleton baseline (it is "
      "still present in the JSON).")
    A("")
    per_type_agg = aggregate_coref_per_type(pair_results)
    for mode in ("exact", "overlap"):
        A(f"### {mode.upper()}")
        A("")
        rows = []
        for type_key in REPORT_TYPES:
            t = per_type_agg.get(type_key)
            if not t:
                continue
            ag = t[mode]
            matched = t["n_matched_exact" if mode == "exact" else "n_matched_overlap"]
            rows.append([TYPE_DISPLAY[type_key], t["n_documents"],
                         t["n_gold"], t["n_system"], matched,
                         f"{ag['MUC']['f1']*100:.2f}",
                         f"{ag['B3']['f1']*100:.2f}",
                         f"{ag['CEAF_e']['f1']*100:.2f}",
                         f"{ag['LEA']['f1']*100:.2f}",
                         f"{ag['CoNLL']['f1']*100:.2f}"])
        A(md_table(["type", "docs", "gold mentions", "sys mentions", "matched",
                    "MUC F1 %", "B3 F1 %", "CEAF_e F1 %", "LEA F1 %",
                    "CoNLL F1 %"], rows))
        A("")

    A("### Per split and mention type")
    A("")
    rows = []
    for sp in split_order:
        docs = by_split[sp]
        split_pt = aggregate_coref_per_type(docs)
        for type_key in REPORT_TYPES:
            t = split_pt.get(type_key)
            if not t:
                continue
            ag = t["overlap"]
            rows.append([sp, TYPE_DISPLAY[type_key], t["n_gold"], t["n_system"],
                         t["n_matched_overlap"],
                         f"{ag['MUC']['f1']*100:.2f}",
                         f"{ag['B3']['f1']*100:.2f}",
                         f"{ag['CEAF_e']['f1']*100:.2f}",
                         f"{ag['LEA']['f1']*100:.2f}",
                         f"{ag['CoNLL']['f1']*100:.2f}"])
    A("OVERLAP span match (the meaningful mode for a citator):")
    A("")
    A(md_table(["split", "type", "gold mentions", "sys mentions", "matched",
                "MUC F1 %", "B3 F1 %", "CEAF_e F1 %", "LEA F1 %",
                "CoNLL F1 %"], rows))
    A("")

    A("## Per-document breakdown (exact match)")
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
    A("## Notes")
    A("")
    A("* The `singleton` unmatched policy (default) counts extraction errors: "
      "mentions found by only one side become singletons on the other side. "
      "This makes Reflex's scores directly comparable with the LLM/pipeline "
      "systems scored with `evaluate_coref.py --match-mode span "
      "--unmatched singleton`.")
    A("* The `drop` policy isolates coreference from extraction errors and is "
      "an upper bound — not comparable with pipeline end-to-end scores.")
    A("* Reflex never links secondary sources, so those clusters are empty on "
      "the system side by construction; the secondary rows in the JSON are a "
      "trivial all-singleton baseline, not a real measurement.")
    A("* Reflex only exposes a ``concatenated-id`` on legislation anchors. "
      "Decision anchors carry no ``concatenated-id``, so every decision "
      "prediction is a singleton cluster: decision MUC is 0 by construction. "
      "Decision B3 / CEAF_e / LEA still respond to the gold clusters that the "
      "singleton predictions land on, so they are reported but they do not "
      "measure decision coreference, which cannot be recovered from these "
      "exports as they stand.")
    A("* Coref metrics (MUC / B3 / CEAF_e / LEA) are implemented inline, "
      "equivalent to the `coval` library used by `src/evaluation/`.")
    A("")
    return "\n".join(lines)


def build_resolution_report(args, pair_results, aggregate,
                            candidate_pool_name) -> str:
    lines: List[str] = []
    A = lines.append
    A(f"# Reflex (CanLII citator) — URI linking (resolution) evaluation")
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

    for mode in ("exact", "overlap"):
        title = ("EXACT span match" if mode == "exact" else "OVERLAP span match")
        ov = aggregate[mode]["overall"]
        A(f"## {mode.upper()} — {title}")
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

    by_split = group_by_split(pair_results)
    split_order = [s for s in ("test", "dev", "train", "silver") if s in by_split]
    split_order += [s for s in sorted(by_split) if s not in split_order]

    A("## Results per split (overlap match)")
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

    def _per_type_counts(docs, mode):
        """Micro-aggregate per-label counts into evaluation groups."""
        counts = {t: {"tp": 0, "fn": 0, "fp": 0} for t in REPORT_TYPES}
        for d in docs:
            mode_data = getattr(d, mode)
            per_label = mode_data.get("per_label", {})
            for type_key in REPORT_TYPES:
                for label in TYPE_GROUPS[type_key]:
                    m = per_label.get(label)
                    if m is None:
                        continue
                    counts[type_key]["tp"] += m["tp"]
                    counts[type_key]["fn"] += m["fn"]
                    counts[type_key]["fp"] += m["fp"]
        out = {}
        for type_key, c in counts.items():
            p, r, f = _prf1(c["tp"], c["fn"], c["fp"])
            out[type_key] = {"tp": c["tp"], "fn": c["fn"], "fp": c["fp"],
                             "precision": p, "recall": r, "f1": f}
        return out

    A("## Per-type breakdown (micro)")
    A("")
    A("Micro-aggregated across documents, separately for decisions and "
      "legislation. `Secondary` is excluded because Reflex never links "
      "secondary sources; the full per-label table remains in the JSON.")
    A("")
    for mode in ("exact", "overlap"):
        rows = []
        for type_key, m in _per_type_counts(pair_results, mode).items():
            rows.append([TYPE_DISPLAY[type_key], m["tp"], m["fn"], m["fp"],
                         f"{m['precision']*100:.2f}",
                         f"{m['recall']*100:.2f}",
                         f"{m['f1']*100:.2f}"])
        A(f"**{mode.upper()} span match**")
        A("")
        A(md_table(["type", "TP", "FN", "FP", "Precision %", "Recall %", "F1 %"], rows))
        A("")

    A("## Per split and mention type (overlap match)")
    A("")
    rows = []
    for sp in split_order:
        docs = [d for d in pair_results if d.split == sp]
        for type_key, m in _per_type_counts(docs, "overlap").items():
            rows.append([sp, TYPE_DISPLAY[type_key], m["tp"], m["fn"], m["fp"],
                         f"{m['precision']*100:.2f}",
                         f"{m['recall']*100:.2f}",
                         f"{m['f1']*100:.2f}"])
    A(md_table(["split", "type", "TP", "FN", "FP", "Precision %", "Recall %",
                "F1 %"], rows))
    A("")

    A("## Per-document breakdown (exact match)")
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

    A("## URI lookup diagnostics")
    A("")
    A(f"* Total concatenated-ids looked up: {aggregate['lookup']['total']}")
    A(f"* Mapped to a URI: {aggregate['lookup']['mapped']}")
    A(f"* Unmapped (no candidate-pool entry): {aggregate['lookup']['unmapped']}")
    A("")
    rows = []
    for type_key in REPORT_TYPES:
        d = aggregate.get("lookup_by_label", {}).get(type_key)
        if d is None:
            continue
        rows.append([TYPE_DISPLAY[type_key], d.get("total", 0),
                     d.get("mapped", 0), d.get("unmapped", 0)])
    A(md_table(["type", "lookups", "mapped", "unmapped"], rows))
    A("")
    A("* Reflex only exposes a ``concatenated-id`` on legislation anchors. "
      "Decision anchors have no ``concatenated-id``, so the current lookup "
      "cannot map them and their URI-linking score is 0 by construction.")
    A("")
    A("* Unmapped concatenated-ids produce system spans with no uri, which "
      "count as false negatives (no uri produced) when the corresponding gold "
      "span has a real uri.")
    A("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


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
    print(f"Unmatched policy: {args.unmatched}")

    coref_results: List[CorefPairResult] = []
    resolution_results: List[ResolutionPairResult] = []

    for i, p in enumerate(usable, 1):
        print(f"\n[{i}/{len(usable)}] {p.reflex_path.name} -> {p.doc_id} [{p.split}]")
        gold_text, gold_mentions = parse_gold_with_docid(p.gold_path)
        reflex_text, reflex_mentions = build_reflex_mentions(p.reflex_path, args.gap)
        alignment = C.Alignment(reflex_text, gold_text)

        if not args.skip_coref:
            cr = evaluate_coref_pair(gold_mentions, reflex_mentions, alignment,
                                     unmatched=args.unmatched)
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

    coref_agg = aggregate_coref(coref_results) if not args.skip_coref else {}
    coref_per_type_agg = (aggregate_coref_per_type(coref_results)
                          if not args.skip_coref else {})
    resolution_agg = aggregate_resolution(resolution_results) if not args.skip_resolution else {}

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "candidate_pool": args.candidate_pool,
        "merge_gap": args.gap,
        "unmatched": args.unmatched,
        "splits": args.splits,
        "pairs": [{"reflex": p.reflex_path.name, "doc_id": p.doc_id, "split": p.split,
                   "match": p.how, "similarity": round(p.text_ratio, 4)}
                  for p in usable],
    }

    if not args.skip_coref:
        coref_report = build_coref_report(args, coref_results, coref_agg,
                                          args.candidate_pool, args.unmatched)
        (out_dir / "reflex_coref_report.md").write_text(coref_report, encoding="utf-8")
        print(f"\nCoref report: {out_dir / 'reflex_coref_report.md'}")
        coref_payload = {
            "generated": payload["generated"],
            "candidate_pool": args.candidate_pool,
            "merge_gap": args.gap,
            "unmatched": args.unmatched,
            "aggregate": coref_agg,
            "aggregate_per_type": coref_per_type_agg,
            "per_doc": {pr.doc_id: {
                "split": pr.split,
                "exact": pr.exact,
                "overlap": pr.overlap,
                "stats": pr.stats,
                "stats_overlap": pr.stats_overlap,
                "n_gold_mentions": pr.n_gold_mentions,
                "n_sys_mentions": pr.n_sys_mentions,
                "n_matched": pr.n_matched,
                "per_type": pr.per_type,
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
                    "by_label": pr.uri_lookup_by_label,
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
    ap.add_argument("--candidate-pool", choices=["full", "gold"], default="gold",
                    help="candidate pool for URI lookup (default: gold = "
                         "gold_candidate_pool.csv)")
    ap.add_argument("--min-ratio", type=float, default=0.85,
                    help="content-token similarity gate for fuzzy id pairing")
    ap.add_argument("--unmatched", choices=["singleton", "drop"], default="singleton",
                    help="singleton = end-to-end twinless (comparable with "
                         "pipelines, default). drop = coref step only (upper bound).")
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
