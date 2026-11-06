"""
LeREaD - Annotation Evaluation Module

Evaluates system-generated annotations against gold-standard human annotations.
Computes span-level Precision / Recall / F1 per label, plus micro-averaged F1
across a list of (gold, system) file pairs.

Methodology (same as the validated IAA code):
  1. Parse HTML with BeautifulSoup
  2. Extract plain text from <body> (all tags stripped)
  3. For each annotation tag, find its text in the plain text
  4. Take 200 characters before + after as context (normalized)
  5. Two spans match when: same label + same text + same normalized context
"""

import os
import re
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

from bs4 import BeautifulSoup


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONTEXT_CHARS = 200  # characters before/after a span used to identify its location

PARENT_LABELS = {"decision", "legislation", "secondary sources"}

# ---------------------------------------------------------------------------
# Span dataclass
# ---------------------------------------------------------------------------

class Span:
    """One annotated span extracted from an HTML file."""

    def __init__(self, text: str, labelname: str, context_text: str, attributes: dict = None, start_pos: int = 0, end_pos: int = None):
        self.text        = " ".join(text.split())          # normalise whitespace
        self.labelname   = labelname.strip().lower()
        self.context_text = " ".join(context_text.split()).lower()  # normalised
        self.attributes   = attributes or {}
        self.start_pos    = start_pos # character offset in the document plain text
        # Character offset one past the end of the span in the normalised
        # plain text. Defaults to start + len(text) so old call sites that
        # only pass start_pos keep working.
        self.end_pos      = end_pos if end_pos is not None else (start_pos + len(self.text) if start_pos >= 0 else -1)

    def __repr__(self):
        preview = self.text[:40] + ("…" if len(self.text) > 40 else "")
        return f"Span(label={self.labelname!r}, text={preview!r})"

    def matches(self, other: "Span") -> bool:
        """
        Two spans match when they have the same label, the same normalised text,
        and the same normalised surrounding context.
        The context check ensures we are comparing the SAME location in the
        document even when the same sentence appears more than once.
        """
        if self.labelname != other.labelname:
            return False
        if self.text != other.text:
            return False
        if not self.context_text or not other.context_text:
            # If context could not be extracted, fall back to text-only match
            return True
        return self.context_text == other.context_text

    def overlap_len(self, other: "Span") -> int:
        """Length (chars) of the character-span intersection, or 0."""
        if self.start_pos < 0 or other.start_pos < 0:
            return 0
        if self.end_pos <= self.start_pos or other.end_pos <= other.start_pos:
            return 0
        return max(0, min(self.end_pos, other.end_pos) - max(self.start_pos, other.start_pos))

    def overlaps(self, other: "Span") -> bool:
        """
        OVERLAP (comparison only — never used by the exact-match path).

        True when both spans share the same label and their character
        offsets in the normalised document text intersect. This credits
        partial/boundary-mismatched predictions (e.g. gold "R. v. Smith,
        2020 SCC 1" vs system "R. v. Smith").

        Falls back to case-insensitive text containment when either span
        has no valid offsets (occurrence lookup failed).
        """
        if self.labelname != other.labelname:
            return False
        if self.start_pos >= 0 and other.start_pos >= 0:
            if self.end_pos > self.start_pos and other.end_pos > other.start_pos:
                return self.overlap_len(other) > 0
        # Fallback: same label + one text contains the other
        a, b = self.text.lower(), other.text.lower()
        if not a or not b:
            return False
        return a in b or b in a


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def _find_nth(text: str, plain: str, n: int) -> int:
    """Return the start index of the nth occurrence (0-based) of text in plain, or -1."""
    start = 0
    count = 0
    while True:
        idx = plain.find(text, start)
        if idx == -1:
            return -1
        if count == n:
            return idx
        count += 1
        start = idx + len(text)  # advance by full match length


def _get_context(elem_text: str, plain_text: str, context_chars: int,
                 seen_counts: dict, labelname: str) -> Tuple[str, int, int]:
    """
    Find the correct occurrence of elem_text using a per-(labelname, text) counter,
    then return (context_window, start_position, end_position).

    NOTE: the context-window computation is unchanged from the exact-match
    version; only the span offsets are additionally returned for overlap
    matching. Exact-match behaviour is unaffected.
    """
    normalised_elem  = " ".join(elem_text.split())
    normalised_plain = " ".join(plain_text.split())

    key = (labelname, normalised_elem)
    n = seen_counts.get(key, 0)
    seen_counts[key] = n + 1

    start = _find_nth(normalised_elem, normalised_plain, n)

    end       = start + len(normalised_elem)
    ctx_start = max(0, start - context_chars)
    ctx_end   = min(len(normalised_plain), end + context_chars)
    return normalised_plain[ctx_start:ctx_end], start, end


def extract_spans(html_path: str, context_chars: int = CONTEXT_CHARS) -> List[Span]:
    html_path = Path(html_path)
    content   = html_path.read_text(encoding="utf-8")
    soup      = BeautifulSoup(content, "html.parser")

    body = soup.find("body")
    if body is None:
        return []

    plain_text = " ".join(body.get_text().split())  # normalise once here
    tags       = body.find_all("manual_label") + body.find_all("auto_label")

    spans      = []
    seen_counts: dict = {}  # key: (labelname, normalised_text)

    for tag in tags:
        labelname = tag.get("labelname", "").strip().lower()
        if not labelname:
            continue

        text = tag.get_text()
        if not " ".join(text.split()):
            continue

        context, start, end = _get_context(text, plain_text, context_chars, seen_counts, labelname)

        attrs = dict(tag.attrs)
        spans.append(Span(text, labelname, context, attributes=attrs, start_pos=start, end_pos=end))

    return spans


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------

def match_spans(gold: List[Span], system: List[Span]) -> Tuple[int, int, int]:
    """
    Greedily match gold spans to system spans.

    A match requires: same label + same normalised text + same context.
    Each span is used at most once (no double-counting).

    Returns
    -------
    (tp, n_gold, n_system)
    """
    used = [False] * len(system)
    tp   = 0

    for g in gold:
        for j, s in enumerate(system):
            if not used[j] and g.matches(s):
                tp     += 1
                used[j] = True
                break

    return tp, len(gold), len(system)


# ---------------------------------------------------------------------------
# Overlap matching (COMPARISON ONLY — exact-match path above is untouched)
# ---------------------------------------------------------------------------

def match_spans_overlap(gold: List[Span], system: List[Span]) -> Tuple[int, int, int]:
    """
    Greedily match gold spans to system spans on OVERLAP.

    A match requires: same label + character-span intersection (see
    Span.overlaps). Each span is used at most once. For each gold span,
    the unused overlapping system span with the largest intersection is
    chosen (ties broken by order).

    This is a lenient comparison metric shown ALONGSIDE the exact-match
    main metric. It never affects exact-match counts.

    Returns
    -------
    (tp, n_gold, n_system)
    """
    used = [False] * len(system)
    tp   = 0

    for g in gold:
        best_j = -1
        best_overlap = 0
        for j, s in enumerate(system):
            if used[j]:
                continue
            if not g.overlaps(s):
                continue
            ov = g.overlap_len(s)
            if ov <= 0:
                # Positional fallback (e.g. containment fallback): still a
                # valid overlap hit, just without a measurable intersection.
                if best_j == -1:
                    best_j = j
                continue
            if ov > best_overlap:
                best_overlap = ov
                best_j = j
        if best_j != -1:
            tp += 1
            used[best_j] = True

    return tp, len(gold), len(system)


def compute_metrics_overlap(gold: List[Span], system: List[Span]) -> Dict[str, dict]:
    """
    Per-label Precision / Recall / F1 under OVERLAP matching.

    Mirrors compute_metrics() exactly, except match_spans_overlap() is
    used instead of match_spans(). Comparison only.
    """
    all_labels = sorted(
        {s.labelname for s in gold} | {s.labelname for s in system}
    )

    results = {}
    for label in all_labels:
        g = [s for s in gold   if s.labelname == label]
        s = [s for s in system if s.labelname == label]
        tp, n_gold, n_system = match_spans_overlap(g, s)
        P, R, F = _prf(tp, n_gold, n_system)
        results[label] = {"tp": tp, "n_gold": n_gold, "n_system": n_system,
                          "P": P, "R": R, "F1": F}
    return results


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _prf(tp: int, n_gold: int, n_system: int) -> Tuple[float, float, float]:
    P = tp / n_system if n_system > 0 else 0.0
    R = tp / n_gold   if n_gold   > 0 else 0.0
    F = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return P, R, F


def compute_metrics(gold: List[Span], system: List[Span]) -> Dict[str, dict]:
    """
    Compute per-label Precision / Recall / F1.

    Returns
    -------
    dict  label -> {"P", "R", "F1", "tp", "n_gold", "n_system"}
    """
    all_labels = sorted(
        {s.labelname for s in gold} | {s.labelname for s in system}
    )

    results = {}
    for label in all_labels:
        g = [s for s in gold   if s.labelname == label]
        s = [s for s in system if s.labelname == label]
        tp, n_gold, n_system = match_spans(g, s)
        P, R, F = _prf(tp, n_gold, n_system)
        results[label] = {"tp": tp, "n_gold": n_gold, "n_system": n_system,
                          "P": P, "R": R, "F1": F}
    return results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def _print_table(metrics: Dict[str, dict], title: str = "",
                 doc_f1s: List[float] | None = None) -> None:
    W = 34
    if title:
        print(f"\n  {title}")

    header = (f"  {'Label':<{W}} {'Gold':>6} {'Sys':>6} {'TP':>6}"
              f" {'P':>8} {'R':>8} {'F1':>8}")
    sep    = "  " + "─" * (len(header) - 2)
    print(sep)
    print(header)
    print(sep)

    for label in sorted(metrics):
        m = metrics[label]
        print(f"  {label:<{W}} {m['n_gold']:>6} {m['n_system']:>6} {m['tp']:>6}"
              f" {m['P']*100:>7.1f}% {m['R']*100:>7.1f}% {m['F1']*100:>7.1f}%")

    print(sep)

    # Micro average across all labels
    total_tp     = sum(m["tp"]       for m in metrics.values())
    total_gold   = sum(m["n_gold"]   for m in metrics.values())
    total_system = sum(m["n_system"] for m in metrics.values())
    P, R, F = _prf(total_tp, total_gold, total_system)
    print(f"  {'MICRO (all labels)':<{W}} {total_gold:>6} {total_system:>6} {total_tp:>6}"
          f" {P*100:>7.1f}% {R*100:>7.1f}% {F*100:>7.1f}%")

    # Macro F1 per document (mean of per-document F1 scores)
    if doc_f1s:
        macro_doc_f1 = sum(doc_f1s) / len(doc_f1s)
        print(f"  {'MACRO F1 (mean per-doc F1)':<{W + 30}} {macro_doc_f1*100:>7.1f}%"
              f"  (over {len(doc_f1s)} documents)")
    print()

# ---------------------------------------------------------------------------
# Single-pair evaluation
# ---------------------------------------------------------------------------

def evaluate(
    gold_path: str,
    system_path: str,
    context_chars: int = CONTEXT_CHARS,
    verbose: bool = True,
) -> Tuple[Dict[str, dict], float]:
    """
    Evaluate one (gold, system) HTML file pair.

    Parameters
    ----------
    gold_path : str
    system_path : str
    context_chars : int
    verbose : bool
        Print a results table.

    Returns
    -------
    metrics  : dict  label -> {"P", "R", "F1", "tp", "n_gold", "n_system"}
    doc_f1   : float  micro F1 over all spans in this document
    """
    gold_spans   = extract_spans(gold_path,   context_chars)
    system_spans = extract_spans(system_path, context_chars)

    if verbose:
        print(f"  Gold  : {Path(gold_path).name}  ({len(gold_spans)} spans)")
        print(f"  System: {Path(system_path).name}  ({len(system_spans)} spans)")

    metrics = compute_metrics(gold_spans, system_spans)

    # Per-document F1: micro over all spans regardless of label
    tp, n_gold, n_system = match_spans(gold_spans, system_spans)
    _, _, doc_f1 = _prf(tp, n_gold, n_system)

    if verbose:
        title = f"{Path(gold_path).name}  vs  {Path(system_path).name}"
        _print_table(metrics, title, doc_f1s=None)

    return metrics, doc_f1


def evaluate_overlap(
    gold_path: str,
    system_path: str,
    context_chars: int = CONTEXT_CHARS,
    verbose: bool = True,
) -> Tuple[Dict[str, dict], float]:
    """
    Evaluate one (gold, system) pair under OVERLAP matching.

    Comparison only — mirrors evaluate() exactly, except overlap
    matching is used. The exact-match evaluate() above is untouched
    and remains the main metric.
    """
    gold_spans   = extract_spans(gold_path,   context_chars)
    system_spans = extract_spans(system_path, context_chars)

    if verbose:
        print(f"  Gold  : {Path(gold_path).name}  ({len(gold_spans)} spans)")
        print(f"  System: {Path(system_path).name}  ({len(system_spans)} spans)")

    metrics = compute_metrics_overlap(gold_spans, system_spans)

    # Per-document F1: micro over all spans regardless of label (overlap)
    tp, n_gold, n_system = match_spans_overlap(gold_spans, system_spans)
    _, _, doc_f1 = _prf(tp, n_gold, n_system)

    if verbose:
        title = f"{Path(gold_path).name}  vs  {Path(system_path).name}  [OVERLAP]"
        _print_table(metrics, title, doc_f1s=None)

    return metrics, doc_f1


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def evaluate_batch(
    pairs: List[Tuple[str, str]],
    context_chars: int = CONTEXT_CHARS,
    verbose_per_file: bool = False,
) -> Dict[str, dict]:
    """
    Evaluate a list of (gold_path, system_path) pairs.

    Returns:
        {
            "overall": {
                "tp": ...,
                "n_gold": ...,
                "n_system": ...,
                "P": ...,
                "R": ...,
                "F1": ...,
            },
            "per_document": {
                "doc_name": {
                    "tp": ...,
                    "n_gold": ...,
                    "n_system": ...,
                    "P": ...,
                    "R": ...,
                    "F1": ...,
                },
                ...
            },
            "per_label": {
                "label": {
                    "tp": ...,
                    "n_gold": ...,
                    "n_system": ...,
                    "P": ...,
                    "R": ...,
                    "F1": ...,
                },
                ...
            }
        }
    """

    accumulated = defaultdict(
        lambda: {"tp": 0, "n_gold": 0, "n_system": 0}
    )

    per_document = {}
    doc_f1s = []

    print(f"\nBatch evaluation — {len(pairs)} file pair(s)")
    print("=" * 60)

    for i, (gold_path, system_path) in enumerate(pairs, 1):
        doc_name = Path(gold_path).stem

        print(f"\n[{i}/{len(pairs)}] {Path(gold_path).name}")

        try:
            metrics, doc_f1 = evaluate(
                gold_path,
                system_path,
                context_chars=context_chars,
                verbose=verbose_per_file,
            )
        except Exception as e:
            print(
                f"  ERROR processing "
                f"{Path(gold_path).name} / {Path(system_path).name}: {e}"
            )
            continue

        doc_f1s.append(doc_f1)

        # ---------------------------------------------------------
        # Per-document counts across ALL labels
        # ---------------------------------------------------------
        doc_tp = sum(m["tp"] for m in metrics.values())
        doc_n_gold = sum(m["n_gold"] for m in metrics.values())
        doc_n_system = sum(m["n_system"] for m in metrics.values())

        doc_P, doc_R, doc_F1 = _prf(
            doc_tp,
            doc_n_gold,
            doc_n_system,
        )

        per_document[doc_name] = {
            "tp": doc_tp,
            "n_gold": doc_n_gold,
            "n_system": doc_n_system,
            "P": doc_P,
            "R": doc_R,
            "F1": doc_F1,
        }

        # ---------------------------------------------------------
        # Accumulate counts across documents, per label
        # ---------------------------------------------------------
        for label, m in metrics.items():
            accumulated[label]["tp"] += m["tp"]
            accumulated[label]["n_gold"] += m["n_gold"]
            accumulated[label]["n_system"] += m["n_system"]

        if not verbose_per_file:
            print(
                f"  TP={doc_tp} | "
                f"Gold={doc_n_gold} | "
                f"System={doc_n_system} | "
                f"F1={doc_F1 * 100:.1f}%"
            )

    # -------------------------------------------------------------
    # Overall counts across ALL documents and ALL labels
    # -------------------------------------------------------------
    total_tp = sum(d["tp"] for d in per_document.values())
    total_n_gold = sum(d["n_gold"] for d in per_document.values())
    total_n_system = sum(d["n_system"] for d in per_document.values())

    total_P, total_R, total_F1 = _prf(
        total_tp,
        total_n_gold,
        total_n_system,
    )

    overall = {
        "tp": total_tp,
        "n_gold": total_n_gold,
        "n_system": total_n_system,
        "P": total_P,
        "R": total_R,
        "F1": total_F1,
    }

    # -------------------------------------------------------------
    # Micro-averaged results per label
    # -------------------------------------------------------------
    per_label = {}

    for label, counts in accumulated.items():
        tp = counts["tp"]
        n_gold = counts["n_gold"]
        n_system = counts["n_system"]

        P, R, F = _prf(tp, n_gold, n_system)

        per_label[label] = {
            "tp": tp,
            "n_gold": n_gold,
            "n_system": n_system,
            "P": P,
            "R": R,
            "F1": F,
        }

    parent_counts   = {"tp": 0, "n_gold": 0, "n_system": 0}
    children_counts = {"tp": 0, "n_gold": 0, "n_system": 0}

    for label, counts in accumulated.items():
        bucket = parent_counts if label in PARENT_LABELS else children_counts
        bucket["tp"]       += counts["tp"]
        bucket["n_gold"]   += counts["n_gold"]
        bucket["n_system"] += counts["n_system"]

    parent_P, parent_R, parent_F1       = _prf(parent_counts["tp"], parent_counts["n_gold"], parent_counts["n_system"])
    children_P, children_R, children_F1 = _prf(children_counts["tp"], children_counts["n_gold"], children_counts["n_system"])

    parent_metrics   = {**parent_counts,   "P": parent_P,   "R": parent_R,   "F1": parent_F1}
    children_metrics = {**children_counts, "P": children_P, "R": children_R, "F1": children_F1}

    # -------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS — PER DOCUMENT")
    print("=" * 60)

    for doc_name, m in per_document.items():
        print(
            f"{doc_name:<40} "
            f"TP={m['tp']:>4} | "
            f"Gold={m['n_gold']:>4} | "
            f"System={m['n_system']:>4} | "
            f"F1={m['F1'] * 100:>5.1f}%"
        )

    print("\n" + "=" * 60)
    print("RESULTS — TOTAL")
    print("=" * 60)

    print(
        f"TP={total_tp} | "
        f"Gold={total_n_gold} | "
        f"System={total_n_system} | "
        f"P={total_P * 100:.1f}% | "
        f"R={total_R * 100:.1f}% | "
        f"F1={total_F1 * 100:.1f}%"
    )

    print("\n" + "=" * 60)
    _print_table(
        per_label,
        title="RESULTS — MICRO-AVERAGED PER LABEL",
        doc_f1s=doc_f1s,
    )

    print("=" * 60)
    print("RESULTS — PARENT vs CHILDREN vs TOTAL")
    print("=" * 60)
    print(f"{'':<20} {'Gold':>6} {'Sys':>6} {'TP':>6} {'P':>8} {'R':>8} {'F1':>8}")
    print(f"{'Parent spans':<20} {parent_metrics['n_gold']:>6} {parent_metrics['n_system']:>6} "
          f"{parent_metrics['tp']:>6} {parent_metrics['P']*100:>7.1f}% "
          f"{parent_metrics['R']*100:>7.1f}% {parent_metrics['F1']*100:>7.1f}%")
    print(f"{'Children spans':<20} {children_metrics['n_gold']:>6} {children_metrics['n_system']:>6} "
          f"{children_metrics['tp']:>6} {children_metrics['P']*100:>7.1f}% "
          f"{children_metrics['R']*100:>7.1f}% {children_metrics['F1']*100:>7.1f}%")
    print(f"{'Total':<20} {total_n_gold:>6} {total_n_system:>6} "
          f"{total_tp:>6} {total_P*100:>7.1f}% {total_R*100:>7.1f}% {total_F1*100:>7.1f}%")
    print()

    return {
        "overall": overall,
        "per_document": per_document,
        "per_label": per_label,
        "parent_child_breakdown": {
            "parent": parent_metrics,
            "children": children_metrics,
        },
    }

def evaluate_batch_overlap(
    pairs: List[Tuple[str, str]],
    context_chars: int = CONTEXT_CHARS,
    verbose_per_file: bool = False,
) -> Dict[str, dict]:
    """
    Batch evaluation under OVERLAP matching (comparison only).

    Mirrors evaluate_batch() exactly, except overlap matching is used.
    The exact-match evaluate_batch() above is untouched and remains the
    main metric. Returned dict has the same shape ("overall",
    "per_document", "per_label", "parent_child_breakdown").
    """

    accumulated = defaultdict(
        lambda: {"tp": 0, "n_gold": 0, "n_system": 0}
    )

    per_document = {}
    doc_f1s = []

    print(f"\nBatch evaluation (OVERLAP — comparison) — {len(pairs)} file pair(s)")
    print("=" * 60)

    for i, (gold_path, system_path) in enumerate(pairs, 1):
        doc_name = Path(gold_path).stem

        print(f"\n[{i}/{len(pairs)}] {Path(gold_path).name}")

        try:
            metrics, doc_f1 = evaluate_overlap(
                gold_path,
                system_path,
                context_chars=context_chars,
                verbose=verbose_per_file,
            )
        except Exception as e:
            print(
                f"  ERROR processing "
                f"{Path(gold_path).name} / {Path(system_path).name}: {e}"
            )
            continue

        doc_f1s.append(doc_f1)

        # ---------------------------------------------------------
        # Per-document counts across ALL labels
        # ---------------------------------------------------------
        doc_tp = sum(m["tp"] for m in metrics.values())
        doc_n_gold = sum(m["n_gold"] for m in metrics.values())
        doc_n_system = sum(m["n_system"] for m in metrics.values())

        doc_P, doc_R, doc_F1 = _prf(
            doc_tp,
            doc_n_gold,
            doc_n_system,
        )

        per_document[doc_name] = {
            "tp": doc_tp,
            "n_gold": doc_n_gold,
            "n_system": doc_n_system,
            "P": doc_P,
            "R": doc_R,
            "F1": doc_F1,
        }

        # ---------------------------------------------------------
        # Accumulate counts across documents, per label
        # ---------------------------------------------------------
        for label, m in metrics.items():
            accumulated[label]["tp"] += m["tp"]
            accumulated[label]["n_gold"] += m["n_gold"]
            accumulated[label]["n_system"] += m["n_system"]

        if not verbose_per_file:
            print(
                f"  TP={doc_tp} | "
                f"Gold={doc_n_gold} | "
                f"System={doc_n_system} | "
                f"F1={doc_F1 * 100:.1f}%"
            )

    # -------------------------------------------------------------
    # Overall counts across ALL documents and ALL labels
    # -------------------------------------------------------------
    total_tp = sum(d["tp"] for d in per_document.values())
    total_n_gold = sum(d["n_gold"] for d in per_document.values())
    total_n_system = sum(d["n_system"] for d in per_document.values())

    total_P, total_R, total_F1 = _prf(
        total_tp,
        total_n_gold,
        total_n_system,
    )

    overall = {
        "tp": total_tp,
        "n_gold": total_n_gold,
        "n_system": total_n_system,
        "P": total_P,
        "R": total_R,
        "F1": total_F1,
    }

    # -------------------------------------------------------------
    # Micro-averaged results per label
    # -------------------------------------------------------------
    per_label = {}

    for label, counts in accumulated.items():
        tp = counts["tp"]
        n_gold = counts["n_gold"]
        n_system = counts["n_system"]

        P, R, F = _prf(tp, n_gold, n_system)

        per_label[label] = {
            "tp": tp,
            "n_gold": n_gold,
            "n_system": n_system,
            "P": P,
            "R": R,
            "F1": F,
        }

    parent_counts   = {"tp": 0, "n_gold": 0, "n_system": 0}
    children_counts = {"tp": 0, "n_gold": 0, "n_system": 0}

    for label, counts in accumulated.items():
        bucket = parent_counts if label in PARENT_LABELS else children_counts
        bucket["tp"]       += counts["tp"]
        bucket["n_gold"]   += counts["n_gold"]
        bucket["n_system"] += counts["n_system"]

    parent_P, parent_R, parent_F1       = _prf(parent_counts["tp"], parent_counts["n_gold"], parent_counts["n_system"])
    children_P, children_R, children_F1 = _prf(children_counts["tp"], children_counts["n_gold"], children_counts["n_system"])

    parent_metrics   = {**parent_counts,   "P": parent_P,   "R": parent_R,   "F1": parent_F1}
    children_metrics = {**children_counts, "P": children_P, "R": children_R, "F1": children_F1}

    # -------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS (OVERLAP — comparison) — PER DOCUMENT")
    print("=" * 60)

    for doc_name, m in per_document.items():
        print(
            f"{doc_name:<40} "
            f"TP={m['tp']:>4} | "
            f"Gold={m['n_gold']:>4} | "
            f"System={m['n_system']:>4} | "
            f"F1={m['F1'] * 100:>5.1f}%"
        )

    print("\n" + "=" * 60)
    print("RESULTS (OVERLAP — comparison) — TOTAL")
    print("=" * 60)

    print(
        f"TP={total_tp} | "
        f"Gold={total_n_gold} | "
        f"System={total_n_system} | "
        f"P={total_P * 100:.1f}% | "
        f"R={total_R * 100:.1f}% | "
        f"F1={total_F1 * 100:.1f}%"
    )

    print("\n" + "=" * 60)
    _print_table(
        per_label,
        title="RESULTS (OVERLAP — comparison) — MICRO-AVERAGED PER LABEL",
        doc_f1s=doc_f1s,
    )

    print("=" * 60)
    print("RESULTS (OVERLAP — comparison) — PARENT vs CHILDREN vs TOTAL")
    print("=" * 60)
    print(f"{'':<20} {'Gold':>6} {'Sys':>6} {'TP':>6} {'P':>8} {'R':>8} {'F1':>8}")
    print(f"{'Parent spans':<20} {parent_metrics['n_gold']:>6} {parent_metrics['n_system']:>6} "
          f"{parent_metrics['tp']:>6} {parent_metrics['P']*100:>7.1f}% "
          f"{parent_metrics['R']*100:>7.1f}% {parent_metrics['F1']*100:>7.1f}%")
    print(f"{'Children spans':<20} {children_metrics['n_gold']:>6} {children_metrics['n_system']:>6} "
          f"{children_metrics['tp']:>6} {children_metrics['P']*100:>7.1f}% "
          f"{children_metrics['R']*100:>7.1f}% {children_metrics['F1']*100:>7.1f}%")
    print(f"{'Total':<20} {total_n_gold:>6} {total_n_system:>6} "
          f"{total_tp:>6} {total_P*100:>7.1f}% {total_R*100:>7.1f}% {total_F1*100:>7.1f}%")
    print()

    return {
        "overall": overall,
        "per_document": per_document,
        "per_label": per_label,
        "parent_child_breakdown": {
            "parent": parent_metrics,
            "children": children_metrics,
        },
    }

def _get_unmatched_system(gold: List[Span], system: List[Span]) -> List[Span]:
    """
    Same greedy matching as match_spans(), but returns the *system* spans that
    failed to find a match, instead of just a count. Uses Span.matches() exactly
    as match_spans() does, so results stay consistent with compute_metrics().
    """
    used_system = [False] * len(system)
    for g in gold:
        for j, s in enumerate(system):
            if not used_system[j] and g.matches(s):
                used_system[j] = True
                break
    return [s for j, s in enumerate(system) if not used_system[j]]


def evaluate_triple_batch(
    pairs_h_hllm_llm: List[Tuple[str, str, str]],
    context_chars: int = CONTEXT_CHARS,
    verbose_per_file: bool = False,
    use_id_shortcut: bool = False,
) -> Dict[str, dict]:
    """
    Each item is (human_path, human_llm_path, llm_path).

    Pass 1 (primary):  human = gold, human+llm = system.
                        Uses compute_metrics(), exactly like evaluate_batch().
    Pass 2 (anchoring): for the human+llm spans that did NOT match the human
                        gold in Pass 1, check whether they match the raw LLM
                        spans (via match_spans() again, per label). A match
                        means the corrector kept the LLM's original span even
                        though it disagrees with the independent human —
                        a direct anchoring signal.

    Returns
    -------
    {
        "human_vs_humanllm": {overall, per_document, per_label}   # same shape as evaluate_batch
        "anchoring": {
            "overall_rate": float,          # matches_llm / unmatched_hllm
            "per_document": {...},
            "per_label": {...},
        }
    }
    """
    accumulated  = defaultdict(lambda: {"tp": 0, "n_gold": 0, "n_system": 0})
    per_document = {}
    doc_f1s      = []

    anchoring_per_document = {}
    anchoring_accum = defaultdict(lambda: {"unmatched_hllm": 0, "matches_llm": 0})

    print(f"\nTriple batch evaluation — {len(pairs_h_hllm_llm)} triple(s)")
    print("=" * 60)

    for i, (human_path, hllm_path, llm_path) in enumerate(pairs_h_hllm_llm, 1):
        doc_name = Path(human_path).stem
        print(f"\n[{i}/{len(pairs_h_hllm_llm)}] {doc_name}")

        try:
            human_spans = extract_spans(human_path, context_chars)
            hllm_spans  = extract_spans(hllm_path,  context_chars)
            llm_spans   = extract_spans(llm_path,   context_chars)
        except Exception as e:
            print(f"  ERROR reading files for {doc_name}: {e}")
            continue

        # ---------------- Pass 1: human (gold) vs human+llm (system) ----------------
        metrics = compute_metrics(human_spans, hllm_spans)   # <- existing function, unchanged

        doc_tp       = sum(m["tp"]       for m in metrics.values())
        doc_n_gold   = sum(m["n_gold"]   for m in metrics.values())
        doc_n_system = sum(m["n_system"] for m in metrics.values())
        doc_P, doc_R, doc_F1 = _prf(doc_tp, doc_n_gold, doc_n_system)

        per_document[doc_name] = {
            "tp": doc_tp, "n_gold": doc_n_gold, "n_system": doc_n_system,
            "P": doc_P, "R": doc_R, "F1": doc_F1,
        }
        doc_f1s.append(doc_F1)

        for label, m in metrics.items():
            accumulated[label]["tp"]       += m["tp"]
            accumulated[label]["n_gold"]   += m["n_gold"]
            accumulated[label]["n_system"] += m["n_system"]

        if not verbose_per_file:
            print(f"  [Human vs Human+LLM] TP={doc_tp} | Gold={doc_n_gold} | "
                  f"System={doc_n_system} | F1={doc_F1*100:.1f}%")

        # ---------------- Pass 2: unmatched human+llm spans vs raw LLM ----------------
        all_labels = sorted({s.labelname for s in human_spans} | {s.labelname for s in hllm_spans})

        doc_unmatched  = 0
        doc_matches_llm = 0

        for label in all_labels:
            g   = [s for s in human_spans if s.labelname == label]
            sys_ = [s for s in hllm_spans  if s.labelname == label]

            unmatched_sys = _get_unmatched_system(g, sys_)   # spans human+llm added/changed that human's gold doesn't confirm
            if not unmatched_sys:
                continue

            llm_label_spans = [s for s in llm_spans if s.labelname == label]

            if use_id_shortcut:
                llm_ids = {s.attributes.get("id") for s in llm_label_spans}
                tp_vs_llm = sum(1 for s in unmatched_sys if s.attributes.get("id") in llm_ids)
            else:
                # reuse match_spans(): "gold" = unmatched human+llm spans,
                # "system" = raw LLM spans. tp = identical to the LLM's raw output.
                tp_vs_llm, n_unmatched, _ = match_spans(unmatched_sys, llm_label_spans)

            n_unmatched = len(unmatched_sys)
            doc_unmatched   += n_unmatched
            doc_matches_llm += tp_vs_llm

            anchoring_accum[label]["unmatched_hllm"] += n_unmatched
            anchoring_accum[label]["matches_llm"]    += tp_vs_llm

        anchor_rate = doc_matches_llm / doc_unmatched if doc_unmatched else 0.0
        anchoring_per_document[doc_name] = {
            "unmatched_hllm": doc_unmatched,
            "matches_llm":    doc_matches_llm,
            "anchor_rate":    anchor_rate,
        }

        print(f"  [Anchoring check] Human+LLM spans not matching Human gold: {doc_unmatched} "
              f"| identical to raw LLM: {doc_matches_llm} ({anchor_rate*100:.1f}%)")

    # ---------------- Overall summary, pass 1 ----------------
    total_tp       = sum(d["tp"]       for d in per_document.values())
    total_n_gold   = sum(d["n_gold"]   for d in per_document.values())
    total_n_system = sum(d["n_system"] for d in per_document.values())
    total_P, total_R, total_F1 = _prf(total_tp, total_n_gold, total_n_system)

    overall = {"tp": total_tp, "n_gold": total_n_gold, "n_system": total_n_system,
               "P": total_P, "R": total_R, "F1": total_F1}

    per_label = {}
    for label, counts in accumulated.items():
        P, R, F = _prf(counts["tp"], counts["n_gold"], counts["n_system"])
        per_label[label] = {**counts, "P": P, "R": R, "F1": F}

    # ---------------- Overall summary, anchoring ----------------
    total_unmatched   = sum(a["unmatched_hllm"] for a in anchoring_per_document.values())
    total_matches_llm = sum(a["matches_llm"]    for a in anchoring_per_document.values())
    overall_anchor_rate = total_matches_llm / total_unmatched if total_unmatched else 0.0

    anchoring_per_label = {}
    for label, counts in anchoring_accum.items():
        rate = counts["matches_llm"] / counts["unmatched_hllm"] if counts["unmatched_hllm"] else 0.0
        anchoring_per_label[label] = {**counts, "anchor_rate": rate}

    print("\n" + "=" * 60)
    print("RESULTS — HUMAN vs HUMAN+LLM (TOTAL)")
    print("=" * 60)
    print(f"TP={total_tp} | Gold={total_n_gold} | System={total_n_system} | "
          f"P={total_P*100:.1f}% | R={total_R*100:.1f}% | F1={total_F1*100:.1f}%")

    print("\n" + "=" * 60)
    _print_table(per_label, title="RESULTS — MICRO-AVERAGED PER LABEL (Human vs Human+LLM)", doc_f1s=doc_f1s)

    print("=" * 60)
    print("RESULTS — ANCHORING CHECK (unmatched Human+LLM spans vs raw LLM)")
    print("=" * 60)
    print(f"{'Label':<34} {'Unmatched':>10} {'=LLM':>8} {'Rate':>8}")
    print("─" * 62)
    for label in sorted(anchoring_per_label):
        a = anchoring_per_label[label]
        print(f"{label:<34} {a['unmatched_hllm']:>10} {a['matches_llm']:>8} {a['anchor_rate']*100:>7.1f}%")
    print("─" * 62)
    print(f"{'TOTAL':<34} {total_unmatched:>10} {total_matches_llm:>8} {overall_anchor_rate*100:>7.1f}%\n")

    return {
        "human_vs_humanllm": {"overall": overall, "per_document": per_document, "per_label": per_label},
        "anchoring": {
            "overall_rate": overall_anchor_rate,
            "total_unmatched": total_unmatched,
            "total_matches_llm": total_matches_llm,
            "per_document": anchoring_per_document,
            "per_label": anchoring_per_label,
        },
    }


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def show_errors(
    gold_path: str,
    system_path: str,
    label_filter: Optional[str] = None,
    context_chars: int = CONTEXT_CHARS,
    max_show: int = 20,
) -> None:
    """
    Print False Negatives (missed) and False Positives (spurious) for one pair.

    Parameters
    ----------
    gold_path, system_path : str
    label_filter : str or None
        Restrict output to one label (e.g. "decision").
    max_show : int
        Maximum number of errors to show per category.
    """
    gold_spans   = extract_spans(gold_path,   context_chars)
    system_spans = extract_spans(system_path, context_chars)

    if label_filter:
        lf = label_filter.strip().lower()
        gold_spans   = [s for s in gold_spans   if s.labelname == lf]
        system_spans = [s for s in system_spans if s.labelname == lf]

    # Greedy matching (same logic as match_spans)
    used_system     = [False] * len(system_spans)
    false_negatives = []

    for g in gold_spans:
        matched = False
        for j, s in enumerate(system_spans):
            if not used_system[j] and g.matches(s):
                used_system[j] = True
                matched = True
                break
        if not matched:
            false_negatives.append(g)

    false_positives = [system_spans[j] for j, u in enumerate(used_system) if not u]

    lf_str = f" [{label_filter}]" if label_filter else ""
    print(f"\n{'─'*60}")
    print(f"  ERROR ANALYSIS{lf_str}")
    print(f"  Gold  : {Path(gold_path).name}")
    print(f"  System: {Path(system_path).name}")
    print(f"{'─'*60}")

    def _preview(s: Span) -> str:
        return s.text[:100] + ("…" if len(s.text) > 100 else "")

    print(f"\n  FALSE NEGATIVES — gold spans missed by system ({len(false_negatives)} total)")
    for s in false_negatives[:max_show]:
        print(f"  [{s.labelname}] {_preview(s)}")
    if len(false_negatives) > max_show:
        print(f"  … and {len(false_negatives) - max_show} more")

    print(f"\n  FALSE POSITIVES — spurious system spans ({len(false_positives)} total)")
    for s in false_positives[:max_show]:
        print(f"  [{s.labelname}] {_preview(s)}")
    if len(false_positives) > max_show:
        print(f"  … and {len(false_positives) - max_show} more")

    print()


# ---------------------------------------------------------------------------
# Entry point — edit the paths below to run
# ---------------------------------------------------------------------------

if __name__ == "__main__":


    pairs = [
    ("data/annotated/test/1989CanLII1415ONCA.html", "data/supplementary/manual_extraction/1989CanLII1415ONCA_extraction.html"),
    ("data/annotated/test/2005QCCA437.html", "data/supplementary/manual_extraction/2005QCCA437_extraction.html"),
    ("data/annotated/train/2016NBOMB12.html", "data/supplementary/manual_extraction/2016NBOMB12_extraction.html"),
    ("data/annotated/train/1994CanLII4528NLCA.html", "data/supplementary/manual_extraction/1994CanLII4528NLCA_extraction.html"),
    ]

    evaluate_batch(pairs)

    """
    # Single pair
    GOLD   = "data/annotated/test/my_document.html"
    SYSTEM = "data/supplementary/llm_extraction/my_document.html"

    evaluate(GOLD, SYSTEM)
    show_errors(GOLD, SYSTEM, label_filter="decision")

    # Batch — Option A: list pairs manually
    # Batch — Option A: list pairs manually
    pairs = [
        ("data/annotated/test/doc1.html", "data/supplementary/llm_extraction/doc1.html"),
        ("data/annotated/test/doc2.html", "data/supplementary/llm_extraction/doc2.html"),

    # …
]

    # Batch — Option B: auto-match by filename
    # GOLD_DIR   = Path("data/annotated/test")
    # SYSTEM_DIR = Path("data/supplementary/llm_extraction")
    # pairs = [
    #     (str(g), str(SYSTEM_DIR / g.name))
    #     for g in sorted(GOLD_DIR.glob("*.html"))
    #     if (SYSTEM_DIR / g.name).exists()
    # ]

    evaluate_batch(pairs)
    """