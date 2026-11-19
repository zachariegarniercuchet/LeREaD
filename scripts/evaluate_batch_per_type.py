#!/usr/bin/env python
"""
Per-mention-type evaluation (OVERLAP matching) of a stage-2/stage-3 batch
folder: coreference (MUC / B3 / CEAF_e / LEA) and URI linking (stage 3).

Decision, legislation and secondary sources are scored separately; no
combined group.  Coreference uses the same inline metrics as
`scripts/reflex/evaluate_reflex_coref_resolution.py` (equivalent to the coval
functions used by `src/evaluation/evaluation_coref.py`).  URI linking reuses
`src/evaluation/evaluate_resolution.compute_uri_metrics_overlap`, i.e. exactly
the numbers `evaluate_resolution.py batch` prints under "OVERLAP".

Usage (from repo root):

    python scripts/evaluate_batch_per_type.py \
        --batch-folder output_coref/1-final/gpt5.2_pip --split test

    # write a markdown report next to the batch folder
    python scripts/evaluate_batch_per_type.py \
        --batch-folder output_coref/<run> --split test --out report.md

The batch folder must be subfolder-per-document, as produced by main_coref.py /
main_resolution.py: <batch>/<doc>/<doc>_coref.html and
<batch>/<doc>/<doc>_resolution.html.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(PROJECT_ROOT), str(PROJECT_ROOT / "scripts" / "reflex")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from bs4 import BeautifulSoup, NavigableString  # noqa: E402

# Inline coref metrics + overlap alignment (single source of truth, no coval).
from evaluate_reflex_coref_resolution import (  # noqa: E402
    Mention,
    align_and_build_clusters,
    align_and_build_clusters_overlap,
    muc_raw,
    b_cubed_raw,
    ceafe_raw,
    lea_raw,
)


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ER = _load_module(
    "evaluate_resolution_per_type",
    PROJECT_ROOT / "src" / "evaluation" / "evaluate_resolution.py",
)

TYPES = ("decision", "legislation", "secondary sources")
TYPE_DISPLAY = {
    "decision": "Decision",
    "legislation": "Legislation",
    "secondary sources": "Secondary",
}
COREF_METRICS = ("MUC", "B3", "CEAF_e", "LEA")


# ---------------------------------------------------------------------------
# Mention extraction (verbatim semantics of evaluate_coref.py)
# ---------------------------------------------------------------------------


def extract_mentions_with_offsets(html_content: str) -> List[Mention]:
    """Every tag carrying both `id` and `docid`, as a Mention whose identity is
    (labelname, start, end) in whitespace-normalised plain text."""
    soup = BeautifulSoup(html_content, "html.parser")

    pieces: List[str] = []
    found = []  # (tag, raw_start, raw_end)
    pos = 0

    def walk(node) -> None:
        nonlocal pos
        for child in node.children:
            if isinstance(child, NavigableString):
                if type(child) is NavigableString:
                    pieces.append(str(child))
                    pos += len(child)
            else:
                start = pos
                walk(child)
                if child.has_attr("docid") and child.has_attr("id"):
                    found.append((child, start, pos))

    walk(soup)
    raw = "".join(pieces)

    norm_pos = [0] * (len(raw) + 1)
    norm_chars: List[str] = []
    prev_ws = False
    for i, ch in enumerate(raw):
        norm_pos[i] = len(norm_chars)
        if ch.isspace():
            if not prev_ws:
                norm_chars.append(" ")
            prev_ws = True
        else:
            norm_chars.append(ch)
            prev_ws = False
    norm_pos[len(raw)] = len(norm_chars)
    norm_text = "".join(norm_chars)

    mentions: List[Mention] = []
    seen: Dict[str, int] = defaultdict(int)
    for tag, s, e in found:
        while s < e and raw[s].isspace():
            s += 1
        while e > s and raw[e - 1].isspace():
            e -= 1
        start, end = norm_pos[s], norm_pos[e]
        label = tag.get("labelname", "")
        base = f"{label}|{start}|{end}"
        n = seen[base]
        seen[base] += 1
        key = base if n == 0 else f"{base}#{n}"
        mentions.append(Mention(
            key=key,
            docid=(tag.get("docid") or "").strip(),
            label=label,
            start=start,
            end=end,
            text=norm_text[start:end],
        ))
    return mentions


def derive_doc_stem(system_path: Path) -> str:
    name = system_path.stem
    for suffix in ("_coref_clusters", "_coref_mapping", "_coref"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _first_nonempty_glob(folder: Path, pattern: str) -> List[Path]:
    return sorted(p for p in folder.glob(pattern) if p.is_file())


def collect_coref_pairs(batch_folder: Path, gold_dir: Path) -> List[Tuple[Path, Path]]:
    gold_files = {p.stem: p for p in gold_dir.glob("*.html")}

    direct = _first_nonempty_glob(batch_folder, "*_coref.html")
    system_files = direct
    if not system_files:
        for sub in sorted(d for d in batch_folder.iterdir() if d.is_dir()):
            candidates = _first_nonempty_glob(sub, "*_coref.html")
            if candidates:
                system_files.append(candidates[0])
            else:
                print(f"  ⚠ No *_coref.html in: {sub.name}")

    pairs = []
    for system_file in system_files:
        stem = derive_doc_stem(system_file)
        gold = gold_files.get(stem)
        if gold is None:
            print(f"  ⚠ No matching gold file for: {stem}")
            continue
        pairs.append((gold, system_file))
    return pairs


# ---------------------------------------------------------------------------
# Coreference (OVERLAP) per type — micro aggregation
# ---------------------------------------------------------------------------


def _empty_counts() -> Dict[str, int]:
    return {"gold": 0, "system": 0, "matched": 0, "docs": 0}


def _add_raw(acc: Dict[str, List[float]], raw, name: str) -> None:
    if name not in acc:
        acc[name] = [0.0, 0.0, 0.0, 0.0]
    for i, value in enumerate(raw):
        acc[name][i] += value


def _finalize_coref(acc: Dict[str, List[float]], counts: Dict[str, int]) -> dict:
    out = {}
    for name in COREF_METRICS:
        p_num, p_den, r_num, r_den = acc.get(name, (0.0, 0.0, 0.0, 0.0))
        precision = p_num / p_den if p_den else 0.0
        recall = r_num / r_den if r_den else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        out[name] = {"precision": precision, "recall": recall, "f1": f1}
    out["CoNLL"] = {
        "f1": (out["MUC"]["f1"] + out["B3"]["f1"] + out["CEAF_e"]["f1"]) / 3,
        "precision": None,
        "recall": None,
    }
    out["_counts"] = counts
    return out


def score_coref(pairs: List[Tuple[Path, Path]], unmatched: str,
                match_mode: str = "overlap") -> Dict[str, dict]:
    align = (align_and_build_clusters_overlap if match_mode == "overlap"
             else align_and_build_clusters)
    accs = {t: {} for t in TYPES}
    counts = {t: _empty_counts() for t in TYPES}

    for gold_path, system_path in pairs:
        gold_mentions = extract_mentions_with_offsets(
            gold_path.read_text(encoding="utf-8", errors="replace"))
        system_mentions = extract_mentions_with_offsets(
            system_path.read_text(encoding="utf-8", errors="replace"))

        for type_key in TYPES:
            fg = [m for m in gold_mentions if m.label == type_key]
            fs = [m for m in system_mentions if m.label == type_key]
            gold_clusters, system_clusters, stats = align(
                fg, fs, unmatched=unmatched)

            _add_raw(accs[type_key], muc_raw(system_clusters, gold_clusters), "MUC")
            _add_raw(accs[type_key], b_cubed_raw(system_clusters, gold_clusters), "B3")
            _add_raw(accs[type_key], ceafe_raw(system_clusters, gold_clusters), "CEAF_e")
            _add_raw(accs[type_key], lea_raw(system_clusters, gold_clusters), "LEA")

            counts[type_key]["gold"] += stats["gold"]
            counts[type_key]["system"] += stats["system"]
            counts[type_key]["matched"] += stats["matched"]
            counts[type_key]["docs"] += 1

    return {
        type_key: _finalize_coref(accs[type_key], counts[type_key])
        for type_key in TYPES
    }


# ---------------------------------------------------------------------------
# URI linking (OVERLAP) per type — reuse evaluate_resolution.py
# ---------------------------------------------------------------------------


def score_resolution(batch_folder: Path, gold_dir: Path,
                     match_mode: str = "overlap") -> Tuple[dict, dict, dict, int]:
    data_dir = gold_dir.parents[1]
    split = gold_dir.name
    pairs = ER.collect_pairs_from_batch_folder(
        batch_folder, data_dir, split, suffix=ER.RESOLUTION_SUFFIX, verbose=False)

    batch_fn = (ER.evaluate_resolution_batch_overlap if match_mode == "overlap"
                else ER.evaluate_resolution_batch)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        overall, per_label, per_category = batch_fn(
            [(str(g), str(s)) for g, s in pairs],
            context_chars=ER.CONTEXT_CHARS,
            verbose_per_file=False,
        )
    return overall, per_label, per_category, len(pairs)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def md_table(header: List[str], rows: List[List[str]]) -> str:
    out = ["| " + " | ".join(header) + " |",
           "| " + " | ".join(["---"] * len(header)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(out)


def _pct(value: float) -> str:
    return f"{value * 100:.2f}"


def build_report(batch_folder: Path, res_folder: Path, split: str, unmatched: str,
                 match_mode: str, coref: Dict[str, dict], coref_docs: int,
                 res_overall: dict, res_per_label: Dict[str, dict],
                 res_docs: int) -> str:
    lines: List[str] = []
    A = lines.append
    A(f"# Per-type evaluation — `{batch_folder.name}`")
    A("")
    A(f"*Split: `{split}` — {match_mode} matching — coref unmatched policy: "
      f"`{unmatched}`*")
    A("")
    A(f"Coref documents: {coref_docs} | URI-linking documents: {res_docs} "
      f"(`{res_folder.name}`)")
    A("")
    A(f"## Coreference ({match_mode.upper()}, micro)")
    A("")
    rows = []
    for type_key in TYPES:
        s = coref[type_key]
        c = s["_counts"]
        rows.append([TYPE_DISPLAY[type_key], c["docs"], c["gold"], c["system"],
                     c["matched"]] +
                    [f"{_pct(s[m]['f1'])}" for m in COREF_METRICS] +
                    [f"{_pct(s['CoNLL']['f1'])}"])
    A(md_table(["type", "docs", "gold mentions", "sys mentions", "matched",
                "MUC F1 %", "B3 F1 %", "CEAF_e F1 %", "LEA F1 %", "CoNLL F1 %"],
               rows))
    A("")
    A("LEA detail:")
    A("")
    rows = []
    for type_key in TYPES:
        s = coref[type_key]["LEA"]
        rows.append([TYPE_DISPLAY[type_key], f"{_pct(s['precision'])}",
                     f"{_pct(s['recall'])}", f"{_pct(s['f1'])}"])
    A(md_table(["type", "LEA P %", "LEA R %", "LEA F1 %"], rows))
    A("")
    A("## URI linking (EXACT, micro)" if match_mode == "exact"
      else "## URI linking (OVERLAP, micro)")
    A("")
    rows = []
    for type_key in TYPES:
        m = res_per_label.get(type_key)
        if m is None:
            rows.append([TYPE_DISPLAY[type_key], "-", "-", "-", "-", "-", "-"])
            continue
        rows.append([TYPE_DISPLAY[type_key], m["tp"], m["fn"], m["fp"],
                     f"{_pct(m['precision'])}", f"{_pct(m['recall'])}",
                     f"{_pct(m['f1'])}"])
    A(md_table(["type", "TP", "FN", "FP", "Precision %", "Recall %", "F1 %"], rows))
    A("")
    A(f"Overall URI linking F1: {_pct(res_overall['f1'])} "
      f"(P {_pct(res_overall['precision'])}, R {_pct(res_overall['recall'])})")
    A("")
    return "\n".join(lines)


def print_report(report: str) -> None:
    print(report)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-type coref + URI linking evaluation (overlap or exact).")
    ap.add_argument("--batch-folder", required=True,
                    help="Coref batch folder (subfolder-per-document).")
    ap.add_argument("--resolution-folder", default=None,
                    help="Optional separate folder holding *_resolution.html "
                         "(defaults to --batch-folder).")
    ap.add_argument("--split", required=True)
    ap.add_argument("--data-dir", default=str(PROJECT_ROOT / "data"))
    ap.add_argument("--unmatched", choices=["singleton", "drop"], default="singleton")
    ap.add_argument("--match-mode", choices=["overlap", "exact"], default="overlap",
                    help="Mention matching mode for coref and URI linking.")
    ap.add_argument("--out", default=None, help="optional markdown output path")
    args = ap.parse_args()

    batch_folder = Path(args.batch_folder)
    res_folder = Path(args.resolution_folder) if args.resolution_folder else batch_folder
    gold_dir = Path(args.data_dir) / "annotated" / args.split
    if not gold_dir.exists():
        raise FileNotFoundError(f"Gold directory not found: {gold_dir}")

    coref_pairs = collect_coref_pairs(batch_folder, gold_dir)
    coref = score_coref(coref_pairs, args.unmatched, args.match_mode)
    res_overall, res_per_label, _res_per_cat, res_docs = score_resolution(
        res_folder, gold_dir, args.match_mode)

    report = build_report(batch_folder, res_folder, args.split, args.unmatched,
                          args.match_mode, coref, len(coref_pairs),
                          res_overall, res_per_label, res_docs)
    print_report(report)
    if args.out:
        Path(args.out).write_text(report, encoding="utf-8")
        print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
