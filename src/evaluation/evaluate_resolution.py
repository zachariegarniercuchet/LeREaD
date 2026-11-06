"""
LeREaD Stage-3 (Resolution) Evaluation Script

Evaluates the `uri` attribute end-to-end, for both <manual_label> and
<auto_label> (no distinction is made), as Precision / Recall / F1 over the
mentions that actually need a uri.

Definitions
-----------
A gold span's uri falls into one of three states:

  1. uri="none"      -> gold says "not resolvable"  -> EXCLUDED from P/R/F1
  2. uri absent       -> label type has no uri concept -> EXCLUDED from P/R/F1
  3. uri="<value>"    -> a genuine instance that must be resolved -> COUNTED

Only category 3 ("real-uri") gold spans are used to build the metric.

  TP   = matched (Level-1) pair, gold has a real uri, system predicted the
         same value.
  FN   = a real-uri gold mention the pipeline failed to resolve correctly,
         whether because:
           (a) extraction never found the span at all (fn_missed_extraction),
           (b) extraction found the span but disambiguation produced no uri
               at all (fn_missing_sys_uri),
           (c) extraction found the span but disambiguation produced the
               WRONG uri (fn_wrong).
  FP   = a uri the system asserted that is wrong:
           (a) fn_wrong above (wrong value on a matched real-uri gold span)
               counts as FP too - "a wrong uri counts once against each",
           (b) fp_spurious: a system span with no gold counterpart at all
               (an extraction hallucination) that nonetheless carries a uri.

  Recall    = TP / (TP + fn_wrong + fn_missing_sys_uri + fn_missed_extraction)
            = TP / (all real-uri gold mentions, matched or not)
  Precision = TP / (TP + fn_wrong + fp_spurious)
            = TP / (all system uri assertions that map onto a real-uri
              gold instance, plus spurious ones that don't)
  F1        = harmonic mean of the two

Sanity check (oracle mode): if you feed the resolver the gold mention exactly
(no extraction step - see your own oracle setup, hiding only the uri), then
there are no missed extractions and no spurious spans, so
fn_missed_extraction = fp_spurious = 0 and precision = recall = accuracy,
i.e. this reduces to your old "conditional accuracy" number.

We ALSO keep that old conditional "accuracy" as a diagnostic
(TP / (TP + fn_wrong + fn_missing_sys_uri), i.e. computed only over spans
that matched at Level 1) since it isolates disambiguation quality alone,
decoupled from extraction misses - useful to see where the pipeline's
errors are coming from when it disagrees with recall.

Expected stage-3 output layout::

    output_coref/
      test_gemma4_coref_fs6_random_gold/     <- batch folder (split = "test")
        2001CanLII21117/
          2001CanLII21117_coref.html         <- stage 2
          2001CanLII21117_resolution.html    <- stage 3  (evaluated)
        2005QCCA437/
          ...

Gold files are looked up in ``<data-dir>/annotated/<split>/<docid>.html``.

Usage:
    # Batch folder (split auto-detected from the folder name)
    python evaluate_resolution.py batch --batch-folder output_coref/test_gemma4_coref_fs6_random_gold/

    # Batch folder with explicit split
    python evaluate_resolution.py batch --batch-folder output_coref/my_run/ --split test

    # Single file
    python evaluate_resolution.py single --split test \
        --system-file output_coref/test_gemma4_coref_fs6_random_gold/2001CanLII21117/2001CanLII21117_resolution.html



python -m src.evaluation.evaluate_resolution batch --batch-folder output_coref/test_gemma4_coref_fs6_random_gold
"""

import argparse
import contextlib
import importlib.util
import io
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Imports (same pattern as evaluate.py: load modules directly by path so that
# we never trigger a full `src` package import)
# ---------------------------------------------------------------------------

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

_spec_l1 = importlib.util.spec_from_file_location(
    "evaluation_l1_util",
    project_root / "src" / "evaluation" / "evaluation_l1_util.py",
)
evaluation_l1_util = importlib.util.module_from_spec(_spec_l1)
_spec_l1.loader.exec_module(evaluation_l1_util)

Span = evaluation_l1_util.Span
extract_spans = evaluation_l1_util.extract_spans
CONTEXT_CHARS = getattr(evaluation_l1_util, "CONTEXT_CHARS", 200)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

URI_ATTR = "uri"
RESOLUTION_SUFFIX = "_resolution.html"
VALID_SPLITS = ("train", "test", "dev", "incoming", "extended_test", "dev_train")


@dataclass
class ResolutionConfig:
    """Configuration for a stage-3 (resolution) evaluation run."""

    mode: str = "batch"                       # "single" | "batch"
    split: Optional[str] = None
    system_file: Optional[str] = None
    batch_folder: Optional[str] = None
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    context_chars: int = CONTEXT_CHARS
    file_suffix: str = RESOLUTION_SUFFIX
    verbose_per_file: bool = False

    def get_ground_truth_dir(self) -> Path:
        if not self.split:
            raise ValueError("split is not set")
        return Path(self.data_dir) / "annotated" / self.split

    def validate(self) -> None:
        if self.mode == "single":
            if not self.system_file:
                raise ValueError("single mode requires --system-file")
            if not self.split:
                raise ValueError("single mode requires --split")
        elif self.mode == "batch":
            if not self.batch_folder:
                raise ValueError("batch mode requires --batch-folder")
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        if self.split and self.split not in VALID_SPLITS:
            raise ValueError(f"Unknown split '{self.split}' (expected one of {VALID_SPLITS})")


# ---------------------------------------------------------------------------
# Level-1 matching (greedy, exact) — same logic as evaluation_l2_util
# ---------------------------------------------------------------------------

def _match_spans_l1(gold: List[Span], system: List[Span]) -> List[Tuple[Span, Span]]:
    """Return (gold_span, system_span) pairs matching at Level 1; each span used once."""
    used = [False] * len(system)
    pairs: List[Tuple[Span, Span]] = []

    for g in gold:
        for j, s in enumerate(system):
            if not used[j] and g.matches(s):
                pairs.append((g, s))
                used[j] = True
                break

    return pairs


# ---------------------------------------------------------------------------
# URI helpers
# ---------------------------------------------------------------------------

# Gold values meaning "there is no URI to resolve to" -> never penalise
NO_ANSWER_VALUES = {"none"}


def _is_no_answer(value: Optional[str]) -> bool:
    """True if the (already stripped) uri value is the 'no answer' placeholder."""
    return value is not None and value.strip().lower() in NO_ANSWER_VALUES


def _get_uri(span: Span) -> Optional[str]:
    """Return the raw uri attribute of a span, or None if absent/empty."""
    value = span.attributes.get(URI_ATTR)
    if value is None:
        return None
    value = value.strip()
    return value if value else None


def _is_real_uri(value: Optional[str]) -> bool:
    """True iff this uri value is a genuine instance that must be resolved
    (i.e. not absent, and not the explicit 'none' no-answer placeholder)."""
    return value is not None and not _is_no_answer(value)


def _uri_agree(v1: Optional[str], v2: Optional[str]) -> bool:
    """True iff both values are non-None and equal after normalisation."""
    if v1 is None or v2 is None:
        return False
    return v1.lower().strip() == v2.lower().strip()


def _category_of(span: Span) -> str:
    """Top-level category of a span (legislation / decision / secondary sources / other)."""
    lname = (span.labelname or "").lower()
    if lname in {"legislation", "decision", "secondary sources"}:
        return lname
    parent = span.attributes.get("parent", "")
    top = parent.split(",")[0].strip().lower() if parent else ""
    return top if top in {"legislation", "decision", "secondary sources"} else "other"


def _prf1(tp: int, fn: int, fp: int) -> Tuple[float, float, float]:
    """Precision / recall / F1 from raw counts. 0.0 on empty denominators."""
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_uri_metrics(gold: List[Span], system: List[Span]) -> Tuple[Dict, Dict[str, Dict], Dict[str, Dict]]:
    """
    Compute end-to-end uri Precision / Recall / F1.

    Only gold spans whose uri is a "real" value (not absent, not the
    explicit "none" no-answer placeholder) are counted as instances that
    need resolving. See module docstring for the exact TP/FN/FP breakdown.

    Returns
    -------
    overall : dict
        tp, fn_wrong, fn_missing_sys_uri, fn_missed_extraction, fp_spurious,
        skipped_gold_none, matched_spans, precision, recall, f1, accuracy
        (the old Level-1-conditional accuracy, kept as a diagnostic).
    per_label : dict
        {labelname -> {tp, fn, fp, precision, recall, f1}}
    per_category : dict
        {category -> {tp, fn, fp, precision, recall, f1}}
    """
    matched_pairs = _match_spans_l1(gold, system)
    matched_gold_ids = {id(g) for g, _ in matched_pairs}
    matched_system_ids = {id(s) for _, s in matched_pairs}

    unmatched_gold = [g for g in gold if id(g) not in matched_gold_ids]
    unmatched_system = [s for s in system if id(s) not in matched_system_ids]

    tp = 0
    fn_wrong = 0             # matched pair, gold real uri, system uri present but WRONG
    fn_missing_sys_uri = 0   # matched pair, gold real uri, system asserted NO uri at all
    fn_missed_extraction = 0  # gold real-uri span never found by extraction
    fp_spurious = 0          # system span with no gold counterpart at all, but carries a uri
    skipped_gold_none = 0

    per_label_acc: Dict[str, dict] = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})
    per_cat_acc: Dict[str, dict] = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})

    for g, s in matched_pairs:
        u_g = _get_uri(g)
        if not _is_real_uri(u_g):
            skipped_gold_none += 1
            continue

        label = g.labelname or "(none)"
        cat = _category_of(g)
        u_s = _get_uri(s)

        if u_s is not None and _uri_agree(u_g, u_s):
            tp += 1
            per_label_acc[label]["tp"] += 1
            per_cat_acc[cat]["tp"] += 1
        elif u_s is not None:
            # system attempted a resolution, but it's the wrong one:
            # counts against BOTH recall (FN) and precision (FP)
            fn_wrong += 1
            per_label_acc[label]["fn"] += 1
            per_label_acc[label]["fp"] += 1
            per_cat_acc[cat]["fn"] += 1
            per_cat_acc[cat]["fp"] += 1
        else:
            # span matched, but disambiguation produced nothing at all:
            # recall hit only (system made no assertion, so no FP)
            fn_missing_sys_uri += 1
            per_label_acc[label]["fn"] += 1
            per_cat_acc[cat]["fn"] += 1

    for g in unmatched_gold:
        u_g = _get_uri(g)
        if not _is_real_uri(u_g):
            continue  # extraction missed a mention that didn't need a uri anyway
        fn_missed_extraction += 1
        label = g.labelname or "(none)"
        cat = _category_of(g)
        per_label_acc[label]["fn"] += 1
        per_cat_acc[cat]["fn"] += 1

    for s in unmatched_system:
        u_s = _get_uri(s)
        if not _is_real_uri(u_s):
            continue  # extraction hallucinated a span, but it carries no uri anyway
        fp_spurious += 1
        label = s.labelname or "(none)"
        cat = _category_of(s)
        per_label_acc[label]["fp"] += 1
        per_cat_acc[cat]["fp"] += 1

    fn_total = fn_wrong + fn_missing_sys_uri + fn_missed_extraction
    fp_total = fn_wrong + fp_spurious
    precision, recall, f1 = _prf1(tp, fn_total, fp_total)

    # Old Level-1-conditional accuracy, kept as a diagnostic: disambiguation
    # quality alone, over spans extraction actually found.
    matched_total = tp + fn_wrong + fn_missing_sys_uri
    accuracy = tp / matched_total if matched_total else 0.0

    overall = {
        "matched_spans": len(matched_pairs),
        "tp": tp,
        "fn_wrong": fn_wrong,
        "fn_missing_sys_uri": fn_missing_sys_uri,
        "fn_missed_extraction": fn_missed_extraction,
        "fp_spurious": fp_spurious,
        "skipped_gold_none": skipped_gold_none,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "matched_total": matched_total,
    }

    def _finalize(acc: Dict[str, dict]) -> Dict[str, dict]:
        out = {}
        for key, d in acc.items():
            p, r, f = _prf1(d["tp"], d["fn"], d["fp"])
            out[key] = {"tp": d["tp"], "fn": d["fn"], "fp": d["fp"],
                        "precision": p, "recall": r, "f1": f}
        return out

    per_label = _finalize(per_label_acc)
    per_category = _finalize(per_cat_acc)

    return overall, per_label, per_category


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_SEP = "─" * 78
_SEP_D = "═" * 78


def print_results(
    overall: Dict,
    per_label: Dict[str, Dict],
    per_category: Dict[str, Dict],
    gold_name: str = "",
    system_name: str = "",
) -> None:
    """Print stage-3 uri resolution P/R/F1 results."""
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " LEVEL 3 – URI RESOLUTION (PRECISION / RECALL / F1)".center(68) + "║")
    print("╚" + "═" * 68 + "╝")

    if gold_name:
        print(f"\n  Gold  : {gold_name}")
    if system_name:
        print(f"  System: {system_name}")

    print("\n  Evaluated attribute : uri")
    print("  Agreement criterion : exact match (case-insensitive, stripped)")
    print("  Counted instances   : gold spans with a REAL uri value only")
    print("                        (uri=\"none\" and absent uri are always excluded)")

    print(f"\n{_SEP}")
    print("  OVERALL")
    print(_SEP)
    print(f"\n  TP                          : {overall['tp']:>6}")
    print(f"  FN - wrong uri (matched)    : {overall['fn_wrong']:>6}")
    print(f"  FN - no uri produced        : {overall['fn_missing_sys_uri']:>6}")
    print(f"  FN - mention never extracted: {overall['fn_missed_extraction']:>6}")
    print(f"  FP - spurious span w/ uri   : {overall['fp_spurious']:>6}")
    print(f"  Skipped (gold uri=none)     : {overall['skipped_gold_none']:>6}")
    print()
    print(f"  PRECISION                   : {overall['precision']*100:>9.2f}%")
    print(f"  RECALL                      : {overall['recall']*100:>9.2f}%")
    print(f"  F1                          : {overall['f1']*100:>9.2f}%")
    print(f"\n  (diagnostic) resolver-only accuracy, i.e. Level-1-matched spans only")
    print(f"  ACCURACY | extraction OK    : {overall['accuracy']*100:>9.2f}%  "
          f"({overall['matched_total']} instances)")

    def _print_breakdown(title: str, table: Dict[str, Dict]) -> None:
        if not table:
            return
        print(f"\n{_SEP}")
        print(f"  {title}")
        print(_SEP)
        W = 32
        print(f"\n  {'Key':<{W}} {'TP':>5} {'FN':>5} {'FP':>5} "
              f"{'Prec':>7} {'Rec':>7} {'F1':>7}")
        print("  " + "─" * (W + 42))
        for key in sorted(table):
            m = table[key]
            print(f"  {key:<{W}} {m['tp']:>5} {m['fn']:>5} {m['fp']:>5} "
                  f"{m['precision']*100:>6.1f}% {m['recall']*100:>6.1f}% {m['f1']*100:>6.1f}%")

    _print_breakdown("PER-LABEL BREAKDOWN", per_label)
    _print_breakdown("PER-CATEGORY BREAKDOWN", per_category)

    print(f"\n{_SEP_D}\n")


# ---------------------------------------------------------------------------
# Single-pair evaluation
# ---------------------------------------------------------------------------

def evaluate_resolution(
    gold_path: str,
    system_path: str,
    context_chars: int = CONTEXT_CHARS,
    verbose: bool = True,
) -> Tuple[Dict, Dict[str, Dict], Dict[str, Dict]]:
    """Stage-3 uri evaluation for one (gold, system) HTML file pair."""
    gold_spans = extract_spans(gold_path, context_chars)
    system_spans = extract_spans(system_path, context_chars)

    overall, per_label, per_category = compute_uri_metrics(gold_spans, system_spans)

    if verbose:
        print_results(
            overall, per_label, per_category,
            gold_name=Path(gold_path).name,
            system_name=Path(system_path).name,
        )

    return overall, per_label, per_category


# ---------------------------------------------------------------------------
# Batch evaluation (micro-averaged)
# ---------------------------------------------------------------------------

_OVERALL_SUM_KEYS = (
    "matched_spans", "tp", "fn_wrong", "fn_missing_sys_uri",
    "fn_missed_extraction", "fp_spurious", "skipped_gold_none",
)


def evaluate_resolution_batch(
    pairs: List[Tuple[str, str]],
    context_chars: int = CONTEXT_CHARS,
    verbose_per_file: bool = False,
) -> Tuple[Dict, Dict[str, Dict], Dict[str, Dict]]:
    """Stage-3 uri evaluation across a list of (gold, system) file pairs (micro-averaged)."""
    acc_overall: Dict[str, int] = defaultdict(int)
    acc_per_label: Dict[str, dict] = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})
    acc_per_cat: Dict[str, dict] = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0})

    per_file_f1: List[float] = []
    n_processed = 0

    print(f"\nLevel-3 (uri) batch evaluation — {len(pairs)} file pair(s)")
    print("=" * 70)

    for i, (gold_path, system_path) in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}] {Path(gold_path).name}")

        try:
            gold_spans = extract_spans(gold_path, context_chars)
            system_spans = extract_spans(system_path, context_chars)
        except Exception as exc:
            print(f"  ⚠ Skipped — {exc}")
            continue

        overall, per_label, per_category = compute_uri_metrics(gold_spans, system_spans)

        if verbose_per_file:
            print_results(
                overall, per_label, per_category,
                gold_name=Path(gold_path).name,
                system_name=Path(system_path).name,
            )
        else:
            print(f"  tp: {overall['tp']:>4}  |  "
                  f"P: {overall['precision']*100:.1f}%  "
                  f"R: {overall['recall']*100:.1f}%  "
                  f"F1: {overall['f1']*100:.1f}%")

        for key in _OVERALL_SUM_KEYS:
            acc_overall[key] += overall[key]

        for label, m in per_label.items():
            acc_per_label[label]["tp"] += m["tp"]
            acc_per_label[label]["fn"] += m["fn"]
            acc_per_label[label]["fp"] += m["fp"]

        for cat, m in per_category.items():
            acc_per_cat[cat]["tp"] += m["tp"]
            acc_per_cat[cat]["fn"] += m["fn"]
            acc_per_cat[cat]["fp"] += m["fp"]

        per_file_f1.append(overall["f1"])
        n_processed += 1

    if n_processed == 0:
        print("\n  No files could be processed.")
        empty = {k: 0 for k in _OVERALL_SUM_KEYS}
        empty.update({"precision": 0.0, "recall": 0.0, "f1": 0.0,
                      "accuracy": 0.0, "matched_total": 0})
        return empty, {}, {}

    fn_total = acc_overall["fn_wrong"] + acc_overall["fn_missing_sys_uri"] + acc_overall["fn_missed_extraction"]
    fp_total = acc_overall["fn_wrong"] + acc_overall["fp_spurious"]
    precision, recall, f1 = _prf1(acc_overall["tp"], fn_total, fp_total)
    matched_total = acc_overall["tp"] + acc_overall["fn_wrong"] + acc_overall["fn_missing_sys_uri"]
    accuracy = acc_overall["tp"] / matched_total if matched_total else 0.0

    final_overall = {
        **{k: acc_overall[k] for k in _OVERALL_SUM_KEYS},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "matched_total": matched_total,
    }

    def _finalize(acc: Dict[str, dict]) -> Dict[str, dict]:
        out = {}
        for key, d in acc.items():
            p, r, f = _prf1(d["tp"], d["fn"], d["fp"])
            out[key] = {"tp": d["tp"], "fn": d["fn"], "fp": d["fp"],
                        "precision": p, "recall": r, "f1": f}
        return out

    final_per_label = _finalize(acc_per_label)
    final_per_cat = _finalize(acc_per_cat)

    print("\n" + "=" * 70)
    print_results(final_overall, final_per_label, final_per_cat)

    macro_f1 = sum(per_file_f1) / len(per_file_f1) if per_file_f1 else 0.0
    print(f"  Macro F1 (mean per-document F1): {macro_f1*100:.2f}%")
    print(f"  Documents evaluated: {n_processed}/{len(pairs)}\n")

    return final_overall, final_per_label, final_per_cat


# ---------------------------------------------------------------------------
# Folder / pairing helpers
# ---------------------------------------------------------------------------

def extract_split_from_folder_name(folder_name: str) -> Optional[str]:
    """Extract the split from a batch folder name, e.g. 'test_gemma4_coref_fs6_random_gold' -> 'test'."""
    parts = folder_name.split("_")
    if parts and parts[0] in VALID_SPLITS:
        return parts[0]
    return None


def find_resolution_html_in_folder(folder_path: Path, suffix: str = RESOLUTION_SUFFIX) -> Optional[Path]:
    """Find the *_resolution.html file inside a document folder (exactly one expected)."""
    candidates = sorted(folder_path.glob(f"*{suffix}"))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        print(f"  ⚠ Several '*{suffix}' files in {folder_path.name}, using {candidates[0].name}")
        return candidates[0]
    return None


def _resolve_gold_stem(subfolder_name: str, html_stem: str, suffix: str, gold_files: Dict[str, Path]) -> Optional[str]:
    """Try several conventions to map a document folder to a gold file stem."""
    candidates = [
        subfolder_name,                                  # 2001CanLII21117
        html_stem[: -len(suffix.replace(".html", ""))]   # 2001CanLII21117_resolution -> 2001CanLII21117
        if html_stem.endswith(suffix.replace(".html", "")) else html_stem,
        subfolder_name.split("_")[0],                    # 2005QCCA437_AIO_... -> 2005QCCA437
    ]
    for cand in candidates:
        if cand in gold_files:
            return cand
    return None


def collect_pairs_from_batch_folder(
    batch_folder: Path,
    data_dir: Path,
    split: str,
    suffix: str = RESOLUTION_SUFFIX,
    verbose: bool = False,
) -> List[Tuple[str, str]]:
    """
    Collect (gold, system) pairs from a stage-3 batch folder.

    Handles:
      1. One subfolder per document, each containing a *_resolution.html file.
      2. Flat *_resolution.html (or *.html) files directly in the batch folder.
    """
    gold_dir = data_dir / "annotated" / split
    if not gold_dir.exists():
        raise ValueError(f"Gold directory does not exist: {gold_dir}")

    gold_files = {p.stem: p for p in gold_dir.glob("*.html")}
    if verbose:
        print(f"Found {len(gold_files)} gold files in {gold_dir}")

    pairs: List[Tuple[str, str]] = []

    # --- Strategy 1: subfolders ------------------------------------------------
    subfolders = sorted(d for d in batch_folder.iterdir() if d.is_dir())
    if subfolders:
        if verbose:
            print(f"Found {len(subfolders)} document subfolder(s)")

        for subfolder in subfolders:
            system_html = find_resolution_html_in_folder(subfolder, suffix)
            if system_html is None:
                print(f"  ⚠ No '*{suffix}' found in: {subfolder.name}")
                continue

            gold_stem = _resolve_gold_stem(subfolder.name, system_html.stem, suffix, gold_files)
            if gold_stem is None:
                print(f"  ⚠ No matching gold file for: {subfolder.name}")
                continue

            pairs.append((str(gold_files[gold_stem]), str(system_html)))
            if verbose:
                print(f"  Paired: {gold_stem} -> {system_html.name}")

        if pairs:
            return pairs

    # --- Strategy 2: flat files ------------------------------------------------
    html_files = sorted(f for f in batch_folder.glob(f"*{suffix}") if f.is_file())
    if not html_files:
        html_files = sorted(f for f in batch_folder.glob("*.html") if f.is_file())

    for system_html in html_files:
        stem = system_html.stem
        if stem.endswith(suffix.replace(".html", "")):
            stem = stem[: -len(suffix.replace(".html", ""))]
        if stem in gold_files:
            pairs.append((str(gold_files[stem]), str(system_html)))
            if verbose:
                print(f"  Paired: {stem} -> {system_html.name}")
        else:
            print(f"  ⚠ No matching gold file for: {stem}")

    if not pairs:
        raise ValueError(f"No evaluable file pairs found in {batch_folder}")

    return pairs


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def save_evaluation_log(folder_path: Path, log_content: str,
                        filename: str = "evaluation_resolution_results.txt") -> None:
    """Save captured evaluation output to a txt file."""
    log_file = folder_path / filename
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(log_content)
    print(f"✓ Saved evaluation results to: {log_file}")


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_single_file_evaluation(config: ResolutionConfig):
    system_path = Path(config.system_file)
    if not system_path.exists():
        raise FileNotFoundError(f"System file not found: {system_path}")

    gold_dir = config.get_ground_truth_dir()

    stem = system_path.stem
    marker = config.file_suffix.replace(".html", "")
    if stem.endswith(marker):
        stem = stem[: -len(marker)]

    gold_path = gold_dir / f"{stem}.html"
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_path}")

    header = (
        f"\n{'='*70}\n"
        f"SINGLE FILE RESOLUTION EVALUATION (uri)\n"
        f"{'='*70}\n"
        f"Split:  {config.split}\n"
        f"Gold:   {gold_path}\n"
        f"System: {system_path}\n"
        f"{'='*70}\n"
    )
    print(header)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        overall, per_label, per_category = evaluate_resolution(
            str(gold_path), str(system_path),
            context_chars=config.context_chars,
            verbose=True,
        )
    captured = buffer.getvalue()
    print(captured)

    save_evaluation_log(system_path.parent, header + captured)

    return overall, per_label, per_category


def run_batch_evaluation(config: ResolutionConfig):
    batch_folder = Path(config.batch_folder)
    if not batch_folder.exists():
        raise FileNotFoundError(f"Batch folder not found: {batch_folder}")

    split = config.split
    if not split:
        detected = extract_split_from_folder_name(batch_folder.name)
        if detected:
            split = detected
            config.split = split
            print(f"Auto-detected split from folder name: {split}")
        else:
            raise ValueError(
                f"Could not auto-detect split from folder name: {batch_folder.name}\n"
                f"Please provide --split explicitly."
            )

    header = (
        f"\n{'='*70}\n"
        f"BATCH RESOLUTION EVALUATION (uri)\n"
        f"{'='*70}\n"
        f"Batch folder: {batch_folder}\n"
        f"Split: {split}\n"
        f"File suffix: {config.file_suffix}\n"
        f"{'='*70}\n"
    )
    print(header)

    pairs = collect_pairs_from_batch_folder(
        batch_folder,
        Path(config.data_dir),
        split,
        suffix=config.file_suffix,
        verbose=config.verbose_per_file,
    )

    print(f"Found {len(pairs)} evaluation pair(s)\n")

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        results = evaluate_resolution_batch(
            pairs,
            context_chars=config.context_chars,
            verbose_per_file=config.verbose_per_file,
        )
    captured = buffer.getvalue()
    print(captured)

    full_log = header + f"Found {len(pairs)} evaluation pair(s)\n\n" + captured
    save_evaluation_log(batch_folder, full_log)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LeREaD Stage-3 Resolution Evaluation (uri Precision/Recall/F1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Evaluation mode")

    # ========== SINGLE ==========
    single_parser = subparsers.add_parser("single", help="Evaluate a single *_resolution.html file")
    single_parser.add_argument("--split", required=True, choices=list(VALID_SPLITS),
                               help="Ground truth split (required for single file evaluation)")
    single_parser.add_argument("--system-file", required=True,
                               help="Path to the stage-3 output HTML file")
    single_parser.add_argument("--data-dir", default="./data",
                               help="Root data directory (default: ./data)")
    single_parser.add_argument("--context-chars", type=int, default=CONTEXT_CHARS,
                               help=f"Context characters for span matching (default: {CONTEXT_CHARS})")
    single_parser.add_argument("--file-suffix", default=RESOLUTION_SUFFIX,
                               help=f"Stage-3 file suffix (default: {RESOLUTION_SUFFIX})")
    single_parser.add_argument("--verbose", action="store_true", help="Verbose output")

    # ========== BATCH ==========
    batch_parser = subparsers.add_parser("batch", help="Evaluate a whole stage-3 output folder")
    batch_parser.add_argument("--batch-folder", required=True,
                              help="Path to the batch folder, e.g. output_coref/test_gemma4_coref_fs6_random_gold")
    batch_parser.add_argument("--split", default=None, choices=list(VALID_SPLITS),
                              help="Ground truth split (auto-detected from folder name if omitted)")
    batch_parser.add_argument("--data-dir", default="./data",
                              help="Root data directory (default: ./data)")
    batch_parser.add_argument("--context-chars", type=int, default=CONTEXT_CHARS,
                              help=f"Context characters for span matching (default: {CONTEXT_CHARS})")
    batch_parser.add_argument("--file-suffix", default=RESOLUTION_SUFFIX,
                              help=f"Stage-3 file suffix (default: {RESOLUTION_SUFFIX})")
    batch_parser.add_argument("--verbose", action="store_true", help="Verbose output per file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "single":
        config = ResolutionConfig(
            mode="single",
            split=args.split,
            system_file=args.system_file,
            data_dir=Path(args.data_dir),
            context_chars=args.context_chars,
            file_suffix=args.file_suffix,
            verbose_per_file=args.verbose,
        )
        config.validate()
        run_single_file_evaluation(config)

    elif args.command == "batch":
        config = ResolutionConfig(
            mode="batch",
            split=args.split,
            batch_folder=args.batch_folder,
            data_dir=Path(args.data_dir),
            context_chars=args.context_chars,
            file_suffix=args.file_suffix,
            verbose_per_file=args.verbose,
        )
        config.validate()
        run_batch_evaluation(config)


if __name__ == "__main__":
    main()