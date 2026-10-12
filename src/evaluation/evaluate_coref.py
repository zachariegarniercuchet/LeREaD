#!/usr/bin/env python
"""
LeREaD Coreference Evaluation Script

Flexible evaluation of a single document or a batch folder, mirroring
evaluate.py (the extraction evaluator). Sibling script, not a --task flag on
evaluate.py: coref inputs (JSON clusters / docid-annotated HTML) and metrics
(MUC/B3/CEAFe/LEA) are different enough from the extraction evaluator's
(span HTML, span-F1) that a shared CLI would just be two disjoint arg sets
glued together.

Two matching modes (--match-mode)
---------------------------------
id   (default, GOLD setting)
     Gold and system share the same mention ids (the coref step was run on
     the gold extraction). Clusters are compared directly by id. System
     output can be `*_coref_clusters.json` or `*_coref.html`.
     This is the historical behaviour; results are unchanged.

span (PIPELINE setting)
     The system extracted its own spans, so ids are NOT shared with gold.
     Both sides are read as docid-annotated HTML (gold: annotated split;
     system: `*_coref.html`). Every tag carrying `id` + `docid` is a mention.
     System mentions are aligned to gold mentions by exact span match
     (labelname + start/end offsets in whitespace-normalised plain text),
     and both clusterings are re-expressed with the shared span key before
     scoring. Only the *grouping* by docid matters, never the docid string.

     Mentions found on one side only (extraction errors) are handled by
     --unmatched:
       singleton (default, end-to-end): a gold mention with no system twin
                 is added to the system as a singleton; a system mention
                 with no gold twin is added to gold as a singleton
                 (Cai & Strube 2010 "twinless" handling). Extraction errors
                 therefore cost recall / precision.
       drop      score only the mentions present on both sides (isolates the
                 coreference step from extraction errors).

Usage:
    # Gold setting (unchanged)
    python evaluate_coref.py single --split test \\
        --system-file output_coref/.../1989CanLII1415ONCA_coref_clusters.json

    python evaluate_coref.py batch \\
        --batch-folder output_coref/test_gpt-5.2_coref_fs6_random_gold/

    # Pipeline setting (span alignment, end-to-end)
    python evaluate_coref.py batch --match-mode span \\
        --batch-folder output_coref/test_gpt-5.2_coref_fs6_greedy_from-<extraction_folder>-final/

    # Pipeline setting, coreference step only
    python evaluate_coref.py batch --match-mode span --unmatched drop \\
        --batch-folder output_coref/test_..._from-<extraction_folder>-final/


# Gold setting: unchanged, still the default
python -m src.evaluation.evaluate_coref batch --batch-folder output_coref/test_..._gold/ --verbose

# Pipeline setting, end-to-end (extraction errors are penalised)
python -m src.evaluation.evaluate_coref batch --match-mode span --batch-folder output_coref/test_..._from-<extraction>-final/

# Pipeline setting, coreference step only
python -m src.evaluation.evaluate_coref batch --match-mode span --unmatched drop --batch-folder ...
"""

import argparse
import sys
import json
import io
import re
import contextlib
import importlib.util
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from datetime import datetime

from bs4 import BeautifulSoup, NavigableString

from configs.config import SPLITS

# Add project root to path for imports (same depth convention as evaluate.py)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import evaluation utilities directly to avoid full src package import
spec = importlib.util.spec_from_file_location(
    "evaluation_coref_util",
    project_root / "src" / "evaluation" / "evaluation_coref.py"
)
evaluation_coref_util = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation_coref_util)
evaluate_coref_raw = evaluation_coref_util.evaluate_coref_raw
aggregate_raw = evaluation_coref_util.aggregate_raw
finalize_raw = evaluation_coref_util.finalize_raw
print_evaluation_table = evaluation_coref_util.print_evaluation_table

# Import config constants
spec_config = importlib.util.spec_from_file_location(
    "config",
    project_root / "configs" / "config.py"
)
config_module = importlib.util.module_from_spec(spec_config)
spec_config.loader.exec_module(config_module)
DATA_DIR = config_module.DATA_DIR


# =============================================================================
# Config
# =============================================================================

@dataclass
class CorefEvaluationConfig:
    mode: str  # "single" | "batch"
    split: Optional[str] = None
    system_file: Optional[str] = None
    batch_folder: Optional[str] = None
    data_dir: Path = Path("./data")
    verbose_per_file: bool = False
    match_mode: str = "id"        # "id" (gold setting) | "span" (pipeline setting)
    unmatched: str = "singleton"  # span mode only: "singleton" | "drop"
    overlap: bool = True          # extra overlap-alignment comparison (exact stays main)

    def get_ground_truth_dir(self) -> Path:
        return Path(self.data_dir) / "annotated" / self.split

    def results_filename(self) -> str:
        # id mode keeps the historical filename so old logs are never overwritten
        if self.match_mode == "id":
            return "evaluation_coref_results.txt"
        return f"evaluation_coref_results_span_{self.unmatched}.txt"

    def overlap_results_filename(self) -> str:
        # overlap comparison always goes to a separate file — exact logs untouched
        if self.match_mode == "id":
            return "evaluation_coref_results_overlap.txt"
        return f"evaluation_coref_results_span_{self.unmatched}_overlap.txt"

    def validate(self) -> None:
        if self.match_mode not in ("id", "span"):
            raise ValueError(f"Unknown match_mode: {self.match_mode}")
        if self.unmatched not in ("singleton", "drop"):
            raise ValueError(f"Unknown unmatched policy: {self.unmatched}")
        if self.mode == "single":
            if not self.split:
                raise ValueError("--split is required for single file evaluation")
            if not self.system_file:
                raise ValueError("--system-file is required for single file evaluation")
            if self.match_mode == "span" and not str(self.system_file).endswith(".html"):
                raise ValueError(
                    "--match-mode span needs the system *_coref.html file "
                    "(ids in the JSON clusters cannot be aligned to gold spans)"
                )
        elif self.mode == "batch":
            if not self.batch_folder:
                raise ValueError("--batch-folder is required for batch evaluation")
        else:
            raise ValueError(f"Unknown mode: {self.mode}")


# =============================================================================
# HTML / cluster helpers  (id mode -- unchanged)
# =============================================================================

def extract_mention_docid_mapping(html_content: str, parser: str = "html.parser") -> Dict[str, str]:
    """id -> docid for every tag in the document carrying both attributes."""
    soup = BeautifulSoup(html_content, parser)
    return {tag["id"]: tag["docid"] for tag in soup.find_all(attrs={"docid": True, "id": True})}


def dict_to_clusters(mapping: Dict[str, str]) -> List[List[str]]:
    clusters: Dict[str, List[str]] = {}
    for mention_id, docid in mapping.items():
        clusters.setdefault(docid, []).append(mention_id)
    return list(clusters.values())


def load_clusters(path: Path) -> List[List[str]]:
    """Load clusters from a `*_coref_clusters.json` file, or derive them from
    a docid-annotated HTML file (gold annotated file, or a `*_coref.html`
    system reconstruction)."""
    if path.suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    if path.suffix == ".html":
        with open(path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return dict_to_clusters(extract_mention_docid_mapping(html_content))
    raise ValueError(f"Unsupported file type for clusters: {path}")


def derive_doc_stem(system_path: Path) -> str:
    """Strip known coref-output suffixes to recover the original filename
    stem, used to look up the matching gold file."""
    name = system_path.stem  # already strips .json / .html
    for suffix in ("_coref_clusters", "_coref_mapping", "_coref"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def extract_split_from_folder_name(folder_name: str) -> Optional[str]:
    """Expected patterns, matching main_coref.py's run_tag(), e.g.
    'test_gpt-5.2_coref_fs6_random_gold' or
    'dev_gpt-5.2_coref_fs6_greedy_from-<extraction_folder>-final'."""
    parts = folder_name.split("_")
    if parts and parts[0] in ["train", "test", "dev", "incoming", "extended_test"]:
        return parts[0]
    return None


# =============================================================================
# Span alignment (span mode -- pipeline setting)
# =============================================================================

@dataclass
class Mention:
    key: str      # shared key used as the mention id after alignment
    docid: str    # cluster label (only used to group mentions)
    label: str    # labelname attribute (e.g. "decision")
    start: int    # offsets in whitespace-normalised plain text
    end: int
    text: str


def extract_mentions_with_offsets(html_content: str, parser: str = "html.parser") -> List[Mention]:
    """Return every tag carrying both `id` and `docid` as a Mention whose
    identity is (labelname, start, end) in the whitespace-normalised plain
    text of the document.

    Offsets ignore markup, so they are identical between the gold HTML and a
    system HTML built from the same source text, whatever the ids are.
    Whitespace runs are collapsed and span boundaries are trimmed so that
    `<x> Foo</x>` and `<x>Foo</x>` (or an inner newline) still match.
    """
    soup = BeautifulSoup(html_content, parser)

    pieces: List[str] = []
    found = []  # (tag, raw_start, raw_end)
    pos = 0

    def walk(node) -> None:
        nonlocal pos
        for child in node.children:
            if isinstance(child, NavigableString):
                # exact NavigableString only: skips comments, doctype, scripts...
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

    # raw offset -> whitespace-normalised offset
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
        key = base if n == 0 else f"{base}#{n}"  # nested spans with identical extent
        mentions.append(Mention(
            key=key,
            docid=(tag.get("docid") or "").strip(),
            label=label,
            start=start,
            end=end,
            text=norm_text[start:end],
        ))
    return mentions


def _group_by_docid(mentions: List[Mention]) -> List[List[str]]:
    groups: Dict[str, List[str]] = defaultdict(list)
    clusters: List[List[str]] = []
    for m in mentions:
        if m.docid:
            groups[m.docid].append(m.key)
        else:
            clusters.append([m.key])  # no docid -> singleton
    return list(groups.values()) + clusters


def align_and_build_clusters(
    gold_mentions: List[Mention],
    system_mentions: List[Mention],
    unmatched: str = "singleton",
) -> Tuple[List[List[str]], List[List[str]], Dict[str, int]]:
    """Re-express gold and system clusters over a shared mention space.

    unmatched="singleton": twinless mentions are added as singletons to the
        opposite side, so both sides score over the union of mentions.
    unmatched="drop": only mentions present on both sides are scored.
    """
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


# =============================================================================
# Overlap alignment (COMPARISON ONLY — exact alignment above is untouched)
# =============================================================================

def _mention_overlap_len(a: Mention, b: Mention) -> int:
    """Character length of the offset intersection, or 0 (labels must agree)."""
    if a.label != b.label:
        return 0
    return max(0, min(a.end, b.end) - max(a.start, b.start))


def align_and_build_clusters_overlap(
    gold_mentions: List[Mention],
    system_mentions: List[Mention],
    unmatched: str = "singleton",
) -> Tuple[List[List[str]], List[List[str]], Dict[str, int]]:
    """Re-express gold and system clusters over a shared mention space, where
    a system mention aligns to a gold mention on OVERLAP (same labelname +
    intersecting character offsets) instead of exact (label, start, end).

    Greedy best-overlap matching: each gold mention takes the unused system
    mention with the largest intersection (ties broken by order); each span
    is used at most once. The shared key of a matched pair is the gold key.
    Twinless-mention handling (`unmatched`) mirrors align_and_build_clusters.

    Comparison only — the exact-match align_and_build_clusters() above is
    untouched and remains the main metric.
    """
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


def format_alignment_stats(stats: Dict[str, int], title: str = "Mention alignment") -> str:
    m, g, s = stats["matched"], stats["gold"], stats["system"]
    p = m / s if s else 0.0
    r = m / g if g else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return (
        f"{title}\n"
        f"  gold mentions: {g} | system mentions: {s} | matched: {m}\n"
        f"  gold-only (missed): {stats['gold_only']} | system-only (spurious): {stats['system_only']}\n"
        f"  mention-level  P={p:.3f}  R={r:.3f}  F1={f:.3f}\n"
    )


def _read(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_cluster_pair(
    gold_path: Path, system_path: Path, config: CorefEvaluationConfig
) -> Tuple[List[List[str]], List[List[str]], Optional[Dict[str, int]]]:
    """Return (gold_clusters, system_clusters, alignment_stats_or_None)."""
    if config.match_mode == "id":
        return load_clusters(gold_path), load_clusters(system_path), None

    if system_path.suffix != ".html":
        raise ValueError(f"span mode needs a system *_coref.html file, got: {system_path}")
    gold_mentions = extract_mentions_with_offsets(_read(gold_path))
    system_mentions = extract_mentions_with_offsets(_read(system_path))
    return align_and_build_clusters(gold_mentions, system_mentions, config.unmatched)


def load_cluster_pair_overlap(
    gold_path: Path, system_path: Path, config: CorefEvaluationConfig
) -> Tuple[List[List[str]], List[List[str]], Optional[Dict[str, int]]]:
    """Overlap-alignment variant of load_cluster_pair (comparison only).

    id mode has no spans to overlap, so overlap is only defined for span
    mode; in id mode this falls back to the exact pairing.
    """
    if config.match_mode == "id":
        return load_clusters(gold_path), load_clusters(system_path), None

    if system_path.suffix != ".html":
        raise ValueError(f"span mode needs a system *_coref.html file, got: {system_path}")
    gold_mentions = extract_mentions_with_offsets(_read(gold_path))
    system_mentions = extract_mentions_with_offsets(_read(system_path))
    return align_and_build_clusters_overlap(gold_mentions, system_mentions, config.unmatched)


# =============================================================================
# Logging helpers
# =============================================================================

def save_evaluation_log(folder_path: Path, log_content: str, filename: str = "evaluation_coref_results.txt") -> None:
    log_file = folder_path / filename
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(log_content)
    print(f"✓ Saved evaluation results to: {log_file}")


def save_version_info(folder_path: Path, config: CorefEvaluationConfig, mode: str, details: str = "") -> None:
    version_file = folder_path / "version_coref.txt"
    content = (
        "Coref Evaluation Version Information\n"
        "================================\n"
        f"Timestamp: {datetime.now().isoformat()}\n"
        f"Mode: {mode}\n"
        f"Split: {config.split}\n"
        f"Match mode: {config.match_mode}\n"
        f"Unmatched policy: {config.unmatched}\n\n"
    )
    if details:
        content += f"Details:\n{details}\n"
    with open(version_file, "w") as f:
        f.write(content)
    print(f"✓ Saved evaluation info to: {version_file}")


# =============================================================================
# Single-file evaluation
# =============================================================================

def run_single_file_evaluation(config: CorefEvaluationConfig):
    system_path = Path(config.system_file)
    if not system_path.exists():
        raise FileNotFoundError(f"System file not found: {system_path}")

    doc_stem = derive_doc_stem(system_path)
    gold_dir = config.get_ground_truth_dir()
    gold_path = gold_dir / f"{doc_stem}.html"
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_path}")

    header = (
        f"\n{'='*70}\n"
        f"SINGLE FILE COREF EVALUATION\n"
        f"{'='*70}\n"
        f"Split: {config.split}\n"
        f"Match mode: {config.match_mode}"
        + (f" (unmatched={config.unmatched})" if config.match_mode == "span" else "") + "\n"
        f"Gold:   {gold_path}\n"
        f"System: {system_path}\n"
        f"{'='*70}\n"
    )
    print(header)

    gold_clusters, system_clusters, stats = load_cluster_pair(gold_path, system_path, config)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        if stats is not None:
            print(format_alignment_stats(stats))
        raw = evaluate_coref_raw(gold_clusters, system_clusters)
        scores = finalize_raw(raw)
        print_evaluation_table(scores, title=f"Coreference Evaluation — {doc_stem}")
    captured = buffer.getvalue()
    print(captured)

    full_log = header + captured
    save_evaluation_log(system_path.parent, full_log, config.results_filename())

    # ---- Overlap alignment (comparison only; exact output above untouched) ----
    if getattr(config, "overlap", True):
        ov_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(ov_buffer):
                print("\n" + "=" * 70)
                print("OVERLAP ALIGNMENT (comparison — main metric remains EXACT above)")
                print("Mentions align on same label + overlapping character offsets.")
                print("=" * 70)
                ov_gold, ov_system, ov_stats = load_cluster_pair_overlap(gold_path, system_path, config)
                if ov_stats is not None:
                    print(format_alignment_stats(ov_stats, title="Mention alignment [OVERLAP]"))
                ov_raw = evaluate_coref_raw(ov_gold, ov_system)
                ov_scores = finalize_raw(ov_raw)
                print_evaluation_table(ov_scores, title=f"Coreference Evaluation [OVERLAP] — {doc_stem}")
        except Exception as e:
            print(f"\n⚠ Overlap evaluation failed (exact results above unaffected): {e}")
            ov_buffer.write(f"\n⚠ Overlap evaluation failed: {e}\n")
        ov_captured = ov_buffer.getvalue()
        print(ov_captured)
        full_log += ov_captured
        save_evaluation_log(system_path.parent, full_log, config.overlap_results_filename())
        return scores, ov_scores

    return scores


# =============================================================================
# Batch evaluation
# =============================================================================

def _first_nonempty_glob(folder: Path, patterns: List[str]) -> List[Path]:
    for pattern in patterns:
        files = [f for f in folder.glob(pattern) if f.is_file()]
        if files:
            return files
    return []


def collect_cluster_pairs_from_batch_folder(
    batch_folder: Path,
    data_dir: Path,
    split: str,
    verbose: bool = False,
    match_mode: str = "id",
) -> List[Tuple[Path, Path]]:
    """Collect (gold_path, system_path) pairs from a batch folder.

    Handles two structures (same convention as evaluate.py):
    1. Direct files inside batch_folder
    2. Subfolder-per-file structure, as produced by main_coref.py's split
       mode: batch_folder/<filename>/<filename>_coref*.{json,html}

    id mode:   `*_coref_clusters.json` (preferred) or `*_coref.html`
    span mode: `*_coref.html` only (ids in the JSON can't be aligned to gold)
    """
    patterns = ["*_coref.html"] if match_mode == "span" else ["*_coref_clusters.json", "*_coref.html"]

    gold_dir = data_dir / "annotated" / split
    if not gold_dir.exists():
        raise ValueError(f"Gold directory does not exist: {gold_dir}")
    gold_files = {p.stem: p for p in gold_dir.glob("*.html")}

    if verbose:
        print(f"Found {len(gold_files)} gold files in {gold_dir}")

    def pair_up(system_files: List[Path]) -> List[Tuple[Path, Path]]:
        pairs = []
        for system_file in system_files:
            stem = derive_doc_stem(system_file)
            if stem in gold_files:
                pairs.append((gold_files[stem], system_file))
                if verbose:
                    print(f"  Paired: {stem}")
            else:
                print(f"  ⚠ No matching gold file for: {stem}")
        return pairs

    # Strategy 1: direct files
    direct_files = _first_nonempty_glob(batch_folder, patterns)
    if direct_files:
        if verbose:
            print(f"Found {len(direct_files)} direct coref output files")
        return pair_up(direct_files)

    # Strategy 2: subfolder-per-file
    subfolders = [d for d in batch_folder.iterdir() if d.is_dir()]
    if subfolders:
        if verbose:
            print(f"Found {len(subfolders)} subfolders")
        system_files = []
        for subfolder in subfolders:
            candidates = _first_nonempty_glob(subfolder, patterns)
            if candidates:
                system_files.append(candidates[0])
            elif verbose:
                print(f"  ⚠ No coref output found in: {subfolder.name}")
        return pair_up(system_files)

    raise ValueError(f"No coref cluster/HTML files or subfolders found in {batch_folder}")


def run_batch_evaluation(config: CorefEvaluationConfig):
    batch_folder = Path(config.batch_folder)
    if not batch_folder.exists():
        raise FileNotFoundError(f"Batch folder not found: {batch_folder}")

    split = config.split
    if not split:
        detected_split = extract_split_from_folder_name(batch_folder.name)
        if detected_split:
            split = detected_split
            print(f"Auto-detected split from folder name: {split}")
        else:
            raise ValueError(
                f"Could not auto-detect split from folder name: {batch_folder.name}\n"
                f"Please provide split parameter explicitly."
            )

    header = (
        f"\n{'='*70}\n"
        f"BATCH COREF EVALUATION\n"
        f"{'='*70}\n"
        f"Batch folder: {batch_folder}\n"
        f"Split: {split}\n"
        f"Match mode: {config.match_mode}"
        + (f" (unmatched={config.unmatched})" if config.match_mode == "span" else "") + "\n"
        f"{'='*70}\n"
    )
    print(header)

    pairs = collect_cluster_pairs_from_batch_folder(
        batch_folder, config.data_dir, split, verbose=False, match_mode=config.match_mode
    )
    if not pairs:
        raise ValueError(f"No file pairs found in batch folder: {batch_folder}")
    print(f"Found {len(pairs)} evaluation pair(s)\n")

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        per_file_raws = []
        total_stats: Dict[str, int] = defaultdict(int)
        for gold_path, system_path in pairs:
            gold_clusters, system_clusters, stats = load_cluster_pair(gold_path, system_path, config)
            raw = evaluate_coref_raw(gold_clusters, system_clusters)
            per_file_raws.append(raw)
            if stats is not None:
                for k, v in stats.items():
                    total_stats[k] += v
            if config.verbose_per_file:
                if stats is not None:
                    print(format_alignment_stats(stats, title=f"Mention alignment — {gold_path.stem}"))
                print_evaluation_table(finalize_raw(raw), title=f"{gold_path.stem}")

        # Corpus-level aggregation: sum raw counts, then compute one overall
        # P/R/F per metric (standard CoNLL-scorer convention), rather than
        # averaging per-document F1s.
        aggregated_scores = finalize_raw(aggregate_raw(per_file_raws))
        if config.match_mode == "span":
            print(format_alignment_stats(dict(total_stats), title=f"Mention alignment (total over {len(pairs)} document(s))"))
        print_evaluation_table(aggregated_scores, title=f"Aggregate over {len(pairs)} document(s)")
    captured = buffer.getvalue()

    print(captured)

    full_log = header + f"Found {len(pairs)} evaluation pair(s)\n\n" + captured
    save_evaluation_log(batch_folder, full_log, config.results_filename())

    # ---- Overlap alignment (comparison only; exact output above untouched) ----
    overlap_scores = None
    if getattr(config, "overlap", True):
        ov_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(ov_buffer):
                print("\n" + "=" * 70)
                print("OVERLAP ALIGNMENT (comparison — main metric remains EXACT above)")
                print("Mentions align on same label + overlapping character offsets.")
                print("=" * 70)
                ov_per_file_raws = []
                ov_total_stats: Dict[str, int] = defaultdict(int)
                for gold_path, system_path in pairs:
                    ov_gold, ov_system, ov_stats = load_cluster_pair_overlap(gold_path, system_path, config)
                    ov_raw = evaluate_coref_raw(ov_gold, ov_system)
                    ov_per_file_raws.append(ov_raw)
                    if ov_stats is not None:
                        for k, v in ov_stats.items():
                            ov_total_stats[k] += v
                    if config.verbose_per_file:
                        if ov_stats is not None:
                            print(format_alignment_stats(ov_stats, title=f"Mention alignment [OVERLAP] — {gold_path.stem}"))
                        print_evaluation_table(finalize_raw(ov_raw), title=f"{gold_path.stem} [OVERLAP]")

                ov_aggregated = finalize_raw(aggregate_raw(ov_per_file_raws))
                if config.match_mode == "span":
                    print(format_alignment_stats(dict(ov_total_stats), title=f"Mention alignment [OVERLAP] (total over {len(pairs)} document(s))"))
                print_evaluation_table(ov_aggregated, title=f"Aggregate over {len(pairs)} document(s) [OVERLAP]")
                overlap_scores = ov_aggregated
        except Exception as e:
            print(f"\n⚠ Overlap evaluation failed (exact results above unaffected): {e}")
            ov_buffer.write(f"\n⚠ Overlap evaluation failed: {e}\n")
        ov_captured = ov_buffer.getvalue()
        print(ov_captured)
        full_log += ov_captured
        save_evaluation_log(batch_folder, full_log, config.overlap_results_filename())

    return aggregated_scores


# =============================================================================
# CLI
# =============================================================================

def _add_match_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--match-mode", choices=["id", "span"], default="id",
                   help="id: gold and system share mention ids (gold setting, default). "
                        "span: align system spans to gold spans (pipeline setting; needs *_coref.html)")
    p.add_argument("--unmatched", choices=["singleton", "drop"], default="singleton",
                   help="span mode only. singleton: unmatched mentions become singletons on the other side "
                        "(end-to-end, default). drop: score only mentions present on both sides")
    p.add_argument("--no-overlap", action="store_true",
                   help="Skip the extra overlap-alignment comparison (exact match is always shown)")


def main():
    parser = argparse.ArgumentParser(
        description="LeREaD Coreference Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Evaluation mode")

    # ========== SINGLE FILE EVALUATION ==========
    single_parser = subparsers.add_parser("single", help="Evaluate a single system coref output")
    single_parser.add_argument("--split", required=True, choices=SPLITS,
                                help="Ground truth split (required for single file evaluation)")
    single_parser.add_argument("--system-file", required=True,
                                help="Path to a *_coref_clusters.json or *_coref.html system output file")
    single_parser.add_argument("--data-dir", default="./data", help="Root data directory (default: ./data)")
    single_parser.add_argument("--verbose", action="store_true", help="Verbose output")
    _add_match_args(single_parser)

    # ========== BATCH EVALUATION ==========
    batch_parser = subparsers.add_parser("batch", help="Evaluate a batch folder with multiple documents")
    batch_parser.add_argument("--batch-folder", required=True, help="Path to a main_coref.py output folder")
    batch_parser.add_argument("--split", default=None, choices=SPLITS,
                               help="Ground truth split (optional, auto-detected from folder name if not provided)")
    batch_parser.add_argument("--data-dir", default="./data", help="Root data directory (default: ./data)")
    batch_parser.add_argument("--verbose", action="store_true",
                               help="Also print a per-document table before the aggregate")
    _add_match_args(batch_parser)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "single":
        config = CorefEvaluationConfig(
            mode="single",
            split=args.split,
            system_file=args.system_file,
            data_dir=Path(args.data_dir),
            verbose_per_file=args.verbose,
            match_mode=args.match_mode,
            unmatched=args.unmatched,
            overlap=not args.no_overlap,
        )
        config.validate()
        run_single_file_evaluation(config)

    elif args.command == "batch":
        config = CorefEvaluationConfig(
            mode="batch",
            split=args.split,
            batch_folder=args.batch_folder,
            data_dir=Path(args.data_dir),
            verbose_per_file=args.verbose,
            match_mode=args.match_mode,
            unmatched=args.unmatched,
            overlap=not args.no_overlap,
        )
        config.validate()
        run_batch_evaluation(config)


if __name__ == "__main__":
    main()