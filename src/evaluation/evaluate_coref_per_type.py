"""
Per-mention-type coreference evaluation (LEA) for LeREaD.

Computes coreference metrics (MUC / B3 / CEAF_e / LEA) restricted to mentions
of a given type (decision / legislation / secondary sources), by
post-processing coref output:

    1. Align gold and system mentions (span mode: same label + offsets).
    2. Filter the aligned mention set to one labelname.
    3. Rebuild clusters (group by docid) on the filtered set.
    4. Score with coval, aggregating raw counts across the corpus.

This is an EVALUATION-TIME operation. The coref extraction module
(main_coref.py) only assigns docids to all parent-level mentions; it does not
compute LEA and should not be type-restricted, because coreference clusters are
built over the full mention set and LEA depends on that full cluster structure.

The mention type is read from the `labelname` attribute of each parent-level
mention, which is present both in the gold annotated HTML and in the system
`*_coref.html` reconstruction. Data labelnames are "decision", "legislation",
"secondary sources" (displayed as Decision / Legislation / Secondary).

Usage:
    python -m src.evaluation.evaluate_coref_per_type batch \
        --batch-folder output_coref/test_gpt-5.2_coref_fs6_random_gold/ \
        --split test --match-mode span --unmatched singleton

Requires the system outputs to be `*_coref.html` files (span mode), which is
what main_coref.py always writes (step 7 of process_file_coref).
"""

import argparse
import sys
import io
import contextlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Tuple

project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import importlib.util
# evaluate_coref.py (the CLI) re-exports the raw metrics from evaluation_coref.py
# AND defines the span-alignment helpers (extract_mentions_with_offsets,
# align_and_build_clusters, collect_cluster_pairs_from_batch_folder, Mention).
spec = importlib.util.spec_from_file_location(
    "evaluate_coref_cli",
    project_root / "src" / "evaluation" / "evaluate_coref.py",
)
evaluate_coref_cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate_coref_cli)

extract_mentions_with_offsets = evaluate_coref_cli.extract_mentions_with_offsets
align_and_build_clusters = evaluate_coref_cli.align_and_build_clusters
collect_cluster_pairs_from_batch_folder = evaluate_coref_cli.collect_cluster_pairs_from_batch_folder
evaluate_coref_raw = evaluate_coref_cli.evaluate_coref_raw
aggregate_raw = evaluate_coref_cli.aggregate_raw
finalize_raw = evaluate_coref_cli.finalize_raw
_read = evaluate_coref_cli._read
Mention = evaluate_coref_cli.Mention

from configs.config import SPLITS, DATA_DIR  # noqa: E402

# Data labelname -> display name used in the table.
LABEL_DISPLAY = {
    "decision": "Decision",
    "legislation": "Legislation",
    "secondary sources": "Secondary",
}
MENTION_TYPES = ["decision", "legislation", "secondary sources"]
METRICS = ["MUC", "B3", "CEAF_e", "LEA"]


@dataclass
class PerTypeConfig:
    batch_folder: Path
    split: str
    data_dir: Path = Path(DATA_DIR)
    match_mode: str = "span"        # "span" only for now (needs *_coref.html)
    unmatched: str = "singleton"    # "singleton" | "drop"


def collect_pairs(config: PerTypeConfig) -> List[Tuple[Path, Path]]:
    """Reuse evaluate_coref's pairing logic (span mode -> *_coref.html)."""
    return collect_cluster_pairs_from_batch_folder(
        config.batch_folder, config.data_dir, config.split,
        verbose=False, match_mode=config.match_mode,
    )


def clusters_for_type(
    gold_mentions: List[Mention],
    system_mentions: List[Mention],
    label: str,
    unmatched: str,
) -> Tuple[List[List[str]], List[List[str]]]:
    """Filter both sides to mentions of `label`, align, and rebuild clusters."""
    fg = [m for m in gold_mentions if m.label == label]
    fs = [m for m in system_mentions if m.label == label]
    gold_clusters, system_clusters, _stats = align_and_build_clusters(
        fg, fs, unmatched=unmatched
    )
    return gold_clusters, system_clusters


def run(config: PerTypeConfig) -> Dict[str, dict]:
    pairs = collect_pairs(config)
    if not pairs:
        raise ValueError(f"No file pairs found in {config.batch_folder}")

    # Accumulate raw LEA counts per type across documents (micro-aggregation).
    raw_by_type: Dict[str, List[dict]] = {t: [] for t in MENTION_TYPES}
    type_mention_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for gold_path, system_path in pairs:
        gold_html = _read(gold_path)
        if config.match_mode == "span":
            if system_path.suffix != ".html":
                raise ValueError(
                    f"span mode needs a system *_coref.html file, got: {system_path}"
                )
            system_html = _read(system_path)
        else:
            raise ValueError(f"Unsupported match_mode for per-type eval: {config.match_mode}")

        gold_mentions = extract_mentions_with_offsets(gold_html)
        system_mentions = extract_mentions_with_offsets(system_html)

        for label in MENTION_TYPES:
            fg = [m for m in gold_mentions if m.label == label]
            fs = [m for m in system_mentions if m.label == label]
            type_mention_counts[label]["gold"] += len(fg)
            type_mention_counts[label]["system"] += len(fs)

            gold_clusters, system_clusters = clusters_for_type(
                gold_mentions, system_mentions, label, config.unmatched
            )
            raw = evaluate_coref_raw(gold_clusters, system_clusters)
            raw_by_type[label].append(raw)

    # Micro-aggregate then finalize per type.
    results: Dict[str, dict] = {}
    for label in MENTION_TYPES:
        raws = raw_by_type[label]
        if not raws:
            continue
        aggregated = aggregate_raw(raws)
        scores = finalize_raw(aggregated)
        results[label] = scores
        results[label]["_counts"] = dict(type_mention_counts[label])

    return results


def print_table(results: Dict[str, dict], title: str) -> str:
    lines = [f"\n=== {title} ==="]
    header = ["Type", "Gold M.", "Sys M."] + [f"{m} F1" for m in METRICS]
    lines.append("  " + " | ".join(header))
    lines.append("  " + "-+-".join("-" * len(h) for h in header))
    for label in MENTION_TYPES:
        if label not in results:
            continue
        s = results[label]
        counts = s.pop("_counts", None)
        row = [
            LABEL_DISPLAY[label],
            str(counts["gold"]) if counts else "-",
            str(counts["system"]) if counts else "-",
        ] + [f"{s[m]['f1']:.4f}" for m in METRICS]
        lines.append("  " + " | ".join(row))
    lines.append("")
    return "\n".join(lines)


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Per-mention-type coreference evaluation (LEA)."
    )
    sub = p.add_subparsers(dest="command", required=True)

    batch = sub.add_parser("batch", help="Evaluate a coref batch folder per type")
    batch.add_argument("--batch-folder", required=True,
                       help="Path to a main_coref.py output folder (subfolder-per-file).")
    batch.add_argument("--split", required=True, choices=SPLITS,
                       help="Ground truth split.")
    batch.add_argument("--data-dir", default=str(DATA_DIR),
                       help="Root data directory.")
    batch.add_argument("--match-mode", choices=["span"], default="span",
                       help="Only 'span' is supported (needs *_coref.html).")
    batch.add_argument("--unmatched", choices=["singleton", "drop"], default="singleton",
                       help="Unmatched-mention policy for span mode.")
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.command == "batch":
        config = PerTypeConfig(
            batch_folder=Path(args.batch_folder),
            split=args.split,
            data_dir=Path(args.data_dir),
            match_mode=args.match_mode,
            unmatched=args.unmatched,
        )
        results = run(config)
        print(print_table(
            results,
            f"Per-type coreference LEA — {config.batch_folder.name} (split={config.split}, "
            f"unmatched={config.unmatched})",
        ))


if __name__ == "__main__":
    main()
