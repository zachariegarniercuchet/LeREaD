#!/usr/bin/env python
"""
LeREaD Evaluation Script

Flexible evaluation of single files or batch folders with automatic split detection.

Usage:
    # Single file evaluation (requires split parameter)
    python evaluate.py single --split test --system-file output/1989CanLII1415ONCA_processed.html
    
    # Batch folder evaluation (auto-detects split from folder name)
    python evaluate.py batch --batch-folder output/test_gpt-5.2_DEC_fs6_greedy_sentence/
    
    # Batch folder evaluation with explicit split
    python evaluate.py batch --batch-folder output/test_gpt-5.2_DEC_fs6_greedy_sentence/ --split test
    
    # Use a config file
    python evaluate.py config --config-file evaluation_configs/test_config.py
"""

import argparse
import sys
import re
import importlib.util
from pathlib import Path
from typing import List, Tuple, Optional
from collections import defaultdict
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import evaluation utilities directly to avoid full src package import
import importlib.util
spec = importlib.util.spec_from_file_location(
    "evaluation_l1_util",
    project_root / "src" / "evaluation" / "evaluation_l1_util.py"
)
evaluation_l1_util = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation_l1_util)
evaluate_batch = evaluation_l1_util.evaluate_batch
evaluate = evaluation_l1_util.evaluate
# Overlap matching (comparison only — exact match stays the main metric)
evaluate_batch_overlap = evaluation_l1_util.evaluate_batch_overlap
evaluate_overlap = evaluation_l1_util.evaluate_overlap

# Import evaluation config directly
spec_config_eval = importlib.util.spec_from_file_location(
    "evaluation_config",
    project_root / "src" / "evaluation" / "evaluation_config.py"
)
evaluation_config_module = importlib.util.module_from_spec(spec_config_eval)
spec_config_eval.loader.exec_module(evaluation_config_module)
EvaluationConfig = evaluation_config_module.EvaluationConfig

# Import config constants
spec_config = importlib.util.spec_from_file_location(
    "config",
    project_root / "configs" / "config.py"
)
config_module = importlib.util.module_from_spec(spec_config)
spec_config.loader.exec_module(config_module)
DATA_DIR = config_module.DATA_DIR


import io
import contextlib

def save_evaluation_log(folder_path: Path, log_content: str) -> None:
    """Save captured evaluation output to evaluation_results.txt."""
    log_file = folder_path / "evaluation_results.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(log_content)
    print(f"✓ Saved evaluation results to: {log_file}")


def save_version_info(folder_path: Path, config: EvaluationConfig, mode: str, details: str = "") -> None:
    """
    Save evaluation version and configuration information to version.txt.
    
    Parameters
    ----------
    folder_path : Path
        Folder where version.txt will be saved
    config : EvaluationConfig
        Evaluation configuration
    mode : str
        Evaluation mode ("single" or "batch")
    details : str
        Additional details to include in version.txt
    """
    version_file = folder_path / "version.txt"
    
    content = f"""Evaluation Version Information
================================
Timestamp: {datetime.now().isoformat()}
Mode: {mode}
Split: {config.split}
Context chars: {config.context_chars}

"""
    
    if details:
        content += f"Details:\n{details}\n"
    
    with open(version_file, "w") as f:
        f.write(content)
    
    print(f"✓ Saved evaluation info to: {version_file}")


def extract_split_from_folder_name(folder_name: str) -> Optional[str]:
    """
    Extract split name from folder name.
    
    Expected patterns:
    - test_gpt-5.2_DEC_fs6_greedy_sentence
    - train_AIO_fs6_greedy_paragraph
    - dev_gpt-4_DEC_fs6_greedy_paragraph
    
    Returns the first element (split name) or None if not recognized.
    """
    parts = folder_name.split("_")
    if parts and parts[0] in ["train", "test", "dev", "incoming", "extended_test"]:
        return parts[0]
    return None


def find_final_html_in_folder(folder_path: Path) -> Optional[Path]:
    """
    Find the *final.html file in a folder.
    
    Expected: exactly one file matching *final.html pattern.
    """
    final_htmls = list(folder_path.glob("*final.html"))
    if len(final_htmls) == 1:
        return final_htmls[0]
    return None


def collect_html_pairs_from_batch_folder(
    batch_folder: Path,
    data_dir: Path,
    split: str,
    verbose: bool = False,
) -> List[Tuple[str, str]]:
    """
    Collect (gold, system) HTML file pairs from a batch folder.
    
    Handles two structures:
    1. Direct HTML files: evaluate all .html files in batch_folder
    2. Subfolder structure: each subfolder contains one *final.html file
       Use the folder name (without timestamp) to find corresponding gold file
    
    Parameters
    ----------
    batch_folder : Path
        The batch output folder
    data_dir : Path
        Root data directory (e.g., ./data)
    split : str
        The split name (train/test/dev/incoming/extended_test)
    verbose : bool
        Print debug information
    
    Returns
    -------
    List[Tuple[str, str]]
        List of (gold_path, system_path) pairs
    """
    pairs = []
    gold_dir = data_dir / "annotated" / split
    
    if not gold_dir.exists():
        raise ValueError(f"Gold directory does not exist: {gold_dir}")
    
    # Get all gold files
    gold_files = {p.stem: p for p in gold_dir.glob("*.html")}
    
    if verbose:
        print(f"Found {len(gold_files)} gold files in {gold_dir}")
    
    # Strategy 1: Check for direct HTML files
    html_files = list(batch_folder.glob("*.html"))
    html_files = [f for f in html_files if f.is_file()]
    
    if html_files:
        if verbose:
            print(f"Found {len(html_files)} direct HTML files")
        
        for system_html in html_files:
            gold_stem = system_html.stem
            if gold_stem in gold_files:
                pairs.append((str(gold_files[gold_stem]), str(system_html)))
                if verbose:
                    print(f"  Paired: {gold_stem}")
            else:
                print(f"  ⚠ No matching gold file for: {gold_stem}")
        
        return pairs
    
    # Strategy 2: Check for subfolders with *final.html
    subfolders = [d for d in batch_folder.iterdir() if d.is_dir()]
    
    if subfolders:
        if verbose:
            print(f"Found {len(subfolders)} subfolders")
        
        for subfolder in subfolders:
            final_html = find_final_html_in_folder(subfolder)
            if final_html:
                # Extract folder base name (remove timestamp if present)
                # e.g., "2005QCCA437_AIO_fs6_greedy_paragraph_20260604_190954" -> "2005QCCA437"
                folder_base = subfolder.name.split("_")[0]
                
                if folder_base in gold_files:
                    pairs.append((str(gold_files[folder_base]), str(final_html)))
                    if verbose:
                        print(f"  Paired: {folder_base} -> {final_html.name}")
                else:
                    print(f"  ⚠ No matching gold file for: {folder_base}")
            else:
                if verbose:
                    print(f"  ⚠ No *final.html found in: {subfolder.name}")
        
        return pairs
    
    raise ValueError(f"No HTML files or subfolders found in {batch_folder}")


def run_single_file_evaluation(config: EvaluationConfig):
    system_path = Path(config.system_file)
    if not system_path.exists():
        raise FileNotFoundError(f"System file not found: {system_path}")

    gold_dir = config.get_ground_truth_dir()
    gold_path = gold_dir / f"{system_path.stem}.html"
    if not gold_path.exists():
        raise FileNotFoundError(f"Gold file not found: {gold_path}")

    header = (
        f"\n{'='*70}\n"
        f"SINGLE FILE EVALUATION\n"
        f"{'='*70}\n"
        f"Split: {config.split}\n"
        f"Gold:   {gold_path}\n"
        f"System: {system_path}\n"
        f"{'='*70}\n"
    )
    print(header)

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        metrics, doc_f1 = evaluate(
            str(gold_path), str(system_path),
            context_chars=config.context_chars,
            verbose=True,
        )
    captured = buffer.getvalue()
    print(captured, end="" if captured.endswith("\n") else "\n")
    print("Main metric is EXACT match (above). "
          "Overlap below is a lenient comparison only.")

    full_log = header + captured

    # ---- Overlap matching (comparison only; exact output above untouched) ----
    if getattr(config, "overlap", True):
        ov_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(ov_buffer):
                print("\n" + "=" * 70)
                print("OVERLAP MATCHING (comparison — main metric remains EXACT above)")
                print("A hit = same label + overlapping character offsets.")
                print("=" * 70)
                evaluate_overlap(
                    str(gold_path), str(system_path),
                    context_chars=config.context_chars,
                    verbose=True,
                )
        except Exception as e:
            print(f"\n⚠ Overlap evaluation failed (exact results above unaffected): {e}")
            ov_buffer.write(f"\n⚠ Overlap evaluation failed: {e}\n")
        ov_captured = ov_buffer.getvalue()
        print(ov_captured, end="" if ov_captured.endswith("\n") else "\n")
        full_log += ov_captured

    save_evaluation_log(system_path.parent, full_log)


def run_batch_evaluation(
    config: EvaluationConfig,
) -> None:
    """Run evaluation for a batch of files in a folder."""
    batch_folder = Path(config.batch_folder)
    
    if not batch_folder.exists():
        raise FileNotFoundError(f"Batch folder not found: {batch_folder}")
    
    # Auto-detect split from folder name if not provided
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
        f"BATCH EVALUATION\n"
        f"{'='*70}\n"
        f"Batch folder: {batch_folder}\n"
        f"Split: {split}\n"
        f"{'='*70}\n"
    )
    print(header)

    
    # Collect pairs
    pairs = collect_html_pairs_from_batch_folder(
        batch_folder,
        config.data_dir,
        split,
        verbose=False,
    )
    
    if not pairs:
        raise ValueError(f"No file pairs found in batch folder: {batch_folder}")
    
    print(f"Found {len(pairs)} evaluation pair(s)\n")
    
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        results = evaluate_batch(
            pairs,
            context_chars=config.context_chars,
            verbose_per_file=config.verbose_per_file,
        )
    captured = buffer.getvalue()

    # Print to console as normal (exact match — main metric, unchanged)
    print(captured, end="" if captured.endswith("\n") else "\n")

    # Build full log: header + captured output
    full_log = header + f"Found {len(pairs)} evaluation pair(s)\n\n" + captured

    # ---- Overlap matching (comparison only; exact output above untouched) ----
    overlap_results = None
    if getattr(config, "overlap", True):
        ov_buffer = io.StringIO()
        try:
            with contextlib.redirect_stdout(ov_buffer):
                overlap_results = evaluate_batch_overlap(
                    pairs,
                    context_chars=config.context_chars,
                    verbose_per_file=config.verbose_per_file,
                )
        except Exception as e:
            print(f"\n⚠ Overlap evaluation failed (exact results above unaffected): {e}")
            ov_buffer.write(f"\n⚠ Overlap evaluation failed: {e}\n")
        ov_captured = ov_buffer.getvalue()
        print(ov_captured, end="" if ov_captured.endswith("\n") else "\n")
        full_log += ov_captured

    save_evaluation_log(batch_folder, full_log)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="LeREaD Evaluation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Evaluation mode")
    
    # ========== SINGLE FILE EVALUATION ==========
    single_parser = subparsers.add_parser(
        "single",
        help="Evaluate a single system output file",
    )
    single_parser.add_argument(
        "--split",
        required=True,
        choices=["train", "test", "dev", "incoming", "extended_test"],
        help="Ground truth split (required for single file evaluation)",
    )
    single_parser.add_argument(
        "--system-file",
        required=True,
        help="Path to system output HTML file",
    )
    single_parser.add_argument(
        "--data-dir",
        default="./data",
        help="Root data directory (default: ./data)",
    )
    single_parser.add_argument(
        "--context-chars",
        type=int,
        default=200,
        help="Context characters for span matching (default: 200)",
    )
    single_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output",
    )
    single_parser.add_argument(
        "--no-overlap",
        action="store_true",
        help="Skip the extra overlap-matching comparison (exact match is always shown)",
    )
    
    # ========== BATCH EVALUATION ==========
    batch_parser = subparsers.add_parser(
        "batch",
        help="Evaluate a batch folder with multiple files",
    )
    batch_parser.add_argument(
        "--batch-folder",
        required=True,
        help="Path to batch output folder",
    )
    batch_parser.add_argument(
        "--split",
        default=None,
        choices=["train", "test", "dev", "incoming", "extended_test"],
        help="Ground truth split (optional, auto-detected from folder name if not provided)",
    )
    batch_parser.add_argument(
        "--data-dir",
        default="./data",
        help="Root data directory (default: ./data)",
    )
    batch_parser.add_argument(
        "--context-chars",
        type=int,
        default=200,
        help="Context characters for span matching (default: 200)",
    )
    batch_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output per file",
    )
    batch_parser.add_argument(
        "--no-overlap",
        action="store_true",
        help="Skip the extra overlap-matching comparison (exact match is always shown)",
    )
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # ========== SINGLE FILE MODE ==========
    if args.command == "single":
        config = EvaluationConfig(
            mode="single",
            split=args.split,
            system_file=args.system_file,
            data_dir=Path(args.data_dir),
            context_chars=args.context_chars,
            verbose_per_file=args.verbose,
            overlap=not args.no_overlap,
        )
        config.validate()
        run_single_file_evaluation(config)
    
    # ========== BATCH MODE ==========
    elif args.command == "batch":
        config = EvaluationConfig(
            mode="batch",
            split=args.split,
            batch_folder=args.batch_folder,
            data_dir=Path(args.data_dir),
            context_chars=args.context_chars,
            verbose_per_file=args.verbose,
            overlap=not args.no_overlap,
        )
        config.validate()
        run_batch_evaluation(config)
    
    

if __name__ == "__main__":
    main()
