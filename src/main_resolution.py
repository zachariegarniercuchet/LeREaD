"""Main processing script for reference *resolution* (stage 3): candidate URI
retrieval via similarity search over the pre-computed candidate pool.

Mirrors main_extraction.py / main_coref.py in structure and CLI so the three
pipeline stages stay easy to read side by side.

Pipeline recap:
  stage 1 (main_extraction.py): raw text  -> extracted <manual_label>/<auto_label>
                                 mentions.
  stage 2 (main_coref.py):      mentions  -> docid assigned per mention via an
                                 assistant + a growing ReferenceProfileRegistry;
                                 reconstructed as "<docname>_coref.html".
  stage 3 (this file):          "<docname>_coref.html" -> re-parsed to rebuild
                                 the same ReferenceProfileRegistry (no LLM calls,
                                 purely deterministic), then every profile is
                                 matched against the pre-computed candidate pool
                                 (see precompute_index.py) to assign a `uri`
                                 attribute to every resolved mention tag;
                                 reconstructed as "<docname>_resolution.html".

The retrieval model is selected with --retriever and is entirely pluggable:
BM25, BGE, SPLADE, Legal-BERT and Qwen3-Embedding all go through the same
Retriever interface (src/retrievers/), so nothing in this file is model-specific.

    python -m src.main_resolution --folder test_gemma4_coref_fs6_random_gold --retriever bge-s
    python -m src.main_resolution --gold --split test --retriever bm25
    python -m src.main_resolution --gold --split test --retriever qwen3-8b --device cuda
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from bs4 import BeautifulSoup
from tqdm import tqdm

from src.ann_extractor import extract_parent_level_annotations
from src.htmlLabel import ReferenceMention
from src.precompute_index import default_cache_dir
from src.retrievers import Retriever, available_retrievers, create_retriever
from src.rpr import ReferenceProfile, ReferenceProfileRegistry, normalize_docid


# =============================================================================
# Candidate pool loading
# =============================================================================

def load_candidate_pool(
    cache_dir: Path,
    retriever_name: str,
    **retriever_kwargs,
) -> Tuple[Retriever, List[dict]]:
    """Load the index + metadata produced by precompute_index.py."""

    candidates_path = cache_dir / "candidates.json"
    if not candidates_path.exists():
        raise FileNotFoundError(
            f"Missing candidates file: {candidates_path}. "
            f"Run: python -m src.precompute_index --retriever {retriever_name}"
        )

    with candidates_path.open("r", encoding="utf-8") as f:
        candidates = json.load(f)

    retriever = create_retriever(retriever_name, **retriever_kwargs)
    retriever.load(cache_dir)

    if retriever.size != len(candidates):
        raise ValueError(
            f"candidates.json has {len(candidates)} entries but the {retriever_name} "
            f"index has {retriever.size} - the cache appears inconsistent. "
            f"Rebuild it with precompute_index.py."
        )

    return retriever, candidates


# =============================================================================
# Profile -> query text
# =============================================================================

def build_profile_text(profile: ReferenceProfile, include_fragments: bool = False) -> str:
    """
    Build the textual representation of a ReferenceProfile used to query the
    candidate pool. Mirrors build_metadata_text() in precompute_index.py
    (docType | docTitle | reference texts), but is built from the profile
    accumulated across every mention sharing a docid instead of from raw
    CanLII metadata.
    """

    parts: List[str] = []

    if profile.doc_type:
        parts.append(str(profile.doc_type))

    if profile.main_title:
        parts.append(str(profile.main_title))

    for title in profile.alternative_titles:
        if title != profile.main_title:
            parts.append(str(title))

    for citation in profile.citations:
        parts.append(str(citation))

    if include_fragments:
        for fragment in profile.fragments_mentioned:
            parts.append(str(fragment))

    for author in profile.authors:
        parts.append(str(author))

    return " | ".join(parts)


# =============================================================================
# Candidate filtering
# =============================================================================

def _normalize_doctype(value: Optional[str]) -> str:
    if not value:
        return ""
    return str(value).strip().upper().replace(" ", "_")


def build_doctype_index(candidates: List[dict]) -> Dict[str, np.ndarray]:
    """
    Precompute doctype -> candidate row indices once per run.

    The previous version rebuilt this list comprehension for every profile of
    every document, which is an O(n_profiles x n_candidates) Python loop for a
    mapping that never changes.
    """

    buckets: Dict[str, List[int]] = defaultdict(list)
    for i, candidate in enumerate(candidates):
        buckets[_normalize_doctype(candidate.get("docType"))].append(i)

    return {k: np.array(v, dtype=int) for k, v in buckets.items()}


def _subset_for_doctype(
    doc_type: Optional[str],
    doctype_index: Dict[str, np.ndarray],
    filter_by_doctype: bool,
) -> Optional[np.ndarray]:
    """
    Return the candidate subset to search, or None for the whole pool.

    Falls back to the full pool whenever filtering isn't requested, the profile
    has no doc_type, or no candidate matches it - rather than silently failing
    to resolve.
    """

    if not filter_by_doctype or not doc_type:
        return None

    subset = doctype_index.get(_normalize_doctype(doc_type))
    if subset is None or subset.size == 0:
        return None

    return subset


# =============================================================================
# HTML reconstruction
# =============================================================================

def apply_uris_to_html(
    html_content: str,
    mapping: Dict[str, dict],
    parser: str = "html.parser",
) -> str:
    """
    Reconstruct the document, adding a `uri` attribute to every
    manual_label/auto_label tag whose (normalized) docid was resolved.

    `uri` is inserted as the first attribute so the output matches the
    "<manual_label uri=... docid=... id=... ...>" ordering from the spec.
    Tags without a docid (e.g. nested title/fragment mentions, or mentions
    that failed coref resolution) are left untouched.
    """

    soup = BeautifulSoup(html_content, parser)

    for tag in soup.find_all(["manual_label", "auto_label"], attrs={"docid": True}):
        docid_norm = normalize_docid(tag.get("docid"))
        entry = mapping.get(docid_norm)
        if entry is None:
            continue

        attrs = dict(tag.attrs)
        attrs.pop("uri", None)
        tag.attrs = {"uri": entry["uri"], **attrs}

    return str(soup)


def strip_uris_from_html(html_content: str, parser: str = "html.parser") -> Tuple[str, int]:
    """
    Remove every `uri` attribute from manual_label/auto_label tags.

    Used in gold mode so the resolver never sees the ground-truth uri it is
    supposed to predict. Everything else (docid, labelname, parent, titletype,
    text, ...) is preserved untouched.

    Returns:
        (stripped_html, n_removed)
    """

    soup = BeautifulSoup(html_content, parser)

    removed = 0
    for tag in soup.find_all(["manual_label", "auto_label"], attrs={"uri": True}):
        del tag.attrs["uri"]
        removed += 1

    return str(soup), removed


# =============================================================================
# Core processing
# =============================================================================

def process_document_resolution(
    html_content: str,
    retriever: Retriever,
    candidates: List[dict],
    doctype_index: Dict[str, np.ndarray],
    filter_by_doctype: bool = True,
    include_fragments: bool = False,
    score_threshold: Optional[float] = None,
    debug_topk: int = 3,
) -> Tuple[str, Dict[str, dict], int]:
    """
    Rebuild the ReferenceProfileRegistry for one coref-resolved document
    (deterministic, no assistant calls), retrieve a matching candidate uri for
    every resolved profile, and return the HTML with `uri` attributes injected.

    Profiles are grouped by doctype subset so each group is searched (and, for
    neural retrievers, encoded) in a single batched call instead of one call
    per profile.

    Returns:
        output_html: the reconstructed HTML with `uri` attributes added.
        mapping: normalized docid -> {"uri", "score", "candidate_id",
                 "candidate_title", "query_text", "topk"}.
        unresolved: number of docid-bearing profiles that could not be
                    confidently assigned a uri (empty query text or below
                    --score-threshold).
    """

    mentions = extract_parent_level_annotations(html_content)

    rpr = ReferenceProfileRegistry()
    for mention in mentions:
        rpr.update_from_mention(ReferenceMention(mention.html_str))

    unresolved = 0

    # Only profiles that actually got a docid during the coref stage are
    # worth resolving; the rest were never coref-resolved and their tags
    # have no `docid` attribute for apply_uris_to_html() to match against.
    #
    # Group key: the doctype bucket the profile will be searched in. None means
    # "search the full pool".
    groups: Dict[Optional[str], List[Tuple[ReferenceProfile, str]]] = defaultdict(list)

    for profile in rpr:
        if not profile.docid:
            continue

        text = build_profile_text(profile, include_fragments=include_fragments)
        if not text.strip():
            unresolved += 1
            continue

        subset = _subset_for_doctype(profile.doc_type, doctype_index, filter_by_doctype)
        group_key = None if subset is None else _normalize_doctype(profile.doc_type)
        groups[group_key].append((profile, text))

    mapping: Dict[str, dict] = {}

    for group_key, items in groups.items():
        subset_idx = None if group_key is None else doctype_index[group_key]
        texts = [text for _, text in items]

        top_idx, top_scores = retriever.search(texts, k=debug_topk, subset_idx=subset_idx)

        for (profile, text), row_idx, row_scores in zip(items, top_idx, top_scores):
            best_score = float(row_scores[0])

            if score_threshold is not None and best_score < score_threshold:
                unresolved += 1
                continue

            best_candidate = candidates[int(row_idx[0])]

            mapping[normalize_docid(profile.docid)] = {
                "uri": best_candidate["original_url"],
                "score": best_score,
                "candidate_id": best_candidate.get("id"),
                "candidate_title": best_candidate.get("docTitle"),
                "query_text": text,
                "topk": [
                    {
                        "uri": candidates[int(i)]["original_url"],
                        "score": float(s),
                        "candidate_title": candidates[int(i)].get("docTitle"),
                    }
                    for i, s in zip(row_idx, row_scores)
                ],
            }

    output_html = apply_uris_to_html(html_content, mapping)

    return output_html, mapping, unresolved


def process_folder_resolution(
    folder: str,
    base_dir: Path,
    retriever: Retriever,
    candidates: List[dict],
    doctype_index: Dict[str, np.ndarray],
    filter_by_doctype: bool,
    include_fragments: bool,
    score_threshold: Optional[float],
    debug_topk: int,
    overwrite: bool,
) -> None:

    folder_dir = base_dir / folder
    if not folder_dir.exists():
        raise FileNotFoundError(f"Folder not found: {folder_dir}")

    doc_dirs = sorted(d for d in folder_dir.iterdir() if d.is_dir())
    print(f"[Setup] Found {len(doc_dirs)} document folder(s) in {folder_dir}\n")

    total_resolved = 0
    total_unresolved = 0

    for doc_dir in tqdm(doc_dirs, desc="Resolving documents"):
        coref_files = sorted(doc_dir.glob("*_coref.html"))
        if not coref_files:
            print(f"  [!] No *_coref.html found in {doc_dir.name} - skipping")
            continue

        coref_path = coref_files[0]
        doc_name = coref_path.name[: -len("_coref.html")]

        output_path = doc_dir / f"{doc_name}_resolution.html"
        mapping_path = doc_dir / f"{doc_name}_resolution_mapping.json"

        if output_path.exists() and not overwrite:
            print(f"  [skip] {doc_name}: resolution output already exists")
            continue

        print(f"\n{'='*80}")
        print(f"Resolving: {doc_name}")
        print(f"{'='*80}")

        with coref_path.open("r", encoding="utf-8") as f:
            html_content = f.read()

        output_html, mapping, unresolved = process_document_resolution(
            html_content=html_content,
            retriever=retriever,
            candidates=candidates,
            doctype_index=doctype_index,
            filter_by_doctype=filter_by_doctype,
            include_fragments=include_fragments,
            score_threshold=score_threshold,
            debug_topk=debug_topk,
        )

        with output_path.open("w", encoding="utf-8") as f:
            f.write(output_html)

        with mapping_path.open("w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)

        print(f"  Resolved {len(mapping)} profile(s), {unresolved} unresolved")
        print(f"  Saved {output_path.name} and {mapping_path.name}")

        total_resolved += len(mapping)
        total_unresolved += unresolved

    print(f"\n{'='*80}")
    print("RESOLUTION COMPLETE")
    print(f"Folder: {folder_dir}")
    print(f"Total profiles resolved: {total_resolved}")
    print(f"Total profiles unresolved: {total_unresolved}")
    print("="*80)


# =============================================================================
# Gold mode
# =============================================================================

def process_gold_split_resolution(
    split: str,
    gold_dir: Path,
    out_dir: Path,
    retriever: Retriever,
    candidates: List[dict],
    doctype_index: Dict[str, np.ndarray],
    filter_by_doctype: bool,
    include_fragments: bool,
    score_threshold: Optional[float],
    debug_topk: int,
    overwrite: bool,
) -> None:
    """
    Oracle run: take every gold annotated file of a split, strip its `uri`
    attributes, and run the exact same resolution as on stage-2 output.

    This isolates stage 3: the docids come from the gold annotation instead of
    from the coref model, so the resulting uri accuracy is an upper bound on
    what the full pipeline can achieve - which makes it the right setting for
    comparing retrievers against each other.
    """

    split_dir = gold_dir / split
    if not split_dir.exists():
        raise FileNotFoundError(f"Gold split directory not found: {split_dir}")

    gold_files = sorted(split_dir.glob("*.html"))
    if not gold_files:
        raise FileNotFoundError(f"No .html files found in {split_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Setup] Gold split   : {split_dir}")
    print(f"[Setup] Found {len(gold_files)} gold document(s)")
    print(f"[Setup] Output folder: {out_dir}\n")

    total_resolved = 0
    total_unresolved = 0
    total_stripped = 0

    for gold_path in tqdm(gold_files, desc="Resolving gold documents"):
        doc_name = gold_path.stem

        doc_dir = out_dir / doc_name
        input_path = doc_dir / f"{doc_name}_coref.html"
        output_path = doc_dir / f"{doc_name}_resolution.html"
        mapping_path = doc_dir / f"{doc_name}_resolution_mapping.json"

        if output_path.exists() and not overwrite:
            print(f"  [skip] {doc_name}: resolution output already exists")
            continue

        print(f"\n{'='*80}")
        print(f"Resolving (gold): {doc_name}")
        print(f"{'='*80}")

        with gold_path.open("r", encoding="utf-8") as f:
            gold_html = f.read()

        stripped_html, n_removed = strip_uris_from_html(gold_html)
        print(f"  - Stripped {n_removed} gold uri attribute(s) from the input")

        output_html, mapping, unresolved = process_document_resolution(
            html_content=stripped_html,
            retriever=retriever,
            candidates=candidates,
            doctype_index=doctype_index,
            filter_by_doctype=filter_by_doctype,
            include_fragments=include_fragments,
            score_threshold=score_threshold,
            debug_topk=debug_topk,
        )

        doc_dir.mkdir(parents=True, exist_ok=True)

        with input_path.open("w", encoding="utf-8") as f:
            f.write(stripped_html)

        with output_path.open("w", encoding="utf-8") as f:
            f.write(output_html)

        with mapping_path.open("w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)

        print(f"  Resolved {len(mapping)} profile(s), {unresolved} unresolved")
        print(f"  Saved {output_path.name} and {mapping_path.name}")

        total_resolved += len(mapping)
        total_unresolved += unresolved
        total_stripped += n_removed

    print(f"\n{'='*80}")
    print("GOLD RESOLUTION COMPLETE")
    print(f"Split: {split}")
    print(f"Output folder: {out_dir}")
    print(f"Gold uri attributes hidden: {total_stripped}")
    print(f"Total profiles resolved: {total_resolved}")
    print(f"Total profiles unresolved: {total_unresolved}")
    print("="*80)


# =============================================================================
# Arg parser
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 3: resolve every reference profile in a coref-resolved "
            "document to a candidate uri via similarity search over the "
            "pre-computed candidate pool."
        )
    )

    parser.add_argument(
        "--folder",
        default=None,
        help=(
            "Name of the stage-2 run folder under --dir, e.g. "
            "test_gemma4_coref_fs6_random_gold. Mutually exclusive with --gold."
        ),
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("output_coref"),
        help="Root directory containing --folder (and where --gold output is written).",
    )

    # --- gold (oracle) mode -------------------------------------------------
    parser.add_argument(
        "--gold",
        action="store_true",
        help=(
            "Run stage 3 on the gold annotated files of --split instead of on a "
            "stage-2 run folder. Gold `uri` attributes are stripped before resolution."
        ),
    )
    parser.add_argument(
        "--split",
        choices=["train", "test", "dev", "incoming", "extended_test"],
        default=None,
        help="Gold split to use with --gold.",
    )
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=Path("data/annotated"),
        help="Root of the gold annotations (default: data/annotated).",
    )
    parser.add_argument(
        "--output-folder",
        default=None,
        help=(
            "Name of the output folder under --dir for --gold mode "
            "(default: <split>_gold_resolution_<retriever>)."
        ),
    )

    # --- retrieval ----------------------------------------------------------
    parser.add_argument(
        "--retriever",
        choices=available_retrievers(),
        default="bge-s",
        help="Retrieval model. Must match the one used by precompute_index.py.",
    )
    parser.add_argument(
        "--encoder",
        default=None,
        help="Deprecated alias for --retriever, kept so old commands keep working.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Candidate pool index directory. Defaults to cache/candidate_pool_index_<retriever>_v1.",
    )
    parser.add_argument("--device", default=None, help="PyTorch device, e.g. cuda or cpu.")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override the HuggingFace checkpoint (must match the one indexed with).",
    )

    parser.add_argument(
        "--no-filter-by-doctype",
        action="store_true",
        help="Disable restricting candidates to the profile's doc_type.",
    )
    parser.add_argument(
        "--include-fragments",
        action="store_true",
        help="Include fragment mentions in the profile's query text.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help=(
            "Minimum score required to assign a uri. Only meaningful for "
            "similarity-based retrievers; rejected for BM25, whose scores are "
            "unbounded and not comparable across queries."
        ),
    )
    parser.add_argument(
        "--debug-topk",
        type=int,
        default=3,
        help="Number of top candidates recorded in the mapping json for inspection.",
    )

    parser.add_argument("--overwrite", action="store_true")

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.encoder:
        args.retriever = args.encoder

    # --- input mode validation ----------------------------------------------
    if args.gold and args.folder:
        parser.error("--gold and --folder are mutually exclusive.")
    if not args.gold and not args.folder:
        parser.error("Provide either --folder (stage-2 run) or --gold --split.")
    if args.gold and not args.split:
        parser.error("--gold requires --split (train/test/dev/incoming).")

    cache_dir = args.cache_dir or default_cache_dir(args.retriever)

    retriever_kwargs: Dict[str, Any] = {}
    if args.device is not None:
        retriever_kwargs["device"] = args.device
    if args.batch_size is not None:
        retriever_kwargs["batch_size"] = args.batch_size
    if args.model_name is not None:
        retriever_kwargs["model_name"] = args.model_name

    print(f"[Setup] Loading candidate pool from: {cache_dir}")
    retriever, candidates = load_candidate_pool(cache_dir, args.retriever, **retriever_kwargs)
    print(f"  Loaded {len(candidates):,} candidates with the {args.retriever} index\n")

    if args.score_threshold is not None and retriever.score_kind != "similarity":
        parser.error(
            f"--score-threshold is not supported for '{args.retriever}': its scores are "
            "unbounded and not comparable across queries, so any fixed cutoff is arbitrary."
        )

    doctype_index = build_doctype_index(candidates)

    common = dict(
        retriever=retriever,
        candidates=candidates,
        doctype_index=doctype_index,
        filter_by_doctype=not args.no_filter_by_doctype,
        include_fragments=args.include_fragments,
        score_threshold=args.score_threshold,
        debug_topk=args.debug_topk,
        overwrite=args.overwrite,
    )

    if args.gold:
        output_folder = args.output_folder or f"{args.split}_gold_resolution_{args.retriever}"
        process_gold_split_resolution(
            split=args.split,
            gold_dir=args.gold_dir,
            out_dir=args.dir / output_folder,
            **common,
        )
    else:
        process_folder_resolution(folder=args.folder, base_dir=args.dir, **common)


if __name__ == "__main__":
    main()