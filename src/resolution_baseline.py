"""Stage-3 *resolution* baseline: rule/string-matching URI linker.

Mirrors ``src/main_resolution.py`` in structure, CLI and output layout, but
replaces the neural/BM25 retriever over the pre-computed candidate-pool index
with a dependency-free, deterministic rule-based matcher (stdlib + bs4 only).

Matching tiers (first hit wins, best candidate within the tier wins):

  1. ``citation_exact`` — a profile citation (e.g. ``2019 SCC 65``,
     ``[2019] 4 S.C.R. 653``) equals a candidate reference string
     (neutral citation, CanLII reference, official-report / other
     references, legislation/doctrine references) after aggressive
     normalisation (tags stripped, accents folded, everything but
     ``[a-z0-9]`` removed, trailing ``canlii`` tolerated).
  2. ``title_exact`` — a profile title equals the candidate ``docTitle``
     (or a ``MAIN_TITLE``/``ALTERNATE_TITLE`` completion) after
     normalisation.
  3. ``fuzzy`` — best :class:`difflib.SequenceMatcher` ratio between any
     profile title and the candidate's titles, above ``--fuzzy-threshold``.
     Candidates are pre-filtered through a token-overlap block so this
     stays fast without any vector index.

Candidates come straight from the reference-repository CSV (same pool the
embedding runs index — default ``data/gold_candidate_pool.csv``, i.e. the
6,519-entry pool behind the ``d_t_gold_resolution_*`` runs), so no
pre-computed index is needed and there is nothing to keep in sync with
``precompute_index.py``.

Doctype handling mirrors ``main_resolution.py``: candidates are restricted
to the profile's ``doc_type`` bucket, falling back to the full pool when
the profile has no doc_type or the bucket is empty.

Example (gold setting on dev+train, as in the paper's Figure)::

    python -m src.resolution_baseline --gold --split dev_train \\
        --dir output_coref/2-resolution_gold

which writes ``output_coref/2-resolution_gold/d_t_gold_resolution_rule/``.
Evaluate exactly like the other retrievers::

    python -m src.evaluation.evaluate_resolution batch \\
        --batch-folder output_coref/2-resolution_gold/d_t_gold_resolution_rule \\
        --split dev_train
"""

import argparse
import csv
import difflib
import html
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from bs4 import BeautifulSoup
from tqdm import tqdm

from src.ann_extractor import extract_parent_level_annotations
from src.htmlLabel import ReferenceMention
from src.rpr import ReferenceProfile, ReferenceProfileRegistry, normalize_docid


# =============================================================================
# Normalisation
# =============================================================================

_TAG_RE = re.compile(r"<[^>]+>")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_WS_RE = re.compile(r"\s+")

# Candidate autoCompletion types used as citation/reference evidence.
# Anything with CITATION or REFERENCE in the type name qualifies; docket
# numbers, jurisdiction / collection names, authors, publishers and subjects
# are deliberately excluded.
_CITATION_TYPE_MARKERS = ("CITATION", "REFERENCE")
_TITLE_TYPE_MARKERS = ("TITLE",)

_MIN_CITE_KEY_LEN = 4
_MIN_TITLE_LEN = 4


def _ascii_fold(text: str) -> str:
    """Strip accents/diacritics (handles FR titles on both sides)."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def _clean_raw(text: str) -> str:
    """Strip HTML tags/entities and lowercase."""
    return html.unescape(_TAG_RE.sub(" ", str(text))).lower()


def normalize_title(text: str) -> str:
    """Normalised title key: alphanumerics separated by single spaces."""
    text = _ascii_fold(_clean_raw(text))
    text = _NON_ALNUM_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def citation_keys(text: str) -> Set[str]:
    """Aggressive citation keys: all non-alphanumerics removed.

    ``[2019] 4 S.C.R. 653`` and ``[2019] 4 SCR 653`` both become
    ``20194scr653``. A trailing ``canlii`` (from ``... (CanLII)`` completion
    strings) is also indexed in stripped form so ``2019 SCC 65`` still hits
    ``2019 SCC 65 (CanLII)``.
    """
    base = _NON_ALNUM_RE.sub("", _ascii_fold(_clean_raw(text)))
    keys = {base}
    if base.endswith("canlii") and len(base) > len("canlii"):
        keys.add(base[: -len("canlii")])
    return {k for k in keys if len(k) >= _MIN_CITE_KEY_LEN}


# =============================================================================
# Candidate pool loading (directly from the reference-repository CSV)
# =============================================================================

def _parse_completions(raw) -> List[dict]:
    if not raw:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list):
        return []
    out = []
    for completion in raw:
        if not isinstance(completion, dict):
            continue
        text = completion.get("text")
        if text is None:
            continue
        text = str(text).strip()
        if text:
            out.append({"type": str(completion.get("type") or ""), "text": text})
    return out


def load_rule_candidates(csv_path: Path) -> List[dict]:
    """Load the reference repository from a candidate-pool CSV.

    Returns one dict per metadata entry with the normalised key sets the
    matcher needs. Accepts both the current ``uri`` column schema and the
    older ``original_url`` cache schema for the resolution target.
    """

    candidates: List[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uri = (row.get("uri") or row.get("original_url") or "").strip()
            if not uri:
                continue
            try:
                metadata = json.loads(row.get("metadata") or "[]")
            except json.JSONDecodeError:
                continue
            if not isinstance(metadata, list):
                continue
            for entry in metadata:
                if not isinstance(entry, dict):
                    continue
                completions = _parse_completions(entry.get("autoCompletions"))

                cite_keys: Set[str] = set()
                title_keys: Set[str] = set()
                for completion in completions:
                    ctype = completion["type"].upper()
                    if any(m in ctype for m in _CITATION_TYPE_MARKERS):
                        cite_keys.update(citation_keys(completion["text"]))
                    if any(m in ctype for m in _TITLE_TYPE_MARKERS):
                        key = normalize_title(completion["text"])
                        if len(key) >= _MIN_TITLE_LEN:
                            title_keys.add(key)

                doc_title = str(entry.get("docTitle") or "").strip()
                doc_title_key = normalize_title(doc_title)
                if len(doc_title_key) >= _MIN_TITLE_LEN:
                    title_keys.add(doc_title_key)

                try:
                    cited_count = int(str(entry.get("citedCount") or "0").strip())
                except ValueError:
                    cited_count = 0

                candidates.append(
                    {
                        "uri": uri,
                        "id": entry.get("id"),
                        "docType": entry.get("docType"),
                        "docTitle": doc_title,
                        "citedCount": cited_count,
                        "cite_keys": cite_keys,
                        "title_keys": title_keys,
                    }
                )

    if not candidates:
        raise ValueError(f"No candidates could be loaded from {csv_path}")
    return candidates


# =============================================================================
# Rule index (inverted maps over the candidate pool, built once per run)
# =============================================================================

def _normalize_doctype(value: Optional[str]) -> str:
    if not value:
        return ""
    return str(value).strip().upper().replace(" ", "_")


class RuleIndex:
    """Inverted lookup structures for the rule-based matcher."""

    def __init__(self, candidates: List[dict]):
        self.candidates = candidates
        self.cite_to_idx: Dict[str, List[int]] = defaultdict(list)
        self.title_to_idx: Dict[str, List[int]] = defaultdict(list)
        self.token_to_idx: Dict[str, List[int]] = defaultdict(list)
        self.doctype_to_idx: Dict[str, List[int]] = defaultdict(list)

        for i, candidate in enumerate(candidates):
            for key in candidate["cite_keys"]:
                self.cite_to_idx[key].append(i)
            for key in candidate["title_keys"]:
                self.title_to_idx[key].append(i)
                for token in set(key.split()):
                    self.token_to_idx[token].append(i)
            self.doctype_to_idx[_normalize_doctype(candidate.get("docType"))].append(i)

    def subset(self, doc_type: Optional[str], filter_by_doctype: bool) -> Optional[Set[int]]:
        """Candidate row ids to search, or None for the full pool.

        Same fallback semantics as main_resolution.py: no filtering when not
        requested, when the profile has no doc_type, or when the bucket is
        empty.
        """
        if not filter_by_doctype or not doc_type:
            return None
        bucket = self.doctype_to_idx.get(_normalize_doctype(doc_type))
        if not bucket:
            return None
        return set(bucket)


# =============================================================================
# Profile -> query text (same representation as main_resolution.py, kept so
# the mapping jsons stay comparable across methods)
# =============================================================================

def build_profile_text(profile: ReferenceProfile, include_fragments: bool = False) -> str:
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
# Rule-based matching
# =============================================================================

# Score bands keep the tiers strictly ordered: citation_exact > title_exact
# > fuzzy, so scores stay comparable across profiles.
_CITE_BASE = 3.0
_TITLE_BASE = 2.0


def _profile_keys(
    profile: ReferenceProfile, include_fragments: bool = False
) -> Tuple[List[str], Set[str], List[str]]:
    """Return (title_keys, cite_keys, fuzzy_strings) for a profile."""
    titles: List[str] = []
    seen: Set[str] = set()

    raw_titles: List[str] = []
    if profile.main_title:
        raw_titles.append(str(profile.main_title))
    for title in profile.alternative_titles:
        if title != profile.main_title:
            raw_titles.append(str(title))

    for raw in raw_titles:
        key = normalize_title(raw)
        if len(key) >= _MIN_TITLE_LEN and key not in seen:
            seen.add(key)
            titles.append(key)

    cite_keys: Set[str] = set()
    for citation in profile.citations:
        cite_keys.update(citation_keys(str(citation)))

    fuzzy_strings = list(titles)
    if include_fragments:
        for fragment in profile.fragments_mentioned:
            key = normalize_title(str(fragment))
            if len(key) >= _MIN_TITLE_LEN and key not in seen:
                fuzzy_strings.append(key)

    return titles, cite_keys, fuzzy_strings


def _fuzzy_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _best_title_ratio(fuzzy_strings: List[str], candidate_title_keys: Set[str]) -> float:
    best = 0.0
    for profile_string in fuzzy_strings:
        for candidate_string in candidate_title_keys:
            # Cheap pre-filter before the full quadratic ratio.
            quick = difflib.SequenceMatcher(None, profile_string, candidate_string)
            if quick.real_quick_ratio() < 0.5:
                continue
            ratio = quick.ratio()
            if ratio > best:
                best = ratio
    return best


def resolve_profile(
    profile: ReferenceProfile,
    rule_index: RuleIndex,
    subset: Optional[Set[int]],
    fuzzy_threshold: float = 0.85,
    no_fuzzy: bool = False,
    include_fragments: bool = False,
    debug_topk: int = 3,
) -> Optional[dict]:
    """Match one reference profile against the candidate pool.

    Returns the mapping entry for the profile, or None when nothing matches
    (caller counts it as unresolved).
    """

    titles, cite_keys, fuzzy_strings = _profile_keys(
        profile, include_fragments=include_fragments
    )
    if not titles and not cite_keys:
        return None

    title_set = set(titles)

    def _in_subset(idx: int) -> bool:
        return subset is None or idx in subset

    # --- Tier 1: exact citation match ------------------------------------
    cite_hits: Dict[int, Set[str]] = defaultdict(set)
    for key in cite_keys:
        for idx in rule_index.cite_to_idx.get(key, ()):
            if _in_subset(idx):
                cite_hits[idx].add(key)

    if cite_hits:
        ranked = []
        for idx, shared in cite_hits.items():
            candidate = rule_index.candidates[idx]
            title_bonus = 0.5 if title_set & candidate["title_keys"] else 0.0
            score = _CITE_BASE + 0.2 * (len(shared) - 1) + title_bonus
            ratio = (
                _best_title_ratio(fuzzy_strings, candidate["title_keys"])
                if fuzzy_strings and candidate["title_keys"]
                else 0.0
            )
            ranked.append((idx, score, ratio, len(shared)))
        ranked.sort(key=lambda r: (-r[1], -r[2], -rule_index.candidates[r[0]]["citedCount"], r[0]))
        return _make_entry(
            profile, rule_index, ranked, "citation_exact", debug_topk, fuzzy_strings
        )

    # --- Tier 2: exact title match ---------------------------------------
    title_hits: List[int] = []
    seen_idx: Set[int] = set()
    for key in titles:
        for idx in rule_index.title_to_idx.get(key, ()):
            if _in_subset(idx) and idx not in seen_idx:
                seen_idx.add(idx)
                title_hits.append(idx)

    if title_hits:
        ranked = []
        for idx in title_hits:
            candidate = rule_index.candidates[idx]
            ratio = (
                _best_title_ratio(fuzzy_strings, candidate["title_keys"])
                if fuzzy_strings and candidate["title_keys"]
                else 0.0
            )
            ranked.append((idx, _TITLE_BASE, ratio, 0))
        ranked.sort(key=lambda r: (-r[2], -rule_index.candidates[r[0]]["citedCount"], r[0]))
        return _make_entry(
            profile, rule_index, ranked, "title_exact", debug_topk, fuzzy_strings
        )

    # --- Tier 3: fuzzy title fallback ------------------------------------
    if no_fuzzy or not fuzzy_strings:
        return None

    overlap: Dict[int, int] = defaultdict(int)
    for profile_string in fuzzy_strings:
        for token in set(profile_string.split()):
            for idx in rule_index.token_to_idx.get(token, ()):
                if _in_subset(idx):
                    overlap[idx] += 1

    if not overlap:
        return None

    # Cap the expensive ratio computations to the best token-overlap block.
    block = sorted(overlap, key=lambda i: (-overlap[i], i))[:400]
    ranked = []
    for idx in block:
        candidate = rule_index.candidates[idx]
        if not candidate["title_keys"]:
            continue
        ratio = _best_title_ratio(fuzzy_strings, candidate["title_keys"])
        if ratio >= fuzzy_threshold:
            ranked.append((idx, ratio, ratio, 0))

    if not ranked:
        return None
    ranked.sort(key=lambda r: (-r[1], -rule_index.candidates[r[0]]["citedCount"], r[0]))
    return _make_entry(profile, rule_index, ranked, "fuzzy", debug_topk, fuzzy_strings)


def _make_entry(
    profile: ReferenceProfile,
    rule_index: RuleIndex,
    ranked: List[Tuple[int, float, float, int]],
    method: str,
    debug_topk: int,
    fuzzy_strings: List[str],
) -> dict:
    best_idx, best_score, _, _ = ranked[0]
    best = rule_index.candidates[best_idx]

    topk = []
    for idx, score, _, _ in ranked[: max(debug_topk, 1)]:
        candidate = rule_index.candidates[idx]
        topk.append(
            {
                "uri": candidate["uri"],
                "score": float(score),
                "candidate_title": candidate["docTitle"],
            }
        )

    # matched_key records the evidence that fired, for inspection.
    matched_key: Optional[str] = None
    if method == "citation_exact":
        _, cite_keys, _ = _profile_keys(profile)
        shared = sorted(cite_keys & best["cite_keys"])
        matched_key = shared[0] if shared else None
    elif method in ("title_exact", "fuzzy"):
        titles, _, _ = _profile_keys(profile)
        shared = sorted(set(titles) & best["title_keys"])
        matched_key = shared[0] if shared else (fuzzy_strings[0] if fuzzy_strings else None)

    return {
        "uri": best["uri"],
        "score": float(best_score),
        "candidate_id": best["id"],
        "candidate_title": best["docTitle"],
        "query_text": build_profile_text(profile),
        "method": method,
        "matched_key": matched_key,
        "topk": topk,
    }


# =============================================================================
# HTML reconstruction (identical semantics to main_resolution.py)
# =============================================================================

def apply_uris_to_html(
    html_content: str,
    mapping: Dict[str, dict],
    parser: str = "html.parser",
) -> str:
    """Add a `uri` attribute to every resolved mention tag (as first attr)."""

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
    """Remove every `uri` attribute (gold mode: hide the answer to predict)."""

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
    rule_index: RuleIndex,
    filter_by_doctype: bool = True,
    fuzzy_threshold: float = 0.85,
    no_fuzzy: bool = False,
    include_fragments: bool = False,
    debug_topk: int = 3,
) -> Tuple[str, Dict[str, dict], int]:
    """Rebuild the profile registry deterministically and resolve each
    docid-bearing profile with the rule-based matcher.

    Returns (output_html, mapping, unresolved) with the same semantics as
    main_resolution.process_document_resolution.
    """

    mentions = extract_parent_level_annotations(html_content)

    rpr = ReferenceProfileRegistry()
    for mention in mentions:
        rpr.update_from_mention(ReferenceMention(mention.html_str))

    unresolved = 0
    mapping: Dict[str, dict] = {}

    for profile in rpr:
        if not profile.docid:
            continue

        titles, cite_keys, _ = _profile_keys(profile, include_fragments=include_fragments)
        if not titles and not cite_keys:
            unresolved += 1
            continue

        subset = rule_index.subset(profile.doc_type, filter_by_doctype)
        entry = resolve_profile(
            profile,
            rule_index,
            subset,
            fuzzy_threshold=fuzzy_threshold,
            no_fuzzy=no_fuzzy,
            include_fragments=include_fragments,
            debug_topk=debug_topk,
        )
        if entry is None:
            unresolved += 1
            continue

        mapping[normalize_docid(profile.docid)] = entry

    output_html = apply_uris_to_html(html_content, mapping)

    return output_html, mapping, unresolved


def process_folder_resolution(
    folder: str,
    base_dir: Path,
    rule_index: RuleIndex,
    filter_by_doctype: bool,
    fuzzy_threshold: float,
    no_fuzzy: bool,
    include_fragments: bool,
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
            rule_index=rule_index,
            filter_by_doctype=filter_by_doctype,
            fuzzy_threshold=fuzzy_threshold,
            no_fuzzy=no_fuzzy,
            include_fragments=include_fragments,
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
    print("BASELINE RESOLUTION COMPLETE")
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
    rule_index: RuleIndex,
    filter_by_doctype: bool,
    fuzzy_threshold: float,
    no_fuzzy: bool,
    include_fragments: bool,
    debug_topk: int,
    overwrite: bool,
) -> None:
    """Oracle run on gold files: strip `uri`s, then resolve as on stage-2 output."""

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
            rule_index=rule_index,
            filter_by_doctype=filter_by_doctype,
            fuzzy_threshold=fuzzy_threshold,
            no_fuzzy=no_fuzzy,
            include_fragments=include_fragments,
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
    print("GOLD BASELINE RESOLUTION COMPLETE")
    print(f"Split: {split}")
    print(f"Output folder: {out_dir}")
    print(f"Gold uri attributes hidden: {total_stripped}")
    print(f"Total profiles resolved: {total_resolved}")
    print(f"Total profiles unresolved: {total_unresolved}")
    print("="*80)


# =============================================================================
# Arg parser
# =============================================================================

def _default_output_folder(split: str) -> str:
    short = "d_t" if split == "dev_train" else split
    return f"{short}_gold_resolution_rule"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 3 baseline: resolve every reference profile to a candidate "
            "uri with deterministic string-matching rules (no embeddings)."
        )
    )

    parser.add_argument(
        "--folder",
        default=None,
        help=(
            "Name of the stage-2 run folder under --dir. "
            "Mutually exclusive with --gold."
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
        choices=["train", "test", "dev", "incoming", "extended_test", "dev_train"],
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
            "(default: <split>_gold_resolution_rule, with dev_train -> d_t)."
        ),
    )

    # --- reference repository ----------------------------------------------
    parser.add_argument(
        "--candidate-pool",
        type=Path,
        default=Path("data/gold_candidate_pool.csv"),
        help=(
            "Reference-repository CSV (default: data/gold_candidate_pool.csv, "
            "the 6,519-entry pool behind the d_t_gold_resolution_* runs). "
            "Use data/full_candidate_pool.csv for the larger pool."
        ),
    )

    # --- rule behaviour ------------------------------------------------------
    parser.add_argument(
        "--no-filter-by-doctype",
        action="store_true",
        help="Disable restricting candidates to the profile's doc_type.",
    )
    parser.add_argument(
        "--include-fragments",
        action="store_true",
        help="Also use fragment mentions as fuzzy-match strings.",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=0.85,
        help="Minimum SequenceMatcher ratio to accept a fuzzy title match (default: 0.85).",
    )
    parser.add_argument(
        "--no-fuzzy",
        action="store_true",
        help="Disable the fuzzy fallback: only exact citation/title matches resolve.",
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

    # --- input mode validation ----------------------------------------------
    if args.gold and args.folder:
        parser.error("--gold and --folder are mutually exclusive.")
    if not args.gold and not args.folder:
        parser.error("Provide either --folder (stage-2 run) or --gold --split.")
    if args.gold and not args.split:
        parser.error("--gold requires --split (train/test/dev/dev_train/incoming).")

    print(f"[Setup] Loading reference repository from: {args.candidate_pool}")
    candidates = load_rule_candidates(args.candidate_pool)
    print(f"  Loaded {len(candidates):,} candidates (no index required)\n")

    rule_index = RuleIndex(candidates)

    common = dict(
        rule_index=rule_index,
        filter_by_doctype=not args.no_filter_by_doctype,
        fuzzy_threshold=args.fuzzy_threshold,
        no_fuzzy=args.no_fuzzy,
        include_fragments=args.include_fragments,
        debug_topk=args.debug_topk,
        overwrite=args.overwrite,
    )

    if args.gold:
        output_folder = args.output_folder or _default_output_folder(args.split)
        out_dir = args.dir / output_folder
        process_gold_split_resolution(
            split=args.split,
            gold_dir=args.gold_dir,
            out_dir=out_dir,
            **common,
        )
        config_path = out_dir / "run_config.json"
        with config_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "method": "rule_baseline",
                    "split": args.split,
                    "candidate_pool": str(args.candidate_pool),
                    "num_candidates": len(candidates),
                    "filter_by_doctype": common["filter_by_doctype"],
                    "fuzzy_threshold": args.fuzzy_threshold,
                    "no_fuzzy": args.no_fuzzy,
                    "include_fragments": args.include_fragments,
                },
                f,
                indent=2,
            )
        print(f"Saved run config to: {config_path}")
    else:
        process_folder_resolution(folder=args.folder, base_dir=args.dir, **common)


if __name__ == "__main__":
    main()
