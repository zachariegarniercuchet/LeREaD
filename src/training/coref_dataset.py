"""Build cross-encoder training pairs for the coreference resolution step.

Pairs are (mention, cluster_profile, label) triples built from the gold
docids carried in the annotated HTML. Mentions are processed in document
order, growing a gold ReferenceProfileRegistry as we go:

  * Positive  : the mention is matched against the profile of its own docid
                already present in the registry (i.e. a previous mention of
                the same authority already resolved). Label = 1.0.
  * Hard negative: the mention is matched against profiles of *other*
                docids of the same mention type in the same document.
                Label = 0.0.

The first mention of a docid yields no positive (it would open a new cluster
at inference time); it only seeds the registry so later mentions can match it.

Output examples (one dict per pair):
    {
        "input_a": "<mention text with left context>",
        "input_b": "<cluster profile text>",
        "label": 0.0 | 1.0,
        "mention_id": "<html id of the mention>",
        "mention_docid": "<gold docid of the mention>",
        "profile_docid": "<docid of the profile (None for none)>",
        "source_file": "<split>/<stem>",
    }

The pair text is consumed by ``src.models.coref_encoder`` which tokenizes
``input_a`` and ``input_b`` as a DeBERTa text pair (mention [SEP] profile).
"""
from __future__ import annotations

import ast
import json
import random
import re
from pathlib import Path
from typing import Iterable

from configs.config import CONTEXT_MAX_TOKENS, DATA_DIR
from src.ann_extractor import extract_parent_level_annotations
from src.rpr import ReferenceProfileRegistry, ReferenceProfile
from src.tokenizer_utils import tokenize, decode
from src.transforme_utils import LabelTransformConfig, prepare_label_tokens


TRAIN_SET_CONDITIONS: dict[str, list[str]] = {
    "train": ["train"],
    "silver": ["silver"],
    "train+silver": ["train", "silver"],
}

ANNOTATED_DIR = DATA_DIR / "annotated"

# Strip noisy attributes (style, uri, verified, ...) from the mention markup,
# keeping only labelname — exactly what the existing coref LLM pipeline feeds
# the assistant (see main_coref.process_document_coref).
_MENTION_INPUT_CFG = LabelTransformConfig(
    use_simplified=True,
    switch_type=False,
    keep_attributes=["labelname"],
)


def _read_annotated_files(splits: Iterable[str]) -> dict[str, dict[str, str]]:
    """{split: {filename: html}} for the requested annotated splits."""
    out: dict[str, dict[str, str]] = {}
    for split in splits:
        folder = ANNOTATED_DIR / split
        if not folder.is_dir():
            print(f"⚠  Annotated folder not found, skipping: {folder}")
            out[split] = {}
            continue
        files: dict[str, str] = {}
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() not in {".html", ".htm"}:
                continue
            files[path.stem] = path.read_text(encoding="utf-8")
        out[split] = files
    return out


def _profile_text(profile: ReferenceProfile) -> str:
    """Render one profile into a compact text string for the cross-encoder."""
    def join(values) -> str:
        if not values:
            return "none"
        return "; ".join(str(v) for v in values)

    titles = join(profile.alternative_titles.keys())
    citations = join(profile.citations.keys())
    fragments = join(profile.fragments_mentioned.keys())
    authors = join(profile.authors.keys())
    return (
        f"type: {profile.doc_type}\n"
        f"titles: {titles}\n"
        f"citations: {citations}\n"
        f"fragments: {fragments}\n"
        f"authors: {authors}"
    )


def _find_mention_token_start(tokens, mention):
    """Locate the index of the first token belonging to `mention` inside `tokens`."""
    tag = mention.html_tag
    for i, tok in enumerate(tokens):
        if str(tag) == tok:
            return i
    return None


def _examples_from_html(html: str, source_file: str, registry: ReferenceProfileRegistry,
                        rng: random.Random, max_hard_negatives: int) -> list[dict]:
    """Build (mention, profile, label) pairs for every parent mention in one doc.

    Mentions are visited in document order; the registry is grown after each
    mention so later mentions of the same authority can match its profile.
    """
    examples: list[dict] = []
    mentions = extract_parent_level_annotations(html)
    if not mentions:
        return examples

    # Tokenize the document once so we can fetch left context cheaply.
    doc_tokens = tokenize(html)

    for mention in mentions:
        mention_id = mention.html_tag.attributes.get("id")
        mention_docid = mention.html_tag.attributes.get("docid")
        mention_type = mention.name  # decision / legislation / secondary sources

        # Left context immediately preceding the mention (same window as the
        # existing coref LLM pipeline: get_mention_upper_context). Strip noisy
        # attributes (docid, uri, style, ...) from context tags so the model
        # cannot read the cluster id from earlier mentions — training and
        # inference must match exactly.
        start_idx = _find_mention_token_start(doc_tokens, mention)
        if start_idx is None:
            context = ""
        else:
            context_start = max(0, start_idx - CONTEXT_MAX_TOKENS)
            context = decode(
                prepare_label_tokens(doc_tokens[context_start:start_idx],
                                     _MENTION_INPUT_CFG))

        # Mention markup with noisy attributes stripped (keeps labelname only).
        mention_markup = decode(
            prepare_label_tokens(tokenize(str(mention.html_str)), _MENTION_INPUT_CFG)
        )

        input_a = (
            f"Mention: {mention_markup}\n"
            f"Context: {context}"
        )

        # Candidate profiles already in the registry (previous mentions),
        # restricted to the same mention type.
        candidates = [p for p in registry if p.doc_type == mention_type]

        # Positive: the profile of this mention's own docid, if present.
        pos_profile = registry.get_profile_by_docid(mention_docid)

        # Hard negatives: every other profile of the same mention type.
        neg_profiles = [p for p in candidates if p.docid != mention_docid]
        # max_hard_negatives=0 (or negative) -> keep ALL negatives; >0 -> sample.
        if max_hard_negatives is not None and max_hard_negatives > 0:
            if len(neg_profiles) > max_hard_negatives:
                neg_profiles = rng.sample(neg_profiles, max_hard_negatives)

        # Positive example (only when the authority was seen before).
        if pos_profile is not None:
            examples.append({
                "input_a": input_a,
                "input_b": _profile_text(pos_profile),
                "label": 1.0,
                "mention_id": mention_id,
                "mention_docid": mention_docid,
                "profile_docid": pos_profile.docid,
                "source_file": source_file,
            })

        # Hard-negative examples.
        for np_ in neg_profiles:
            examples.append({
                "input_a": input_a,
                "input_b": _profile_text(np_),
                "label": 0.0,
                "mention_id": mention_id,
                "mention_docid": mention_docid,
                "profile_docid": np_.docid,
                "source_file": source_file,
            })

        # Grow the gold registry with this mention so later mentions of the
        # same authority can match its profile.
        try:
            registry.update_from_mention(mention)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  ⚠  could not update registry for mention {mention_id}: {exc}")

    return examples


def build_coref_training_examples(
    train_on: str = "train",
    max_hard_negatives: int = 5,
    use_cache: bool = True,
    seed: int = 42,
) -> list[dict]:
    """Build cross-encoder training pairs for one training condition.

    Args:
        train_on: one of TRAIN_SET_CONDITIONS ("train" | "silver" | "train+silver").
        max_hard_negatives: max hard-negative profiles sampled per mention (0 = all).
        use_cache: kept for API compatibility (context is built from the HTML).
        seed: seed for hard-negative sampling.

    Returns:
        list of dicts with keys input_a / input_b / label / mention_id /
        mention_docid / profile_docid / source_file.
    """
    if train_on not in TRAIN_SET_CONDITIONS:
        raise ValueError(
            f"Unknown train_on={train_on!r}. Choose from {list(TRAIN_SET_CONDITIONS)}"
        )

    splits = TRAIN_SET_CONDITIONS[train_on]
    by_split = _read_annotated_files(splits)

    examples: list[dict] = []
    rng = random.Random(seed)

    for split in splits:
        files = by_split.get(split, {})
        if not files:
            continue
        print(f"[coref-dataset] {train_on}: processing {len(files)} file(s) from '{split}'")
        for filename, html in files.items():
            file_examples = _examples_from_html(
                html, f"{split}/{filename}", ReferenceProfileRegistry(), rng,
                max_hard_negatives)
            examples.extend(file_examples)
            print(f"  {filename}: {len(file_examples)} pairs")

    print(f"[coref-dataset] {train_on}: built {len(examples)} pairs")
    return examples


def save_dataset(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"[coref-dataset] saved {len(examples)} examples -> {path}")


def load_dataset(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_or_build_dataset(
    train_on: str,
    max_hard_negatives: int = 5,
    cache_dir: Path | None = None,
    force: bool = False,
    seed: int = 42,
) -> list[dict]:
    """Load the cached coref cross-encoder dataset, building it on first use."""
    from configs.config import CACHE_DIR

    cache_dir = cache_dir or (CACHE_DIR / "training")
    path = cache_dir / f"coref_ce_{train_on}_hn{max_hard_negatives}_s{seed}.json"
    if not force and path.is_file():
        print(f"[coref-dataset] cache hit -> {path}")
        return load_dataset(path)
    examples = build_coref_training_examples(
        train_on=train_on,
        max_hard_negatives=max_hard_negatives,
        seed=seed,
    )
    save_dataset(examples, path)
    return examples


# ---------------------------------------------------------------------------
# Arbitrary-split builder (used for the dev set / threshold tuning)
# ---------------------------------------------------------------------------
def build_coref_examples_for_split(
    split: str,
    max_hard_negatives: int = 5,
    seed: int = 42,
) -> list[dict]:
    """Build cross-encoder pairs from one annotated split folder (e.g. "dev").

    Reuses the same document-order / registry logic as the training builder but
    reads directly from DATA_DIR/annotated/<split>/ so the dev set is fully
    disjoint from the training condition.
    """
    folder = DATA_DIR / "annotated" / split
    if not folder.is_dir():
        print(f"⚠  Annotated folder not found, skipping: {folder}")
        return []

    examples: list[dict] = []
    rng = random.Random(seed)
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in {".html", ".htm"})
    print(f"[coref-dataset] split '{split}': processing {len(files)} file(s)")

    for path in files:
        html = path.read_text(encoding="utf-8")
        file_examples = _examples_from_html(
            html, f"{split}/{path.stem}", ReferenceProfileRegistry(), rng,
            max_hard_negatives)
        examples.extend(file_examples)
        print(f"  {path.stem}: {len(file_examples)} pairs")

    print(f"[coref-dataset] split '{split}': built {len(examples)} pairs")
    return examples
