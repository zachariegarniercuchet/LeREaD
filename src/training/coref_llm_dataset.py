"""Build (input, output) SFT pairs for LLM coreference resolution (main_coref).

Replays ``main_coref.process_document_coref`` on gold documents with teacher
forcing: mentions are visited in document order and the Reference Profile
Registry is grown from the *gold* docids, so each example shows the model the
registry it would see at inference had every earlier mention been resolved
correctly.

  * input  = ``src.coref_input.build_coref_user_input`` (mention markup +
             upper context + registry) — the exact builder main_coref uses
  * output = the mention markup with ``docid="<gold docid>"`` on the outer tag

Two decisions are learned from the same format:
  * the authority is already in the registry -> copy that profile's docid
  * it is new -> emit a fresh canonical docid (the gold one)

Evaluation (MUC/B3/CEAFe/LEA) only compares clusters, so the exact spelling of
a new docid does not matter — only that later mentions reuse it.

``max_profiles`` (training only) caps the registry shown in very long
documents: the gold profile (if any) is always kept, other profiles are
sampled, original registry order is preserved. Inference always shows the full
registry; the cap only keeps late-document examples within ``max_seq_len``
instead of dropping them. None = no cap.

Training conditions (``--train-on``): train | silver | train+silver; dev is
built the same way for the dev-loss / best-checkpoint selection.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

from configs.config import DATA_DIR
from src.ann_extractor import extract_parent_level_annotations
from src.coref_input import (
    DEFAULT_PROFILE_ATTRIBUTES,
    build_coref_user_input,
    coref_target,
)
from src.htmlLabel import ReferenceMention
from src.rpr import ReferenceProfileRegistry, normalize_docid

TRAIN_SET_CONDITIONS: dict[str, list[str]] = {
    "train": ["train"],
    "silver": ["silver"],
    "train+silver": ["train", "silver"],
    "dev": ["dev"],
}

ANNOTATED_DIR = DATA_DIR / "annotated"


def _read_annotated_files(splits: Iterable[str]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for split in splits:
        folder = ANNOTATED_DIR / split
        if not folder.is_dir():
            print(f"⚠  Annotated folder not found, skipping: {folder}")
            out[split] = {}
            continue
        out[split] = {
            p.stem: p.read_text(encoding="utf-8")
            for p in sorted(folder.iterdir())
            if p.suffix.lower() in {".html", ".htm"}
        }
    return out


def _visible_profiles(registry, target_profile, max_profiles, rng):
    """Registry shown in the prompt; capped (gold profile kept) if requested."""
    profiles = list(registry)
    if max_profiles is None or len(profiles) <= max_profiles:
        return profiles
    others = [p for p in profiles if p is not target_profile]
    n_others = max_profiles - (1 if target_profile is not None else 0)
    keep = set(map(id, rng.sample(others, n_others)))
    if target_profile is not None:
        keep.add(id(target_profile))
    return [p for p in profiles if id(p) in keep]


def examples_from_html(
    html: str,
    source_file: str,
    max_fragments: int = 10,
    profile_attributes: list[str] | None = None,
    max_profiles: int | None = None,
    seed: int = 42,
) -> list[dict]:
    profile_attributes = profile_attributes or DEFAULT_PROFILE_ATTRIBUTES
    rng = random.Random(f"{seed}:{source_file}")
    registry = ReferenceProfileRegistry()
    examples: list[dict] = []

    for mention in extract_parent_level_annotations(html):
        mention_id = mention.html_tag.attributes.get("id")
        docid = normalize_docid(mention.html_tag.attributes.get("docid"))
        if not docid:
            continue

        target_profile = registry.get_profile_by_docid(docid)
        n_registry = len(registry)
        shown = _visible_profiles(registry, target_profile, max_profiles, rng)
        try:
            user_input = build_coref_user_input(
                html_content=html,
                mention=mention,
                profiles=shown,
                max_fragments=max_fragments,
                profile_attributes=profile_attributes,
            )
        except ValueError as exc:  # mention not locatable in the cleaned html
            print(f"  ⚠  {source_file} mention {mention_id}: {exc}")
            user_input = None

        if user_input is not None:
            examples.append({
                "input": user_input,
                "output": coref_target(mention, docid),
                "source_file": source_file,
                "mention_id": mention_id,
                "docid": docid,
                "is_new": target_profile is None,
                "n_profiles": len(shown),
                "n_registry": n_registry,
            })

        # Teacher forcing: grow the registry with the gold docid, the same way
        # main_coref does with the predicted one.
        mention.html_tag.set_attribute("docid", docid)
        registry.update_from_mention(ReferenceMention(mention.html_str))

    return examples


def build_coref_llm_examples(
    train_on: str = "train",
    max_fragments: int = 10,
    max_profiles: int | None = None,
    seed: int = 42,
) -> list[dict]:
    if train_on not in TRAIN_SET_CONDITIONS:
        raise ValueError(f"Unknown train_on={train_on!r}. Choose from {list(TRAIN_SET_CONDITIONS)}")
    examples: list[dict] = []
    for split, files in _read_annotated_files(TRAIN_SET_CONDITIONS[train_on]).items():
        print(f"[coref-llm-dataset] {train_on}: {len(files)} file(s) from '{split}'")
        for filename, html in files.items():
            ex = examples_from_html(
                html, f"{split}/{filename}", max_fragments=max_fragments,
                max_profiles=max_profiles, seed=seed,
            )
            examples.extend(ex)
            print(f"  {filename}: {len(ex)} mentions ({sum(e['is_new'] for e in ex)} new)")
    print(f"[coref-llm-dataset] {train_on}: built {len(examples)} examples")
    return examples


def get_or_build_dataset(
    train_on: str,
    max_fragments: int = 10,
    max_profiles: int | None = None,
    seed: int = 42,
    cache_dir: Path | None = None,
    force: bool = False,
) -> list[dict]:
    from configs.config import CACHE_DIR

    cache_dir = cache_dir or (CACHE_DIR / "training")
    mp = "all" if max_profiles is None else max_profiles
    path = cache_dir / f"coref_llm_{train_on}_mf{max_fragments}_mp{mp}_s{seed}.json"
    if not force and path.is_file():
        print(f"[coref-llm-dataset] cache hit -> {path}")
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    examples = build_coref_llm_examples(
        train_on, max_fragments=max_fragments, max_profiles=max_profiles, seed=seed,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"[coref-llm-dataset] saved {len(examples)} examples -> {path}")
    return examples
