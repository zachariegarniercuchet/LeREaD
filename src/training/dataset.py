"""Build (input, output) SFT pairs for the extraction task from annotated HTML.

The fine-tuning task is the AIO copy-and-annotate formulation already used at
inference: given a plain-text chunk of a decision, produce the same chunk with
``<decision>/<legislation>/<secondary sources>`` mentions and their nested
``<title>/<citation>/<fragment>/<authors>/<source>`` components inserted as
simplified tags.

We get the (input, output) pairs for free from the same machinery that builds
the few-shot examples (``src.extractor.extract_few_shot_examples``):

  * input  = the annotated chunk with every label tag stripped  (plain text)
  * output = the annotated chunk rendered with the AIO output config
             (simplified tags, docid/uri/attributes dropped)

so a fine-tuned model is trained to emit exactly the string format the
existing pipeline, post-processing, and ``evaluate.py`` already understand.

Training conditions (``--train-on``):
  * ``train``        gold train split only          (data/annotated/train)
  * ``silver``       silver supplement only         (data/annotated/silver)
  * ``train+silver`` the union of the two
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from configs.config import DATA_DIR, MIN_TOKENS, PROMPT_DIR
from src.chunkers import ChunkerFactory
from src.extractor import extract_few_shot_examples
from src.transforme_utils import LabelTransformConfig

# The 8 labels the AIO extraction pass predicts (see
# src.main_extraction.get_method_config("AIO")["new_labels"]). Docid/uri and
# component attributes are deliberately excluded — they belong to the
# coreference and linking stages, not to span extraction.
AIO_LABELS = [
    "decision", "legislation", "secondary sources",
    "title", "citation", "source", "authors", "fragment",
]

# Which annotated splits each --train-on condition pulls from.
TRAIN_SET_CONDITIONS: dict[str, list[str]] = {
    "train": ["train"],
    "silver": ["silver"],
    "train+silver": ["train", "silver"],
}

ANNOTATED_DIR = DATA_DIR / "annotated"


def load_system_prompt(prompt_type: str = "long") -> str:
    """Return the AIO system prompt used at inference (src/prompts/allInOne_*.txt).

    The fine-tuned model is trained under this same system prompt so that it
    can be dropped into the existing AIO pipeline unchanged.
    """
    with open(PROMPT_DIR / f"allInOne_{prompt_type}.txt", "r", encoding="utf-8") as f:
        return f.read()


def _aio_output_config() -> LabelTransformConfig:
    """How the AIO pass renders its output (mirrors load_fewshot_examples).

    use_simplified -> ``<decision>`` instead of ``<manual_label labelname=...>``
    switch_type    -> manual_label -> auto_label (the type the model must emit)
    keep_labels    -> keep all 8 predicted labels, strip any other tag's name
    keep_attributes-> keep only labelname, dropping docid/uri/titletype/...
    """
    return LabelTransformConfig(
        use_simplified=True,
        switch_type=True,
        keep_labels=list(AIO_LABELS),
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


def build_extraction_examples(
    train_on: str = "train",
    chunker: str = "paragraph",
    min_tokens: int = MIN_TOKENS,
    use_cache: bool = True,
) -> list[dict]:
    """Build the (input, output) SFT pairs for one training condition.

    Args:
        train_on: one of TRAIN_SET_CONDITIONS ("train" | "silver" | "train+silver").
        chunker: "paragraph" (default, matches inference) or "sentence".
        min_tokens: chunk-merge threshold. Inference uses MIN_TOKENS (500); a
            smaller value yields more, shorter training sequences.
        use_cache: route chunking through the shared annotated chunk cache.

    Returns:
        list of dicts, each {"input": plain_text, "output": tagged_text,
        "source_file": "<split>/<stem>"} — the same shape as the few-shot
        example cache, ready to feed to the chat template in trainer.py.
    """
    if train_on not in TRAIN_SET_CONDITIONS:
        raise ValueError(
            f"Unknown train_on={train_on!r}. Choose from {list(TRAIN_SET_CONDITIONS)}"
        )

    splits = TRAIN_SET_CONDITIONS[train_on]
    by_split = _read_annotated_files(splits)
    output_cfg = _aio_output_config()

    examples: list[dict] = []
    for split in splits:
        files = by_split.get(split, {})
        if not files:
            continue
        print(f"[dataset] {train_on}: chunking {len(files)} file(s) from '{split}'")
        for filename, html in files.items():
            chunks = ChunkerFactory.get_chunks(
                html,
                method=chunker,
                annotated=True,
                min_tokens=min_tokens,
                split=split if use_cache else None,
                filename=filename if use_cache else None,
            )
            file_examples = extract_few_shot_examples(
                chunks, output_cfg, source_file=f"{split}/{filename}"
            )
            # extract_few_shot_examples nests the pair under "example" and adds
            # pattern metadata we don't need for SFT — flatten to input/output.
            for ex in file_examples:
                examples.append(
                    {
                        "input": ex["example"]["input"],
                        "output": ex["example"]["output"],
                        "source_file": ex["source_file"],
                    }
                )

    print(f"[dataset] {train_on}: built {len(examples)} (input, output) pairs")
    return examples


def save_dataset(examples: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"[dataset] saved {len(examples)} examples -> {path}")


def load_dataset(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_or_build_dataset(
    train_on: str,
    chunker: str = "paragraph",
    min_tokens: int = MIN_TOKENS,
    cache_dir: Path | None = None,
    force: bool = False,
) -> list[dict]:
    """Load the cached dataset for *train_on*, building it on first use.

    Cache key encodes everything that changes the pairs, so switching chunker
    or min_tokens writes a separate file instead of silently reusing stale data.
    """
    from configs.config import CACHE_DIR

    cache_dir = cache_dir or (CACHE_DIR / "training")
    path = cache_dir / f"aio_{train_on}_{chunker}_mt{min_tokens}.json"

    if not force and path.is_file():
        print(f"[dataset] cache hit -> {path}")
        return load_dataset(path)

    examples = build_extraction_examples(
        train_on=train_on, chunker=chunker, min_tokens=min_tokens
    )
    save_dataset(examples, path)
    return examples
