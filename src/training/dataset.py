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
from src.extractor import (
    build_processing_segments,
    extract_few_shot_examples,
    get_list_of_mention,
)
from src.tokenizer_utils import decode, tokenize
from src.transforme_utils import LabelTransformConfig, prepare_label_tokens

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

# Per-pass configuration for decomposed (DEC) extraction fine-tuning.
#
# Each pass is a copy-and-annotate task:
#   * input  = pre-annotated text (already_labeled labels kept) + task token
#   * output = same text with new_labels added
#
# scope  : "chunk"  -> DEC0, one example per paragraph/sentence chunk
#          "mention"-> DEC1-3, one example per parent mention span
# parent_scope: None -> all parents in `parents` are processable
#                list -> only mentions whose parent label is in this list
DEC_PASS_CONFIGS: dict[str, dict] = {
    "DEC0": {
        "task_token": "[DEC0]",
        "scope": "chunk",
        "parents": ["decision", "legislation", "secondary sources"],
        "already_labeled": [],
        "new_labels": ["decision", "legislation", "secondary sources"],
        "parent_scope": None,
    },
    "DEC1": {
        "task_token": "[DEC1]",
        "scope": "mention",
        "parents": ["decision", "legislation", "secondary sources"],
        "already_labeled": ["decision", "legislation", "secondary sources"],
        "new_labels": ["title", "fragment"],
        "parent_scope": None,
    },
    "DEC2": {
        "task_token": "[DEC2]",
        "scope": "mention",
        "parents": ["secondary sources"],
        "already_labeled": [
            "decision", "legislation", "secondary sources", "title", "fragment",
        ],
        "new_labels": ["source", "authors"],
        "parent_scope": ["secondary sources"],
    },
    "DEC3": {
        "task_token": "[DEC3]",
        "scope": "mention",
        "parents": ["decision", "legislation"],
        "already_labeled": [
            "decision", "legislation", "secondary sources",
            "title", "fragment", "source", "authors",
        ],
        "new_labels": ["citation"],
        "parent_scope": ["decision", "legislation"],
    },
}

# All 8 labels the AIO extraction pass predicts (kept for reference).
DEC_ALL_LABELS = [
    "decision", "legislation", "secondary sources",
    "title", "citation", "source", "authors", "fragment",
]


def _all_passes() -> list[str]:
    """Return the list of all DEC pass ids in pipeline order."""
    return ["DEC0", "DEC1", "DEC2", "DEC3"]

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


def build_dec_training_examples(
    pass_id: str = "DEC0",
    train_on: str = "train",
    chunker: str = "paragraph",
    min_tokens: int = MIN_TOKENS,
    use_cache: bool = True,
) -> list[dict]:
    """Build the (input, output) SFT pairs for one decomposed (DEC) pass.

    Each example is prefixed with the pass task token (``[DEC0]`` … ``[DEC3]``)
    so the model learns to condition on which pass it is performing.

    Args:
        pass_id: one of "DEC0" | "DEC1" | "DEC2" | "DEC3" | "all".
        train_on: one of TRAIN_SET_CONDITIONS ("train" | "silver" | "train+silver").
        chunker: "paragraph" or "sentence" (only used by DEC0 chunking).
        min_tokens: chunk-merge threshold for DEC0.
        use_cache: route DEC0 chunking through the shared annotated chunk cache.

    Returns:
        list of dicts, each {"input": "<TASK_TOKEN> plain_text", "output": tagged_text,
        "source_file": "<split>/<stem>"}.
    """
    if pass_id == "all":
        examples: list[dict] = []
        for p in _all_passes():
            examples.extend(
                build_dec_training_examples(
                    pass_id=p, train_on=train_on, chunker=chunker,
                    min_tokens=min_tokens, use_cache=use_cache,
                )
            )
        print(f"[dataset] {train_on}/all: built {len(examples)} (input, output) pairs")
        return examples

    if pass_id not in DEC_PASS_CONFIGS:
        raise ValueError(
            f"Unknown pass_id={pass_id!r}. Choose from "
            f"{list(DEC_PASS_CONFIGS)} or 'all'."
        )
    if train_on not in TRAIN_SET_CONDITIONS:
        raise ValueError(
            f"Unknown train_on={train_on!r}. Choose from {list(TRAIN_SET_CONDITIONS)}"
        )

    cfg = DEC_PASS_CONFIGS[pass_id]
    task_token = cfg["task_token"]

    input_cfg = LabelTransformConfig(
        use_simplified=True,
        switch_type=True,
        keep_labels=list(cfg["already_labeled"]),
        keep_attributes=["labelname"],
    )
    output_cfg = LabelTransformConfig(
        use_simplified=True,
        switch_type=True,
        keep_labels=list(cfg["already_labeled"] + cfg["new_labels"]),
        keep_attributes=["labelname"],
    )

    splits = TRAIN_SET_CONDITIONS[train_on]
    by_split = _read_annotated_files(splits)

    examples: list[dict] = []
    for split in splits:
        files = by_split.get(split, {})
        if not files:
            continue
        print(
            f"[dataset] {pass_id}/{train_on}: processing {len(files)} file(s) "
            f"from '{split}' (scope={cfg['scope']})"
        )
        for filename, html in files.items():
            if cfg["scope"] == "chunk":
                # ── DEC0: chunk-based, one example per chunk ──────────────
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
                for ex in file_examples:
                    examples.append(
                        {
                            "input": f"{task_token} {ex['example']['input']}",
                            "output": ex["example"]["output"],
                            "source_file": ex["source_file"],
                        }
                    )
            else:
                # ── DEC1-3: mention-based, one example per parent mention ─
                tokens = tokenize(html)
                parent_mentions = get_list_of_mention(
                    tokens=tokens,
                    keep_labels=cfg["parents"],
                    label_type=None,
                )
                segments = build_processing_segments(tokens, parent_mentions)
                for segment in segments:
                    if not segment["process"]:
                        continue
                    html_label = segment["meta"]["label"]
                    # Enforce parent_scope (DEC2 -> secondary sources only,
                    # DEC3 -> decision/legislation only).
                    if (
                        cfg["parent_scope"] is not None
                        and html_label.name not in cfg["parent_scope"]
                    ):
                        continue
                    mention_tokens = segment["tokens"]

                    input_tokens = prepare_label_tokens(mention_tokens, input_cfg)
                    output_tokens = prepare_label_tokens(mention_tokens, output_cfg)

                    examples.append(
                        {
                            "input": f"{task_token} {decode(input_tokens)}",
                            "output": decode(output_tokens),
                            "source_file": f"{split}/{filename}",
                        }
                    )

    print(f"[dataset] {pass_id}/{train_on}: built {len(examples)} (input, output) pairs")
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
    pass_id: str | None = None,
) -> list[dict]:
    """Load the cached dataset for *train_on*, building it on first use.

    Cache key encodes everything that changes the pairs, so switching chunker,
    min_tokens, or DEC pass writes a separate file instead of silently reusing
    stale data.

    Args:
        train_on: one of TRAIN_SET_CONDITIONS ("train" | "silver" | "train+silver").
        chunker: "paragraph" or "sentence".
        min_tokens: chunk-merge threshold.
        cache_dir: override cache directory.
        force: ignore cache and rebuild.
        pass_id: one of "DEC0" | "DEC1" | "DEC2" | "DEC3" | "all" | None.
            None (default) builds the AIO dataset; any DEC value builds the
            decomposed pass dataset (with task tokens prepended).
    """
    from configs.config import CACHE_DIR

    cache_dir = cache_dir or (CACHE_DIR / "training")

    if pass_id is None:
        path = cache_dir / f"aio_{train_on}_{chunker}_mt{min_tokens}.json"
        if not force and path.is_file():
            print(f"[dataset] cache hit -> {path}")
            return load_dataset(path)
        examples = build_extraction_examples(
            train_on=train_on, chunker=chunker, min_tokens=min_tokens
        )
    else:
        if pass_id not in DEC_PASS_CONFIGS and pass_id != "all":
            raise ValueError(
                f"Unknown pass_id={pass_id!r}. Choose from "
                f"{list(DEC_PASS_CONFIGS)} or 'all'."
            )
        path = cache_dir / f"dec_{pass_id}_{train_on}_{chunker}_mt{min_tokens}.json"
        if not force and path.is_file():
            print(f"[dataset] cache hit -> {path}")
            return load_dataset(path)
        examples = build_dec_training_examples(
            pass_id=pass_id, train_on=train_on, chunker=chunker,
            min_tokens=min_tokens, use_cache=True,
        )

    save_dataset(examples, path)
    return examples
