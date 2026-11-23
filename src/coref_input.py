"""The coref user message, shared by inference (main_coref) and LoRA training.

Keeping one builder is what guarantees a fine-tuned coref model sees, at
inference, exactly the input format it was trained on.
"""
from __future__ import annotations

from typing import List, Optional

from configs.config import CONTEXT_MAX_TOKENS
from src.ann_extractor import get_mention_upper_context
from src.extractor import LabelTransformConfig, prepare_label_tokens
from src.fewshot.coref_helper import format_profile_for_prompt
from src.tokenizer_utils import decode, tokenize
from src.transforme_utils import clean_tokens

COREF_USER_PREFIX = "Annotate this mention: "

DEFAULT_PROFILE_ATTRIBUTES = [
    "docid",
    "alternative_titles",
    "citations",
    "fragments_mentioned",
    "authors",
]

# Mention markup as shown to the model: simplified tags, labelname only (no
# docid/uri/style leak).
MENTION_INPUT_CFG = LabelTransformConfig(
    use_simplified=True,
    switch_type=False,
    keep_attributes=["labelname"],
)


def mention_markup(mention) -> str:
    """The mention as the model sees it, e.g. ``<decision><title>X</title></decision>``."""
    return decode(prepare_label_tokens(tokenize(mention.html_str), MENTION_INPUT_CFG))


def build_coref_user_input(
    html_content: str,
    mention,
    profiles,
    max_fragments: int,
    profile_attributes: List[str],
    context_max_tokens: int = CONTEXT_MAX_TOKENS,
) -> str:
    """Mention + upper context + Reference Profile Registry (no prefix).

    ``profiles`` is any iterable of ReferenceProfile (the registry, or a subset
    of it at training time).
    """
    mention_id = mention.html_tag.attributes["id"]
    context = get_mention_upper_context(
        html=decode(clean_tokens(
            tokenize(html_content),
            keep_manual_label=False,
            keep_auto_label=False,
            protected_id=mention_id,
        )),
        mention=mention,
        max_tokens=context_max_tokens,
    )

    profiles_formatted = [
        format_profile_for_prompt(p.to_dict(attributes=profile_attributes), max_fragments=max_fragments)
        for p in profiles
    ]
    lines = ["Reference Profile Registry:"]
    lines += [f"  Profile {i}: {p}" for i, p in enumerate(profiles_formatted)]
    profiles_str = "\n".join(lines)

    return mention_markup(mention) + "\n Upper Context : \n" + context + "\n" + profiles_str


def coref_target(mention, docid: Optional[str]) -> str:
    """Training target: the mention markup with ``docid="..."`` on the outer tag.

    Same shape as the examples in src/prompts/coref_long.txt and what
    main_coref.extract_docid_from_generation parses.
    """
    markup = mention_markup(mention)
    i = markup.index(">")
    return f'{markup[:i]} docid="{docid}"{markup[i:]}'
