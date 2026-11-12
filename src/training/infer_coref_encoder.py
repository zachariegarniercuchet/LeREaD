"""CorefEncoderAssistant: plug a trained cross-encoder into main_coref.py.

Same BaseAssistant interface as the LLM assistants. generate() receives the
usual chat messages built by main_coref.py (mention + left context +
Reference Profile Registry), scores the mention against every profile with the
trained DeBERTa cross-encoder, and returns the best-matching profile's
``docid="..."`` (so the existing extract_docid_from_generation / registry
growth logic in main_coref.py is untouched). If no profile scores above the
threshold, a new cluster is opened and the mention's own title is returned as
the new canonical docid.

Usage:
    assistant = CorefEncoderAdapter(adapter_path=<run folder>)
    # or via the factory:
    assistant = AssistantFactory.create("deberta", adapter_path=<run folder>)
"""
from __future__ import annotations

import ast
import re
from typing import List, Optional, Tuple

from src.models.base import BaseAssistant
from src.htmlLabel import ReferenceMention


_PROFILE_LINE_RE = re.compile(
    r"^\s*Profile\s+\d+:\s*(\{.*\})\s*$", re.DOTALL
)
_UPPER_CONTEXT_SPLIT = "\n Upper Context : \n"
_REGISTRY_SPLIT = "\nReference Profile Registry:"


class CorefEncoderAssistant(BaseAssistant):
    def __init__(self, adapter_path: str, threshold: float = 0.5,
                 has_system_role: bool = True, user_prefix: str = "Annotate this mention:",
                 max_len: int = 512, batch_size: int = 16):
        if not adapter_path:
            raise ValueError(
                "CorefEncoderAssistant requires adapter_path: a trained run folder "
                "(e.g. <MODELS_DIR>/output_training/<run_name>/) written by "
                "src.main_training_coref_encoder. Pass it explicitly."
            )
        super().__init__(adapter_path, temperature=0.0,
                         has_system_role=has_system_role)
        self.user_prefix = user_prefix
        self.threshold = threshold
        self.max_len = max_len
        self.batch_size = batch_size
        from src.models.coref_encoder import load_coref_encoder, predict_scores

        self._load_coref_encoder = load_coref_encoder
        self._predict_scores = predict_scores
        self.model, self.tokenizer, self.cfg = load_coref_encoder(adapter_path)
        # Threshold saved at training time (best on dev) wins unless overridden.
        self.threshold = float(self.cfg.get("best_threshold", threshold))
        self.device = next(self.model.parameters()).device
        print(f"[coref-encoder] loaded {adapter_path} | threshold={self.threshold}")

    # ------------------------------------------------------------------ #
    # Parsing of the user message built by main_coref.process_document_coref
    # ------------------------------------------------------------------ #
    @staticmethod
    def _parse_user_input(content: str) -> Tuple[str, str, List[dict]]:
        """Split ``mention | context | [profiles]`` from the user message.

        Returns (mention_html, context, profile_dicts).
        """
        # Strip the user prefix if present.
        if content.startswith("Annotate this mention: "):
            content = content[len("Annotate this mention: "):]

        mention_html = content
        context = ""
        profiles: List[dict] = []

        if _UPPER_CONTEXT_SPLIT in content:
            mention_html, rest = content.split(_UPPER_CONTEXT_SPLIT, 1)
        else:
            rest = ""

        if _REGISTRY_SPLIT in rest:
            context, registry_part = rest.split(_REGISTRY_SPLIT, 1)
        else:
            context = rest
            registry_part = ""

        for line in registry_part.splitlines():
            m = _PROFILE_LINE_RE.match(line)
            if not m:
                continue
            dict_text = m.group(1)
            try:
                profile = ast.literal_eval(dict_text)
                if isinstance(profile, dict):
                    profiles.append(profile)
            except (ValueError, SyntaxError):
                continue

        return mention_html.strip(), context.strip(), profiles

    def _new_cluster_docid(self, mention_html: str) -> str:
        """Derive a canonical docid for a brand-new cluster from the mention.

        Prefers the mention's title; falls back to its plain text; then to a
        generic placeholder so the registry can still grow.
        """
        try:
            mention = ReferenceMention(mention_html)
            titles = mention.get_sublabel_texts().get("title", [])
            for t in titles:
                if t and t.strip():
                    return t.strip()
            text = (mention.text or "").strip()
            if text:
                return text
        except Exception:
            pass
        return "new_cluster"

    # ------------------------------------------------------------------ #
    # BaseAssistant interface
    # ------------------------------------------------------------------ #
    def generate(self, messages: List[dict], max_new_tokens: int | None = None) -> str:
        user = [m for m in messages if m["role"] == "user"][-1]["content"]
        mention_html, context, profiles = self._parse_user_input(user)

        if not profiles:
            # No existing profiles -> always open a new cluster.
            return f'docid="{self._new_cluster_docid(mention_html)}"'

        # Build one (mention+context, profile) pair per candidate profile.
        input_a = f"Mention: {mention_html}\nContext: {context}"
        pairs = [
            {"input_a": input_a, "input_b": _profile_text(p)}
            for p in profiles
        ]

        from src.models.coref_encoder import predict_scores
        scores = predict_scores(
            self.model, self.tokenizer, pairs, self.device,
            self.max_len, self.batch_size,
        )

        best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        best_score = scores[best_idx]
        best_profile = profiles[best_idx]
        best_docid = best_profile.get("docid")

        if best_score >= self.threshold and best_docid:
            return f'docid="{best_docid}"'

        # Below threshold -> open a new cluster from the mention itself.
        return f'docid="{self._new_cluster_docid(mention_html)}"'


def _profile_text(profile: dict) -> str:
    """Render one profile dict (from format_profile_for_prompt) to text."""
    def join(values) -> str:
        if not values:
            return "none"
        return "; ".join(str(v) for v in values)

    return (
        f"docid: {profile.get('docid') or 'none'}\n"
        f"type: {profile.get('doc_type') or 'none'}\n"
        f"titles: {join(profile.get('alternative_titles') or [])}\n"
        f"citations: {join(profile.get('citations') or [])}\n"
        f"fragments: {join(profile.get('fragments_mentioned') or [])}\n"
        f"authors: {join(profile.get('authors') or [])}"
    )
