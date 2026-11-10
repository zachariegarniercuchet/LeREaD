"""EncoderAssistant: plug a trained encoder into the existing AIO pipeline.

Same BaseAssistant interface as LoraAssistant. generate() receives the usual
chat messages, pulls the chunk out of the last user message, and returns the
chunk with tags inserted (verbatim copy), so chunking, verification,
post-processing and evaluate.py are untouched.
"""
from __future__ import annotations

from typing import List

from src.models.base import BaseAssistant


class EncoderAssistant(BaseAssistant):
    def __init__(self, adapter_path: str, temperature: float = 0.0,
                 has_system_role: bool = True, user_prefix: str = "Annotate this text:"):
        if not adapter_path:
            raise ValueError(
                "EncoderAssistant requires adapter_path: a trained run folder "
                "(e.g. <MODELS_DIR>/output_training/<run_name>/) written by "
                "src.main_training_encoder. Pass it explicitly, e.g. "
                "AssistantFactory.create('deberta-v3-large', adapter_path='...')."
            )
        super().__init__(adapter_path, temperature, has_system_role=has_system_role)
        from src.models.encoder import load_encoder

        self.user_prefix = user_prefix
        self.model, self.tokenizer, self.cfg = load_encoder(adapter_path)
        self.device = next(self.model.parameters()).device

    def generate(self, messages: List[dict], max_new_tokens: int | None = None) -> str:
        from src.models.encoder import predict_kept, render

        user = [m for m in messages if m["role"] == "user"][-1]["content"]
        text = user[len(self.user_prefix):] if user.startswith(self.user_prefix) else user
        kept = predict_kept(self.model, self.tokenizer, text, self.device,
                            self.cfg["max_len"], self.cfg["stride"])
        return render(text, kept)