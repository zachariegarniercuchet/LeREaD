"""LoraAssistant: run a fine-tuned (LoRA) model through the existing pipeline.

Implements the same ``BaseAssistant`` interface as the inference assistants, so
a fine-tuned checkpoint plugs into ``main_extraction.py``'s AIO path with no
change to chunking, verification, post-processing, or evaluation.

The adapter is merged into the base weights at load time (``merge_and_unload``)
so generation is a plain forward pass with no extra PEFT overhead.
"""
from __future__ import annotations

import re
from typing import List

from src.models.base import BaseAssistant


class LoraAssistant(BaseAssistant):
    """Base model + LoRA adapter, exposed through the standard generate() API.

    Args:
        adapter_path: directory holding the saved adapter + tokenizer
            (the ``output_training/<run_name>/`` folder written by trainer.py).
        model_path: optional explicit base-model path. If None, it is read from
            the adapter's own ``run_config.json`` via the model registry.
        temperature: sampling temperature.
        max_tokens: max new tokens per generation.
        quantization: base-model quantization ("4bit" | "8bit" | "fp16" | "bf16").
        has_system_role: whether the chat template uses a system role.
        merge: merge the adapter into base weights (True) or keep it PEFT-wrapped
            (False). Merging is faster at inference.
    """

    def __init__(
        self,
        adapter_path: str,
        model_short: str | None = None,
        model_path: str | None = None,
        temperature: float = 0.3,
        max_tokens: int = 2048,
        quantization: str = "4bit",
        has_system_role: bool = True,
        strip_thinking: bool = True,
        merge: bool = True,
    ):
        super().__init__(model_short or adapter_path, temperature, has_system_role=has_system_role)
        self.adapter_path = adapter_path
        self.model_short = model_short
        self.model_path = model_path
        self.max_tokens = max_tokens
        self.quantization = quantization
        self.strip_thinking = strip_thinking
        self.merge = merge
        self._load()

    def _resolve_base_path(self) -> str:
        if self.model_path:
            return self.model_path
        # Try to recover the short name from the saved run_config.json.
        import json
        from pathlib import Path

        cfg_path = Path(self.adapter_path) / "run_config.json"
        short = self.model_short
        if short is None and cfg_path.is_file():
            with cfg_path.open("r", encoding="utf-8") as f:
                short = json.load(f).get("model_short")
        if short is None:
            raise ValueError(
                "Could not determine the base model. Pass model_short or model_path."
            )
        from .lora import resolve_model_path

        base_path, _ = resolve_model_path(short)
        return base_path

    def _load(self):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        base_path = self._resolve_base_path()
        print(f"[LoraAssistant] base={base_path} adapter={self.adapter_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.adapter_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict = dict(device_map="auto", low_cpu_mem_usage=True)
        if self.quantization == "4bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif self.quantization == "8bit":
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        elif self.quantization == "bf16":
            load_kwargs["torch_dtype"] = torch.bfloat16
        else:
            load_kwargs["torch_dtype"] = torch.float16

        base = AutoModelForCausalLM.from_pretrained(base_path, **load_kwargs)
        model = PeftModel.from_pretrained(base, self.adapter_path)

        if self.merge:
            model = model.merge_and_unload()

        self.model = model
        self.model.config.use_cache = True
        self.model.eval()

    @staticmethod
    def _strip_think_block(text: str) -> str:
        return re.sub(r".*?", "", text, flags=re.DOTALL).strip()

    def generate(self, messages: List[dict], max_new_tokens: int | None = None) -> str:
        import torch

        max_new_tokens = max_new_tokens or self.max_tokens
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated = outputs[0][inputs.input_ids.shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        if self.strip_thinking:
            response = self._strip_think_block(response)
        return response
