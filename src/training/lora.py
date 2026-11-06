"""LoRA / QLoRA configuration and base-model loading for extraction fine-tuning.

Model paths come from the same gitignored ``src/models/registry.py`` the
inference factory uses (``AssistantFactory``), so nothing here invents paths or
quantization. If the registry is missing, pass an explicit ``--model-path``.

Two dataclasses hold every knob:

  * ``LoRAConfig`` — the adapter itself (rank, alpha, dropout, target modules).
  * ``SFTConfig``  — the optimization schedule (lr, epochs, batch, seq len…).

``DEFAULT_GRIDS`` gives a small, deliberate search space per model so the
train/dev sweep stays reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from configs.constants import MODEL_MAPPING_NAME


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfig:
    """The adapter. Sensible defaults follow the QLoRA recipe (Dettmers 2023)."""
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"                      # "none" | "all" | "lora_only"
    # "all-linear" covers q/k/v/o + MLP across every architecture we train
    # (Gemma, Qwen, SaulLM, Muse) without hard-coding per-model layer names.
    target_modules: Any = "all-linear"
    task_type: str = "CAUSAL_LM"

    def to_peft(self):
        from peft import LoraConfig as PeftLoraConfig, TaskType

        return PeftLoraConfig(
            r=self.r,
            lora_alpha=self.alpha,
            lora_dropout=self.dropout,
            bias=self.bias,
            target_modules=self.target_modules,
            task_type=TaskType.CAUSAL_LM,
        )


@dataclass
class SFTConfig:
    """Optimization + data settings for one training run."""
    # schedule
    learning_rate: float = 2e-4
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    # data
    max_seq_len: int = 2048                # covers a ~500-token chunk + tags
    # precision / memory
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    quantization: str = "4bit"             # "4bit" (QLoRA) | "8bit" | "fp16"/"bf16"
    # bookkeeping
    logging_steps: int = 10
    save_strategy: str = "epoch"
    eval_steps: float | None = None
    seed: int = 42
    output_root: str = "output_training"

    def run_name(self, model_short: str, train_on: str, lora: LoRAConfig) -> str:
        return (
            f"{model_short}_{train_on}"
            f"_r{lora.r}_a{lora.alpha}_lr{self.learning_rate:g}"
            f"_ep{self.num_train_epochs:g}_q{self.quantization}"
        )


# Small, deliberate search space. Kept intentionally tight: 4 models x 3 data
# conditions is already 12 headline runs, so the sweep only probes rank, alpha
# (via rank scaling) and learning rate.
DEFAULT_GRIDS: dict[str, list[Any]] = {
    "r": [8, 16, 32],
    "alpha": [32],                         # often kept at 2*r
    "learning_rate": [1e-4, 2e-4],
}


def iter_grid(grids: dict[str, list[Any]] | None = None) -> list[dict[str, Any]]:
    """Expand a {field: values} dict into the list of override dicts."""
    import itertools

    grids = grids or DEFAULT_GRIDS
    keys = list(grids.keys())
    combos = itertools.product(*(grids[k] for k in keys))
    return [dict(zip(keys, combo)) for combo in combos]


# ---------------------------------------------------------------------------
# Model resolution + loading
# ---------------------------------------------------------------------------

def _registry() -> dict:
    try:
        from src.models.registry import MODEL_REGISTRY
    except ImportError as e:  # registry.py is gitignored / local-only
        raise ImportError(
            "src/models/registry.py not found (it is gitignored). Provide it, "
            "or pass an explicit model_path to load_base_model()."
        ) from e
    return MODEL_REGISTRY


def resolve_model_path(model_short: str) -> tuple[str, dict]:
    """Map a CLI short name -> (hf_model_path, registry_entry).

    Accepts either a short name from MODEL_MAPPING_NAME (e.g. "gemma4") or a
    MODEL_MAPPING_NAME value directly (e.g. "gemma-4-31B-it").
    """
    registry = _registry()
    hf_name = MODEL_MAPPING_NAME.get(model_short, model_short)
    if hf_name not in registry:
        raise KeyError(
            f"'{hf_name}' not in MODEL_REGISTRY. Known: {list(registry.keys())}"
        )
    return registry[hf_name]["path"], registry[hf_name]


def load_base_model(
    model_short: str,
    model_path: str | None = None,
    quantization: str = "4bit",
    trust_remote_code: bool | None = None,
    gradient_checkpointing: bool = True,
):
    """Load (tokenizer, peft-wrapped model) ready for LoRA training.

    Quantization mapping:
      * "4bit" -> QLoRA (nf4 + double quant, fp16 compute)
      * "8bit" -> 8-bit load
      * "fp16"/"bf16" -> no quantization, cast on load
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    if model_path is None:
        model_path, entry = resolve_model_path(model_short)
        if trust_remote_code is None:
            trust_remote_code = entry.get("trust_remote_code", False)
    trust_remote_code = bool(trust_remote_code)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict[str, Any] = dict(
        trust_remote_code=trust_remote_code,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    if quantization == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    elif quantization == "8bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization == "bf16":
        load_kwargs["torch_dtype"] = torch.bfloat16
    else:  # "fp16"
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    return tokenizer, model


def attach_lora(model, lora: LoRAConfig):
    """Wrap *model* with a fresh LoRA adapter and return it trainable."""
    from peft import get_peft_model

    peft_cfg = lora.to_peft()
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model


def config_to_dict(*configs) -> dict:
    """Flatten one or more dataclasses/dicts into a plain dict (for saving)."""
    out: dict[str, Any] = {}
    for c in configs:
        out.update(c if isinstance(c, dict) else asdict(c))
    return out
