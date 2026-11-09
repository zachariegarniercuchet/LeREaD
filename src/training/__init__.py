"""Supervised (LoRA) fine-tuning baselines for the extraction task.

This package turns the annotated HTML (gold `train`, auto `silver`, or their
union) into copy-and-annotate (input, output) pairs — the exact format the
AIO pipeline already uses at inference — and fine-tunes an open-weight model
on them with LoRA / QLoRA.

Layout:
    dataset.py   build + cache (input, output) SFT pairs from annotated HTML
    lora.py      LoRA / QLoRA hyperparameter config and model loading
    trainer.py   chat-template + loss-masking SFT training loop
    infer.py     LoraAssistant — drop a fine-tuned model into main_extraction

Entry point (run from repo root):
    python -m src.main_training --model gemma4 --train-on train --dry-run

The dataset helpers (``dataset``) are import-safe without torch/transformers.
The training and inference helpers (``lora``, ``trainer``, ``infer``) import
torch lazily on first use.
"""
from .dataset import (
    AIO_LABELS,
    DEC_ALL_LABELS,
    DEC_PASS_CONFIGS,
    TRAIN_SET_CONDITIONS,
    build_dec_training_examples,
    build_extraction_examples,
    get_or_build_dataset,
    load_system_prompt,
)

_LAZY = {
    "LoRAConfig": "src.training.lora",
    "SFTConfig": "src.training.lora",
    "resolve_model_path": "src.training.lora",
    "load_base_model": "src.training.lora",
    "SFTDataset": "src.training.trainer",
    "train_lora": "src.training.trainer",
    "LoraAssistant": "src.training.infer",
}

__all__ = [
    "AIO_LABELS",
    "DEC_ALL_LABELS",
    "DEC_PASS_CONFIGS",
    "TRAIN_SET_CONDITIONS",
    "build_dec_training_examples",
    "build_extraction_examples",
    "get_or_build_dataset",
    "load_system_prompt",
    *list(_LAZY),
]


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(_LAZY[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
