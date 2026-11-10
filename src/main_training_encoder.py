"""Train an encoder (token classification) on the extraction task.

    python -m src.main_training_encoder --model deberta-v3-large --train-on silver --dry-run
    python -m src.main_training_encoder --model deberta-v3-large --train-on silver
    python -m src.main_training_encoder --model deberta-v3-large --train-on train          # gold-only baseline
    python -m src.main_training_encoder --model ModernBERT-large --train-on silver --max-len 1024

--model is a key of MODEL_REGISTRY (src/models/registry.py), i.e. the folder
name under MODELS_DIR. Adapters/weights go to <MODELS_DIR>/output_training.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from configs.config import MIN_TOKENS
from src.training import dataset as ds
from src.models.encoder import (build_records, default_output_root,
                                tokenization_ceiling, train_encoder)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="deberta-v3-large")
    p.add_argument("--model-path", default=None)
    p.add_argument("--train-on", default="silver", choices=["train", "silver", "train+silver"])
    p.add_argument("--output-dir", default=None)
    p.add_argument("--chunker", default="paragraph", choices=["paragraph", "sentence"])
    p.add_argument("--min-tokens", type=int, default=MIN_TOKENS)
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--stride", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-dev", action="store_true", help="skip dev evaluation / model selection")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()

    train_ex = ds.get_or_build_dataset(a.train_on, a.chunker, a.min_tokens, pass_id=None)

    dev_ex = []
    if not a.no_dev:
        ds.TRAIN_SET_CONDITIONS.setdefault("dev", ["dev"])     # data/annotated/dev
        dev_ex = ds.get_or_build_dataset("dev", a.chunker, a.min_tokens, pass_id=None)

    if a.dry_run:
        from transformers import AutoTokenizer
        from src.training.lora import resolve_model_path
        path = a.model_path or resolve_model_path(a.model)[0]
        tok = AutoTokenizer.from_pretrained(path, use_fast=True)
        recs, mism = build_records(train_ex, tok, a.max_len, a.stride)
        print(f"train chunks={len(train_ex)} windows={len(recs)} text-mismatch chunks={mism}")
        print("tokenization ceiling (exact F1) train:", tokenization_ceiling(train_ex, tok, a.max_len, a.stride))
        if dev_ex:
            print("tokenization ceiling (exact F1) dev  :", tokenization_ceiling(dev_ex, tok, a.max_len, a.stride))
        return

    train_encoder(a.model, a.train_on, train_ex, dev_ex or None,
                  output_root=a.output_dir or default_output_root(), model_path=a.model_path,
                  max_len=a.max_len, stride=a.stride, lr=a.lr, head_lr=a.head_lr,
                  epochs=a.epochs, batch_size=a.batch_size, grad_accum=a.grad_accum, seed=a.seed)


if __name__ == "__main__":
    main()