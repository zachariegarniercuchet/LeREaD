"""Train a cross-encoder (DeBERTa) for the coreference resolution step.

    python -m src.main_training_coref_encoder --model deberta-v3-large --train-on silver --dry-run
    python -m src.main_training_coref_encoder --model deberta-v3-large --train-on silver --seed 42
    python -m src.main_training_coref_encoder --model deberta-v3-large --train-on train+silver --epochs 8 --seed 7

--model is a key of MODEL_REGISTRY (src/models/registry.py), i.e. the folder
name under MODELS_DIR. Weights go to <MODELS_DIR>/output_training/<run_name>/.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from src.training import coref_dataset as cd
from src.models.coref_encoder import train_coref_encoder, default_output_root
from src.training.lora import resolve_model_path


def main():
    p = argparse.ArgumentParser(
        description="Train a DeBERTa cross-encoder for coreference resolution.")
    p.add_argument("--model", default="deberta-v3-large",
                   help="Short name from configs.constants.MODEL_MAPPING_NAME "
                        "(e.g. deberta-v3-large).")
    p.add_argument("--model-path", default=None,
                   help="Explicit base-model path (overrides the registry).")
    p.add_argument("--train-on", default="silver",
                   choices=["train", "silver", "train+silver"],
                   help="Training data condition.")
    p.add_argument("--output-dir", default=None,
                   help="Output root (default: <MODELS_DIR>/output_training).")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-hard-negatives", type=int, default=0,
                   help="Max hard-negative profiles sampled per mention (0 = all).")
    p.add_argument("--dev-split", default="dev",
                   help="Annotated split used for sequential checkpoint selection.")
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--no-dev", action="store_true",
                   help="skip dev evaluation / threshold selection")
    p.add_argument("--dry-run", action="store_true",
                   help="Build the dataset and print stats, then exit (no GPU).")
    p.add_argument("--rebuild-dataset", action="store_true",
                   help="Ignore the dataset cache and rebuild.")
    a = p.parse_args()

    # ── Build / load the cross-encoder training dataset ───────────────── #
    train_ex = cd.get_or_build_dataset(
        a.train_on,
        max_hard_negatives=a.max_hard_negatives,
        force=a.rebuild_dataset,
        seed=a.seed,
    )
    if not train_ex:
        print("✗ No training examples built — check the annotated data.")
        return None

    dev_ex = []
    if not a.no_dev:
        # Dev set is built from the annotated dev split for threshold tuning.
        dev_ex = cd.build_coref_examples_for_split(
            a.dev_split, max_hard_negatives=a.max_hard_negatives, seed=a.seed)
        print(f"[coref-training] dev pairs={len(dev_ex)}")

    if a.dry_run:
        from transformers import AutoTokenizer
        path = a.model_path or resolve_model_path(a.model)[0]
        tok = AutoTokenizer.from_pretrained(path, use_fast=True)
        from src.models.coref_encoder import build_records
        recs, n_empty = build_records(train_ex, tok, a.max_len)
        print(f"[dry-run] pairs={len(train_ex)} records={len(recs)} empty={n_empty}")
        n_pos = sum(1 for r in recs if r["label"] >= 0.5)
        n_neg = len(recs) - n_pos
        print(f"[dry-run] positives={n_pos} negatives={n_neg}")
        print(f"[dry-run] dev pairs={len(dev_ex)}")
        return None

    out_dir = train_coref_encoder(
        a.model, a.train_on, train_ex, dev_ex or None,
        output_root=a.output_dir or default_output_root(), model_path=a.model_path,
        max_len=a.max_len, lr=a.lr, head_lr=a.head_lr,
        epochs=a.epochs, batch_size=a.batch_size, grad_accum=a.grad_accum,
        warmup_ratio=a.warmup_ratio, weight_decay=a.weight_decay,
        seed=a.seed, max_hard_negatives=a.max_hard_negatives,
        dev_split=None if a.no_dev else a.dev_split,
    )
    print(f"[coref-training] done -> {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()
