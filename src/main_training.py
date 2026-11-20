"""Fine-tune an open-weight model on the extraction task with LoRA.

Runs from repo root. Builds (or reuses) the copy-and-annotate dataset from the
annotated HTML, then trains with LoRA/QLoRA and saves the adapter under
``output_training/<run_name>/``.

Single run:
    python -m src.main_training --model gemma4 --train-on train
    python -m src.main_training --model qwen38-27b --train-on silver --lora-r 32
    python -m src.main_training --model saul-54b --train-on train+silver --lr 1e-4

Sweep (expands src.training.lora.DEFAULT_GRIDS):
    python -m src.main_training --model gemma4 --train-on train --grid

Dataset-only / smoke test (no model load, no GPU needed):
    python -m src.main_training --model gemma4 --train-on train+silver --dry-run

Fast end-to-end rehearsal (~5 min, 1 GPU) that exercises the exact code path
that used to die at the end of epoch 1 — first checkpoint, dev eval, adapter
save, run_config.json — then verifies the artefacts:
    python -m src.main_training --model qwen38-27b --train-on train+silver \\
        --dec-pass all --chunker paragraph --smoke-test

Check whether a run finished / can be resumed (no GPU, no torch import):
    python -m src.training.check_run --root output_training

python -m src.main_training --model gemma4 --train-on silver --dec-pass all --dry-run --chunker paragraph
python -m src.main_training --model gemma4 --train-on silver --dec-pass all --chunker paragraph
python -m src.main_training --model gemma4 --train-on train+silver --dec-pass all --chunker paragraph

The best dev-loss checkpoint is restored and saved as the run directory. Runs
include the seed in their name and default to the registry's scratch folder.

To force a fresh start (ignore existing checkpoints)
python -m src.main_training --model gemma4 --train-on silver --dec-pass all --resume false
"""
import argparse
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reduce allocator fragmentation before CUDA is first initialised (the logs
# showed the trainer one allocation away from OOM). setdefault so the sbatch
# script can still override it.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from configs.config import MIN_TOKENS
from src.training.dataset import (
    DEC_PASS_CONFIGS,
    get_or_build_dataset,
    load_system_prompt,
)
from src.training.lora import (
    DEFAULT_GRIDS,
    LoRAConfig,
    SFTConfig,
    iter_grid,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LoRA fine-tuning for extraction.")

    # what / where
    p.add_argument("--model", default="gemma4",
                   help="Short name from configs.constants.MODEL_MAPPING_NAME.")
    p.add_argument("--train-on", default="train",
                   choices=["train", "silver", "train+silver"],
                   help="Training data condition.")
    p.add_argument("--model-path", default=None,
                   help="Explicit base-model path (overrides the registry).")
    p.add_argument("--output-dir", default=None,
                   help="Output root (default: SFTConfig.output_root).")

    # data
    p.add_argument("--chunker", default="paragraph", choices=["paragraph", "sentence"])
    p.add_argument("--dec-pass", default=None,
                   help=f"Decomposed pass to train: {list(DEC_PASS_CONFIGS)} or 'all' "
                        "(union of all 4 passes with task tokens). Default=None trains AIO.")
    p.add_argument("--min-tokens", type=int, default=MIN_TOKENS,
                   help="Chunk-merge threshold; smaller => more, shorter examples.")
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--rebuild-dataset", action="store_true",
                   help="Ignore the dataset cache and rebuild.")
    p.add_argument("--prompt-type", default="long", choices=["short", "long"])

    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    # optimization
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)

    # memory
    p.add_argument("--quantization", default="4bit",
                   choices=["4bit", "8bit", "fp16", "bf16"])
    p.add_argument("--no-gradient-checkpointing", action="store_true")
    p.add_argument("--no-bf16", action="store_true",
                   help="Disable bf16 (useful on older GPUs).")

    # bookkeeping / checkpointing
    p.add_argument("--save-strategy", default="steps",
                   choices=["steps", "epoch", "no"],
                   help="Default 'steps': an epoch-only save loses the whole run "
                        "if the end-of-epoch eval crashes.")
    p.add_argument("--save-steps", type=int, default=200,
                   help="Checkpoint every N optimizer steps (save-strategy=steps).")
    p.add_argument("--save-total-limit", type=int, default=2,
                   help="Keep only the N newest checkpoints (-1 = keep all).")
    p.add_argument("--eval-strategy", default=None, choices=["steps", "epoch", "no"],
                   help="Dev-loss cadence (default: mirror --save-strategy).")
    p.add_argument("--eval-steps", type=int, default=None,
                   help="Evaluate every N steps (default: mirror --save-steps).")
    p.add_argument("--eval-batch-size", type=int, default=1,
                   help="per_device_eval_batch_size (default 1; the Trainer's "
                        "own default of 8 is what OOMs a 27B QLoRA at eval).")
    p.add_argument("--eval-max-examples", type=int, default=None,
                   help="Evaluate on at most N dev examples (memory/time cap).")
    p.add_argument("--logging-steps", type=int, default=10)

    # modes
    p.add_argument("--grid", action="store_true",
                   help="Run every combo in DEFAULT_GRIDS instead of one run.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build the dataset and print stats, then exit (no GPU).")
    p.add_argument("--smoke-test", action="store_true",
                   help="Fast rehearsal: few examples, 1 epoch, checkpoint + eval "
                        "every couple of steps, then verify the saved artefacts.")
    p.add_argument("--smoke-train-examples", type=int, default=32,
                   help="Training examples kept in --smoke-test.")
    p.add_argument("--smoke-eval-examples", type=int, default=8,
                   help="Dev examples kept in --smoke-test.")
    p.add_argument("--smoke-steps", type=int, default=2,
                   help="Wanted checkpoint/eval interval in --smoke-test "
                        "(clamped to one epoch's worth of steps).")
    p.add_argument("--resume", default="auto", choices=["auto", "true", "false"],
                   help="Checkpoint resumption: 'auto' (default) resumes from the "
                        "latest checkpoint-* in the output folder if one exists, "
                        "'true' forces resume (errors if none), 'false' starts fresh.")
    return p


def make_configs(args, overrides: dict | None = None):
    """Build (LoRAConfig, SFTConfig) from argparse args + optional grid overrides."""
    lora = LoRAConfig(
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
    )
    sft = SFTConfig(
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        max_seq_len=args.max_seq_len,
        bf16=not args.no_bf16,
        fp16=args.no_bf16,           # if bf16 disabled, fall back to fp16
        gradient_checkpointing=not args.no_gradient_checkpointing,
        quantization=args.quantization,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=None if args.save_total_limit < 0 else args.save_total_limit,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        per_device_eval_batch_size=args.eval_batch_size,
        name_suffix="_smoke" if args.smoke_test else "",
        seed=args.seed,
    )
    if overrides:
        for key, value in overrides.items():
            if hasattr(lora, key):
                setattr(lora, key, value)
            elif hasattr(sft, key):
                setattr(sft, key, value)
            else:
                raise KeyError(f"Unknown override {key!r}")
    return lora, sft


def _apply_smoke_defaults(args):
    """Shrink a real run into a few-minute rehearsal of the same code path.

    The point is to reach *and survive* the end-of-epoch checkpoint + dev eval,
    which is where the 8 h runs died. Only run length and data size change;
    quantization, batch size, grad accumulation and the collator stay real.
    """
    args.smoke_test = True
    args.epochs = 1.0
    args.save_strategy = "steps"
    args.eval_strategy = "steps"
    args.save_total_limit = 1
    args.logging_steps = 1


def _clamp_smoke_cadence(args, n_train: int):
    """Make sure at least two checkpoints fit inside one (tiny) epoch."""
    batches = math.ceil(n_train / max(1, args.batch_size))
    steps_per_epoch = max(1, batches // max(1, args.grad_accum))
    args.save_steps = max(1, min(args.smoke_steps, steps_per_epoch))
    args.eval_steps = args.save_steps
    print(
        f"[smoke] {n_train} train examples -> ~{steps_per_epoch} step(s)/epoch, "
        f"checkpoint + eval every {args.save_steps} step(s)"
    )


def run_one(args, overrides: dict | None = None):
    if args.smoke_test:
        _apply_smoke_defaults(args)

    examples = get_or_build_dataset(
        train_on=args.train_on,
        chunker=args.chunker,
        min_tokens=args.min_tokens,
        force=args.rebuild_dataset,
        pass_id=args.dec_pass,
    )
    if not examples:
        print("✗ No training examples built — check the annotated data.")
        return None

    if args.smoke_test:
        _clamp_smoke_cadence(
            args, len(examples) if args.dry_run else min(len(examples), args.smoke_train_examples)
        )
        if not args.dry_run:
            examples = examples[: args.smoke_train_examples]

    lora, sft = make_configs(args, overrides)

    if args.dry_run:
        total_in = sum(len(e["input"]) for e in examples)
        total_out = sum(len(e["output"]) for e in examples)
        print("\n[dry-run] dataset summary")
        print(f"  examples     : {len(examples)}")
        print(f"  avg input ch : {total_in / len(examples):.0f}")
        print(f"  avg output ch: {total_out / len(examples):.0f}")
        if args.dec_pass == "all":
            # per-pass breakdown
            from collections import Counter
            pass_counts = Counter()
            for e in examples:
                for tok in ("[DEC0]", "[DEC1]", "[DEC2]", "[DEC3]"):
                    if e["input"].startswith(tok):
                        pass_counts[tok] += 1
                        break
            for p in ("[DEC0]", "[DEC1]", "[DEC2]", "[DEC3]"):
                print(f"  {p:<8s}: {pass_counts.get(p, 0)} examples")
        print(f"  lora         : {lora}")
        print(f"  sft          : {sft}")
        return None

    system_prompt = load_system_prompt(args.prompt_type)
    dev_examples = get_or_build_dataset(
        train_on="dev",
        chunker=args.chunker,
        min_tokens=args.min_tokens,
        force=True,
        pass_id=args.dec_pass,
    )
    eval_cap = args.eval_max_examples
    if args.smoke_test:
        eval_cap = eval_cap if eval_cap is not None else args.smoke_eval_examples
    if eval_cap and dev_examples:
        dev_examples = dev_examples[:eval_cap]
        print(f"[eval] dev set capped at {len(dev_examples)} examples")

    from src.training.trainer import train_lora

    trainer, out_dir = train_lora(
        model_short=args.model,
        train_on=args.train_on,
        examples=examples,
        system_prompt=system_prompt,
        lora_cfg=lora,
        sft_cfg=sft,
        eval_dataset=dev_examples,
        model_path=args.model_path,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        resume=args.resume,
        dec_pass=args.dec_pass,
        prompt_type=args.prompt_type,
    )
    return out_dir


def main():
    args = build_arg_parser().parse_args()

    if args.grid:
        combos = iter_grid(DEFAULT_GRIDS)
        print(f"[grid] {len(combos)} runs: {DEFAULT_GRIDS}")
        results = []
        for i, overrides in enumerate(combos, 1):
            print(f"\n{'=' * 80}\n[grid] run {i}/{len(combos)}: {overrides}\n{'=' * 80}")
            out_dir = run_one(args, overrides)
            results.append({"overrides": overrides, "output_dir": str(out_dir)})
        _save_grid_results(args, results)
        return

    run_one(args)


def _save_grid_results(args, results: list[dict]):
    import json

    root = Path(args.output_dir) if args.output_dir else Path("output_training")
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"grid_{args.model}_{args.train_on}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[grid] results manifest -> {path}")


if __name__ == "__main__":
    main()
