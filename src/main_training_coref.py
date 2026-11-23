"""Fine-tune an open-weight LLM (LoRA/QLoRA) for coreference resolution.

Trains the model to do exactly what main_coref asks of it: given a mention, its
upper context and the Reference Profile Registry, return the mention with a
``docid``. Data comes from src/training/coref_llm_dataset.py (gold docids,
teacher-forced registry); the system prompt is src/prompts/coref_<type>.txt.
Training, checkpointing, NaN guard and best-dev-loss selection are shared with
main_training (src/training/trainer.py).

Inspect the dataset first (no GPU):
    python -m src.main_training_coref --model Qwen3.8-27B --train-on silver --dry-run

Rehearsal (~5 min, 1 GPU):
    python -m src.main_training_coref --model Qwen3.8-27B --train-on train --smoke-test

The three conditions:
    python -m src.main_training_coref --model Qwen3.8-27B --train-on train
    python -m src.main_training_coref --model Qwen3.8-27B --train-on silver
    python -m src.main_training_coref --model Qwen3.8-27B --train-on train+silver

Inference with the result (zero-shot, gold mentions):
    python -m src.main_coref --split test --input-source gold --model qwen38-27b \\
        --adapter-path <MODELS_DIR>/output_training/Qwen3.8-27B_silver_coref_... \\
        --temperature 0
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import PROMPT_DIR
from src.main_training import (
    _apply_smoke_defaults,
    _clamp_smoke_cadence,
    add_common_training_args,
    make_configs,
)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LoRA fine-tuning for coreference resolution.")
    add_common_training_args(p)
    p.add_argument("--max-fragments", type=int, default=10,
                   help="Fragments per profile shown in the prompt (= main_coref default).")
    p.add_argument("--max-profiles", type=int, default=50,
                   help="Training-only cap on registry profiles shown per example (gold "
                        "profile always kept, others sampled). One train doc reaches 278 "
                        "profiles (~24k tokens); without a cap most of its examples exceed "
                        "--max-seq-len and are dropped. Test docs have <= 71 authorities. "
                        "Pass -1 for the full registry.")
    # Coref examples are long (registry grows through the document) and the
    # extraction runs peaked on dev after ~1 epoch.
    p.set_defaults(model="Qwen3.8-27B", max_seq_len=8192, epochs=2.0, lr=1e-4)
    return p


def _token_estimate(text: str) -> int:
    return len(text) // 3  # rough chars->tokens for English legal text + markup


def _dry_run_summary(examples: list[dict], system_prompt: str, max_seq_len: int) -> None:
    import statistics

    sys_tok = _token_estimate(system_prompt)
    lens = sorted(sys_tok + _token_estimate(e["input"]) + _token_estimate(e["output"]) for e in examples)
    n = len(lens)
    new = sum(e["is_new"] for e in examples)
    print("\n[dry-run] coref dataset summary")
    print(f"  examples           : {n}  (new authority: {new} = {new / n:.0%}, "
          f"reuse existing docid: {n - new})")
    print(f"  profiles shown     : median {statistics.median(e['n_profiles'] for e in examples):.0f}, "
          f"max {max(e['n_profiles'] for e in examples)}")
    print(f"  est. tokens / ex   : median {lens[n // 2]}, p90 {lens[int(n * .9)]}, max {lens[-1]}")
    for cap in (4096, 8192, 16384):
        print(f"  est. kept @ {cap:>5}: {sum(l <= cap for l in lens) / n:.0%}")
    print(f"  (--max-seq-len {max_seq_len}; exact counts are printed by SFTDataset at training time)")
    ex = examples[min(5, n - 1)]
    print("\n[dry-run] example input (truncated):\n" + ex["input"][:1500])
    print("\n[dry-run] example output:\n" + ex["output"])


def run(args):
    from src.training.coref_llm_dataset import get_or_build_dataset
    from src.coref_input import COREF_USER_PREFIX

    if args.max_profiles is not None and args.max_profiles < 0:
        args.max_profiles = None

    if args.smoke_test:
        _apply_smoke_defaults(args)

    examples = get_or_build_dataset(
        args.train_on, max_fragments=args.max_fragments,
        max_profiles=args.max_profiles, seed=args.seed, force=args.rebuild_dataset,
    )
    if not examples:
        print("✗ No coref training examples built — check the annotated data.")
        return None

    with open(PROMPT_DIR / f"coref_{args.prompt_type}.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read()

    if args.smoke_test:
        n = len(examples) if args.dry_run else min(len(examples), args.smoke_train_examples)
        _clamp_smoke_cadence(args, n)
        if not args.dry_run:
            examples = examples[: args.smoke_train_examples]

    if args.dry_run:
        _dry_run_summary(examples, system_prompt, args.max_seq_len)
        return None

    dev_examples = get_or_build_dataset(
        "dev", max_fragments=args.max_fragments,
        max_profiles=args.max_profiles, seed=args.seed, force=args.rebuild_dataset,
    )
    eval_cap = args.eval_max_examples
    if args.smoke_test and eval_cap is None:
        eval_cap = args.smoke_eval_examples
    if eval_cap and dev_examples:
        dev_examples = dev_examples[:eval_cap]
        print(f"[eval] dev set capped at {len(dev_examples)} examples")

    lora, sft = make_configs(args)

    from src.training.trainer import train_lora

    _, out_dir = train_lora(
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
        prompt_type=args.prompt_type,
        task="coref",
        user_prefix=COREF_USER_PREFIX,
    )
    return out_dir


def main():
    run(build_arg_parser().parse_args())


if __name__ == "__main__":
    main()
