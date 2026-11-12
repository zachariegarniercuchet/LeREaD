"""SFT training loop: chat-template formatting + loss masking + HF Trainer.

Each example is rendered exactly as the AIO pipeline builds its messages
(``src.models.base.get_messages``), i.e.

    system : the AIO prompt (src/prompts/allInOne_long.txt)
    user   : "Annotate this text:" + plain-text chunk
    assistant : the tagged chunk            <- only these tokens get loss

The prompt tokens (system + user) are masked with -100 so the model is only
trained to *produce* the annotated output, not to regurgitate the prompt.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import math
import os
import re

from torch.utils.data import Dataset


class SFTDataset(Dataset):
    """Tokenizes (input, output) pairs with the chat template and masks the prompt.

    Args:
        examples: list of {"input": str, "output": str, ...} dicts.
        tokenizer: the base model's tokenizer (must have a chat template).
        system_prompt: the AIO prompt, used as the system message.
        max_seq_len: sequences longer than this are dropped (with a warning)
            rather than truncated, so no output is ever half-cut.
        user_prefix: matches get_messages()' default, "Annotate this text:".
    """

    def __init__(
        self,
        examples: list[dict],
        tokenizer,
        system_prompt: str,
        max_seq_len: int = 2048,
        user_prefix: str = "Annotate this text:",
    ):
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.user_prefix = user_prefix
        self.max_seq_len = max_seq_len
        self._records: list[dict[str, Any]] = []

        kept = dropped = 0
        for ex in examples:
            rec = self._encode(ex["input"], ex["output"])
            if rec is None:
                dropped += 1
                continue
            self._records.append(rec)
            kept += 1

        if dropped:
            print(
                f"[SFTDataset] kept {kept} examples, dropped {dropped} "
                f"longer than {max_seq_len} tokens"
            )
        print(f"[SFTDataset] {kept} training examples")

    # -- internals ---------------------------------------------------------

    def _encode(self, chunk_in: str, chunk_out: str) -> dict[str, Any] | None:
        tok = self.tokenizer

        # Prompt-only rendering (what the model sees at inference).
        prompt_messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"{self.user_prefix}{chunk_in}"},
        ]
        prompt_text = tok.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True
        )

        # Full rendering = prompt + the assistant output we want to learn.
        full_messages = prompt_messages + [{"role": "assistant", "content": chunk_out}]
        full_text = tok.apply_chat_template(
            full_messages, tokenize=False, add_generation_prompt=False
        )

        prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tok(full_text, add_special_tokens=False)["input_ids"]

        # Some templates don't make prompt_text a strict prefix of full_text
        # (trailing whitespace / eos). Fall back to a defensive prefix trim.
        if full_ids[: len(prompt_ids)] != prompt_ids:
            prompt_len = self._common_prefix_len(prompt_ids, full_ids)
        else:
            prompt_len = len(prompt_ids)

        if len(full_ids) > self.max_seq_len:
            return None

        # Guard: never mask the whole sequence (would yield NaN loss).
        if prompt_len >= len(full_ids):
            prompt_len = max(0, len(full_ids) - 1)

        labels = list(full_ids)
        labels[:prompt_len] = [-100] * prompt_len

        return {
            "input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels,
        }

    @staticmethod
    def _common_prefix_len(a: list[int], b: list[int]) -> int:
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    # -- public Dataset API ------------------------------------------------

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._records[idx]


def _collate(records: list[dict[str, Any]], pad_token_id: int):
    """Pad a batch to its longest sequence; -100 pads labels (ignored by loss)."""
    import torch

    max_len = max(len(r["input_ids"]) for r in records)
    input_ids, attention_mask, labels = [], [], []
    for r in records:
        pad = max_len - len(r["input_ids"])
        input_ids.append(r["input_ids"] + [pad_token_id] * pad)
        attention_mask.append(r["attention_mask"] + [0] * pad)
        labels.append(r["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def train_lora(
    model_short: str,
    train_on: str,
    examples: list[dict],
    system_prompt: str,
    lora_cfg,
    sft_cfg,
    model_path: str | None = None,
    eval_dataset: Dataset | list[dict] | None = None,
    output_dir: Path | None = None,
    resume: str = "auto",
):
    """Fine-tune *model_short* on *examples* with LoRA; return (trainer, out_dir).

    Saves the adapter + both configs under
    ``<output_root>/<run_name>/`` so every run is reproducible and can be
    reloaded for inference via ``LoraAssistant``.

    Checkpoint resumption (controlled by *resume*):
        * "auto" -> if ``output_dir`` contains ``checkpoint-*`` folders, resume
          from the latest one; otherwise start fresh.
        * "true" -> force resume (raises if no checkpoint found).
        * "false" -> always start from scratch.
    """
    from transformers import Trainer, TrainingArguments

    from .lora import attach_lora, config_to_dict, load_base_model

    tokenizer, base_model = load_base_model(
        model_short,
        model_path=model_path,
        quantization=sft_cfg.quantization,
        gradient_checkpointing=sft_cfg.gradient_checkpointing,
    )

    train_dataset = SFTDataset(
        examples,
        tokenizer,
        system_prompt=system_prompt,
        max_seq_len=sft_cfg.max_seq_len,
    )
    if isinstance(eval_dataset, list):
        eval_dataset = SFTDataset(
            eval_dataset,
            tokenizer,
            system_prompt=system_prompt,
            max_seq_len=sft_cfg.max_seq_len,
        )

    model = attach_lora(base_model, lora_cfg)

    run_name = sft_cfg.run_name(model_short, train_on, lora_cfg)
    from .lora import default_training_root
    output_dir = Path(output_dir or sft_cfg.output_root or default_training_root()) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Automatic checkpoint detection / resumption ──────────────────────
    checkpoint_dirs = [
        d for d in output_dir.iterdir()
        if d.is_dir() and d.name.startswith("checkpoint-")
    ]

    def _step(p: Path) -> int:
        m = re.search(r"checkpoint-(\d+)", p.name)
        return int(m.group(1)) if m else -1

    resume_from_checkpoint = False
    if checkpoint_dirs:
        latest = max(checkpoint_dirs, key=_step)
        if resume == "auto":
            resume_from_checkpoint = True
            print(
                f"[train_lora] Found existing checkpoints in {output_dir.name} — "
                f"resuming from {latest.name}"
            )
        elif resume == "true":
            resume_from_checkpoint = True
            print(f"[train_lora] Forced resume from {latest.name}")
        else:  # "false"
            print(
                f"[train_lora] Checkpoints found in {output_dir.name} but "
                f"resume=false — starting from scratch"
            )
    elif resume == "true":
        raise FileNotFoundError(
            f"resume=true requested but no checkpoint-* folders found in {output_dir}"
        )

    pad_id = tokenizer.pad_token_id
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # Estimate optimizer steps per epoch.
    batches_per_epoch = math.ceil(
        len(train_dataset)
        / (sft_cfg.per_device_train_batch_size * world_size)
    )

    steps_per_epoch = max(
        1,
        batches_per_epoch // sft_cfg.gradient_accumulation_steps
    )

    total_steps = math.ceil(
        steps_per_epoch * sft_cfg.num_train_epochs
    )

    warmup_steps = round(sft_cfg.warmup_ratio * total_steps)

    print(
        f"[training] examples={len(train_dataset)}, "
        f"estimated_steps={total_steps}, "
        f"warmup_steps={warmup_steps}"
    )
    training_kwargs = dict(
        output_dir=str(output_dir),
        run_name=run_name,
        # schedule
        learning_rate=sft_cfg.learning_rate,
        num_train_epochs=sft_cfg.num_train_epochs,
        per_device_train_batch_size=sft_cfg.per_device_train_batch_size,
        gradient_accumulation_steps=sft_cfg.gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        weight_decay=sft_cfg.weight_decay,
        lr_scheduler_type=sft_cfg.lr_scheduler_type,
        max_grad_norm=sft_cfg.max_grad_norm,
        # precision
        fp16=sft_cfg.fp16,
        bf16=sft_cfg.bf16,
        gradient_checkpointing=sft_cfg.gradient_checkpointing,
        # bookkeeping
        logging_steps=sft_cfg.logging_steps,
        save_strategy=sft_cfg.save_strategy,
        report_to=[],
        seed=sft_cfg.seed,
        remove_unused_columns=False,
    )
    if eval_dataset is not None:
        import inspect
        supported = inspect.signature(TrainingArguments.__init__).parameters
        strategy_key = "eval_strategy" if "eval_strategy" in supported else "evaluation_strategy"
        training_kwargs.update({
            strategy_key: "epoch",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_loss",
            "greater_is_better": False,
        })
    args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda recs: _collate(recs, pad_id),
    )

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Save the adapter (not the full base model) + the configs used.
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    import json

    with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_short": model_short,
                "train_on": train_on,
                "lora": config_to_dict(lora_cfg),
                "sft": config_to_dict(sft_cfg),
                "n_examples": len(train_dataset),
                "best_dev_loss": trainer.state.best_metric,
                "best_checkpoint": trainer.state.best_model_checkpoint,
            },
            f,
            indent=2,
        )
    print(f"[train_lora] adapter + configs saved -> {output_dir}")

    return trainer, output_dir
