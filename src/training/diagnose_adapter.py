"""Check whether a LoRA training run diverged (NaN) — CPU only, no base model.

    python -m src.training.diagnose_adapter <run_dir> [<run_dir> ...]

For each run (and each checkpoint-* inside it) reports:
  * non-finite tensors in adapter_model.safetensors
  * from trainer_state.json: first logged step with loss 0 / nan or grad_norm
    nan/inf, and the eval_loss history (what load_best_model_at_end compared)
  * best_dev_loss / best_checkpoint from run_config.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path


def _bad(v) -> bool:
    return isinstance(v, (int, float)) and not math.isfinite(v)


def check_weights(folder: Path) -> str:
    f = folder / "adapter_model.safetensors"
    if not f.is_file():
        return "no adapter_model.safetensors"
    import torch
    from safetensors.torch import load_file

    tensors = load_file(str(f))
    bad = [k for k, t in tensors.items() if not torch.isfinite(t.float()).all()]
    return f"{len(bad)}/{len(tensors)} tensors non-finite" + (f" (e.g. {bad[0]})" if bad else "")


def check_state(folder: Path) -> list[str]:
    f = folder / "trainer_state.json"
    if not f.is_file():
        return []
    state = json.loads(f.read_text(encoding="utf-8"))
    out = []
    first = next(
        (h for h in state.get("log_history", [])
         if "loss" in h and (h["loss"] == 0 or _bad(h["loss"]) or _bad(h.get("grad_norm")))),
        None,
    )
    out.append(
        f"first diverged log: step {first.get('step')} epoch {first.get('epoch'):.3f} "
        f"loss={first.get('loss')} grad_norm={first.get('grad_norm')}"
        if first else "no diverged training log"
    )
    evals = [(h["step"], h["eval_loss"]) for h in state.get("log_history", []) if "eval_loss" in h]
    out.append(f"eval_loss: {evals}")
    out.append(f"best: {state.get('best_metric')} @ {state.get('best_model_checkpoint')}")
    return out


def main(paths: list[str]) -> None:
    for p in map(Path, paths):
        print(f"\n=== {p}")
        cfg = p / "run_config.json"
        if cfg.is_file():
            c = json.loads(cfg.read_text(encoding="utf-8"))
            print(f"run_config: best_dev_loss={c.get('best_dev_loss')} "
                  f"best_checkpoint={c.get('best_checkpoint')}")
        print(f"final adapter: {check_weights(p)}")
        for ck in sorted(p.glob("checkpoint-*"), key=lambda d: int(d.name.split("-")[-1])):
            print(f"  {ck.name}: {check_weights(ck)}")
            for line in check_state(ck):
                print(f"    {line}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    main(sys.argv[1:])
