"""Answer one question: did this training run finish, and can it be resumed?

A run is only *finished* once ``run_config.json`` exists, because ``train_lora``
writes it after ``trainer.train()`` returns and ``trainer.save_model()``
succeeds. Anything else means the process died somewhere after the last
``checkpoint-<step>/`` it managed to write.

Usage (from the repo root, on the cluster, no GPU / no torch import):

    python -m src.training.check_run output_training/Qwen3.8-27B_train_dec_all_...
    python -m src.training.check_run --root output_training      # every run
    python -m src.training.check_run --root output_training --watch   # repeat

Exit code is 0 when every inspected run is complete, 1 otherwise, so a Slurm
script can test it directly.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ADAPTER_FILES = ("adapter_model.safetensors", "adapter_model.bin")

COMPLETE = "COMPLETE"
RESUMABLE = "INCOMPLETE (resumable from checkpoint)"
NO_CHECKPOINT = "INCOMPLETE (nothing to resume)"
ADAPTER_ONLY = "ADAPTER SAVED (no run_config.json)"


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _step_of(path: Path) -> int:
    match = re.search(r"checkpoint-(\d+)", path.name)
    return int(match.group(1)) if match else -1


def _checkpoints(run_dir: Path) -> list[Path]:
    if not run_dir.is_dir():
        return []
    return sorted(
        (d for d in run_dir.glob("checkpoint-*") if d.is_dir()),
        key=_step_of,
    )


def _seconds_per_step(ckpts: list[Path]) -> float | None:
    """Estimate s/step from the mtimes of the two newest checkpoints."""
    if len(ckpts) < 2:
        return None
    try:
        newest, previous = ckpts[-1], ckpts[-2]
        dt = newest.stat().st_mtime - previous.stat().st_mtime
        dstep = _step_of(newest) - _step_of(previous)
    except OSError:
        return None
    if dt <= 0 or dstep <= 0:
        return None
    return dt / dstep


def summarize(run_dir: str | Path) -> dict:
    """Collect everything needed to judge a run without loading the model."""
    run_dir = Path(run_dir)
    state = _read_json(run_dir / "trainer_state.json")
    config = _read_json(run_dir / "run_config.json")
    ckpts = _checkpoints(run_dir)
    adapter = next((f for f in ADAPTER_FILES if (run_dir / f).is_file()), None)

    last_ckpt = ckpts[-1] if ckpts else None
    last_state = _read_json(last_ckpt / "trainer_state.json") if last_ckpt else None
    global_step = (last_state or state or {}).get("global_step")
    max_steps = (last_state or state or {}).get("max_steps")
    epoch = (last_state or state or {}).get("epoch")

    if config is not None and adapter is not None:
        status = COMPLETE
    elif adapter is not None:
        status = ADAPTER_ONLY
    elif ckpts:
        status = RESUMABLE
    else:
        status = NO_CHECKPOINT

    per_step = _seconds_per_step(ckpts)
    remaining = (
        max_steps - global_step
        if isinstance(max_steps, int) and isinstance(global_step, int)
        else None
    )
    eta = (
        remaining * per_step
        if remaining is not None and per_step is not None and remaining > 0
        else None
    )

    return {
        "run_dir": run_dir,
        "status": status,
        "finished": status is COMPLETE,
        "global_step": global_step,
        "max_steps": max_steps,
        "epoch": epoch,
        "checkpoints": [_step_of(c) for c in ckpts],
        "adapter": adapter,
        "run_config": config,
        "seconds_per_step": per_step,
        "eta_seconds": eta,
    }


def _human(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    if seconds < 1:
        return "<1s"
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def format_report(info: dict) -> str:
    run_dir = info["run_dir"]
    lines = [f"run      : {run_dir.name}", f"status   : {info['status']}"]
    if info["global_step"] is not None or info["max_steps"] is not None:
        lines.append(
            f"progress : step {info['global_step']}/{info['max_steps']}"
            + (f"  epoch {info['epoch']}" if info["epoch"] is not None else "")
        )
    ckpts = info["checkpoints"]
    lines.append(
        "ckpts    : " + (", ".join(f"checkpoint-{s}" for s in ckpts) or "none")
    )
    if info["adapter"]:
        lines.append(f"adapter  : {info['adapter']}")
    if info["run_config"]:
        cfg = info["run_config"]
        loss = cfg.get("best_dev_loss")
        lines.append(
            f"best eval: {loss if loss is not None else 'n/a'}"
            + f"  ({cfg.get('n_examples')} examples)"
        )
    if info["eta_seconds"] is not None:
        lines.append(
            f"remaining: ~{_human(info['eta_seconds'])} "
            f"at {info['seconds_per_step']:.1f}s/step"
        )
    if info["status"] is RESUMABLE:
        lines.append(
            "resume   : re-run the same command (default --resume auto) or "
            "pass --resume true"
        )
    return "\n".join(lines)


def report_run_status(run_dir: str | Path) -> dict:
    """Print (and return) the one-screen status report for a run folder."""
    info = summarize(run_dir)
    print("[check_run] " + format_report(info).replace("\n", "\n[check_run] "))
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", nargs="*", help="One or more run folders.")
    parser.add_argument(
        "--root",
        default=None,
        help="Inspect every sub-folder of this training root.",
    )
    parser.add_argument(
        "--watch",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Repeat the report every SECONDS until every run is complete.",
    )
    args = parser.parse_args()

    def collect() -> list[Path]:
        targets = [Path(p) for p in args.run_dir]
        if args.root:
            targets.extend(
                sorted(d for d in Path(args.root).iterdir() if d.is_dir())
            )
        return targets or [Path("output_training")]

    while True:
        infos = [summarize(t) for t in collect() if t.is_dir()]
        for info in infos:
            print(format_report(info))
            print()
        if not infos:
            print("No run folders found.", file=sys.stderr)
            return 1
        if args.watch is None or all(i["finished"] for i in infos):
            return 0 if all(i["finished"] for i in infos) else 1
        time.sleep(args.watch)


if __name__ == "__main__":
    raise SystemExit(main())
