"""
Plot coverage curves (surface or structural) for already-precomputed
greedy vs random few-shot selections.

Does NOT run any selection — only loads the two cached selection files
and replays coverage accumulation in the order examples already appear
in each file. Raises if either cache is missing.

Usage:
    python plot_fewshot_comparison.py --metric surface
    python plot_fewshot_comparison.py --metric structural
    python plot_fewshot_comparison.py --metric both   # single combined figure, dual y-axes
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from configs.config import FEWSHOT_CACHE_DIR, IMG_DIR, GREEDY_CONFIG
from src.fewshot.patterns.builder import (
    load_surface_pattern_dict,
    load_structural_pattern_dict,
    surface_pattern_dict_exists,
    structural_pattern_dict_exists,
)
# Adjust this import to wherever greedy_select_examples actually lives
from src.fewshot.selector.greedy import _KEY_MAP, _build_score_lookup


RANDOM_PATH = FEWSHOT_CACHE_DIR / "examples_random.json"

METRIC_TO_ANN_KEY = {
    "surface": "surface_pattern",
    "structural": "structural_pattern",
}

METRIC_TO_FIG_NAME = {
    "surface": "Component-Level",
    "structural": "Mention-Level",
}

GREEDY_PATH_BY_METRIC = {
    "surface": FEWSHOT_CACHE_DIR / f"examples_greedy_surf-{1.0}_struct-{0.0}.json",
    "structural": FEWSHOT_CACHE_DIR / f"examples_greedy_surf-{0.0}_struct-{1.0}.json",
}

# Distinct color per metric, distinct line style/marker per selection strategy,
# so the combined plot stays legible even in grayscale printouts.
METRIC_COLOR = {
    "surface": "#1f77b4",     # blue
    "structural": "#d62728",  # red
}
STRATEGY_STYLE = {
    "greedy": {"linestyle": "-", "marker": "o", "markersize": 3},
    "random": {"linestyle": "--", "marker": "x", "markersize": 4},
}


def _load_selection(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Expected cached selection at {path}, but it does not exist.\n"
            f"Run precompute_fewshot_examples.py first."
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["examples"]


def _load_pattern_dict(metric: str) -> dict:
    if metric == "surface":
        if not surface_pattern_dict_exists():
            raise RuntimeError("Surface pattern dict not found. Run precompute_pattern_dict.py first.")
        return load_surface_pattern_dict()
    else:
        if not structural_pattern_dict_exists():
            raise RuntimeError("Structural pattern dict not found. Run precompute_pattern_dict.py first.")
        return load_structural_pattern_dict()


def _make_hashable(p):
    return tuple(p) if isinstance(p, list) else p


def _cumulative_coverage_curve(examples: list[dict], ann_key: str, score_lookup: dict) -> list[float]:
    """Cumulative (ever-growing) coverage score, replaying examples in file order."""
    seen = set()
    total = 0.0
    curve = []
    for ex in examples:
        label_patterns = ex.get(ann_key, {})
        for label_key, patterns in label_patterns.items():
            dict_key = _KEY_MAP.get(label_key)
            if dict_key is None:
                continue
            lookup = score_lookup[dict_key]
            for p in patterns:
                p = _make_hashable(p)
                key = (dict_key, p)
                if key not in seen:
                    seen.add(key)
                    total += lookup.get(p, 0.0)
        curve.append(total)
    return curve


def _compute_curves(metric: str) -> dict:
    """Load caches + pattern dict for one metric and return everything needed to plot it."""
    ann_key = METRIC_TO_ANN_KEY[metric]
    greedy_path = GREEDY_PATH_BY_METRIC[metric]

    greedy_examples = _load_selection(greedy_path)
    random_examples = _load_selection(RANDOM_PATH)

    if len(greedy_examples) != len(random_examples):
        raise ValueError(
            f"Selections have different sizes: greedy={len(greedy_examples)}, "
            f"random={len(random_examples)}. Expected same size."
        )

    pattern_dict = _load_pattern_dict(metric)
    score_lookup = _build_score_lookup(pattern_dict)
    max_score = len(pattern_dict.keys()) * 100

    greedy_curve = _cumulative_coverage_curve(greedy_examples, ann_key, score_lookup)
    random_curve = _cumulative_coverage_curve(random_examples, ann_key, score_lookup)

    n = len(greedy_curve)
    return {
        "steps": list(range(1, n + 1)),
        "greedy_pct": [s / max_score * 100 for s in greedy_curve],
        "random_pct": [s / max_score * 100 for s in random_curve],
    }


def _plot_single(metric: str) -> None:
    if metric not in METRIC_TO_ANN_KEY:
        raise ValueError(f"metric must be one of {list(METRIC_TO_ANN_KEY)}")

    data = _compute_curves(metric)

    plt.figure(figsize=(10, 5))
    plt.plot(data["steps"], data["greedy_pct"], label="greedy", marker="o", markersize=3)
    plt.plot(data["steps"], data["random_pct"], label="random", marker="o", markersize=3)
    plt.xlabel("Number of examples selected")
    plt.ylabel(f"{METRIC_TO_FIG_NAME[metric].capitalize()} pattern coverage (%)")
    plt.title(f"{METRIC_TO_FIG_NAME[metric].capitalize()} coverage: greedy vs random")
    plt.legend()
    plt.grid(alpha=0.3)

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = IMG_DIR / f"coverage_comparison_{METRIC_TO_FIG_NAME[metric]}.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"✅ Plot saved → {plot_path}")


def _plot_combined() -> None:
    """Single figure, dual y-axes: surface on the left axis, structural on the right."""
    surface = _compute_curves("surface")
    structural = _compute_curves("structural")

    if surface["steps"] != structural["steps"]:
        raise ValueError(
            "Surface and structural selections have different lengths "
            f"({len(surface['steps'])} vs {len(structural['steps'])}); "
            "cannot share a single x-axis in the combined plot."
        )
    steps = surface["steps"]

    fig, ax_left = plt.subplots(figsize=(10, 5))
    

    lines = []

    for metric in (("surface", "structural")):
        curves = surface if metric == "surface" else structural
        color = METRIC_COLOR[metric]
        for strategy in ("greedy", "random"):
            style = STRATEGY_STYLE[strategy]
            line, = ax_left.plot(
                steps,
                curves[f"{strategy}_pct"],
                color=color,
                label=f"{METRIC_TO_FIG_NAME[metric]} – {strategy}",
                **style,
            )
            lines.append(line)
        ax_left.tick_params(axis="y")

    ax_left.set_xlabel("Number of examples selected")
    ax_left.set_ylabel(
        f"Coverage (%)",
        #color=METRIC_COLOR["surface"],
    )
    #ax_right.set_ylabel(
    #    f"{METRIC_TO_FIG_NAME['structural'].capitalize()} coverage (%)",
    #    color=METRIC_COLOR["structural"],
    #)

    ax_left.set_title("Coverage: greedy vs random")
    ax_left.grid(alpha=0.3)
    ax_left.legend(lines, [l.get_label() for l in lines], loc="lower right")

    IMG_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = IMG_DIR / "coverage_comparison_combined.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"✅ Combined plot saved → {plot_path}")


def main(metric: str) -> None:
    if metric == "both":
        _plot_combined()
    else:
        _plot_single(metric)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", choices=["surface", "structural", "both"], default="surface")
    args = parser.parse_args()
    main(args.metric)