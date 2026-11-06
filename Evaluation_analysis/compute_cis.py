"""95% CIs for manual (test, n=4) vs assisted (incoming, n=34) micro-F1.

Parses stored per-document counts from each run's evaluation_results.txt
(no re-evaluation needed), then:
  - micro-F1 95% CI: document-level bootstrap (10k resamples, seed 42),
    recompute micro-F1 from summed TP/Gold/Sys, 2.5/97.5 percentiles.
  - macro-F1 95% CI: t-interval over per-doc F1s.

Usage from repo root:
    python Evaluation_analysis/compute_cis.py
"""

import json
import re
from pathlib import Path

import numpy as np
from scipy import stats

FOLDERS = [
    "output/test_gemma4_DEC_fs6_greedy_sentence_long",
    "output/test_gpt-5.2_DEC_fs6_greedy_sentence_long",
    "output/test_muse_DEC_fs6_greedy_sentence_long",
    "output/test_qwen38-27b_DEC_fs6_greedy_sentence_long",
    "output/test_saul-54b_DEC_fs6_greedy_sentence_long",
    "output/incoming_gemma4_DEC_fs6_greedy_sentence_long",
    "output/incoming_gpt-5.2_DEC_fs6_greedy_sentence_long",
    "output/incoming_muse_DEC_fs6_greedy_sentence_long",
    "output/incoming_qwen38-27b_DEC_fs6_greedy_sentence_long",
    "output/incoming_saul-54b_DEC_fs6_greedy_sentence_long",
]

PAT = re.compile(
    r"^(\S+)\s+TP=\s*(\d+)\s*\|\s*Gold=\s*(\d+)\s*\|\s*System=\s*(\d+)\s*\|\s*F1=\s*([\d.]+)%"
)
B = 10000
SEED = 42


def parse_per_doc(folder: str):
    """Extract per-document TP/Gold/Sys/F1 from evaluation_results.txt."""
    txt = Path(folder, "evaluation_results.txt").read_text(encoding="utf-8")
    docs = []
    in_section = False
    for line in txt.splitlines():
        if "RESULTS" in line and "PER DOCUMENT" in line:
            in_section = True
            continue
        if "RESULTS" in line and "TOTAL" in line and docs:
            break
        if in_section:
            m = PAT.search(line.strip())
            if m:
                name, tp, gold, sys, f1 = m.groups()
                docs.append(
                    {
                        "doc": name,
                        "tp": int(tp),
                        "gold": int(gold),
                        "sys": int(sys),
                        "f1": float(f1) / 100,
                    }
                )
    return docs


def main():
    rng = np.random.default_rng(SEED)
    summary = {}
    print(f"{'run':<55} {'n':>3} {'micro':>6} {'boot95':>16} {'macro':>6} {'t95':>16}")
    for folder in FOLDERS:
        docs = parse_per_doc(folder)
        f1s = np.array([d["f1"] for d in docs])
        tps = np.array([d["tp"] for d in docs])
        golds = np.array([d["gold"] for d in docs])
        syss = np.array([d["sys"] for d in docs])
        n = len(docs)
        macro = float(f1s.mean())
        p = tps.sum() / syss.sum() if syss.sum() else 0.0
        r = tps.sum() / golds.sum() if golds.sum() else 0.0
        micro = 2 * p * r / (p + r) if (p + r) else 0.0
        sem = stats.sem(f1s) if n > 1 else 0.0
        t_low, t_high = stats.t.interval(0.95, df=n - 1, loc=macro, scale=sem) if n > 1 else (macro, macro)
        idx = rng.integers(0, n, size=(B, n))
        boot_macro = f1s[idx].mean(axis=1)
        b_tp, b_g, b_s = tps[idx].sum(axis=1), golds[idx].sum(axis=1), syss[idx].sum(axis=1)
        b_p = np.divide(b_tp, b_s, out=np.zeros(B, dtype=float), where=b_s > 0)
        b_r = np.divide(b_tp, b_g, out=np.zeros(B, dtype=float), where=b_g > 0)
        boot_micro = np.divide(
            2 * b_p * b_r, b_p + b_r, out=np.zeros(B, dtype=float), where=(b_p + b_r) > 0
        )
        m_low, m_high = np.percentile(boot_macro, [2.5, 97.5])
        u_low, u_high = np.percentile(boot_micro, [2.5, 97.5])
        summary[folder] = {
            "n": n,
            "micro": micro,
            "micro_boot95": [float(u_low), float(u_high)],
            "macro": macro,
            "macro_t95": [float(t_low), float(t_high)],
            "macro_boot95": [float(m_low), float(m_high)],
            "per_doc": docs,
        }
        print(
            f"{folder:<55} {n:>3} {micro*100:>5.1f}% [{u_low*100:>4.1f},{u_high*100:>4.1f}]"
            f" {macro*100:>5.1f}% [{t_low*100:>5.1f},{t_high*100:>5.1f}]"
        )
    out = Path("Evaluation_analysis/ci_results.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
