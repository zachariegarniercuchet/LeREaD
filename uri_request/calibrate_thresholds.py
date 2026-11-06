"""Calibrate per-doctype SPLADE thresholds from LLM-judge labels.

Input: ``judge/judge_sample.json`` (72 stratified profiles with top-3 hits).
Labels: LLM-as-judge verdicts on top1 (CORRECT / WRONG / UNCERTAIN), recorded
in ``JUDGE`` below after reading every sampled query against its top-3
candidate metadata.

Rule for CORRECT: top1 names the SAME work (same decision with matching
neutral/reported citation, or same enactment with matching jurisdiction +
citation). Title-only matches across jurisdictions (e.g. Alberta vs
Newfoundland Interpretation Act) count as WRONG for URI purposes — a URI
points at one exact document.

Also evaluates a citation/title VERIFIER (does the query's distinctive
citation or exact main title appear in the candidate's metadata?) because the
presence audit showed score alone cannot separate right from wrong: almost
every WRONG top1 is a similar-but-different work whose true URI is ABSENT
from the pool (verified by grep over candidates.json), some with scores > 30.

Usage: ``python uri_request/calibrate_thresholds.py`` (from repo root).
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

MISSION_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = MISSION_DIR / "judge" / "judge_sample.json"

# #idx -> verdict. See module docstring for the rule.
JUDGE: dict[int, str] = {
    0: "U", 1: "W", 2: "W", 3: "W", 4: "W", 5: "W", 6: "W", 7: "W", 8: "W",
    9: "W", 10: "W", 11: "C", 12: "W", 13: "W", 14: "W", 15: "W", 16: "W",
    17: "W", 18: "W", 19: "W", 20: "W", 21: "W", 22: "C", 23: "W", 24: "W",
    25: "W", 26: "C", 27: "W", 28: "W", 29: "W", 30: "W", 31: "W", 32: "W",
    33: "W", 34: "C", 35: "W", 36: "W", 37: "W", 38: "W", 39: "W", 40: "W",
    41: "U", 42: "W", 43: "W", 44: "W", 45: "W", 46: "W", 47: "U", 48: "W",
    49: "W", 50: "W", 51: "C", 52: "W", 53: "C", 54: "W", 55: "C", 56: "W",
    57: "W", 58: "W", 59: "W", 60: "W", 61: "W", 62: "W", 63: "W", 64: "W",
    65: "W", 66: "W", 67: "W", 68: "U", 69: "W", 70: "W", 71: "C",
}

NEUTRAL_RX = None  # retired: verify() now delegates to apply_decision.Verifier
CANLII_RX = None
STATUTE_RX = None


def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", "", s)


def query_main_title(query_text: str) -> str:
    parts = [p.strip() for p in (query_text or "").split("|")]
    return parts[1] if len(parts) > 1 else ""


def extract_cites(text: str) -> set[str]:
    from apply_decision import extract_cites as _ec
    return _ec(text)


_VERIFIER = None


def verify(hit: dict, query_text: str) -> tuple[bool, str]:
    """Delegates to the single source of truth (apply_decision.Verifier)."""
    global _VERIFIER
    if _VERIFIER is None:
        from apply_decision import Verifier
        from pathlib import Path as _P
        cands = json.loads((_P(__file__).resolve().parent.parent
                            / "cache" / "candidate_pool_index_splade_v1"
                            / "candidates.json").read_text(encoding="utf-8"))
        _VERIFIER = Verifier(cands)
    return _VERIFIER.check(hit, query_text)


def main() -> None:
    sample = json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))
    assert len(sample) == len(JUDGE), (len(sample), len(JUDGE))

    rows = []
    for i, s in enumerate(sample):
        ok1, why1 = verify(s["top3"][0], s["query_text"]) if s["top3"] else (False, "-")
        rescued = None
        for h in s["top3"][1:]:
            ok, _ = verify(h, s["query_text"])
            if ok:
                rescued = h["rank"]
                break
        rows.append({
            "idx": i, "dtype": s["doc_type"], "band": s["band"],
            "score": s["top1_score"],
            "margin": s["margin_top1_top2"],
            "label": JUDGE[i], "verify": ok1, "why": why1, "rescued_rank": rescued,
            "key": s["profile_key"],
        })

    print(f"=== judge labels: {sum(1 for r in rows if r['label']=='C')} C / "
          f"{sum(1 for r in rows if r['label']=='W')} W / "
          f"{sum(1 for r in rows if r['label']=='U')} U (n={len(rows)}) ===\n")

    print("--- verifier vs judge (top1) ---")
    for want in ("C", "W"):
        sub = [r for r in rows if r["label"] == want]
        print(f"  label={want}: verify-pass {sum(1 for r in sub if r['verify'])}/{len(sub)}")
    print("  UNCERTAIN verify-pass:",
          [(r["idx"], r["why"]) for r in rows if r["label"] == "U" and r["verify"]])
    print("  WRONG-but-verified:", [(r["idx"], r["why"]) for r in rows
                                    if r["label"] == "W" and r["verify"]])
    print("  CORRECT-but-unverified:", [r["idx"] for r in rows
                                        if r["label"] == "C" and not r["verify"]])
    print("  WRONG top1 rescuable at rank2/3:",
          [(r["idx"], r["rescued_rank"]) for r in rows
           if r["label"] == "W" and r["rescued_rank"]])

    print("\n--- per-type score-only rule: accept iff score >= T ---")
    print("    (precision = C/(C+W) on decided; recall = C kept / C total)")
    for dtype in ("decision", "legislation", "secondary sources"):
        sub = [r for r in rows if r["dtype"] == dtype and r["label"] in ("C", "W")]
        nc = sum(1 for r in sub if r["label"] == "C")
        print(f"  [{dtype}] n={len(sub)} correct={nc}")
        for t in (5, 8, 10, 12, 15, 18, 20, 25, 30, 35):
            acc = [r for r in sub if r["score"] >= t]
            c = sum(1 for r in acc if r["label"] == "C")
            w = len(acc) - c
            prec = f"{c / len(acc):.2f}" if acc else "-"
            rec = f"{c / nc:.2f}" if nc else "-"
            print(f"    T={t:>5}: accept {len(acc):>3}  prec={prec:>4}  recall={rec:>4}")

    print("\n--- score + verifier rule: accept iff score >= T AND verify(top1) ---")
    for dtype in ("decision", "legislation", "secondary sources"):
        sub = [r for r in rows if r["dtype"] == dtype and r["label"] in ("C", "W")]
        nc = sum(1 for r in sub if r["label"] == "C")
        print(f"  [{dtype}] correct={nc}")
        for t in (5, 8, 10, 12, 15):
            acc = [r for r in sub if r["score"] >= t and r["verify"]]
            c = sum(1 for r in acc if r["label"] == "C")
            w = len(acc) - c
            prec = f"{c / len(acc):.2f}" if acc else "-"
            rec = f"{c / nc:.2f}" if nc else "-"
            print(f"    T={t:>5}: accept {len(acc):>3} (wrong={w})  prec={prec:>4}  recall={rec:>4}")

    print("\n--- all CORRECT verdicts (score, margin, verify) ---")
    for r in rows:
        if r["label"] == "C":
            print(f"  #{r['idx']} [{r['dtype']}] score={r['score']:.2f} "
                  f"margin={r['margin']:.2f} verify={r['verify']}({r['why']}) {r['key']}")
    print("\n--- UNCERTAIN ---")
    for r in rows:
        if r["label"] == "U":
            print(f"  #{r['idx']} [{r['dtype']}] score={r['score']:.2f} "
                  f"verify={r['verify']}({r['why']}) {r['key']}")


if __name__ == "__main__":
    main()
