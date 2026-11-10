"""Sanity-check a split: raw HTML tag counts vs cached examples vs hierarchy survival.

    python -m src.check_data dev train silver

Columns per label:
  raw      occurrences of labelname="..." in data/annotated/<split>/*.html
  cached   spans parsed from the cached AIO examples (what the encoder trains/evals on)
  assemble spans that survive the hierarchy rules (non-overlapping mentions,
           components inside a mention, allowed component types)
"""
import json
import re
import sys
from collections import Counter

from configs.config import CACHE_DIR, DATA_DIR
from src.models.encoder import COMPONENT_LABELS, MENTION_LABELS, assemble, parse_tagged

ORDER = MENTION_LABELS + COMPONENT_LABELS

for split in sys.argv[1:] or ["dev", "train"]:
    raw = Counter()
    folder = DATA_DIR / "annotated" / split
    for f in folder.glob("*.htm*"):
        raw.update(re.findall(r'labelname=["\']([^"\']+)["\']', f.read_text(encoding="utf-8")))

    cache_path = CACHE_DIR / "training" / f"aio_{split}_paragraph_mt500.json"
    cache = json.load(open(cache_path, encoding="utf-8"))
    cached, survive = Counter(), Counter()
    for ex in cache:
        _, spans = parse_tagged(ex["output"])
        cached.update(n for *_, n in spans)
        kept = assemble([s for s in spans if s[2] in MENTION_LABELS],
                        [s for s in spans if s[2] in COMPONENT_LABELS])
        for _, _, n, comps in kept:
            survive[n] += 1
            survive.update(c[2] for c in comps)

    print(f"\n[{split}] {len(cache)} cached chunks | label: raw / cached / assemble")
    for lab in ORDER:
        print(f"  {lab:18s} {raw[lab]:6d} {cached[lab]:6d} {survive[lab]:6d}")
    print(f"  {'TOTAL':18s} {sum(raw[l] for l in ORDER):6d} "
          f"{sum(cached[l] for l in ORDER):6d} {sum(survive[l] for l in ORDER):6d}")
    extra = {k: v for k, v in raw.items() if k not in ORDER}
    if extra:
        print(f"  other labelnames in raw HTML (not used): {extra}")
