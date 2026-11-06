#!/usr/bin/env python
"""
Explore the Reflex (CanLII citator) exports in data/reflex/.

Answers, for the currently loaded documents:
  1. Which link classes / href patterns does Reflex use, and how do they map
     to our mention labels (legislation / decision / secondary sources)?
  2. Which gold document (any split under data/annotated/) each reflex file
     corresponds to (exact id, then fuzzy id).
  3. How close is the reflex plain text to the gold plain text (this decides
     whether we can feed reflex files straight into evaluation_l1_util or
     must align spans into the gold text first).

Run from repo root:
    python scripts/reflex/explore.py
"""

from __future__ import annotations

import re
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from bs4 import BeautifulSoup

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

REFLEX_DIR = PROJECT_ROOT / "data" / "reflex"
ANNOTATED_DIR = PROJECT_ROOT / "data" / "annotated"

# href path segment -> mention label (parent level only)
HREF_RULES = [
    ("/laws/", "legislation"),          # statutes, regs, historical statutes
    ("/jnl/", "secondary sources"),     # law journal articles
    ("/book", "secondary sources"),
    ("/comm/", "secondary sources"),    # CanLII comments / connects
    ("/doc/", "decision"),              # court decisions
]
IGNORED_CLASSES = {"reflex-parag", "reflex-toc"}  # internal navigation


def classify(href: str) -> str | None:
    """Map a reflex href to a parent mention label, or None if not a mention."""
    for needle, label in HREF_RULES:
        if needle in href:
            return label
    return None


def norm_id(doc_id: str) -> str:
    """Aggressively normalise a document id for fuzzy matching."""
    s = doc_id.lower()
    s = re.sub(r"[^a-z0-9]", "", s)          # drop punctuation
    # strip trailing court abbreviations so 1993CanLII1889PESCTD ~ ...PESCTSD
    s = re.sub(r"(canlii\d+).*$", r"\1", s)
    return s


def parse_reflex(path: Path) -> dict:
    """Parse one reflex html into a structured record."""
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    body = soup.find("body")
    links = []
    class_counts = Counter()
    if body:
        for a in body.find_all("a"):
            classes = a.get("class") or []
            cls = " ".join(classes) if classes else "(noclass)"
            class_counts[cls] += 1
            href = a.get("href", "")
            text = " ".join(a.get_text().split())
            if cls in IGNORED_CLASSES or href.startswith("#"):
                continue
            label = classify(href)
            links.append(
                {
                    "text": text,
                    "class": cls,
                    "href": href,
                    "label": label,
                    "s": a.get("s"),
                    "ss": a.get("ss"),
                }
            )
    plain = " ".join(body.get_text().split()) if body else ""
    return {"path": path, "links": links, "class_counts": class_counts, "plain": plain}


def gold_index() -> dict[str, list[str]]:
    """doc id -> list of 'split' names it is annotated in."""
    idx: dict[str, list[str]] = defaultdict(list)
    for p in sorted(ANNOTATED_DIR.glob("*/*.html")):
        idx[p.stem].append(p.parent.name)
    return idx


def gold_plain(path: Path) -> str:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    body = soup.find("body")
    return " ".join(body.get_text().split()) if body else ""


def main() -> None:
    reflex_files = sorted(REFLEX_DIR.glob("*_reflex.html"))
    if not reflex_files:
        print(f"No reflex files in {REFLEX_DIR}")
        return

    print("=" * 78)
    print(f"REFLEX INVENTORY — {len(reflex_files)} file(s) in {REFLEX_DIR.relative_to(PROJECT_ROOT)}")
    print("=" * 78)

    idx = gold_index()
    by_norm: dict[str, list[str]] = defaultdict(list)
    for stem in idx:
        by_norm[norm_id(stem)].append(stem)

    all_class_counts = Counter()
    unmatched, matched = [], []

    for path in reflex_files:
        rec = parse_reflex(path)
        all_class_counts.update(rec["class_counts"])
        doc_id = path.name[: -len("_reflex.html")]

        gold_split = idx.get(doc_id)
        how = "exact"
        if gold_split is None:
            cands = by_norm.get(norm_id(doc_id), [])
            gold_split = idx.get(cands[0]) if cands else None
            if gold_split:
                doc_id = cands[0]
                how = "fuzzy"
        if gold_split:
            matched.append((path.name, doc_id, gold_split, how, rec))
        else:
            unmatched.append(path.name)

        # label counts
        lc = Counter(l["label"] for l in rec["links"] if l["label"])
        lc_str = ", ".join(f"{k}={v}" for k, v in sorted(lc.items()))
        print(f"\n{path.name}")
        print(f"   links: {len(rec['links']):4d}   [{lc_str or 'no mention links'}]")
        print(f"   gold : {doc_id} in {gold_split if gold_split else '*** NO GOLD ***'} ({how})")

    print("\n" + "=" * 78)
    print("LINK CLASSES SEEN")
    print("=" * 78)
    for cls, n in all_class_counts.most_common():
        print(f"  {n:5d}  {cls}")

    print("\n" + "=" * 78)
    print(f"PAIRING WITH GOLD — {len(matched)} matched, {len(unmatched)} without gold")
    print("=" * 78)
    for name, doc_id, splits, how, _ in matched:
        print(f"  OK ({how:5s}) {name}  ->  {doc_id} [{', '.join(splits)}]")
    for name in unmatched:
        print(f"  --        {name}  ->  no gold document yet")

    # ------------------------------------------------------------------
    # Plain-text comparison for matched documents
    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("PLAIN TEXT: reflex vs gold  (decides the alignment strategy)")
    print("=" * 78)
    for name, doc_id, splits, how, rec in matched:
        # prefer the human splits for comparison
        split = next((s for s in ("test", "dev", "train") if s in splits), splits[0])
        gpath = ANNOTATED_DIR / split / f"{doc_id}.html"
        gtext = gold_plain(gpath)
        rtext = rec["plain"]
        ratio = SequenceMatcher(None, gtext, rtext).ratio() if gtext and rtext else 0.0
        print(f"\n  {doc_id}  [{split}]")
        print(f"     gold chars={len(gtext):7d}   reflex chars={len(rtext):7d}   ratio={ratio:.4f}")

        # how many reflex mention texts appear verbatim in gold text
        gnorm = re.sub(r"\s+", " ", gtext)
        hits = misses = 0
        miss_examples = []
        for l in rec["links"]:
            if not l["label"]:
                continue
            t = re.sub(r"\s+", " ", l["text"]).strip()
            if t and t in gnorm:
                hits += 1
            else:
                misses += 1
                if len(miss_examples) < 5:
                    miss_examples.append(t)
        print(f"     mention texts found verbatim in gold text: {hits}  missing: {misses}")
        for m in miss_examples:
            print(f"        missing: {m!r}")

    # show a raw diff sample for the first matched doc
    if matched:
        name, doc_id, splits, how, rec = matched[0]
        split = next((s for s in ("test", "dev", "train") if s in splits), splits[0])
        gtext = gold_plain(ANNOTATED_DIR / split / f"{doc_id}.html")
        sm = SequenceMatcher(None, gtext, rec["plain"])
        print("\n" + "=" * 78)
        print(f"FIRST DIFFERENCES ({doc_id}, gold -> reflex)")
        print("=" * 78)
        shown = 0
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                continue
            print(f"  [{tag}] gold={gtext[i1:i2][:90]!r}  reflex={rec['plain'][j1:j2][:90]!r}")
            shown += 1
            if shown >= 12:
                break


if __name__ == "__main__":
    main()
