"""
Core helpers to evaluate Reflex (CanLII citator) extractions against LeREaD gold.

What Reflex gives us
--------------------
`data/reflex/<id>_reflex.html` is the citator's own rendering of the document,
where every reference CanLII resolved is an anchor:

    <A class="reflex2-link" href=".../laws/stat/...">Patent Act</A>        -> legislation
    <A class="reflex3-caselaw" href=".../scc/doc/...">1982 CanLII 19 (SCC)</A> -> decision

Reflex anchors are *pieces* of a mention ("Federal Court Act" + "s. 18" are two
anchors of one gold mention), and Reflex also **injects** neutral citations into
the text ("1982 CanLII 19 (SCC)" does not exist in the gold text).  So a plain
run of `src.evaluation.evaluate` on the reflex file does not work:

  * reflex text != gold text        -> exact matching can never fire
  * one gold mention -> many anchors -> precision collapses

This module therefore
  1. reads both documents into their *whitespace-normalised* plain text while
     recording character offsets of every mention tag,
  2. merges adjacent same-target anchors into one predicted mention,
  3. aligns the reflex text onto the gold text (token diff, insertions of the
     injected citations handled explicitly),
  4. expresses the prediction in gold coordinates, then re-uses the paper's
     Span + compute_metrics / compute_metrics_overlap so the numbers stay
     comparable with Table 1.

Only parent mentions are considered (decision / legislation / secondary
sources) — Reflex has no component (title, fragment, ...) annotation.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from bs4 import BeautifulSoup, NavigableString, Tag

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REFLEX_DIR = PROJECT_ROOT / "data" / "reflex"
ANNOTATED_DIR = PROJECT_ROOT / "data" / "annotated"

PARENT_LABELS = ("decision", "legislation", "secondary sources")

# reflex href path segment -> parent mention label
HREF_RULES: Sequence[Tuple[str, str]] = (
    ("/laws/", "legislation"),
    ("/jnl/", "secondary sources"),
    ("/book", "secondary sources"),
    ("/comm/", "secondary sources"),
    ("/doc/", "decision"),
)

# external hosts Reflex links to when CanLII does not host the document
EXTERNAL_HOST_RULES: Sequence[Tuple[str, str]] = (
    ("westlaw.com", "decision"),            # Westlaw Canada case documents
    ("lexis.com", "decision"),              # Lexis case documents
    ("reports.fja-cmf.gc.ca", "decision"),  # Federal Court Reports ([2016] 2 F.C.R. 351)
    ("canlii.ca", "decision"),              # CanLII short link, anchor = case name
)

# reflex anchor classes that are real reference links (others = navigation)
MENTION_CLASSES = {"reflex2-link", "reflex3-caselaw", "reflex3-secondary"}

# human splits first; duplicated/derived splits are ignored for the report
SPLIT_PRIORITY = ("test", "dev", "train", "silver")
SKIP_SPLITS = {"test_assisted", "test_manual", "dev_train", "incoming", "no-gold"}


# ---------------------------------------------------------------------------
# normalised plain text + offsets
# ---------------------------------------------------------------------------

@dataclass
class Mention:
    """One mention in the normalised text of its own document."""
    label: str
    text: str
    start: int
    end: int
    href: str = ""
    n_anchors: int = 1
    anchor_texts: List[str] = field(default_factory=list)


def _track(rng: Dict[int, List[int]], key: int, pos: int) -> None:
    if key in rng:
        rng[key][1] = pos + 1
    else:
        rng[key] = [pos, pos + 1]


def normalised_text_with_tags(
    root: Tag,
    tag_names: Sequence[str],
    label_of,
    extra_keep=None,
) -> Tuple[str, Dict[int, Tuple[str, int, int, dict]]]:
    """
    Walk `root` producing the whitespace-normalised plain text
    (" ".join(get_text().split()), exactly like evaluation_l1_util) together
    with the [start, end) offset of every tag of interest.

    Returns (norm_text, {tag_index: (label, start, end, attrs)}).
    """
    chunks: List[Tuple[str, Tuple[int, ...]]] = []
    tag_meta: Dict[int, Tuple[str, dict]] = {}
    stack: List[int] = []

    def walk(node) -> None:
        for child in node.children:
            if isinstance(child, NavigableString):
                chunks.append((str(child), tuple(stack)))
            elif isinstance(child, Tag):
                keep = child.name in tag_names and label_of(child) is not None
                if keep:
                    idx = len(tag_meta)
                    tag_meta[idx] = (label_of(child), dict(child.attrs))
                    stack.append(idx)
                if extra_keep is not None:
                    extra_keep(child)
                walk(child)
                if keep:
                    stack.pop()

    walk(root)

    # collapse whitespace, tracking offsets per tag index
    ranges: Dict[int, List[int]] = {}
    out: List[str] = []
    pos = 0
    ws_pending = False
    started = False
    for text, owners in chunks:
        for ch in text:
            if ch.isspace():
                if started:
                    ws_pending = True
                continue
            if ws_pending:
                out.append(" ")
                pos += 1
                ws_pending = False
            for idx in owners:
                _track(ranges, idx, pos)
            out.append(ch)
            pos += 1
            started = True

    norm = "".join(out)
    result: Dict[int, Tuple[str, int, int, dict]] = {}
    for idx, (label, attrs) in tag_meta.items():
        if idx in ranges:
            s, e = ranges[idx]
            result[idx] = (label, s, e, attrs)
    return norm, result


# ---------------------------------------------------------------------------
# Reflex parsing
# ---------------------------------------------------------------------------

def classify_href(href: str) -> Optional[str]:
    """Label of a reference link, from its href.

    Handles both CanLII targets (`/laws/`, `/doc/`, `/jnl/`, ...) and the
    external targets Reflex uses when CanLII does not host the document
    (Westlaw / Lexis / Federal Court Reports -> decision, `laws.gnb.ca` ->
    legislation, CanLII short links -> decision).
    """
    if not href.startswith("http"):
        return None
    for needle, label in HREF_RULES:
        if needle in href:
            return label
    host = href.split("/")[2].lower() if href.count("/") >= 2 else ""
    if host.startswith("laws."):
        return "legislation"
    for needle, label in EXTERNAL_HOST_RULES:
        if needle in host:
            return label
    return None


def _base_href(href: str) -> str:
    return href.split("#", 1)[0]


_REFLEX_CACHE: Dict[Path, "ReflexDoc"] = {}


def parse_reflex(path: Path) -> "ReflexDoc":
    """Parse a reflex export (anchor offsets only — merging is done later so
    that several merge-gap settings can be tried on the same parse)."""
    path = Path(path)
    if path in _REFLEX_CACHE:
        return _REFLEX_CACHE[path]
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    body = soup.find("body")
    if body is None:
        return ReflexDoc(path=path, text="", anchors=[], unclassified=[])

    def label_of(tag: Tag) -> Optional[str]:
        if tag.name != "a":
            return None
        # CanLII-classed links (`reflex2-link`, `reflex3-caselaw`) and the
        # unclassed external links (Westlaw/Lexis/FCR) are both references;
        # navigation anchors (`reflex-parag`, `#_edn…`) never classify.
        return classify_href(tag.get("href", ""))

    text, tags = normalised_text_with_tags(body, ("a",), label_of)

    # unclassified http links (for coverage reporting)
    unclassified = []
    for a in body.find_all("a"):
        href = a.get("href", "")
        if href.startswith("http") and classify_href(href) is None:
            unclassified.append((href, " ".join(a.get_text().split())))

    anchors: List[Tuple[str, str, str, int, int]] = []   # label, href, text, start, end
    for idx in sorted(tags):
        label, s, e, attrs = tags[idx]
        if not text[s:e].strip():
            continue        # empty anchors are footnote markers, not mentions
        anchors.append((label, attrs.get("href", ""), text[s:e], s, e))

    doc = ReflexDoc(path=path, text=text, anchors=anchors, unclassified=unclassified)
    _REFLEX_CACHE[path] = doc
    return doc


def merge_anchors(anchors: List[Tuple[str, str, str, int, int]], merge_gap: int,
                  doc_text: str) -> List[Mention]:
    """
    Merge consecutive anchors that resolve to the same reference (same base
    href) and are at most `merge_gap` normalised characters apart into one
    predicted mention:  "Federal Court Act" + "s. 18"  ->  "Federal Court Act, s. 18".
    The merged text is the raw document slice between both anchor ends, so
    punctuation between the pieces is kept (the evaluator compares texts).
    """
    merged: List[Mention] = []
    for label, href, atext, s, e in anchors:
        if merged:
            last = merged[-1]
            gap = s - last.end
            if last.label == label and _base_href(last.href) == _base_href(href) and 0 <= gap <= merge_gap:
                last.end = e
                last.text = " ".join(doc_text[last.start:e].split())
                last.href = last.href or href
                last.n_anchors += 1
                last.anchor_texts.append(atext)
                continue
        merged.append(Mention(label=label, text=atext, start=s, end=e, href=href,
                              n_anchors=1, anchor_texts=[atext]))
    return merged


@dataclass
class ReflexDoc:
    path: Path
    text: str
    anchors: List[Tuple[str, str, str, int, int]]
    unclassified: List[Tuple[str, str]]
    _cache: Dict[int, List[Mention]] = field(default_factory=dict)

    def mentions(self, merge_gap: int) -> List[Mention]:
        if merge_gap not in self._cache:
            self._cache[merge_gap] = merge_anchors(self.anchors, merge_gap, self.text)
        return self._cache[merge_gap]

    @property
    def n_anchor_classes(self) -> Counter:
        return Counter(a[0] for a in self.anchors)


# ---------------------------------------------------------------------------
# Gold parsing
# ---------------------------------------------------------------------------

def _gold_label(tag: Tag) -> Optional[str]:
    if not (tag.get("parent") or "").strip():        # parent-level mention
        lab = (tag.get("labelname") or "").strip().lower()
        if lab in PARENT_LABELS:
            return lab
    return None


def parse_gold(path: Path) -> "GoldDoc":
    path = Path(path)
    if path in _PARSE_CACHE:
        return _PARSE_CACHE[path]
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
    body = soup.find("body")
    if body is None:
        doc = GoldDoc(path=path, text="", parents=[])
    else:
        text, tags = normalised_text_with_tags(body, ("manual_label", "auto_label"), _gold_label)
        parents = [
            Mention(label=label, text=text[s:e], start=s, end=e)
            for idx, (label, s, e, _) in sorted(tags.items())
        ]
        doc = GoldDoc(path=path, text=text, parents=parents)
    _PARSE_CACHE[path] = doc
    return doc


# an LLM / rules system output is read exactly like a gold file: parent-level
# manual_label/auto_label tags with their own offsets in that file's text.
parse_system = parse_gold

_PARSE_CACHE: Dict[Path, "GoldDoc"] = {}


@dataclass
class GoldDoc:
    path: Path
    text: str
    parents: List[Mention]


# ---------------------------------------------------------------------------
# pairing reflex file -> gold file
# ---------------------------------------------------------------------------

def _norm_id(doc_id: str) -> str:
    s = re.sub(r"[^a-z0-9]", "", doc_id.lower())
    return re.sub(r"(canlii\d+).*$", r"\1", s)


def _digit_key(doc_id: str):
    """(year, all digits) — catches id pairs such as 2026ABCA2 vs 2026ABCA2L or
    2009SKQB465 vs 2009KQB465, which _norm_id() does not relate."""
    m = re.match(r"(\d{4})", doc_id)
    if not m:
        return None
    digits = re.sub(r"\D", "", doc_id)
    return (m.group(1), digits) if digits else None


def gold_index() -> Dict[str, List[str]]:
    idx: Dict[str, List[str]] = {}
    for p in sorted(ANNOTATED_DIR.glob("*/*.html")):
        if p.parent.name in SKIP_SPLITS:
            continue
        idx.setdefault(p.stem, []).append(p.parent.name)
    return idx


def _candidate_index(idx: Dict[str, List[str]], key_fn) -> Dict:
    out: Dict = {}
    for stem in idx:
        k = key_fn(stem)
        if k is not None:
            out.setdefault(k, []).append(stem)
    return out


@dataclass
class Pair:
    reflex_path: Path
    gold_path: Path
    doc_id: str
    split: str
    how: str            # exact | fuzzy | none
    text_ratio: float = 0.0


def pair_documents(min_ratio: float = 0.85, verbose: bool = False) -> List[Pair]:
    """Match every reflex export with its gold document (exact id first, then
    fuzzy id candidates — verified with the content-token similarity gate so we
    never pair two different decisions)."""
    idx = gold_index()
    by_norm = _candidate_index(idx, _norm_id)
    by_digit = _candidate_index(idx, _digit_key)

    pairs: List[Pair] = []
    for rpath in sorted(REFLEX_DIR.glob("*_reflex.html")):
        doc_id = rpath.name[: -len("_reflex.html")]
        how = "exact"
        cand = doc_id
        if cand not in idx:
            options = list(dict.fromkeys(
                by_norm.get(_norm_id(cand), []) + by_digit.get(_digit_key(cand), [])))
            if not options:
                pairs.append(Pair(rpath, Path(""), doc_id, "", "none"))
                continue
            # try every candidate id, keep the one with the closest text
            best_id, best_ratio = None, -1.0
            for opt in options:
                split = next((s for s in SPLIT_PRIORITY if s in idx[opt]), sorted(idx[opt])[0])
                gp = ANNOTATED_DIR / split / f"{opt}.html"
                ratio = text_similarity(gold_text_for(gp), parse_reflex(rpath).text)
                if ratio > best_ratio:
                    best_id, best_ratio = opt, ratio
            cand, how = best_id, "fuzzy"

        split = next((s for s in SPLIT_PRIORITY if s in idx[cand]), sorted(idx[cand])[0])
        gpath = ANNOTATED_DIR / split / f"{cand}.html"
        ratio = text_similarity(gold_text_for(gpath), parse_reflex(rpath).text)
        if how == "fuzzy" and ratio < min_ratio:
            how = "rejected"
        if verbose:
            print(f"  {rpath.name} -> {cand} [{split}] {how} ratio={ratio:.3f}")
        pairs.append(Pair(rpath, gpath, cand, split, how, ratio))

    return pairs


_GOLD_TEXT_CACHE: Dict[Path, str] = {}


def gold_text_for(path: Path) -> str:
    if path not in _GOLD_TEXT_CACHE:
        _GOLD_TEXT_CACHE[path] = parse_gold(path).text
    return _GOLD_TEXT_CACHE[path]


def text_similarity(a: str, b: str) -> float:
    """
    Cheap content-token Jaccard over tokens of length >= 4.

    Used as the pairing gate for fuzzy ids.  Measured separation on the corpus:
    same decision rendered twice  ~0.96-1.00, two different decisions ~0.07-0.14
    (a full difflib ratio gives the same separation but costs ~5 s/document).
    """
    if not a or not b:
        return 0.0
    ca = Counter(t for t in a.split() if len(t) >= 4)
    cb = Counter(t for t in b.split() if len(t) >= 4)
    if not ca or not cb:
        return 0.0
    inter = sum((ca & cb).values())
    union = sum((ca | cb).values())
    return inter / union if union else 0.0


# ---------------------------------------------------------------------------
# alignment: reflex text -> gold text
# ---------------------------------------------------------------------------

class Alignment:
    """
    Maps character positions of the reflex text onto the gold text.

    Built from a token-level diff, which handles exactly the difference between
    the two renderings: Reflex *inserts* neutral citations
        gold  : "... Intervenant (1971), 65 C.P.R. 1 (Ex. Ct.)"
        reflex: "... Intervenant (1971), 1982 CanLII 19 (SCC), 65 C.P.R. 1 ..."
    so an inserted mention maps to the gold insertion point (typically *inside*
    the gold parent mention that names the case).
    """

    def __init__(self, text_reflex: str, text_gold: str):
        self.tok_r, self.off_r = _tokens(text_reflex)
        self.tok_g, self.off_g = _tokens(text_gold)
        self.len_r = len(text_reflex)
        self.len_g = len(text_gold)
        sm = SequenceMatcher(None, self.tok_r, self.tok_g, autojunk=False)
        self.ops = sm.get_opcodes()          # (tag, a1, b1, a2, b2)
        self.ratio = sm.ratio()
        # reflex char index -> gold char index (per opcode region)
        self._spans: List[Tuple[int, int, str, int, int]] = []
        for tag, a1, b1, a2, b2 in self.ops:
            rs = self.off_r[a1] if a1 < len(self.off_r) else self.len_r
            re_ = self.off_r[b1 - 1] + len(self.tok_r[b1 - 1]) if b1 > a1 else rs
            gs = self.off_g[a2] if a2 < len(self.off_g) else self.len_g
            ge = self.off_g[b2 - 1] + len(self.tok_g[b2 - 1]) if b2 > a2 else gs
            self._spans.append((rs, re_, tag, gs, ge))

    def map_pos(self, x: int) -> int:
        """
        Gold coordinate of reflex character position `x`.

        `equal` regions map linearly.  Reflex-only regions (the injected
        citations, i.e. `delete` opcodes because the reflex token list is the
        first argument) collapse onto the single gold insertion point, which
        normally sits *inside* the gold mention that names the same case.
        """
        for rs, re_, tag, gs, ge in self._spans:
            if rs <= x < re_:
                if tag == "equal":
                    return gs + (x - rs)
                if ge > gs:                       # replace: proportional
                    return min(ge - 1, gs + int((ge - gs) * (x - rs) / max(1, re_ - rs)))
                return gs                         # reflex-only: gold insertion point
            if rs == re_ and x == rs:             # gold-only gap between reflex tokens
                return gs
        return self.len_g

    def map_span(self, s: int, e: int) -> Tuple[int, int]:
        gs, ge = self.map_pos(s), self.map_pos(e)
        if ge <= gs:
            ge = min(gs + 1, self.len_g) if self.len_g else gs + 1
            if ge <= gs:
                ge = gs + 1
        return gs, ge


def _tokens(text: str) -> Tuple[List[str], List[int]]:
    toks = re.findall(r"\S+", text)
    offs = [m.start() for m in re.finditer(r"\S+", text)]
    return toks, offs


# ---------------------------------------------------------------------------
# scoring (re-uses the paper's Span / metrics)
# ---------------------------------------------------------------------------

def _load_l1():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "evaluation_l1_util", PROJECT_ROOT / "src" / "evaluation" / "evaluation_l1_util.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_L1 = None


def l1():
    global _L1
    if _L1 is None:
        _L1 = _load_l1()
    return _L1


def _context(text: str, s: int, e: int, ctx: int) -> str:
    return text[max(0, s - ctx): min(len(text), e + ctx)].lower()


def build_spans(
    gold: GoldDoc,
    system: List[Mention],
    alignment: Alignment,
    ctx: int = 200,
) -> Tuple[list, list]:
    """Return (gold_spans, system_spans) as evaluation_l1_util.Span objects,
    with both sides expressed in GOLD coordinates."""
    L1 = l1()
    gold_spans = [
        L1.Span(g.text, g.label, _context(gold.text, g.start, g.end, ctx),
                start_pos=g.start, end_pos=g.end)
        for g in gold.parents
    ]
    sys_spans = []
    for m in system:
        gs, ge = alignment.map_span(m.start, m.end)
        gs = min(max(0, gs), max(0, alignment.len_g - 1))
        ge = max(gs + 1, min(ge, alignment.len_g))
        sys_spans.append(
            L1.Span(m.text, m.label, _context(gold.text, gs, ge, ctx),
                    start_pos=gs, end_pos=ge)
        )
    return gold_spans, sys_spans


def score_pair(gold: GoldDoc, system: List[Mention], alignment: Alignment, ctx: int = 200) -> dict:
    """Per-label P/R/F1 for one document under EXACT and OVERLAP matching."""
    L1 = l1()
    gold_spans, sys_spans = build_spans(gold, system, alignment, ctx)
    return {
        "exact": L1.compute_metrics(gold_spans, sys_spans),
        "overlap": L1.compute_metrics_overlap(gold_spans, sys_spans),
        "n_gold": len(gold_spans),
        "n_system": len(sys_spans),
    }
