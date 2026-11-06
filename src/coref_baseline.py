"""Trivial string-match coreference baselines (no model calls).

Mirrors the structure/CLI of main_coref.py but replaces the LLM registry
with deterministic string clustering.

Methods (--method):
  - exact:      group mentions with identical raw surface text.
  - normalized: group on lowercased/whitespace-collapsed text (per labelname).
  - strong (default): multi-sieve deterministic pipeline (Stanford-style:
    high-precision sieves first, each only ever merges):
      S0 full-text equality (normalized, leading article stripped)
      S1 shared title sublabel (normalized or alnum-equal)
      S2 shared citation sublabel (normalized or alnum-equal)
      S3 stripped-full-text equality (", supra" / "at para. N" trailers removed)
      S4 supra/pinpoint head -> nearest preceding same-label title containing
         all head tokens ("Schwartz, supra" -> "Schwartz v. Canada, ...")
      S5 token containment (shorter mention's significant tokens all appear
         in a longer same-label mention; needs >= 2 tokens, no generic-only)
      S6 acronyms ("s. 5 of the JRPA", "Section 5(a)(ii) C.R.A." -> nearest
         preceding title with matching initials or parens definition)
      S7 proximity fallback (mentions with no title/citation sublabel attach
         to the nearest preceding same-label mention that has one)

Input/output conventions match main_coref.py exactly so the same
evaluation script works unchanged:

  - --input-source gold:       read from DATA_DIR/<variant>/<split>/<file>.html
  - --input-source extraction: read from a main_extraction.py output folder
  - split mode writes per-file subfolders:
      <output_root>/<filename>/<filename>_coref_mapping.json
      <output_root>/<filename>/<filename>_coref_clusters.json
      <output_root>/<filename>/<filename>_coref.html

Predicted docid strings are arbitrary (evaluation only compares the
grouping, never the docid values).

Usage:
    python -m src.coref_baseline --split test --input-source gold
    python -m src.coref_baseline --split test --input-source gold --method normalized
    python -m src.coref_baseline --filename 2002SCC33 --single-split test --method exact
"""

import argparse
import json
import re
import string
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from bs4 import BeautifulSoup

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import DATA_DIR, SPLITS
from src.ann_extractor import extract_parent_level_annotations
from src.htmlLabel import ReferenceMention


# =============================================================================
# HTML reconstruction (same helper as main_coref.py)
# =============================================================================

def dict_to_clusters(mapping: Dict[str, str]) -> List[List[str]]:
    clusters = defaultdict(list)
    for mention_id, docid in mapping.items():
        clusters[docid].append(mention_id)
    return list(clusters.values())


def apply_coref_docids(html_content: str, mapping: Dict[str, str], parser: str = "html.parser") -> str:
    """Strip gold docid/uri attributes, then apply predicted docids."""
    soup = BeautifulSoup(html_content, parser)

    for tag in soup.find_all(attrs={"docid": True}):
        del tag["docid"]

    for tag in soup.find_all(attrs={"uri": True}):
        del tag["uri"]

    missing = []
    for mention_id, docid in mapping.items():
        tag = soup.find(id=mention_id)
        if tag is None:
            missing.append(mention_id)
            continue
        tag["docid"] = docid

    if missing:
        print(
            f"  [warn] {len(missing)} mention id(s) from the mapping were not found "
            f"in the document while reconstructing HTML: {missing}"
        )

    return str(soup)


# =============================================================================
# Normalization
# =============================================================================

_WS_RE = re.compile(r"\s+")
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _clean(s: str) -> str:
    return (s.replace("\xa0", " ")
             .replace("\u2011", "-").replace("\u2010", "-")
             .replace("\u2013", "-").replace("\u2014", "-"))


def strip_leading_article(s: str) -> str:
    return re.sub(r"^(the|a|an|le|la|les)\s+", "", s.strip(), flags=re.I)


def norm_key(s: str) -> str:
    s = _WS_RE.sub(" ", _clean(s).strip().lower())
    s = s.strip(string.punctuation + " \t\n\r")
    return _WS_RE.sub(" ", s)


def alnum_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", _clean(s).lower())


def normalize_text(text: str, method: str) -> str:
    if method == "exact":
        return text
    return norm_key(text)


STOPWORDS = {
    "of", "the", "a", "an", "and", "in", "on", "at", "to", "for", "with",
    "by", "from", "or", "v", "vs", "de", "la", "le", "les", "des", "du",
    "et", "en", "d", "l", "au", "aux", "un", "une", "est", "que", "as",
}

# Single-significant-token heads in this set never trigger the surname
# sieve (S4) and never count as the "shorter" side of containment (S5);
# they are resolved by the proximity fallback (S7) instead.
GENERIC = {
    "act", "acts", "charter", "charters", "code", "codes", "regulation",
    "regulations", "statute", "statutes", "bylaw", "bylaws", "law", "laws",
    "rule", "rules", "section", "sections", "paragraph", "paragraphs",
    "para", "paras", "article", "articles", "art", "part", "parts",
    "chapter", "chapters", "schedule", "schedules", "division", "divisions",
    "title", "titles", "decision", "decisions", "order", "orders",
    "loi", "code", "article", "articles", "paragraphe", "annexe",
}

TOKEN_ALIASES = {
    "s": "section", "ss": "sections",
    "para": "paragraph", "paras": "paragraphs",
    "art": "article", "arts": "articles",
    "sch": "schedule", "reg": "regulation", "regs": "regulations",
}


def significant_tokens(text: str) -> List[str]:
    toks = _TOKEN_RE.findall(_clean(text).lower())
    toks = [TOKEN_ALIASES.get(t, t) for t in toks]
    return [t for t in toks if t not in STOPWORDS]


# ", supra ..." / ", at para. N ..." trailers signal short-form anaphora.
_SUPRA_RE = re.compile(r",?\s*\bsupra\b.*$", re.I)
_AT_RE = re.compile(r",?\s*\bat\b.*$", re.I)


def strip_trailing_refs(text: str) -> str:
    head = _SUPRA_RE.sub("", text)
    head = _AT_RE.sub("", head)
    return head.strip(" ,;:.").strip()


# =============================================================================
# Acronyms
# =============================================================================

# Parens-defined acronyms: ("JRPA"), (EAB), (C.R.A.) — but not docket
# numbers like (500-09-006851-984) or prose like (Guardian ad litem of).
_ACR_DEF_RE = re.compile(r'\(\s*"?([A-Za-z0-9.&\-\/]{2,12})"?\s*\)')


def acronym_defs_in(text: str) -> List[str]:
    found = []
    for m in _ACR_DEF_RE.finditer(text):
        cand = m.group(1)
        letters = re.sub(r"[^A-Za-z]", "", cand)
        if len(re.sub(r"[^A-Za-z0-9]", "", cand)) < 2:
            continue
        if sum(1 for ch in letters if ch.isupper()) < 2:
            continue
        if re.fullmatch(r"[\d\-\s.]+", cand):
            continue
        found.append(re.sub(r"[^A-Z0-9]", "", cand.upper()))
    return found


def initials_of(title: str) -> Optional[str]:
    toks = [t for t in _TOKEN_RE.findall(_clean(title).lower())
            if t not in STOPWORDS and t[:1].isalpha()]
    init = "".join(t[0] for t in toks).upper()
    return init if len(init) >= 2 else None


_OF_THE_RE = re.compile(r"^(?P<prefix>.+?)\bof\s+the\s+(?P<rest>.+)$", re.I)
_LEAD_PIN_RE = re.compile(
    r"^(?:s{1,2}\.?|sections?|para\.?|paragraphs?|art\.?|articles?|§+|pp?\.?|no\.?|sub?s?\.?)\s+[\d().a-zivx,\-–—\s]*",
    re.I,
)
_PINPOINT_WORDS = {
    "s", "ss", "section", "sections", "para", "paragraph", "paragraphs",
    "paras", "article", "articles", "art", "chapter", "part", "schedule",
    "division", "rule", "rules", "subsection", "paragraphs",
}


def bare_reference(text: str) -> Optional[str]:
    """Remainder of '<pinpoint> of the <X>' or '<pinpoint> <X>' references."""
    m = _OF_THE_RE.match(text.strip())
    if m:
        pre = _TOKEN_RE.findall(m.group("prefix").lower())
        if any(t in _PINPOINT_WORDS or t.isdigit() for t in pre):
            return m.group("rest").strip(" ,;:")
    m2 = _LEAD_PIN_RE.match(text.strip())
    if m2 and m2.group(0).strip():
        rest = text.strip()[m2.end():].strip(" ,;:")
        if rest:
            return rest
    return None


def as_acronym(s: str) -> Optional[str]:
    s = s.strip().strip('"').strip()
    if not re.fullmatch(r"[A-Za-z](?:\.?[A-Za-z0-9]+)*\.?", s):
        return None
    key = re.sub(r"[^A-Z0-9]", "", s.upper())
    if not 2 <= len(key) <= 8:
        return None
    if s == s.lower() and "." not in s:
        return None
    return key


# =============================================================================
# Mention representation + union-find
# =============================================================================

@dataclass
class _Mention:
    idx: int
    mid: str
    label: str
    text: str
    norm: str
    titles: List[str] = field(default_factory=list)
    title_keys: Set[str] = field(default_factory=set)
    title_alnums: Set[str] = field(default_factory=set)
    title_tokens: List[Set[str]] = field(default_factory=list)
    title_initials: Set[str] = field(default_factory=set)
    cite_keys: Set[str] = field(default_factory=set)
    cite_alnums: Set[str] = field(default_factory=set)
    has_full: bool = False
    sig: Set[str] = field(default_factory=set)
    stripped_key: str = ""
    had_anaphora: bool = False
    head_sig: Set[str] = field(default_factory=set)
    acronym_defs: List[str] = field(default_factory=list)


def build_infos(mentions: List[ReferenceMention]) -> List[_Mention]:
    infos = []
    for idx, m in enumerate(mentions):
        mid = m.html_tag.attributes.get("id")
        if mid is None:
            continue
        label = m.html_tag.attributes.get("labelname", "")
        text = m.text
        subs = m.get_sublabel_texts()
        titles = subs.get("title", [])
        cites = subs.get("citation", [])

        title_keys, title_alnums, title_tokens, initials = set(), set(), [], set()
        for t in titles:
            k = norm_key(strip_leading_article(t))
            a = alnum_key(t)
            if k:
                title_keys.add(k)
            if a:
                title_alnums.add(a)
            title_tokens.append(set(significant_tokens(t)))
            ini = initials_of(t)
            if ini:
                initials.add(ini)

        cite_keys, cite_alnums = set(), set()
        for c in cites:
            k, a = norm_key(c), alnum_key(c)
            if k:
                cite_keys.add(k)
            if a:
                cite_alnums.add(a)

        stripped = strip_trailing_refs(text)
        stripped_key = norm_key(strip_leading_article(stripped))
        had_anaphora = bool(stripped_key and stripped_key != norm_key(text))

        infos.append(_Mention(
            idx=idx, mid=mid, label=label, text=text,
            norm=norm_key(strip_leading_article(text)),
            titles=titles, title_keys=title_keys, title_alnums=title_alnums,
            title_tokens=title_tokens, title_initials=initials,
            cite_keys=cite_keys, cite_alnums=cite_alnums,
            has_full=bool(titles or cites),
            sig=set(significant_tokens(text)),
            stripped_key=stripped_key,
            had_anaphora=had_anaphora,
            head_sig=set(significant_tokens(stripped)) if had_anaphora else set(),
            acronym_defs=acronym_defs_in(text),
        ))
    return infos


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        self.parent[rb] = ra
        return True


# =============================================================================
# Clustering
# =============================================================================

def mention_key(mention: ReferenceMention, method: str) -> str:
    """Clustering key for the exact/normalized methods."""
    labelname = mention.html_tag.attributes.get("labelname", "")
    return f"{labelname}||{normalize_text(mention.text, method)}"


def cluster_mentions(mentions: List[ReferenceMention], method: str) -> Dict[str, str]:
    """mention_id -> synthetic docid via single-key string-match grouping."""
    groups: Dict[str, List[str]] = defaultdict(list)
    for mention in mentions:
        mention_id = mention.html_tag.attributes.get("id")
        if mention_id is None:
            continue
        groups[mention_key(mention, method)].append(mention_id)

    mapping: Dict[str, str] = {}
    for idx, (_key, ids) in enumerate(groups.items()):
        docid = f"baseline_{idx}"
        for mid in ids:
            mapping[mid] = docid
    return mapping


def cluster_mentions_strong(mentions: List[ReferenceMention]) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Multi-sieve deterministic clustering. Returns (mapping, sieve_stats)."""
    infos = build_infos(mentions)
    n = len(infos)
    uf = UnionFind(n)
    stats: Dict[str, int] = {}

    def group_union(key_fn) -> int:
        buckets: Dict[str, List[int]] = defaultdict(list)
        for i, info in enumerate(infos):
            for k in key_fn(info):
                buckets[k].append(i)
        count = 0
        for members in buckets.values():
            first = members[0]
            for other in members[1:]:
                count += uf.union(first, other)
        return count

    # S0: full-text equality (same label).
    stats["S0_full"] = group_union(lambda info: {(info.label, info.norm)} if info.norm else set())

    # S1: shared title (same label).
    stats["S1_title"] = group_union(
        lambda info: {(info.label, k) for k in info.title_keys}
        | {(info.label, "A:" + a) for a in info.title_alnums}
    )

    # S2: shared citation (same label).
    stats["S2_cite"] = group_union(
        lambda info: {(info.label, k) for k in info.cite_keys}
        | {(info.label, "A:" + a) for a in info.cite_alnums}
    )

    # S3: stripped-full-text equality (supra/pinpoint trailers removed).
    stats["S3_stripped"] = group_union(
        lambda info: {(info.label, info.stripped_key)} if info.stripped_key else set()
    )

    # S4: supra/pinpoint head -> nearest preceding title containing all head tokens.
    count = 0
    for i, info in enumerate(infos):
        if not info.had_anaphora or not info.head_sig:
            continue
        if len(info.head_sig) == 1:
            (tok,) = tuple(info.head_sig)
            if tok in GENERIC or len(tok) < 2:
                continue
        for j in range(i - 1, -1, -1):
            other = infos[j]
            if other.label != info.label:
                continue
            if any(info.head_sig <= tt for tt in other.title_tokens if tt):
                count += uf.union(i, j)
                break
    stats["S4_supra"] = count

    # S5: token containment (same label, shorter fully contained in longer).
    count = 0
    by_label: Dict[str, List[int]] = defaultdict(list)
    for i, info in enumerate(infos):
        by_label[info.label].append(i)
    for members in by_label.values():
        for x in range(len(members)):
            for y in range(x + 1, len(members)):
                a, b = infos[members[x]], infos[members[y]]
                sa, sb = a.sig, b.sig
                if not sa or not sb or sa == sb:
                    continue
                short, long = (a, b) if len(sa) < len(sb) else (b, a)
                ss, ls = (sa, sb) if len(sa) < len(sb) else (sb, sa)
                if len(ss) < 2 or not any(t not in GENERIC for t in ss):
                    continue
                if ss <= ls:
                    count += uf.union(members[x], members[y])
    stats["S5_contain"] = count

    # S6: bare-acronym references -> nearest preceding matching title.
    count = 0
    for i, info in enumerate(infos):
        rest = bare_reference(info.text)
        if not rest:
            continue
        key = as_acronym(rest)
        if not key:
            continue
        for j in range(i - 1, -1, -1):
            other = infos[j]
            if other.label != info.label:
                continue
            if key in other.acronym_defs or key in other.title_initials:
                count += uf.union(i, j)
                break
    stats["S6_acronym"] = count

    # S7: proximity fallback for mentions without title/citation sublabels.
    root_members: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        root_members[uf.find(i)].append(i)
    root_has_full = {r: any(infos[i].has_full for i in mem) for r, mem in root_members.items()}
    count = 0
    for i, info in enumerate(infos):
        if info.has_full or root_has_full[uf.find(i)]:
            continue
        for j in range(i - 1, -1, -1):
            if infos[j].label == info.label and infos[j].has_full:
                count += uf.union(i, j)
                break
    stats["S7_proximity"] = count

    # Connected components -> synthetic docids, deterministic order.
    comp_members: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        comp_members[uf.find(i)].append(i)
    ordered = sorted(comp_members.values(), key=lambda mem: min(infos[i].idx for i in mem))

    mapping: Dict[str, str] = {}
    for cidx, members in enumerate(ordered):
        docid = f"baseline_{cidx}"
        for i in members:
            mapping[infos[i].mid] = docid
    return mapping, stats


def extract_mentions_robust(html_content: str) -> List[ReferenceMention]:
    """Same mentions as main_coref.py, but robust to missing docid attrs.

    extract_parent_level_annotations() filters on docid=True, which holds
    for gold files and for extraction outputs that keep an (empty) docid
    attribute. If it ever returns nothing (e.g. an extractor variant that
    drops the attribute entirely), fall back to any parent-level tag.
    """
    mentions = extract_parent_level_annotations(html_content)
    if mentions:
        return mentions
    soup = BeautifulSoup(html_content, "html.parser")
    parents = soup.find_all(["manual_label", "auto_label"], attrs={"parent": ""})
    return [ReferenceMention(str(p)) for p in parents]


# =============================================================================
# Per-file processing
# =============================================================================

def resolve_input_path(
    filename: str,
    mode: str,
    split: Optional[str],
    single_split: str,
    input_source: str,
    data_dir: Path,
    variant: str,
    extraction_dir: Optional[Path],
    extraction_stage: str,
) -> Path:
    if input_source == "gold":
        split_dir = split if mode == "split" else single_split
        return Path(data_dir) / variant / split_dir / f"{filename}.html"
    suffix = f"_{extraction_stage}"
    nested = extraction_dir / filename / f"{filename}{suffix}.html"
    flat = extraction_dir / f"{filename}{suffix}.html"
    return nested if nested.exists() else flat


def get_filenames(
    mode: str,
    filename: Optional[str],
    split: Optional[str],
    input_source: str,
    data_dir: Path,
    variant: str,
    extraction_dir: Optional[Path],
) -> List[str]:
    if mode == "single":
        return [filename]
    if input_source == "gold":
        folder = Path(data_dir) / variant / split
        return sorted(p.stem for p in folder.glob("*.html"))
    return sorted(p.name for p in extraction_dir.iterdir() if p.is_dir())


def process_file_baseline(filename: str, html_content: str, method: str, exp_dir: Path) -> Dict[str, str]:
    mentions = extract_mentions_robust(html_content)

    stats: Optional[Dict[str, int]] = None
    if method == "strong":
        mapping, stats = cluster_mentions_strong(mentions)
    else:
        mapping = cluster_mentions(mentions, method)

    mapping_path = exp_dir / f"{filename}_coref_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    clusters_path = exp_dir / f"{filename}_coref_clusters.json"
    with open(clusters_path, "w", encoding="utf-8") as f:
        json.dump(dict_to_clusters(mapping), f, indent=2, ensure_ascii=False)

    html_path = exp_dir / f"{filename}_coref.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(apply_coref_docids(html_content, mapping))

    n_clusters = len(set(mapping.values()))
    detail = ""
    if stats:
        detail = " | " + " ".join(k + ":" + str(v) for k, v in stats.items())
    print(f"  [ok] {len(mapping)}/{len(mentions)} mentions -> {n_clusters} clusters{detail}")
    return mapping


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="String-match coref baseline (mirrors main_coref.py CLI).")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--filename", type=str)
    mode_group.add_argument("--split", choices=SPLITS)
    parser.add_argument("--single-split", default="dev", choices=SPLITS,
                        help="Split to look up gold input in when using --filename (single mode).")

    parser.add_argument("--input-source", default="gold", choices=["gold", "extraction"])
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--variant", default="annotated", choices=["original", "annotated"])
    parser.add_argument("--extraction-dir", type=Path, default=None)
    parser.add_argument("--extraction-stage", default="final",
                        choices=["processed", "final", "0", "1", "2", "3"])

    parser.add_argument("--method", default="strong", choices=["exact", "normalized", "strong"],
                        help="exact/normalized: single-key grouping; strong: multi-sieve pipeline.")
    parser.add_argument("--output-dir", type=Path, default=Path("./output_coref"))
    parser.add_argument("--with-timestamp", action="store_true")
    return parser


def run_tag(split: str, method: str, input_source: str, extraction_dir: Optional[Path], extraction_stage: str) -> str:
    tag = f"{split}_baseline_coref_{method}"
    if input_source == "extraction":
        tag += f"_from-{extraction_dir.name}-{extraction_stage}"
    else:
        tag += "_gold"
    return tag


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.input_source == "extraction" and args.extraction_dir is None:
        parser.error("--extraction-dir is required when --input-source=extraction")

    mode = "single" if args.filename else "split"
    data_dir = args.data_dir or Path(DATA_DIR)
    split = args.split or args.single_split

    tag = run_tag(split, args.method, args.input_source, args.extraction_dir, args.extraction_stage)
    output_root = Path(args.output_dir) / tag
    if args.with_timestamp:
        output_root = Path(args.output_dir) / f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_root.mkdir(parents=True, exist_ok=True)

    filenames = get_filenames(mode, args.filename, args.split, args.input_source,
                              data_dir, args.variant, args.extraction_dir)
    print(f"[Setup] Mode: {mode.upper()} | Source: {args.input_source} | "
          f"Method: {args.method} | Files: {len(filenames)} | Output: {output_root}\n")

    for i, filename in enumerate(filenames, start=1):
        print(f"Processing: {filename}  (#{i}/{len(filenames)})")
        path = resolve_input_path(filename, mode, args.split, args.single_split,
                                  args.input_source, data_dir, args.variant,
                                  args.extraction_dir, args.extraction_stage)
        if not path.exists():
            print(f"  [skip] Input not found at {path} — skipping {filename}")
            continue
        with open(path, encoding="utf-8") as f:
            html_content = f.read()

        exp_dir = output_root / filename if mode == "split" else output_root
        exp_dir.mkdir(exist_ok=True, parents=True)
        process_file_baseline(filename, html_content, args.method, exp_dir)

    print("=" * 80)
    print("[done] BASELINE COREF COMPLETE")
    print(f"[done] Experiment folder: {output_root}")
    print("=" * 80)


if __name__ == "__main__":
    main()
