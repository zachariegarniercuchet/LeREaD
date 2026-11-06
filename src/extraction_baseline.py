"""Rule-based extraction baseline (no model calls).

Mirrors the structure/CLI of main_extraction.py: raw HTML documents are read
from data/<variant>/<split>/ (default variant "original"), annotated with
<auto_label> tags, and written to output/<run-tag>/ so that the standard
evaluation script (src/evaluation/evaluate.py) scores them unchanged.

Detection is two-pass per document:
  1. citation anchors, full decision mentions (X v. Y / X c. Y), full
     secondary-source mentions, legislation mentions;
  2. short forms licensed by pass 1 (harvested surnames / authors / acronyms
     in the same document, bare pinpoints labelled from the nearest
     preceding parent).

Labels: decision, legislation, secondary sources (+ children title,
citation, fragment, source, authors).

Output layout mirrors main_extraction.py (per-file subfolders in split
mode, one *_final.html per subfolder, so batch evaluation pairs them):
    output/<split>_baseline_rules/<filename>/<filename>_final.html
Single mode writes output/<run>/<filename>.html (exact stem, so single-file
evaluation pairs it directly with gold).

Usage:
    python -m src.extraction_baseline --split test
    python -m src.extraction_baseline --filename 2002SCC33 --split test
"""

import argparse
import json
import re
import sys
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from bs4 import BeautifulSoup, NavigableString

PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import DATA_DIR, SPLITS


def _load_gazetteer() -> dict:
    try:
        with open(Path(__file__).parent / "extraction_baseline_data.json", encoding="utf-8") as f:
            return json.load(f)
    except OSError:
        return {"secondary_titles": [], "secondary_authors": [], "publisher_words": []}


_GAZ = _load_gazetteer()

# Curated publisher distinctive words (gate for "Place: Publisher, Year").
PUBLISHERS = {
    "press", "presses", "publishing", "éditions", "editions", "university",
    "université", "universities", "association", "associations", "carswell",
    "butterworths", "thomson", "reuters", "lexisnexis", "lexis", "emond",
    "irwin", "wilson", "lafleur", "blais", "yvon", "west", "pitblado",
    "mcgill", "oxford", "cambridge", "books", "barreau", "dictionnaires",
    "robert", "geddes", "grosset", "éditeur", "looseleaf", "yvon",
}


def _gazetteer_pattern(phrases: List[str]):
    phrases = sorted(set(phrases), key=len, reverse=True)
    if not phrases:
        return None
    # case-sensitive: lowercase prose occurrences ("statutory interpretation")
    # are usually not annotated, title-case ones are
    return re.compile(
        "|".join(re.escape(p).replace(r"\ ", r"\s+") for p in phrases),
    )


_SEC_TITLE_RE = _gazetteer_pattern(_GAZ.get("secondary_titles", []))
_SEC_AUTH_RE = _gazetteer_pattern([a for a in _GAZ.get("secondary_authors", []) if len(a.split()) >= 2])
BARE_NAMES = set(_GAZ.get("bare_names", []))
BARE_PHRASES = set(_GAZ.get("bare_phrases", []))


# =============================================================================
# Gazetteers (reporter abbreviations mined from train/dev gold + standard
# Canadian references; no test data involved)
# =============================================================================

REPORTERS = {
    "S.C.R.", "D.L.R.", "C.C.C.", "C.R.R.", "C.R.", "N.R.", "W.W.R.",
    "F.C.R.", "F.C.", "O.A.C.", "O.R.", "O.J.", "N.B.R.", "N.B.J.",
    "U.S.", "S. Ct.", "S.Ct.", "All E.R.", "E.R.", "Imm. L.R.",
    "Alta. L.R.", "C.P.R.", "Admin. L.R.", "C.C.E.L.", "C.L.R.",
    "C.L.L.C.", "Q.A.C.", "R.F.L.", "C.T.C.", "F.T.R.", "L.A.C.",
    "B.C.L.R.", "B.C.A.C.", "A.R.", "D.T.C.", "A.P.R.", "R.J.Q.",
    "J.E.", "N.Z.L.R.", "A.L.R.", "U.N.T.S.", "N.E.", "C.E.R.",
    "T.B.R.", "N.H.", "A.C.", "T.C.", "F.", "Q.B.", "Que. K.B.",
    "U.N.B.L.J.", "McGill L.J.", "Dalhousie L.J.", "Queen's L.J.",
    "U.T.L.J.", "Osgoode Hall L.J.", "Sask. L. Rev.", "Alta. L. Rev.",
    "Case Western L. Rev.", "Harv. L. Rev.", "Yale L.J.", "Colum. L. Rev.",
    "L.J.", "L. Rev.", "L.R.", "Sask. L.R.", "M.P.L.R.", "M.V.R.",
    "Ex. C.R.", "N.J.Eq.", "Nfld. & P.E.I.R.", "P.E.I.R.", "Man. R.",
    "W.C.B.", "Cir.", "I.L.R.", "C.P.C.", "R.R.A.", "K.B.",
    "R.P.R.",
}

COURTS = {
    "SCC", "FCA", "FC", "TCC", "BCCA", "BCSC", "ABCA", "ABQB", "SKCA",
    "SKQB", "MBCA", "MBQB", "ONCA", "ONSC", "QCCA", "QCCS", "QCCQ",
    "NBCA", "NBQB", "NSSC", "NSCA", "PECA", "PEISC", "NLCA", "NLTD",
    "NLSCTD", "YKCA", "YKSC", "NTCA", "NTSC", "NUCA", "NUCJ", "CMAC",
    "BCHRT", "HRTO", "CHRC", "CHRT", "NBOMB", "ON IPC",
}

STATUTE_PREFIXES = [
    "R.S.C.", "S.C.", "R.S.O.", "S.O.", "R.S.Q.", "L.R.Q.", "R.R.Q.",
    "L.Q.", "S.Q.", "R.S.N.B.", "S.N.B.", "S.N.", "R.S.N.", "R.S.A.",
    "S.A.", "S.B.C.", "R.S.B.C.", "S.S.", "S.M.", "R.S.M.", "C.C.Q.",
    "S.N.S.", "R.S.N.S.", "S.N.L.", "R.S.N.L.", "S.P.E.I.", "S.Y.",
    "R.S.Y.", "S.N.W.T.", "S.Nu.", "C.R.C.", "L.R.C.", "C.c.Q.", "C.p.c.",
    "C.cr.", "C.t.", "C.C.P.", "C.m.", "RSNB", "CCSM", "C.C.S.M.",
]

CIT_CODE = re.compile(r"\b\d+\s+(?:C\.C\.P\.|C\.c\.Q\.|C\.p\.c\.|C\.cr\.|C\.t\.|C\.m\.)")

ACRONYM_GAZETTEER = {
    "PSLRA", "JRPA", "ATA", "CRA", "FMIOA", "VCDR", "CCP", "CCQ",
    "CHRC", "CUPE", "CUP", "LRA", "OHSA", "HRA",
}

CONNECTORS = {
    "of", "the", "de", "la", "des", "les", "du", "van", "von", "der",
    "le", "et", "al", "and", "&", "for", "on", "en", "au", "aux", "a",
    "d'", "l'", "par", "pour", "dans", "sur", "avec", "sans", "sous",
    "chez", "vers", "entre", "to", "into",
}
CONNECTORS_LOWER = {c.lower() for c in CONNECTORS}

# Words that may appear at the edge of a case title but are not surnames.
HARVEST_STOP = CONNECTORS_LOWER | {
    "the", "a", "an", "city", "town", "ship", "village", "county", "state",
    "district", "board", "school", "company", "companies", "co", "corp",
    "ltd", "inc", "brothers", "sons", "department", "minister", "deputy",
    "queen", "king", "crown", "attorney", "general", "her", "his",
    "majesty", "royal", "national", "revenue", "customs", "excise",
    "commissioner", "commission", "minister", "ministry", "tribunal",
    "court", "tribunals", "courts", "agency", "service", "services",
    "council", "association", "society", "bank", "appeal", "education",
    "canada", "ontario", "quebec", "defendants", "plaintiffs",
    "appellants", "respondents", "applicant", "intervenor", "intervener",
    "interveners", "judgment", "conseil", "ministre", "ministère",
    "procureur", "gouvernement", "école", "ville", "municipalité",
    "habitants", "citoyens", "fédération", "confédération", "ordre",
    "syndicat", "toronto", "montreal", "vancouver", "ottawa", "calgary",
    "edmonton", "winnipeg", "halifax", "victoria", "saskatoon", "regina",
    "fredericton", "charlottetown", "yellowknife", "whitehorse", "iqaluit",
    "columbia", "manitoba", "saskatchewan", "nova", "scotia",
    "newfoundland", "labrador", "brunswick", "prince", "edward", "alberta",
    "british", "canadian", "new", "imperial", "federal", "provincial",
    "superior", "supreme", "public", "groupe", "group", "bureau",
    "procureure", "ibid", "scc", "qcca", "onca", "fca", "tcc", "bcca",
    "bcsc", "abca", "abqb", "skca", "skqb", "mbca", "mbqb", "onsc",
    "qccs", "qccq", "nbca", "nbqb", "nssc", "nsca", "peca", "nlca",
    "london", "paris", "english", "french", "anglais", "francais",
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december", "janvier",
    "février", "mars", "avril", "mai", "juin", "juillet", "août",
    "septembre", "octobre", "novembre", "décembre", "trial", "division",
    "divisions", "food", "drug", "drugs", "director", "directors",
    "employment", "immigration", "welfare", "this", "that", "these",
    "those",
}

# Procedural roles: a "party" that is only a role is not a case title.
ROLES = {
    "intervenor", "interveners", "intervener", "appellant", "appellants",
    "respondent", "respondents", "applicant", "applicants", "plaintiff",
    "plaintiffs", "defendant", "defendants", "petitioner", "petitioners",
    "claimant", "claimants", "party", "parties", "intimé", "intimée",
    "appelant", "appelante", "requérant", "requérante", "intervenant",
    "intervenante", "mis",
}


# =============================================================================
# Regexes
# =============================================================================

_rep_alt = "|".join(sorted((re.escape(r) for r in REPORTERS), key=len, reverse=True))
_court_alt = "|".join(sorted((re.escape(c).replace(r"\ ", r"\s+") for c in COURTS), key=len, reverse=True))
_stat_alt = "|".join(sorted((re.escape(s) for s in STATUTE_PREFIXES), key=len, reverse=True))

CIT_SCR = re.compile(r"\[\s*\d{4}\s*\]\s*\d+\s*S\.?\s*C\.?\s*R\.?\s*\d+[a-z]?")
CIT_NEUTRAL = re.compile(r"\b(?:19|20)\d{2}\s+(?:" + _court_alt + r")\s+\d+[A-Z]?\b")
CIT_REPORT = re.compile(
    r"(?:[\(\[]\d{4}[\)\]]\s*,?\s*)?"
    r"\b\d{1,3}\s+(?:" + _rep_alt + r")(?:\s*\(\d+(?:[a-z]{2}|\s+[A-Z][A-Za-z.]*\.?)\))?"
    r"\s+\d+[a-z]?(?:\s*,\s*\d+)?"
    r"(?:\s*\((?=[^()]*\.)[^()]{1,40}\))?"
)
# reporter + page without volume ("Ex. C.R. 828")
CIT_NOVOL = re.compile(
    r"(?:[\(\[]\d{4}[\)\]]\s*,?\s*)?"
    r"\b(?:" + _rep_alt + r")\s+\d+[a-z]?"
    r"(?:\s*\((?=[^()]*\.)[^()]{1,40}\))?"
)
# docket-style ("[1997] O.J. No. 1548", "40 N.B.J. No. 327 (QL)", "[2008] OJ 1799")
CIT_DOCKET = re.compile(
    r"(?:\[\d{4}\]\s*)?\b(?:O\.J\.|N\.B\.J\.|M\.J\.|S\.J\.|A\.J\.|B\.C\.J\.|N\.S\.J\.|[A-Z][A-Za-z.]*J\.)\s+No\.\s*\d+"
    r"(?:\s*\(QL\))?"
    r"|\[\d{4}\]\s+(?:OJ|QL)\s+\d+"
)
CIT_US = re.compile(r"\b\d+\s+(?:U\.S\.|S\.\s*Ct\.)\s*\d+(?:\s*\(\d{4}\))?")
CIT_STATUTE = re.compile(
    r"\b(?:" + _stat_alt + r")\s*,?\s*\d{4}(?:\s*[-\u2010-\u2015]\s*\d{2,4})?"
    r"(?:\s*,?\s*c\.\s*[\w][\w.-]*)?(?:\s*,?\s*r\.\s*[\w][\w.-]*)?"
    r"|\b(?:" + _stat_alt + r")\s*,?\s*\d{4}(?:\s*[-\u2010-\u2015]\s*\d{2,4})?"
    r"\s*,?\s*c\s+(?=[A-Z]-\d|[A-Z]\d)[\w][\w.-]*(?:\s*,?\s*r\.\s*[\w][\w.-]*)?"
    r"|\b(?:" + _stat_alt + r")\s*,\s*c\.\s*[\w][\w.-]*(?:\s*,?\s*r\.\s*[\w][\w.-]*)?"
    r"|\b(?:" + _stat_alt + r")\s+c\s+(?=[A-Z]-\d|[A-Z]\d)[\w][\w.-]*"
)
CIT_JOURNAL = re.compile(
    r"(?:\(\d{4}(?:-\d{2,4})?\)\s*,\s*)?"
    r"\b\d+\s+(?:" + _rep_alt + r")\s+\d+(?:\s*[-\u2010-\u2015]\s*\d+)?"
)
ED_ANCHOR = re.compile(r"\(\d+(?:st|nd|rd|th) ed\.[^()]{0,20}\d{4}\)")
PLACE_PUB = re.compile(
    r"\b[A-Z][A-Za-z.'’\-]*(?:\s+[A-Z][A-Za-z.'’\-]*){0,2}:\s+"
    r"[A-Z][^,;()]{2,60}?,\s*\d{4}\b"
)

_FRAG_LEADS = (
    r"s{1,2}\.|ss\.|subs\.|paras?\.|pp?\.|Section|section|subsection|Subsection"
    r"|paragraph|Paragraph|article|Article|art\.|al\.|par\.|Part|Schedule"
    r"|Division|Appendix|Annex"
)
_PIN_DIG = (
    r"\d+(?:\.\d+)*[a-z]?(?:\([^()]*\))?"
    r"(?:\s*[-\u2010-\u2015]\s*\d+(?:\.\d+)*[a-z]?(?:\([^()]*\))?)?"
    r"(?:\s+and\s+\d+(?:\.\d+)*)?"
)
_PIN_ROM = (
    r"(?:\d+(?:\.\d+)*|[IVXLCDM]{1,6})[a-z]?(?:\([^()]*\))?"
    r"(?:\s*[-\u2010-\u2015]\s*(?:\d+(?:\.\d+)*|[IVXLCDM]{1,6})[a-z]?(?:\([^()]*\))?)?"
)
_ROMAN_LEADS = r"Part|Schedule|Division|Appendix|Annex"
_PIN = _PIN_DIG
FRAG_RE = re.compile(
    r"\b(?:" + _FRAG_LEADS + r")\s*" + _PIN_DIG
    + r"|\b(?:" + _ROMAN_LEADS + r")\s*" + _PIN_ROM
)

SUPRA_RE = re.compile(r",\s*supra\b", re.I)
AT_TAIL_RE = re.compile(
    r",\s*at\s+(?:(?:paras?\.|paragraph|pp?\.|p\.|s\.)\s*" + _PIN_DIG + r"(?:\s*,\s*\d+)?"
    r"|pages?\s+\d+(?:\s*[-–—]\s*\d+)?)"
)
COURT_PAREN_RE = re.compile(r"\s*\((?=[^()]*\.)[^()]{1,40}\)")
QUAL_PAREN_RE = re.compile(
    r"\s*\((?:Re|Town|City|Village|County|District|judge alone|en banc|QL"
    r"|dissent|majority|concurring)(?:[^()]*)\)"
)
SUPRA_WORD_RE = re.compile(r"\bsupra\b", re.I)


def _rstrip_term(text: str, end: int, start: int = 0) -> int:
    """Strip trailing sentence punctuation (., ;, :) from a span end."""
    while end > start and text[end - 1] in ".;:":
        # keep dotted abbreviations (C.C.P., U.S.): dot preceded by a capital
        if text[end - 1] == "." and end >= 2 and text[end - 2].isupper():
            break
        end -= 1
    return end


def _rstrip_title(text: str, end: int, start: int = 0) -> int:
    """Strip a trailing sentence period from title/author spans.

    Keeps abbreviation dots (Ltd./Co./R./St./U.S.): only strips when the
    final token is long, has a single dot, and is not a known abbreviation.
    """
    while end > start and text[end - 1] in ".;:":
        if text[end - 1] in ";:":
            end -= 1
            continue
        m = re.search(r"(\S+)\.$", text[start:end])
        if not m:
            break
        tok = m.group(1)
        clean = tok.strip("()[]{}'’\"“”.")
        if clean in Detector._ABBREVS or tok.count(".") >= 2 \
                or (len(clean) <= 2 and not clean.isdigit()):
            break
        end -= 1
    return end


def _fold(s: str) -> str:
    """Lowercase + accent-stripped comparison form."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", s.lower()) if not unicodedata.combining(c)
    )


HARVEST_STOP_FOLDED = {_fold(w) for w in HARVEST_STOP}


# Honorifics / titles: never part of a case name span.
HONORIFICS = {
    "mr", "mrs", "ms", "m", "mme", "me", "mtre", "hon", "honourable",
    "honorable", "justice", "judge", "juge", "chief", "dr", "dre",
    "master", "prothonotary",
}

# Lowercase abbreviations allowed inside titles (don't end a tentative run).
LOW_ABBREVS = {
    "inc", "ltd", "corp", "llc", "co", "cie", "ltee", "sa", "sarl",
    "art", "arts", "eds", "ed", "vol", "no", "st", "ave", "blvd", "ste",
    "al", "etc", "viz", "vs",
}


DICT_CORES = [
    r"Black'?s Law Dictionary",
    r"Shorter Oxford English Dictionary",
    r"Oxford English Dictionary",
    r"Funk\s*&\s*Wagnalls",
    r"Webster'?s",
    r"Houghton Mifflin",
    r"Petit Robert",
    r"Gage Canadian",
    r"Canadian Law Dictionary",
]
DICT_RE = re.compile(
    r"\b(?i:(?:The|Le|La)\s+)?(?:" + "|".join(DICT_CORES) + r")"
)


# =============================================================================
# Span model
# =============================================================================

@dataclass
class Child:
    start: int
    end: int
    kind: str  # title | citation | fragment | source | authors


@dataclass
class Parent:
    start: int
    end: int
    label: str
    children: List[Child] = field(default_factory=list)
    prio: int = 1
    src: str = ""


def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
    return a_start < b_end and b_start < a_end


# =============================================================================
# Detector
# =============================================================================

class Detector:
    def __init__(self, text: str):
        self.text = text
        self.parents: List[Parent] = []
        self.case_surnames: Set[str] = set()
        self.author_surnames: Set[str] = set()
        self.acronyms: Dict[str, str] = {}
        self.case_acronyms: Set[str] = set()
        self.acronym_ctx: Dict[str, str] = {}  # acronym -> defining parent label
        self.paren_terms: Dict[str, str] = {}  # "(the X)" term -> defining parent label
        self.defined_terms: Set[str] = set()  # hereinafter/quoted defined phrases
        self.short_licensed: Set[str] = set()  # words used in short-form refs
        self.phrase_cands: Set[str] = set()  # party-head phrases (folded)
        self.phrase_labels: Dict[str, str] = {}  # licensed phrase -> parent label
        self.acronym_defpos: Dict[str, List[int]] = defaultdict(list)
        self._statute_ranges: List[Tuple[int, int]] = []

    # -- helpers ---------------------------------------------------------

    def _is_free(self, s: int, e: int) -> bool:
        return not any(_overlaps(s, e, p.start, p.end) for p in self.parents)

    def _add(self, p: Parent):
        try:
            p.src = sys._getframe(1).f_code.co_name
        except Exception:
            p.src = "?"
        self.parents.append(p)

    @staticmethod
    def _clean_tok(tok: str) -> str:
        return tok.strip(" \t\n\r\xa0,\";:!?()[]{}'’\"“”.")

    _ABBREVS = {"R", "St", "No", "Jr", "Sr", "J", "C", "Ltd", "Inc", "Co", "Esq",
                "Corp", "Imp", "Bros", "Dept", "Assn", "Intl", "Fed", "Rep",
                "Sen", "Gen", "Col", "Messrs", "Mlle", "Ste", "Ave", "Blvd",
                "Rd", "Vol", "Ed", "Eds", "Art", "Sec", "Para", "Chap"}

    _TRIM_START = {
        "what", "when", "where", "which", "who", "whom", "whose", "why",
        "how", "this", "that", "these", "those", "it", "they", "them",
        "their", "there", "here", "such", "un", "une", "des", "ces",
        "cette", "son", "sa", "ses", "leur", "leurs", "notre", "nos",
        "votre", "vos", "que", "qui", "quoi", "dont", "où", "comme",
        "tout", "toute", "tous", "toutes", "plus", "moins", "in",
        "see", "also", "cf", "eg", "ie",
    }

    def _is_name_tok(self, tok: str) -> bool:
        if re.search(r"[\[\];]", tok):
            return False  # footnotes ([3]), cite tails (; ...) end a title
        c = self._clean_tok(tok)
        if not c:
            return False
        if c in CONNECTORS_LOWER or c == "&":
            return True
        if c in self._ABBREVS:
            return True
        if re.match(r"^[A-ZÀ-Þ]", c):
            return True
        if re.match(r"^\d", c):
            return True
        return False

    @staticmethod
    def _is_allcaps(tok: str) -> bool:
        c = tok.strip(" \t\n\r\xa0,\";:!?()[]{}'’\"“”.")
        return len(c) >= 2 and c.isupper() and c.isalpha()

    def _tok_class(self, tok: str) -> str:
        """Classify a token for title expansion: solid | tent | stop."""
        if re.search(r"[\[\];]", tok):
            return "stop"  # footnotes ([3]), cite tails (; ...) end a title
        c = self._clean_tok(tok)
        if not c:
            return "stop"
        if _fold(c) in HONORIFICS:
            return "stop"  # Mr./Me/Mtre/Justice/Juge... never in a title
        if c[:1].isupper() or c in CONNECTORS_LOWER or c == "&" \
                or c[:1].isdigit() or c in self._ABBREVS:
            return "solid"
        if len(c) >= 3 and (not tok.rstrip("\"'“”").endswith(".") or c in LOW_ABBREVS):
            return "tent"  # lowercase long word: French adjectives etc.
        return "stop"

    def _trim_left_run(self, run: List[Tuple[str, str]], drop_articles: bool = False) -> int:
        """Drop far-side tentative/connector/role/start-word tokens."""
        n = 0
        for tok, cls in reversed(run):  # run[0] is nearest the core
            c = self._clean_tok(tok)
            fc = _fold(c)
            if cls == "tent" or c in CONNECTORS_LOWER or c == "&" \
                    or self._is_allcaps(tok) or fc in ROLES or fc in self._TRIM_START \
                    or (drop_articles and fc in {"the", "a", "an", "le", "la", "les"}):
                n += 1
            else:
                break
        return n

    def _paren_group_ahead(self, toks, idx: int) -> Optional[List[str]]:
        """Tokens of the paren group starting at toks[idx], or None if it
        doesn't close within 10 tokens."""
        depth = 0
        group = []
        for k in range(idx, min(idx + 10, len(toks))):
            t = toks[k][0]
            depth += t.count("(") - t.count(")")
            group.append(t)
            if depth <= 0:
                return group
        return None

    def _paren_left_ok(self, toks, i: int) -> bool:
        group = self._paren_group_ahead(toks, i)
        if not group:
            return False
        inner = " ".join(group)
        words = re.findall(r"[A-Za-zÀ-ÿ][\w'’\-]*", inner)
        if not words:
            return False
        if re.search(r"\d", inner):
            return False  # years/cites attach on the right
        if any(t.count(".") >= 2 for t in group):
            return False  # (Ont. C.A.), (2 Beasl.)
        if len(group) == 1:
            return not (words[0].isupper() and len(words[0]) > 1)
        if all(re.match(r"^[a-zà-ÿ]", w) for w in words):
            return False  # (en banc)
        return any(re.match(r"^[A-ZÀ-Þ]", w) for w in words)

    def _expand_left(self, pos: int, drop_articles: bool = False) -> int:
        start = max(0, pos - 1500)  # titles are short; bound the scan
        toks = [(m.group(0), start + m.start(), start + m.end())
                for m in re.finditer(r"\S+", self.text[start:pos])]
        i = len(toks) - 1
        run: List[Tuple[str, str]] = []  # (tok, cls) from core outward
        caps_run = 0
        while i >= 0 and len(run) < 18:
            tok, a, b = toks[i][0], toks[i][1], toks[i][2]
            if tok and tok[0] in "\"'“”":
                j = i - 1
                while j >= 0 and not toks[j][0].rstrip(",;:").endswith(("\"", "'", "“", "”")):
                    if not self._is_name_tok(toks[j][0]):
                        break
                    j -= 1
                if j >= 0 and toks[j][0].rstrip(",;:").endswith(("\"", "'", "“", "”")):
                    run.append((tok, "solid"))
                    i = j - 1
                    caps_run = 0
                    continue
                break
            if tok and tok[0] == "(":
                # include titular parenthesized qualifiers: (City), (Re),
                # (Guardian ad litem of); but not cite tags (Ont. C.A.),
                # series (2 Beasl.), or bare years (handled as citations)
                if self._paren_left_ok(toks, i):
                    run.append((tok, "solid"))
                    i -= 1
                    caps_run = 0
                    continue
                break
            cls = self._tok_class(tok)
            if cls == "stop":
                break
            # stop before citations ("1 S.C.R. 69 and Manitoba ..." -> Manitoba ...)
            if any(pat.match(self.text, a) for pat in (CIT_SCR, CIT_NEUTRAL, CIT_REPORT, CIT_STATUTE)):
                break
            if tok.rstrip("\"'“”").endswith((",", ";", ":")):
                # boundary token (e.g. "Cited:", "562;"): stop AFTER it
                break
            if self._is_allcaps(tok):
                caps_run += 1
                if caps_run >= 2:
                    break
            else:
                caps_run = 0
            run.append((tok, cls))
            i -= 1
        drop = self._trim_left_run(run, drop_articles=drop_articles)
        if len(run) - drop <= 0:
            return pos
        # consumed tokens are toks[i+1:]; dropping `drop` from the far side
        idx = i + 1 + drop
        if idx < len(toks):
            return toks[idx][1]
        return pos

    def _expand_right(self, pos: int) -> int:
        toks = [(m.group(0), pos + m.start(), pos + m.end())
                for m in re.finditer(r"\S+", self.text[pos:pos + 1500])]
        run: List[Tuple[str, int, int, str]] = []  # (tok, a, b, cls) core outward
        absorbed_year = False
        caps_run = 0
        skip_to = 0
        for idx, (tok, a, b) in enumerate(toks):
            if idx < skip_to:
                continue
            if len(run) >= 18:
                break
            if tok.startswith("(") and re.match(r"^\(\d{4}\)[,;:]?$", tok) and not absorbed_year:
                # ...but not when it belongs to a US citation ("466 U.S. 485 (1984)")
                if re.search(r"\d+\s+(?:U\.S\.|S\.\s*Ct\.)\s+\d+\s*$", self.text[pos:a]):
                    break
                absorbed_year = True
                run.append((tok, a, b, "solid"))
                break  # year absorbed into title; citations continue via tail expansion
            if tok and tok[0] in "\"'“”(":
                if tok[0] in "\"'“”":
                    # consume a balanced quoted run ("Kathy K") when the title
                    # continues after it (comma+citation, "at", year, or end)
                    def _ends_quote(tk: str) -> bool:
                        return tk.rstrip(",;:").endswith(("\"", "'", "“", "”"))
                    k = idx + 1
                    while k < len(toks) and not _ends_quote(toks[k][0]):
                        k += 1
                    if k < len(toks):
                        after = toks[k + 1][0] if k + 1 < len(toks) else ""
                        aba = toks[k + 1][1] if k + 1 < len(toks) else pos
                        ok_after = (
                            after == ""
                            or after[0] in ",;:"
                            or after.lower() in {"at", "supra"}
                            or re.match(r"^\(\d{4}\)", after)
                            or any(pat.match(self.text, aba) for pat in
                                   (CIT_SCR, CIT_NEUTRAL, CIT_REPORT, CIT_STATUTE))
                        )
                        if ok_after:
                            for t2, a2, b2 in toks[idx:k + 1]:
                                run.append((t2, a2, b2, "solid"))
                            skip_to = k + 1
                            continue
                    break
                # single- or multi-token titular qualifiers: (City), (Re),
                # (Attorney General); but not all-caps tags (QL), dotted
                # cites (S.C.C./Ont. C.A.), years, or lowercase asides (en banc)
                group = self._paren_group_ahead(toks, idx)
                if group is not None:
                    inner = " ".join(group)
                    words = re.findall(r"[A-Za-zÀ-ÿ][\w'’\-]*", inner)
                    if not re.search(r"\d", inner) \
                            and not any(t.count(".") >= 2 for t in group) \
                            and words \
                            and (len(group) == 1
                                 or any(re.match(r"^[A-ZÀ-Þ]", w) for w in words)) \
                            and not (len(group) == 1 and words[0].isupper()
                                     and len(words[0]) > 1) \
                            and not all(re.match(r"^[a-zà-ÿ]", w) for w in words):
                        for t2, a2, b2 in toks[idx:idx + len(group)]:
                            run.append((t2, a2, b2, "solid"))
                        continue
                break
            low = self._clean_tok(tok).lower()
            if low in {"at", "supra"}:
                break
            cls = self._tok_class(tok)
            if cls == "stop":
                break
            if self._is_allcaps(tok):
                caps_run += 1
                if caps_run >= 2:
                    break
            else:
                caps_run = 0
            # stop before citations (tail expansion handles them)
            if any(pat.match(self.text, a) for pat in (CIT_SCR, CIT_NEUTRAL, CIT_REPORT, CIT_STATUTE)):
                break
            run.append((tok, a, b, cls))
            if tok[-1] in ",;:":
                # comma: maybe a multi-party title ("A, B v. C (Year)")?
                if tok[-1] == "," and self._comma_continues(b):
                    continue
                break
        # trim trailing tentative/connector tokens
        while run:
            tok, a, b, cls = run[-1]
            c = self._clean_tok(tok)
            if cls == "tent" or c in CONNECTORS_LOWER or c == "&":
                run.pop()
            else:
                break
        if not run:
            return pos
        end = run[-1][2]
        if run[-1][0][-1] in ",;:":
            end -= 1
        # strip a trailing period that is sentence punctuation, not an abbreviation
        m = re.search(r"(\S+)\.$", self.text[pos:end])
        if m:
            tok = m.group(1)
            clean = tok.strip("()[]{}'’\"“”.")
            if clean not in self._ABBREVS and tok.count(".") < 2 \
                    and (len(clean) > 2 or clean.isdigit()):
                end -= 1
        return end

    def _comma_continues(self, pos: int) -> bool:
        """Lookahead: does a comma inside a title lead to more parties?

        Yes when a (Year) follows; no when a citation follows (tail handles
        it); also yes for a short name-run ending at a sentence boundary
        ("... O.S.S.T.F., District 15.").
        """
        toks = [(m.group(0), pos + m.start()) for m in re.finditer(r"\S+", self.text[pos:pos + 200])]
        run: List[str] = []
        for tok, abs_a in toks[:10]:
            if re.match(r"^\(\d{4}\)[,;:]?$", tok):
                return True
            c = self._clean_tok(tok)
            if not c:
                continue
            if c.lower() in {"at", "supra"} or tok[0] in ";:\"'“”(":
                return False
            for pat in (CIT_SCR, CIT_NEUTRAL, CIT_REPORT, CIT_NOVOL, CIT_DOCKET, CIT_US, CIT_STATUTE):
                if pat.match(self.text, abs_a):
                    return False
            if not self._is_name_tok(tok):
                return False
            run.append(tok)
            if tok[-1] == "." and len(run) >= 1:
                # sentence end after a short run: include it (unless abbreviation)
                if c not in Detector._ABBREVS and len(run) <= 6:
                    return True
                if c in Detector._ABBREVS:
                    continue
                return False
        return False

    # -- pass 1a: statute citations (masked first) ------------------------

    # -- pass 1a: statute citations (mask first, parents for leftovers later)

    def mask_statute_cites(self):
        for pat in (CIT_STATUTE, CIT_CODE):
            for m in pat.finditer(self.text):
                self._statute_ranges.append((m.start(), m.end()))

    def detect_statute_leftovers(self):
        for pat in (CIT_STATUTE, CIT_CODE):
            for m in pat.finditer(self.text):
                s, e = m.start(), m.end()
                if not self._is_free(s, e):
                    continue
                # left-expand over "Title, " (title-led may have missed it)
                tstart = self._title_before(s)
                kids = [Child(m.start(), m.end(), "citation")]
                pstart = s
                if tstart is not None:
                    tend = tstart
                    while tend < s and self.text[tend] not in ",;":
                        tend += 1
                    # strip a leading article only for single generics
                    am0 = re.match(r"(the|a|an|le|la|les)\s+", self.text[tstart:tend], re.I)
                    if am0 and len(self.text[tstart + am0.end():tend].split()) == 1 \
                            and self.text[tstart + am0.end():tend].strip().lower() in {
                                "act", "charter", "code", "regulation", "statute",
                                "provision", "clause", "charte", "loi"}:
                        tstart += am0.end()
                    kids.append(Child(tstart, tend, "title"))
                    pstart = tstart
                tail, extra = self._expand_tail(e, legislation=True, title_parens=True)
                kids.extend(extra)
                self._add(Parent(pstart, tail, "legislation", kids, prio=0))

    def _title_before(self, pos: int) -> Optional[int]:
        """Start of a 'Title, ' immediately before pos, else None."""
        seg = self.text[max(0, pos - 150):pos]
        m = re.search(r"([A-ZÀ-Þ][^,;]{2,120}?)\s*,\s*$", seg)
        if not m:
            return None
        title = m.group(1)
        words = title.split()
        if not words:
            return None
        # must look like a title: Act-word inside or >= 2 capitalized words
        caps = sum(1 for w in words if re.match(r"^[A-ZÀ-Þ]", w))
        if not re.search(r"\b(Acts?|Charters?|Codes?|Regulations?|Tariffs?|Charte|Loi)\b", title) \
                and caps < 2:
            return None
        return max(0, pos - 150) + m.start(1)

    def _expand_tail(self, pos: int, legislation: bool = False,
                     title_parens: bool = False, year_source: bool = False
                     ) -> Tuple[int, List[Child]]:
        """Absorb trailing citations / pinpoints / court parens into a parent span.

        Returns (end, extra_children). In legislation parents, trailing
        fragments (", subs.54(2) and s. 60") and "(the Act)"-style parens
        also yield children.
        """
        end = pos
        extra: List[Child] = []
        while True:
            m = re.match(r"\.?,\s*", self.text[end:end + 7])
            if m:
                nxt = end + m.end()
                chained = False
                chain_pats = (CIT_SCR, CIT_NEUTRAL, CIT_REPORT, CIT_NOVOL, CIT_DOCKET, CIT_US)
                if legislation:
                    chain_pats = chain_pats + (CIT_STATUTE, CIT_CODE)
                for pat in chain_pats:
                    cm = pat.match(self.text, nxt)
                    if cm and self._is_free(nxt, cm.end()):
                        end = cm.end()
                        chained = True
                        break
                if chained:
                    continue
                if legislation:
                    fm = FRAG_RE.match(self.text, nxt)
                    if fm:
                        extra.append(Child(fm.start(), fm.end(), "fragment"))
                        end = fm.end()
                        am = re.match(r"\s+and\s+", self.text[end:end + 8])
                        if am:
                            fm2 = FRAG_RE.match(self.text, end + am.end())
                            if fm2:
                                extra.append(Child(fm2.start(), fm2.end(), "fragment"))
                                end = fm2.end()
                        continue
                # ", para. 6" pinpoint tails (no "at")
                pm = re.match(r"(paras?\.|paragraph|pp?\.|p\.|art\.|s\.)\s*" + _PIN, self.text[nxt:nxt + 40])
                if pm:
                    end = nxt + pm.end()
                    continue
                at = AT_TAIL_RE.match(self.text, end)
                if at:
                    end = at.end()
                    continue
                # ", write at pp. 316-17" (verb between comma and at)
                va = re.match(r",\s*[a-zà-ÿ]+\s+at\s+", self.text[end:end + 30])
                if va:
                    vm = re.match(
                        r"(?:paras?\.|paragraph|pp?\.|p\.|page)\s*" + _PIN,
                        self.text[end + va.end():end + va.end() + 30],
                    )
                    if vm:
                        end = end + va.end() + vm.end()
                        continue
                # bare-year tails (", 1944"; a source in secondary mentions)
                ym = re.match(r"(19|20)\d{2}\b", self.text[nxt:nxt + 6])
                if ym and not any(pat.match(self.text, nxt) for pat in
                                  (CIT_SCR, CIT_NEUTRAL, CIT_REPORT, CIT_NOVOL, CIT_STATUTE)):
                    if year_source:
                        extra.append(Child(nxt, nxt + ym.end(), "source"))
                    end = nxt + ym.end()
                    continue
                sm = SUPRA_RE.match(self.text, end)
                if sm:
                    end = sm.end()
                    continue
                cp = COURT_PAREN_RE.match(self.text, nxt)
                if cp:
                    end = cp.end()
                    continue
                qp = QUAL_PAREN_RE.match(self.text, nxt)
                if qp:
                    end = qp.end()
                    continue
                break
            cp = COURT_PAREN_RE.match(self.text, end)
            if cp:
                end = cp.end()
                continue
            qp = QUAL_PAREN_RE.match(self.text, end)
            if qp:
                end = qp.end()
                continue
            at = re.match(r"\s+at\s+(?:paras?\.|paragraph|pp?\.|p\.|s\.)\s*" + _PIN_DIG,
                          self.text[end:end + 40], re.I)
            if at:
                m_end = end + at.end()
                # extend over a second page number (", 82")
                m2 = re.match(r"\s*,\s*\d+", self.text[m_end:m_end + 8])
                if m2:
                    m_end += m2.end()
                end = m_end
                continue
            if title_parens:
                pm = re.match(r"\s*\((the|this|said|such)\s+([^()]{1,25})\)", self.text[end:end + 40], re.I)
                if pm:
                    inner = self.text[end + pm.start(2):end + pm.end(2)]
                    if any(q in inner for q in '"“”'):
                        # "(the "Act")" -> title is Act
                        words = re.findall(r"[A-Za-z][\w'’\-]*", inner)
                        if words:
                            w = words[-1]
                            wi = inner.rfind(w)
                            extra.append(Child(end + pm.start(2) + wi,
                                               end + pm.start(2) + wi + len(w), "title"))
                            end += pm.end()
                            continue
                    extra.append(Child(end + pm.start(2), end + pm.end(2), "title"))
                    end += pm.end()
                    continue
            fm = re.match(r"\[\d+\]", self.text[end:end + 8])
            if fm and end > 0 and self.text[end - 1] != "." \
                    and re.match(r"\s*\(", self.text[end + fm.end():end + fm.end() + 3]):
                end += fm.end()
                continue
            break
        return end, extra

    # -- pass 1b: full decision mentions ----------------------------------

    def detect_decisions(self):
        for m in re.finditer(r"\b(?P<vc>v\.|c\.|v(?=\s))\s+(?P<r1>[A-ZÀ-Þ][\w'’.\-]*)", self.text):
            vc = m.group("vc")
            if vc == "v" and (m.start() > 0 and self.text[m.start() - 1] == "-"):
                continue  # hyphenated prose ("rendez-vous"), not versus
            core_s, core_e = m.start(), m.end()
            if any(_overlaps(core_s, core_e, s, e) for s, e in self._statute_ranges):
                continue
            if vc == "c.":
                r1 = m.group("r1")
                if re.match(r"^[A-Z]-\d", r1):
                    continue
                pre = self.text[max(0, core_s - 24):core_s]
                if re.search(r"(R\.S\.|S\.C\.|L\.R\.|R\.R\.|S\.S\.|L\.Q\.)\s*$", pre):
                    continue
            left = self._expand_left(core_s)
            right = self._expand_right(core_e)
            if right <= left:
                continue
            # multi-party continuation: ", Ft. W. & C. R. Co. (1910)"
            cont = self._party_paren_ahead(right)
            if cont is not None:
                right = cont
            # qualifier parens belong to the title: "Caraquet (Town) (Re)"
            while True:
                qm = re.match(r"\s*\((?:Re|Town|City|Village|County|District)\)",
                              self.text[right:right + 20])
                if not qm:
                    break
                right += qm.end()
            title = self.text[left:right]
            if len(re.findall(r"[A-Za-zÀ-Þ]", title)) < 3:
                continue
            if not re.match(r"[A-ZÀ-Þ\(\"\d]", title.strip()):
                continue  # title must start with a capital/paren/digit
            # reject procedural-role "parties" (headers, not cases)
            sides = re.split(r"\s+(?:v\.|c\.|v(?=\s))\s+", title, maxsplit=1)
            if len(sides) == 2:
                lp = re.findall(r"[\w'’\-]+", sides[0])
                rp = re.findall(r"[\w'’\-]+", sides[1])
                if (lp and lp[-1].lower() in ROLES) or (rp and rp[0].lower() in ROLES):
                    continue
                # parties must not start/end with bare connectors ("City of v. Boyd"),
                # and must contain a real word (not just "Inc." or initials)
                _bare_conn = CONNECTORS_LOWER - {"the", "a", "an", "le", "la", "les",
                                                 "un", "une", "l'", "d'"}
                if (lp and _fold(lp[-1]) in _bare_conn) \
                        or (rp and _fold(rp[0]) in _bare_conn):
                    continue
                if not self._side_ok(sides[0]) or not self._side_ok(sides[1]):
                    continue
            if not self._is_free(left, right):
                continue
            end, _ = self._expand_tail(right)
            kids = [Child(left, right, "title")]
            for pat in (CIT_SCR, CIT_NEUTRAL, CIT_REPORT, CIT_NOVOL, CIT_DOCKET, CIT_US):
                for cm in pat.finditer(self.text, right, end):
                    kids.append(Child(cm.start(), cm.end(), "citation"))
            for fm in FRAG_RE.finditer(self.text, right, end):
                if not any(_overlaps(fm.start(), fm.end(), k.start, k.end) for k in kids):
                    kids.append(Child(fm.start(), fm.end(), "fragment"))
            self._add(Parent(left, end, "decision", kids, prio=0))
            self._harvest_surnames(title)

    def _party_paren_ahead(self, pos: int) -> Optional[int]:
        """End offset of a ", Second Party ... (Year)" continuation, else None."""
        cm = re.match(r",\s*", self.text[pos:pos + 4])
        if not cm:
            return None
        toks = [(m.group(0), pos + cm.end() + m.start(), pos + cm.end() + m.end())
                for m in re.finditer(r"\S+", self.text[pos + cm.end():pos + cm.end() + 160])]
        end = None
        for tok, a, b in toks[:10]:
            if re.match(r"^\(\d{4}\)[,;:]?$", tok):
                return b - (1 if tok[-1] in ",;:" else 0)
            if not self._is_name_tok(tok):
                return None
            if tok[-1] in ";:\"'“”":
                return None
            if tok[-1] == ",":
                continue
            end = b
        return None

    @staticmethod
    def _side_ok(side: str) -> bool:
        """A party must contain a real word (not only abbreviations/initials),
        except the canonical Crown respondent "R."."""
        if side.strip() in {"R.", "R"}:
            return True
        for w in re.findall(r"[A-Za-zÀ-Þ'’\-‑–—]+", side.replace(".", "")):
            if len(w) >= 3 and w not in Detector._ABBREVS:
                return True
        return False

    def _harvest_surnames(self, title: str):
        parts = re.split(r"\s+(?:v\.|c\.)\s+", title)
        for part in parts:
            words = re.findall(r"[A-ZÀ-Þ][\w'’\-‑–—]*", part)
            if not words:
                continue
            for w in (words[0], words[-1]):
                c = w.strip(".,;:")
                if len(c) >= 3 and _fold(c) not in HARVEST_STOP_FOLDED:
                    self.case_surnames.add(_fold(c))
            # party-head phrases (first 3 words, or full party if <= 4)
            for n in (3, 4):
                if len(words) >= n and (n == 3 or len(words) == 4):
                    phrase = " ".join(words[:n])
                    fwords = [_fold(x.strip(".,;:")) for x in phrase.split()]
                    if all(f not in HARVEST_STOP_FOLDED - CONNECTORS_LOWER for f in fwords):
                        self.phrase_cands.add(" ".join(fwords))

    # -- pass 1c: lone citations -> parents -------------------------------

    def detect_lone_cites(self):
        for pat in (CIT_SCR, CIT_NEUTRAL, CIT_REPORT, CIT_NOVOL, CIT_DOCKET, CIT_US):
            for m in pat.finditer(self.text):
                s, e = m.start(), m.end()
                if not self._is_free(s, e):
                    continue
                pre = re.search(r"[\(\[]\d{4}[\)\]]\s*,?\s*$", self.text[max(0, s - 12):s])
                if pre:
                    s = max(0, s - 12) + pre.start()
                kids = [Child(m.start(), m.end(), "citation")]
                end, _ = self._expand_tail(e)
                for pat2 in (CIT_SCR, CIT_NEUTRAL, CIT_REPORT, CIT_NOVOL, CIT_DOCKET, CIT_US):
                    for cm in pat2.finditer(self.text, e, end):
                        if not any(_overlaps(cm.start(), cm.end(), k.start, k.end) for k in kids):
                            kids.append(Child(cm.start(), cm.end(), "citation"))
                for fm in FRAG_RE.finditer(self.text, e, end):
                    if not any(_overlaps(fm.start(), fm.end(), k.start, k.end) for k in kids):
                        kids.append(Child(fm.start(), fm.end(), "fragment"))
                self._add(Parent(s, end, "decision", kids, prio=0))

    # -- pass 1d: legislation ---------------------------------------------

    def detect_legislation(self):
        for m in FRAG_RE.finditer(self.text):
            s, e = m.start(), m.end()
            lead = m.group(0).split()[0].rstrip(".").lower()
            if lead in {"para", "paras", "paragraph", "p", "pp"}:
                continue  # handled by the pinpoint rule
            if not self._is_free(s, e):
                continue
            pos = e
            # absorb ", Part XII, Schedule III"-style chains into the same fragment
            cm = re.match(
                r"(?:\s*,\s*(?:Part|Schedule|Division|Appendix|Annex|Section|section"
                r"|subsection|paragraph|article)\s+(?:\d+(?:\.\d+)*|[IVXLCDM]{1,6})"
                r"[a-z]?(?:\([^()]*\))?)+",
                self.text[pos:pos + 80],
            )
            if cm:
                pos += cm.end()
            kids = [Child(s, pos, "fragment")]
            # "s. 97(2.1) and s. 100.1" and-chains (possibly different leads)
            am = re.match(r"\s+and\s+", self.text[pos:pos + 8])
            if am:
                fm2 = FRAG_RE.match(self.text, pos + am.end())
                if fm2:
                    kids.append(Child(fm2.start(), fm2.end(), "fragment"))
                    pos = fm2.end()
            else:
                # hierarchical pinpoint right after ("article 355 para. 1")
                lead2 = re.match(
                    r"\s+(paras?\.|paragraph|pp?\.|p\.|al\.|par\.)\s*(?=\d|[IVXLCDM]{1,6}\b)",
                    self.text[pos:pos + 14],
                )
                if lead2:
                    ws = re.match(r"\s*", self.text[pos:pos + 6])
                    fm2 = FRAG_RE.match(self.text, pos + (ws.end() if ws else 0))
                    if fm2:
                        kids.append(Child(fm2.start(), fm2.end(), "fragment"))
                        pos = fm2.end()
            om = re.match(r"\s+of\s+(?:the\s+)?", self.text[pos:pos + 12], re.I)
            if om:
                tstart = pos + om.end()
                probe = self.text[tstart:tstart + 150]
                cut = len(probe)
                for ch in ",;(":
                    i = probe.find(ch)
                    if i >= 0:
                        cut = min(cut, i)
                head = probe[:cut]
                if re.search(r"\b(?:subsections?|paragraphs?|paras?\.|articles?|sections?)\b", head, re.I) \
                        and not re.search(r"\b(Acts?|Charters?|Codes?|Regulations?|Tariffs?)\b", head):
                    # "clause (b) of the definition of "institution" in subsection (1)"
                    # is one fragment, not a title
                    cm2 = re.match(r"((?:\s+(?:of|in)\s+(?:the\s+)?[^,;()]{1,60}){1,3})",
                                   self.text[pos:pos + 150])
                    if cm2:
                        cend = _rstrip_term(self.text, pos + cm2.end(), pos)
                        kids[0] = Child(s, cend, "fragment")
                        pos = cend
                else:
                    tend, _ = self._title_phrase_right(tstart, allow_lower=True)
                    ym = re.match(r",\s*\d{4}\b", self.text[tend:tend + 8])
                    if ym:
                        tend += ym.end()
                    if tend > tstart:
                        # strip a leading article only for single generics
                        # ("the Act" -> "Act"; "The Rural ... Act" keeps "The")
                        am2 = re.match(r"(the|a|an|le|la|les)\s+", self.text[tstart:tend], re.I)
                        if am2 and len(self.text[tstart + am2.end():tend].split()) == 1 \
                                and self.text[tstart + am2.end():tend].strip().lower() in {
                                    "act", "charter", "code", "regulation", "statute",
                                    "provision", "clause", "charte", "loi"}:
                            tstart += am2.end()
                        kids.append(Child(tstart, tend, "title"))
                        pos = tend
            # code cites without comma ("article 355 para. 1 C.C.P.")
            ws0 = re.match(r"\s*", self.text[pos:pos + 4])
            cmt0 = CIT_CODE.match(self.text, pos + (ws0.end() if ws0 else 0))
            if cmt0 and self._is_free(cmt0.start(), cmt0.end()):
                kids.append(Child(cmt0.start(), cmt0.end(), "citation"))
                pos = cmt0.end()
            cm = re.match(r",\s*", self.text[pos:pos + 4])
            if cm:
                for pat in (CIT_STATUTE, CIT_SCR, CIT_NEUTRAL, CIT_REPORT, CIT_NOVOL):
                    cmt = pat.match(self.text, pos + cm.end())
                    if cmt and self._is_free(cmt.start(), cmt.end()):
                        kids.append(Child(cmt.start(), cmt.end(), "citation"))
                        pos = cmt.end()
                        break
            end, extra = self._expand_tail(pos, legislation=True, title_parens=True)
            kids.extend(extra)
            self._add(Parent(s, end, "legislation", kids, prio=1))
        # title-led: "... Act/Charter/Code ..."
        for m in re.finditer(
            r"\b(?P<kw>Acts?|Charters?|Codes?|Regulations?|Tariffs?|Charte|Loi)\b", self.text
        ):
            kend = m.end()
            left = self._expand_left(m.start(), drop_articles=True)
            # shrink past header/role words leaking in from the left
            seg = self.text[left:m.start()]
            bad = None
            for tm in re.finditer(r"\S+", seg):
                tok = tm.group(0)
                c = self._clean_tok(tok)
                if self._is_allcaps(tok) or _fold(c) in ROLES or _fold(c) in self._TRIM_START:
                    bad = tm.end()
            if bad is not None:
                left += bad
                # skip whitespace after the bad token
                wm = re.match(r"\s*", self.text[left:m.start()])
                left += wm.end()
            words_left = self.text[left:m.start()].split()
            if len(words_left) > 9:
                continue
            if not re.search(r"[A-ZÀ-Þ]", self.text[left:m.start()]):
                continue
            right, _ = self._title_phrase_right(kend, allow_lower=True, max_lower=3)
            ym = re.match(r",\s*\d{4}\b", self.text[right:right + 8])
            if ym:
                right += ym.end()
            # drop bare "Tariff" (gold never tags it bare)
            if not self.text[left:right].strip() or (
                len(self.text[left:right].split()) == 1
                and self._clean_tok(m.group("kw")).lower() in {"tariff", "tariffs"}
            ):
                continue
            if not self._is_free(left, right):
                continue
            # strip a leading article only for single generics
            # ("the Act" -> "Act"; "The Rural ... Act" keeps "The")
            tstart = left
            am = re.match(r"(the|a|an|le|la|les)\s+", self.text[tstart:right], re.I)
            if am and len(self.text[tstart + am.end():right].split()) == 1 \
                    and self.text[tstart + am.end():right].strip().lower() in {
                        "act", "charter", "code", "regulation", "statute",
                        "provision", "clause", "charte", "loi"}:
                tstart += am.end()
            # drop bare "Tariff" (gold never tags it bare; Act/Charter/Code
            # *are* tagged bare, so those stay)
            if not self.text[tstart:right].strip() or (
                len(self.text[tstart:right].split()) == 1
                and self._clean_tok(m.group("kw")).lower() in {"tariff", "tariffs"}
            ):
                continue
            if not self._is_free(tstart, right):
                continue
            left = tstart  # bare single generics exclude the article too
            kids = [Child(tstart, right, "title")]
            end = right
            dm = re.match(
                r"\s*\(\s*[\"“]?([A-Z][A-Z0-9.]{1,7})[\"”]?(?:\s+or\s+[\"“]?([A-Z][A-Za-z]*?)[\"”]?)?\s*\)",
                self.text[end:end + 40],
            )
            if dm:
                for g in dm.groups():
                    if g:
                        key = re.sub(r"[^A-Z0-9]", "", g.upper())
                        self.acronyms[key] = self.text[left:right]
                        self.acronym_defpos[key].append(end)
                end += dm.end()
            cm = re.match(r",\s*", self.text[end:end + 4])
            if cm:
                cmt = CIT_STATUTE.match(self.text, end + cm.end())
                if cmt and self._is_free(cmt.start(), cmt.end()):
                    kids.append(Child(cmt.start(), cmt.end(), "citation"))
                    end = cmt.end()
            end, extra2 = self._expand_tail(end, legislation=True, title_parens=True)
            kids.extend(extra2)
            self._add(Parent(left, end, "legislation", kids, prio=1))

    def _title_phrase_right(self, pos: int, allow_lower: bool, max_lower: int = 6) -> Tuple[int, bool]:
        toks = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", self.text[pos:])]
        end = pos
        end_before_lower = pos
        lower_run = 0
        caps_run = 0
        hist: List[int] = []
        for tok, a, b in toks:
            a += pos
            b += pos
            c = tok.strip(" \t\n\r\xa0,\";:!?()[]{}'’\"“”.")
            if not c:
                break
            if "[" in tok or "]" in tok:
                break  # footnotes ([2]) never belong to a title
            if tok[0] == "(":
                break  # parens belong to tail expansion, not the title
            if c[0].isupper() or c in CONNECTORS_LOWER or c == "&" or re.match(r"^\(\d{4}\)$", c):
                if self._is_allcaps(tok):
                    caps_run += 1
                    if caps_run >= 2:
                        break
                else:
                    caps_run = 0
                end = b
                end_before_lower = b
                lower_run = 0
                hist.append(b)
                if len(hist) > 12:
                    end = end_before_lower = hist[-13]
                    break
                if tok[-1] in ",;:":
                    end = end_before_lower = b - 1
                    break
                # sentence period ends the phrase (but not abbreviations)
                if tok.endswith(".") and c not in self._ABBREVS and len(c) > 2 \
                        and tok.count(".") < 2:
                    end = end_before_lower = b - 1
                    break
                continue
            if allow_lower and re.match(r"^[a-zà-ÿ]", c) and lower_run < max_lower:
                if "(" in tok or ")" in tok:
                    break
                core = c.rstrip(".")
                if "." in core and core not in self._ABBREVS and not re.match(r"^[A-Z]\.$", tok):
                    break  # sentence period ends the phrase
                if lower_run == 0:
                    end_before_lower = end
                end = b
                lower_run += 1
                if tok[-1] in ",;:":
                    end -= 1
                    break
                continue
            break
        if lower_run:
            # keep the lowercase run only if followed by (, comma/semicolon or a citation
            nxt = self.text[end:end + 4]
            rest = self.text[end:]
            mm = re.match(r"\s*,?\s*", rest)
            keep = re.match(r"\s*[(,;]", nxt) or any(
                pat.match(rest, mm.end()) is not None
                for pat in (CIT_STATUTE, CIT_SCR, CIT_NEUTRAL, CIT_REPORT)
            )
            if not keep:
                end = end_before_lower
        return end, True

    # -- pass 1e: secondary sources ---------------------------------------

    def detect_secondary(self):
        _INI = r"(?:[A-Z]\.\s*+)"
        _NAME = r"[A-Z][\w'’\-]++"
        # Last, First T. | Last, F. | R. Sullivan | First Last | E.A. | single surname
        _AUTH = (
            r"(?:"
            + _NAME + r",\s*+" + _NAME + r"(?:\s*+[A-Z]\.)?+"
            + r"|" + _NAME + r",\s*+" + _INI + r"++"
            + r"|" + _INI + r"++" + _NAME
            + r"|" + r"[A-Z][a-zÀ-ÿ]++\s++" + _NAME
            + r"|" + r"(?:[A-Z]\.){2,},?"
            + r"|" + _NAME
            + r")(?:\s*+,\s*+et al\.)?+"
        )
        _journal_src = (
            r"\(\d{4}(?:-\d{2,4})?+\)\s*+,\s*+\d++\s++[\w.'’&\- ]*?\d+"
            r"|\([^()]{0,30}+\d{4}[^()]*+\)"
        )
        # articles: Authors, "Title" [(Year), Vol Journal Page]
        pat = re.compile(
            r"(?P<authors>" + _AUTH + r")"
            r"(?P<sep>\s*\.?,?\s*)[“\"](?P<title>[^”\"]{5,220}+)[”\"]"
            r"(?:\s*+(?P<source>" + _journal_src + r"))?"
        )
        for m in pat.finditer(self.text):
            s, e = m.start(), m.end()
            if not self._is_free(s, e):
                continue
            # single-word authors need a comma/dot separator ("For "..." is prose)
            amatch = m.group("authors")
            if re.fullmatch(r"[A-ZÀ-Þ][\w'’\-]+", amatch) and not re.search(r"[.,]", m.group("sep")):
                continue
            kids = [
                Child(m.start("authors"), m.end("authors"), "authors"),
                Child(m.start("title"), m.end("title"), "title"),
            ]
            if m.group("source"):
                kids.append(Child(m.start("source"), m.end("source"), "source"))
                for fm in FRAG_RE.finditer(self.text, m.start("source"), e):
                    kids.append(Child(fm.start(), fm.end(), "fragment"))
            end, _ = self._expand_tail(e, title_parens=True, year_source=True)
            for fm in FRAG_RE.finditer(self.text, e, end):
                kids.append(Child(fm.start(), fm.end(), "fragment"))
            self._add(Parent(s, end, "secondary sources", kids, prio=0))
            self._harvest_authors(m.group("authors"))
        # standalone lowercase quotes: dictionary headwords ("office") —
        # gated on dictionary/sub-verbo context (emphasis quotes are not mentions)
        for m in re.finditer(r"[“\"](?P<title>[a-zà-ÿ][\w'’\-]*?)[”\"]", self.text):
            if not self._is_free(m.start(), m.end()):
                continue
            window = self.text[max(0, m.start() - 300):m.end() + 100]
            if not (DICT_RE.search(window) or re.search(r"(?i)sub verbo|\bs\.v\.|définition|definition", window)):
                continue
            self._add(Parent(m.start(), m.end(), "secondary sources",
                             [Child(m.start("title"), m.end("title"), "title")], prio=2))
        for m in re.finditer(
            r"[“\"](?P<title>[^”\"]{5,220}+)[”\"]\s+in\s+(?P<source>"
            r"(?:\(\d{4}(?:-\d{2,4})?+\)\s*+,?\s*+)?\d++\s++[\w.'’&\- ]*?\d+"
            r"|[A-Z][A-Za-z.'’\-]*+(?:\s++[A-Z][A-Za-z.'’\-]*+){0,2}+:\s++[^,;()]{2,60}?,\s*+\d{4})",
            self.text,
        ):
            s, e = m.start(), m.end()
            if not self._is_free(s, e):
                continue
            self._add(Parent(s, e, "secondary sources",
                             [Child(m.start("title"), m.end("title"), "title"),
                              Child(m.start("source"), m.end("source"), "source")], prio=1))
        # books: Authors, Title ... (Nth ed. YYYY) / Place: Publisher, YYYY
        for m in re.finditer(
            r"(?P<authors>" + _AUTH + r")"
            r"\s*[.,]?\s*(?P<title>[A-ZÀ-Þ][^,;()]{5,120}?)"
            r"(?P<source>\(\d++(?:st|nd|rd|th) ed\.[^()]{0,24}+\)"
            r"|\([^()]{0,30}+[Ee]dition[^()]*+\)"
            r"|(?:\(\s*)?[A-Z][A-Za-z.'’\-]*+(?:\s++[A-Z][A-Za-z.'’\-]*+){0,2}+:\s++[^,;()]{2,60}?,\s*+\d{4}(?:\s*\))?)",
            self.text,
        ):
            s, e = m.start(), m.end()
            if not self._is_free(s, e):
                continue
            src = m.group("source")
            # gate "Place: Publisher, Year": publisher must be recognizable
            if re.search(r":\s*", src) and not any(
                w.lower() in PUBLISHERS for w in re.findall(r"[A-Za-zÀ-ÿ']+", src)
            ):
                continue
            # title must start with a real word, not initials ("L.S.U.C. ...")
            if re.match(r"(?:[A-Z]\.\s*)+", m.group("title")):
                continue
            kids = [
                Child(m.start("authors"), m.end("authors"), "authors"),
                Child(m.start("title"), m.end("title"), "title"),
                Child(m.start("source"), m.end("source"), "source"),
            ]
            end, extra3 = self._expand_tail(e, title_parens=True, year_source=True)
            kids.extend(extra3)
            self._add(Parent(s, end, "secondary sources", kids, prio=0))
            self._harvest_authors(m.group("authors"))

    def _harvest_authors(self, authors: str):
        for w in re.findall(r"[A-ZÀ-Þ][\w'’\-‑–—]*", authors):
            c = w.strip(".,;:")
            if len(c) < 3 or c.isupper():
                continue  # TRAFFIC/CAN-style fragments, not surnames
            fc = _fold(c)
            if fc not in HARVEST_STOP_FOLDED and fc not in HONORIFICS:
                self.author_surnames.add(fc)

    # -- pass 1f: dictionaries (secondary sources) ---------------------

    def detect_dictionaries(self):
        for m in DICT_RE.finditer(self.text):
            s = m.start()
            dpos = m.end()
            dm = re.match(r"\s+(?:dictionary|dictionnaire|thesaurus|thésaurus)\b",
                          self.text[dpos:dpos + 16], re.I)
            if dm:
                dpos += dm.end()
            e, _ = self._title_phrase_right(dpos, allow_lower=True, max_lower=4)
            if e <= s:
                continue
            if not self._is_free(s, e):
                continue
            kids = [Child(s, e, "title")]
            pos = e
            # source tails: "(1963)" -> inner; "(Place: Pub, Year)" -> inner;
            # "(Second Edition)" / ", 8th ed. ..." -> whole
            sm = re.match(
                r"\s*,?\s*(\([^()]{1,60}\)|,\s*\d+(?:st|nd|rd|th) ed\.[^,;]{1,60}?,\s*\d{4})",
                self.text[pos:pos + 80],
            )
            if sm:
                inner = sm.group(1)
                mplace = re.match(r"\(\s*([A-Z][^,;()]{1,40}?,\s*\d{4})\s*\)$", inner)
                myear = re.match(r"\(\s*(\d{4})\s*\)$", inner)
                if mplace:
                    a, b = sm.start(1), sm.end(1)
                    inner_text = self.text[pos + a:pos + b]
                    mi = re.search(re.escape(mplace.group(1)), inner_text)
                    kids.append(Child(pos + a + mi.start(), pos + a + mi.end(), "source"))
                elif myear:
                    a, b = sm.start(1), sm.end(1)
                    kids.append(Child(pos + a + 1, pos + b - 1, "source"))
                elif re.search(r"\d", inner) or re.search(r"(?i)ed\.|edition", inner):
                    kids.append(Child(pos + sm.start(1), pos + sm.end(1), "source"))
                else:
                    sm = None
            if sm:
                pos += sm.end()
            end, dextra = self._expand_tail(pos, year_source=True)
            kids.extend(dextra)
            for fm in FRAG_RE.finditer(self.text, pos, end):
                kids.append(Child(fm.start(), fm.end(), "fragment"))
            self._add(Parent(s, end, "secondary sources", kids, prio=1))

    # -- pass 1g: organization definitions + gazetteer -----------------

    def detect_halsbury(self):
        """'9 Halsbury's Laws of England (4th), para. 230' (encyclopedia)."""
        for m in re.finditer(
            r"\b\d+\s+Halsbury's Laws of England\s*\(\d+(?:th|st|nd|rd|d)\)", self.text
        ):
            if not self._is_free(m.start(), m.end()):
                continue
            kids = [Child(m.start(), m.end(), "title")]
            end, _ = self._expand_tail(m.end())
            for fm in FRAG_RE.finditer(self.text, m.end(), end):
                if not any(_overlaps(fm.start(), fm.end(), k.start, k.end) for k in kids):
                    kids.append(Child(fm.start(), fm.end(), "fragment"))
            self._add(Parent(m.start(), end, "secondary sources", kids, prio=1))

    def detect_hansard(self):
        """'Journal des débats (Hansard), 35' (parliamentary debates)."""
        for m in re.finditer(
            r"\b(?:Journal des débats|House of Commons Debates|Debates of the Senate|Hansard)"
            r"\s*(?:\([^()]*\))?\s*,?\s*\d+", self.text
        ):
            if not self._is_free(m.start(), m.end()):
                continue
            jm = re.match(
                r"(?:Journal des débats|House of Commons Debates|Debates of the Senate|Hansard)",
                self.text[m.start():],
            )
            kids = [Child(m.start(), m.start() + len(jm.group(0)), "title")]
            nm = re.search(r"\d+\s*$", self.text[m.start():m.end()])
            if nm:
                kids.append(Child(m.start() + nm.start(), m.end(), "fragment"))
            self._add(Parent(m.start(), m.end(), "secondary sources", kids, prio=1))

    def detect_report_bare(self):
        """Bare 'The Report'/'Report' (commission reports; lowercase
        'the report' prose is left alone)."""
        for m in re.finditer(r"\bThe\s+Report\b|\bReport\b", self.text):
            if not self._is_free(m.start(), m.end()):
                continue
            self._add(Parent(m.start(), m.end(), "secondary sources",
                             [Child(m.start(), m.end(), "title")], prio=3))

    _ORG_WORD = r"(?:[A-ZÀ-Þ][^,;:.()\"'“”]*?|[A-ZÀ-Þ]\.|of|the|de|la|and|&|for|on|des|du|d'|l')"

    def _org_before(self, paren_start: int) -> Optional[Tuple[int, int]]:
        """Trailing organization-name run before a paren (docket headers etc.)."""
        seg = self.text[max(0, paren_start - 200):paren_start]
        m = re.search(
            r"((?:" + self._ORG_WORD + r"\s+){1,7}[A-ZÀ-Þ][^,;:.()\"'“”]*?)\s*$", seg
        )
        if not m:
            return None
        org = m.group(1)
        caps = re.findall(r"[A-ZÀ-Þ][\w'’\-]*", org)
        if len(caps) < 2:
            return None
        if len(caps[-1].strip(".,;:")) < 2:
            return None
        s = max(0, paren_start - 200) + m.start(1)
        # strip a leading article from the span
        am = re.match(r"(the|a|an|le|la|les)\s+", org, re.I)
        if am:
            s += am.end()
        return (s, max(0, paren_start - 200) + m.end(1))

    def detect_org_defs(self):
        """'Ministry of X (the Ontario code)' / 'Association ... (AASHTO)'."""
        for m in re.finditer(r"\((?P<inner>[^()]{2,60})\)", self.text):
            inner = m.group("inner").strip()
            org_span = self._org_before(m.start())
            if org_span is None:
                continue
            os_, oe = org_span
            org = self.text[os_:oe]
            if re.search(r"\bv\.|\bc\.\s+[A-Z]|S\.?C\.?R\.?|SCC|\(\d{4}\)|\d{4}\s+[A-Z]{2,}", org):
                continue  # case/citation context, not an organization
            if not self._is_free(os_, m.end()):
                continue
            if re.fullmatch(r"[\"“]?[A-Z][A-Z0-9.]{1,7}[\"”]?", inner):
                # acronym definition: authors cover the whole span
                self._add(Parent(os_, m.end(), "secondary sources",
                                 [Child(os_, m.end(), "authors")], prio=1))
                key = re.sub(r"[^A-Z0-9]", "", inner.upper())
                if len(key) >= 2:
                    self.acronym_ctx[key] = "secondary sources"
                    self.acronym_defpos[key].append(m.start())
                continue
            # defined-term definitions must be short and end with a class word
            mt = re.match(r"(?:the|a|an)\s+(.+)$|(.+)$", inner, re.I | re.S)
            term = (mt.group(1) or mt.group(2)).strip(" \t\n\r\xa0\"'“”.,;:")
            words = term.split()
            if not words or len(words) > 6 or words[-1].lower() not in {
                "code", "act", "charter", "statute", "regulation", "clause", "provision",
            }:
                continue
            if inner.lower() in {"majority", "dissent", "judge alone", "en banc"}:
                continue
            kids = [Child(os_, oe, "authors")]
            im = re.search(re.escape(inner), self.text[os_:m.end()])
            kids.append(Child(os_ + im.start(), os_ + im.end(), "title"))
            self._add(Parent(os_, m.end(), "secondary sources", kids, prio=1))
            bare = re.sub(r"^(the|a|an)\s+", "", term, flags=re.I)
            if len(bare.split()) >= 2:
                self.paren_terms[bare] = "secondary sources"

    def detect_gazetteer(self):
        if _SEC_TITLE_RE is not None:
            for m in _SEC_TITLE_RE.finditer(self.text):
                if not self._is_free(m.start(), m.end()):
                    continue
                self._add(Parent(m.start(), m.end(), "secondary sources",
                                 [Child(m.start(), m.end(), "title")], prio=2))
        if _SEC_AUTH_RE is not None:
            for m in _SEC_AUTH_RE.finditer(self.text):
                if not self._is_free(m.start(), m.end()):
                    continue
                self._add(Parent(m.start(), m.end(), "secondary sources",
                                 [Child(m.start(), m.end(), "authors")], prio=2))

    # -- pass 2: short forms ----------------------------------------------

    def detect_shorts(self):
        # harvest defined terms: (hereinafter [referred to as] the X)
        # (only in legislation context: bare factual shorthands like
        # "goods in issue" are not annotated, but statute-defined ones are)
        for m in re.finditer(r"\(hereinafter(?: referred to as)? the ([^()]{2,50}?)\)", self.text, re.I):
            term = m.group(1).strip(" \t\n\r\xa0\"'“”.,;:")
            if not term:
                continue
            window = self.text[max(0, m.start() - 400):m.start()]
            if CIT_STATUTE.search(window) or any(
                p.label == "legislation" and p.end > m.start() - 400 for p in self.parents
            ):
                self.defined_terms.add(term)
        # harvest quoted short titles: ("Laicity Act" or "Act")
        for m in re.finditer(r'\(\s*"([^()"]{2,60})"\s+or\s+"([^()"]{2,60})"\s*\)', self.text):
            for g in m.groups():
                self.defined_terms.add(g.strip())
        # "Pfizer case (supra)" / "The Olympia ... case (supra)"
        for m in re.finditer(
            r"(?P<title>(?:The\s+)?[A-ZÀ-Þ][^,;()]{1,80}?\s+case)\s+\(supra\)", self.text
        ):
            if not self._is_free(m.start(), m.end()):
                continue
            words = m.group("title").split()
            core = [w for w in words if w.lower() not in {"the", "case"}]
            if words and words[0].lower() in {"in", "this", "that", "such", "the", "a", "an"} \
                    and not any(re.match(r"^[A-ZÀ-Þ]", w) for w in core[1:]):
                continue
            self.short_licensed.add(_fold(words[0]))
            self.phrase_labels[" ".join(_fold(m.group("title")).split())] = "decision"
            self._add(Parent(m.start(), m.end(), "decision",
                             [Child(m.start("title"), m.end("title"), "title")], prio=2))
        # "Ibid. at p. 838" / "Ibid., at p. 5" / "ibid, at paragraph 59" / "Ibid. page 21"
        for m in re.finditer(
            r"\b[Ii]bid\.?\s*,?\s*(?:at\s+)?(?P<pin>(?:paras?\.|paragraph|pp?\.|p\.|page)\s*" + _PIN + r")",
            self.text,
        ):
            if not self._is_free(m.start(), m.end()):
                continue
            label = self._preceding_label(m.start(), default="decision")
            self._add(Parent(m.start(), m.end(), label,
                             [Child(m.start("pin"), m.end("pin"), "fragment")], prio=2))
        # bare defined terms ("exemption clause", "Act" from ("Laicity Act" or "Act"))
        if self.defined_terms:
            _d_multi = {t for t in self.defined_terms if len(t.split()) > 1}
            _d_single = {t for t in self.defined_terms if len(t.split()) == 1}
            for _terms, _flags in ((_d_multi, re.I), (_d_single, 0)):
                if not _terms:
                    continue
                _terms_alt = "|".join(
                    sorted((re.escape(t) for t in _terms), key=len, reverse=True))
                for m in re.finditer(r"(?P<term>" + _terms_alt + r")", self.text, _flags):
                    if not self._is_free(m.start(), m.end()):
                        continue
                    # word boundaries on both sides
                    before = self.text[m.start() - 1] if m.start() > 0 else " "
                    after = self.text[m.end()] if m.end() < len(self.text) else " "
                    if re.match(r"[\w'’]", before) or re.match(r"[\w'’]", after):
                        continue
                    self._add(Parent(m.start(), m.end(), "legislation",
                                     [Child(m.start(), m.end(), "title")], prio=3))
        # in-doc definitions license same-label short refs:
        # ("ACR") inside legislation/decision/secondary parents,
        # "(the X [code|act|...])" paren terms, "(X code)" terms.
        ranges_by_label: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for p in self.parents:
            ranges_by_label[p.label].append((p.start, p.end))

        def _def_label(s: int, e: int) -> Optional[str]:
            for lab in ("legislation", "decision", "secondary sources"):
                if any(_overlaps(s, e, a, b) for a, b in ranges_by_label[lab]):
                    return lab
            return None

        for m in re.finditer(r"\(\s*[\"“]?([A-Z][A-Z0-9]{1,6})[\"”]?\s*\)", self.text):
            key = m.group(1)
            if len(key) >= 2:
                lab = _def_label(m.start(), m.end())
                if lab == "legislation":
                    self.acronyms.setdefault(key, "")
                    self.acronym_defpos[key].append(m.start())
                elif lab == "decision":
                    self.case_acronyms.add(key)
                    self.acronym_defpos[key].append(m.start())
                elif lab == "secondary sources":
                    self.acronym_ctx[key] = lab
                    self.acronym_defpos[key].append(m.start())
        for m in re.finditer(r"\((the|a|an)\s+([^()]{2,50}?)\)", self.text, re.I):
            raw = m.group(2).strip(" \t\n\r\xa0\"'“”.,;:")
            if not raw or raw.lower() in {"majority", "dissent", "court"}:
                continue
            words = raw.split()
            # strip the article, even for single generics ("the Act" -> "Act")
            if len(words) >= 2:
                term = raw
            elif words[-1].lower() in {
                "code", "act", "charter", "statute", "regulation", "provision", "clause",
            }:
                term = words[-1]
            else:
                continue
            lab = _def_label(m.start(), m.end())
            if lab:
                self.paren_terms[term] = lab
        for m in re.finditer(r"\(([Tt]he\s+)?([A-Za-z][^()]{1,40}?\s+[Cc]ode)\b", self.text):
            term = m.group(2).strip()
            lab = _def_label(m.start(), m.end())
            if lab:
                self.paren_terms[term] = lab
            elif re.search(r"\scode$", term):
                self.paren_terms[term] = "secondary sources"  # lowercase "code"
            else:
                self.paren_terms[term] = "legislation"
        # bare "(the X)" terms with their defining label (multiword: any case;
        # single generics: capitalized only, else prose floods like "act")
        multi = {t: l for t, l in self.paren_terms.items() if len(t.split()) > 1}
        single = {t: l for t, l in self.paren_terms.items() if len(t.split()) == 1}
        for terms, flags in ((multi, re.I), (single, 0)):
            if not terms:
                continue
            _pt_alt = "|".join(
                sorted((re.escape(t) for t in terms), key=len, reverse=True))
            for m in re.finditer(r"(?P<term>" + _pt_alt + r")", self.text, flags):
                if not self._is_free(m.start(), m.end()):
                    continue
                before = self.text[m.start() - 1] if m.start() > 0 else " "
                after = self.text[m.end()] if m.end() < len(self.text) else " "
                if re.match(r"[\w'’]", before) or re.match(r"[\w'’]", after):
                    continue
                lab = self.paren_terms[
                    next(t for t in self.paren_terms if t.lower() == m.group("term").lower())]
                self._add(Parent(m.start(), m.end(), lab,
                                 [Child(m.start(), m.end(), "title")], prio=3))
        _pin = _PIN
        for m in re.finditer(
            r"\b(?P<head>[A-ZÀ-Þ][^,;()]{1,100}?)\s*,\s*"
            r"(?P<ref>supra\b|at\s+(?:paras?\.|paragraph|pp?\.|p\.|page)\s*" + _pin + r")",
            self.text,
        ):
            s, e = m.start(), m.end()
            if not self._is_free(s, e):
                continue
            head = m.group("head").strip()
            ref = m.group("ref")
            if re.match(r"(?i)^(and|the|in|at|for|on|see|also)\b", head):
                continue
            if len(head.split()) > 8:
                continue
            if SUPRA_WORD_RE.search(ref):
                label = self._short_label(head, default="decision")
            elif "para" in ref.lower():
                label = self._short_label(head, default="decision")
            else:
                label = self._short_label(head, default=None)
                if label is None:
                    label = self._preceding_label(s, default="decision")
            if not label:
                continue  # role-word heads (Intervenant, supra) are not mentions
            self.short_licensed.add(_fold(head.split()[0].strip(",.;:")))
            self.phrase_labels[" ".join(_fold(head).split())] = label
            kind = "title" if label == "decision" else "authors"
            kids = [Child(m.start("head"), m.end("head"), kind)]
            fm = re.search(r"(?:paras?\.|paragraph|pp?\.|p\.|page)\s*" + _pin, ref)
            if fm:
                fs = m.start("ref") + fm.start()
                kids.append(Child(fs, fs + len(fm.group(0)), "fragment"))
            end, _ = self._expand_tail(e)
            for fm2 in FRAG_RE.finditer(self.text, e, end):
                if not any(_overlaps(fm2.start(), fm2.end(), k.start, k.end) for k in kids):
                    kids.append(Child(fm2.start(), fm2.end(), "fragment"))
            self._add(Parent(s, end, label, kids, prio=2))
        for m in re.finditer(r"\b(?P<head>[A-ZÀ-Þ][\w'’\-]*)\s+et al\.", self.text):
            if not self._is_free(m.start(), m.end()):
                continue
            head = m.group("head")
            if head in Detector._ABBREVS or len(head.strip(".,;:")) < 3:
                continue  # "Ltd et al." inside full titles is not a mention
            label = self._short_label(head, default="decision")
            if not label:
                continue
            self.short_licensed.add(_fold(m.group("head")))
            kind = "title" if label == "decision" else "authors"
            self._add(Parent(m.start(), m.end(), label,
                             [Child(m.start(), m.end(), kind)], prio=2))
        for m in re.finditer(r"\b[A-Z][A-Z0-9]{1,6}\b|\b(?:[A-Z]\.){2,}", self.text):
            key = re.sub(r"[^A-Z0-9]", "", m.group(0))
            if not self._is_free(m.start(), m.end()):
                continue
            if key in ACRONYM_GAZETTEER:
                self._add(Parent(m.start(), m.end(), "legislation",
                                 [Child(m.start(), m.end(), "title")], prio=2))
            elif key in self.acronyms:
                if not self._near_def(key, m.start()):
                    continue
                self._add(Parent(m.start(), m.end(), "legislation",
                                 [Child(m.start(), m.end(), "title")], prio=2))
            elif key in self.case_acronyms:
                if not self._near_def(key, m.start()):
                    continue
                self._add(Parent(m.start(), m.end(), "decision",
                                 [Child(m.start(), m.end(), "title")], prio=2))
            elif key in self.acronym_ctx:
                if not self._near_def(key, m.start()):
                    continue
                lab = self.acronym_ctx[key]
                kind = "authors" if lab == "secondary sources" else "title"
                self._add(Parent(m.start(), m.end(), lab,
                                 [Child(m.start(), m.end(), kind)], prio=2))
        # multiword bare phrases (licensed or gazetteer), longest first
        phrases: Dict[str, str] = dict(self.phrase_labels)
        for ph in BARE_PHRASES:
            phrases.setdefault(ph, "decision")
        if phrases:
            _ph_alt = "|".join(
                r"\s+".join(re.escape(w) for w in ph.split())
                for ph in sorted(phrases, key=len, reverse=True)
            )
            for m in re.finditer(
                r"(?<![\w'’\]])(?P<ph>" + _ph_alt + r")(?![\w'’\[])", self.text, re.I
            ):
                if not self._is_free(m.start(), m.end()):
                    continue
                key = " ".join(_fold(m.group("ph")).split())
                lab = phrases.get(key)
                if lab is None:
                    continue
                kind = "authors" if lab == "secondary sources" else "title"
                self._add(Parent(m.start(), m.end(), lab,
                                 [Child(m.start(), m.end(), kind)], prio=2))
        # initials + harvested surname ("R.J. Sharpe")
        for m in re.finditer(r"\b((?:[A-Z]\.\s*){1,2}([A-ZÀ-Þ][\w'’\-]*))", self.text):
            if _fold(m.group(2)) not in self.author_surnames:
                continue
            if not self._is_free(m.start(), m.end()):
                continue
            self._add(Parent(m.start(), m.end(), "secondary sources",
                             [Child(m.start(), m.end(), "authors")], prio=2))
        for m in re.finditer(r"[A-ZÀ-Þ][\w'’\-]*", self.text):
            w = _fold(m.group(0).strip(".,;:"))
            if len(w) < 3 or w in ROLES:
                continue
            in_case = w in self.case_surnames \
                and (w in self.short_licensed or w in BARE_NAMES)
            in_auth = w in self.author_surnames
            if not in_case and not in_auth:
                continue
            if not self._is_free(m.start(), m.end()):
                continue
            if not self._bare_context_ok(m.start(), m.end()):
                continue
            if in_case and not in_auth:
                label, kind = "decision", "title"
            elif in_auth and not in_case:
                label, kind = "secondary sources", "authors"
            else:
                label = self._preceding_label(m.start(), default="decision")
                kind = "title" if label == "decision" else "authors"
            self._add(Parent(m.start(), m.end(), label, [Child(m.start(), m.end(), kind)], prio=3))
        # docket/file numbers ("MO-2393", "PO-1234"): tribunal order refs
        for m in re.finditer(r"\b[A-Z]{2,5}-\d{2,5}\b", self.text):
            if not self._is_free(m.start(), m.end()):
                continue
            self._add(Parent(m.start(), m.end(), "decision",
                             [Child(m.start(), m.end(), "title")], prio=2))
        for m in FRAG_RE.finditer(self.text):
            if not self._is_free(m.start(), m.end()):
                continue
            lead = m.group(0).split()[0].rstrip(".").lower()
            kids = [Child(m.start(), m.end(), "fragment")]
            end = m.end()
            # Quebec-style "p. 44, art. 36" (no code cite): one secondary parent
            label = None
            if lead in {"p", "pp"}:
                am = re.match(r",\s*(art\.\s*" + _PIN_DIG + r")", self.text[end:end + 30])
                ckm = CIT_CODE.search(self.text[end:end + 60])
                if am:
                    kids.append(Child(end + am.start(1), end + am.end(1), "fragment"))
                    end += am.end()
                    # with a code cite it's legislation ("art. 355 para. 1 C.C.P."),
                    # otherwise a Quebec secondary pinpoint ("p. 44, art. 36")
                    if ckm is not None and ckm.start() <= 30:
                        label = "legislation"
                    else:
                        label = "secondary sources"
            if label is None:
                # full-word "paragraph" leans legislation, "para./p." lean decision
                if lead in {"paragraph", "paragraphs"}:
                    label = self._preceding_label(m.start(), default="legislation")
                elif lead in {"para", "paras", "p", "pp"}:
                    label = self._preceding_label(m.start(), default="decision")
                else:
                    label = self._preceding_label(m.start(), default="legislation")
            self._add(Parent(m.start(), end, label, kids, prio=1 if len(kids) > 1 else 3))

    def _short_label(self, head: str, default: Optional[str]) -> str:
        first = _fold(head.split()[0].strip(",.;:"))
        if first in ROLES:
            return ""  # role words are never names; caller skips
        if first in self.case_surnames and first not in self.author_surnames:
            return "decision"
        if first in self.author_surnames and first not in self.case_surnames:
            return "secondary sources"
        if default is not None:
            return default
        return "decision"

    def _bare_context_ok(self, s: int, e: int) -> bool:
        """Local-context gate for bare single-word mentions."""
        after = self.text[e:e + 16]
        # possessives ("Anderson's") and judge titles ("Smith J.") are not mentions
        if re.match(r"\s*['’]s\b", after):
            return False
        if re.match(r",?\s*(?:JJ?|JA|JSC|JCA|J\.|JJ\.|JA\.)\b", after):
            return False
        # part of a longer capitalized name on the same line ("Baker McKenzie",
        # "Southam Inc.", "Southam\nInc." — but not "Dunsmuir, Southam")
        nm = re.match(r"[ \t\xa0\n]{1,8}([A-ZÀ-Þ][\w'’\-]*)", after)
        if nm:
            return False
        # preceded by a capitalized word in the same phrase ("John Smith",
        # "Justice Southam" — but not "v. Dunsmuir" or "Dunsmuir, Southam")
        pm = re.search(r"([A-ZÀ-Þ][\w'’\-]*)([^.?!;:,()]{0,40})$", self.text[max(0, s - 40):s])
        if pm and _fold(pm.group(1)) not in {
            "v", "c", "the", "a", "an", "le", "la", "les", "de", "van", "von",
            "der", "r", "st", "et", "d", "l",
        }:
            return False
        return True

    def _near_def(self, key: str, pos: int, window: int = 1500) -> bool:
        """Is there a recorded definition of this acronym near pos?"""
        return any(abs(d - pos) <= window for d in self.acronym_defpos.get(key, []))

    def _preceding_label(self, pos: int, default: str) -> str:
        best = None
        for p in self.parents:
            if p.end <= pos and (best is None or p.end > best.end):
                best = p
        if best is not None and pos - best.end < 800:
            return best.label
        return default

    # -- finalize ----------------------------------------------------------

    def finalize(self) -> List[Parent]:
        for p in self.parents:
            # strip trailing sentence punctuation, then unbalanced closers
            end = _rstrip_term(self.text, p.end, p.start)
            while end > p.start and self.text[end - 1] in ")]":
                seg = self.text[p.start:end]
                if seg.count("(") + seg.count("[") >= seg.count(")") + seg.count("]"):
                    break
                end -= 1
            p.end = _rstrip_term(self.text, end, p.start)
            for k in p.children:
                if k.kind in ("citation", "fragment"):
                    k.end = _rstrip_term(self.text, k.end, k.start)
                elif k.kind in ("title", "authors"):
                    k.end = _rstrip_title(self.text, k.end, k.start)
        ordered = sorted(self.parents, key=lambda p: (p.prio, -(p.end - p.start), p.start))
        accepted: List[Parent] = []
        for p in ordered:
            if not any(_overlaps(p.start, p.end, q.start, q.end) for q in accepted):
                kids = [k for k in p.children
                        if k.start >= p.start and k.end <= p.end and k.end > k.start]
                kids = sorted(kids, key=lambda k: (k.start, -(k.end - k.start)))
                kept: List[Child] = []
                for k in kids:
                    if not any(_overlaps(k.start, k.end, q.start, q.end) for q in kept):
                        kept.append(k)
                p.children = kept
                accepted.append(p)
        return sorted(accepted, key=lambda p: p.start)


def detect_all(text: str) -> List[Parent]:
    det = Detector(text)
    det.mask_statute_cites()
    det.detect_secondary()
    det.detect_dictionaries()
    det.detect_decisions()
    det.detect_legislation()
    det.detect_lone_cites()
    det.detect_statute_leftovers()
    det.detect_halsbury()
    det.detect_hansard()
    det.detect_report_bare()
    det.detect_org_defs()
    det.detect_gazetteer()
    det.detect_shorts()
    return det.finalize()


# =============================================================================
# Tag insertion (text-preserving: only adds <auto_label> elements)
# =============================================================================

def _text_nodes(root) -> list:
    return [n for n in root.descendants if type(n) is NavigableString]


def _apply_level(soup: BeautifulSoup, container, spans: List[dict]) -> List:
    """Insert tags for disjoint spans (each: start/end/attrs) inside container.

    Returns the created tag for each input span (None when the span could
    not be aligned exactly, in which case nothing is inserted).
    """
    ordered = sorted([s for s in spans if s["end"] > s["start"]], key=lambda s: s["start"])
    created: List = [None] * len(ordered)
    if not ordered:
        return created
    nodes = _text_nodes(container)
    offs: List[int] = []
    pos = 0
    for n in nodes:
        offs.append(pos)
        pos += len(n)
    total = pos

    bounds: Set[int] = set()
    for s in spans:
        bounds.add(max(0, s["start"]))
        bounds.add(min(total, s["end"]))

    # split text nodes at boundaries (reverse order keeps earlier offsets valid)
    for idx in range(len(nodes) - 1, -1, -1):
        n = nodes[idx]
        a = offs[idx]
        cuts = sorted(b - a for b in bounds if a < b < a + len(n))
        if not cuts:
            continue
        parent_el = n.parent
        current = n
        for c in reversed(cuts):
            s = str(current)
            head_node = NavigableString(s[:c])
            tail_node = NavigableString(s[c:])
            current.replace_with(head_node, tail_node)
            current = head_node  # earlier cuts lie in the head part

    # re-walk, then wrap exact ranges
    nodes = _text_nodes(container)
    offs = []
    pos = 0
    for n in nodes:
        offs.append(pos)
        pos += len(n)
    for idx, s in enumerate(ordered):
        start = max(0, s["start"])
        end = min(pos, s["end"])
        group = [n for n, a in zip(nodes, offs) if a >= start and a + len(n) <= end]
        if not group:
            continue
        covered = sum(len(n) for n in group)
        if covered != end - start:
            continue  # misaligned (should not happen); skip rather than corrupt
        tag = soup.new_tag("auto_label")
        for k, v in s["attrs"].items():
            tag[k] = v
        group[0].insert_before(tag)
        for n in group:
            tag.append(n.extract())
        created[idx] = tag
    return created


def annotate_body(soup: BeautifulSoup, parents: List[Parent]):
    body = soup.find("body")
    if body is None:
        return 0, 0
    top = [{"start": p.start, "end": p.end,
            "attrs": {"labelname": p.label, "parent": ""}} for p in parents]
    ordered_parents = sorted(parents, key=lambda q: q.start)
    created = _apply_level(soup, body, top)
    # children inside each successfully wrapped parent element
    n_children = 0
    for p, el in zip(ordered_parents, created):
        if el is None:
            continue
        kids = [{"start": k.start - p.start, "end": k.end - p.start,
                 "attrs": {"labelname": k.kind, "parent": p.label}} for k in p.children]
        # offsets relative to parent text: recompute from element text
        _apply_level(soup, el, kids)
        n_children += len(kids)
    # sequential ids in document order (parents before their children)
    nxt = 0
    for tag in body.find_all("auto_label"):
        tag["id"] = str(nxt)
        nxt += 1
    for tag in body.find_all("auto_label", attrs={"parent": ""}):
        tag["docid"] = ""
        tag["uri"] = ""
    return len(parents), n_children


# =============================================================================
# Run plumbing (mirrors main_extraction.py)
# =============================================================================

def process_file(filename: str, input_path: Path, exp_dir: Path, output_name: str):
    with open(input_path, encoding="utf-8") as f:
        html_content = f.read()
    soup = BeautifulSoup(html_content, "html.parser")
    body = soup.find("body")
    if body is None:
        print("  [skip] no <body> found")
        return
    nodes = _text_nodes(body)
    parts: List[str] = []
    for n in nodes:
        parts.append(str(n))
    text = "".join(parts)

    parents = detect_all(text)
    # map global offsets: _text_nodes order == concatenation order
    n_par, n_kid = annotate_body(soup, parents)

    out_path = exp_dir / output_name
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(str(soup))
    print(f"  [ok] {n_par} parents, {n_kid} children -> {output_name}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rule-based extraction baseline.")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--filename", type=str)
    mode_group.add_argument("--split", choices=SPLITS)
    parser.add_argument("--single-split", default="test", choices=SPLITS,
                        help="Split to read from in single-file mode.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--variant", default="original", choices=["original", "annotated"])
    parser.add_argument("--method", default="rules", choices=["rules"])
    parser.add_argument("--output-dir", type=Path, default=Path("./output"))
    parser.add_argument("--with-timestamp", action="store_true")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    mode = "single" if args.filename else "split"
    data_dir = args.data_dir or Path(DATA_DIR)
    split = args.split or args.single_split

    tag = f"{split}_baseline_{args.method}"
    output_root = Path(args.output_dir) / tag
    if args.with_timestamp:
        output_root = Path(args.output_dir) / f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_root.mkdir(parents=True, exist_ok=True)

    if mode == "single":
        filenames = [args.filename]
    else:
        folder = Path(data_dir) / args.variant / split
        filenames = sorted(p.stem for p in folder.glob("*.html"))

    total = len(filenames)
    start = args.start if args.start is not None else 1
    end = args.end if args.end is not None else total
    if start < 1 or end < start:
        parser.error(f"Invalid range: --start {start} --end {end}")
    if start > total:
        parser.error(f"--start {start} is greater than the number of documents ({total})")
    end = min(end, total)
    filenames = filenames[start - 1:end]

    print(f"[Setup] Mode: {mode.upper()} | Variant: {args.variant} | "
          f"Files: {len(filenames)} | Output: {output_root}")

    for i, filename in enumerate(filenames, start=start):
        print(f"Processing: {filename} (#{i}/{total})")
        if mode == "split":
            input_path = Path(data_dir) / args.variant / split / f"{filename}.html"
            exp_dir = output_root / filename
            exp_dir.mkdir(exist_ok=True)
            output_name = f"{filename}_final.html"
        else:
            input_path = Path(data_dir) / args.variant / args.single_split / f"{filename}.html"
            exp_dir = output_root
            output_name = f"{filename}.html"
        if not input_path.exists():
            print(f"  [skip] not found: {input_path}")
            continue
        process_file(filename, input_path, exp_dir, output_name)

    print("=" * 80)
    print("[done] BASELINE EXTRACTION COMPLETE")
    print(f"[done] Experiment folder: {output_root}")
    print("=" * 80)


if __name__ == "__main__":
    main()
