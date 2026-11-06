"""Verified URI decision for the no-gold request (SPLADE shortlist + verifier).

Why this exists: calibration (``calibrate_thresholds.py``, 72 LLM-judged
samples) showed the SPLADE dot score ALONE cannot deliver high precision —
similar-but-different works score up to 31+ when the true URI is absent from
the pool. The score only tells "looks similar"; an exact check must confirm
"is the same document". So:

    EXISTS  <=>  top1_score >= T  AND  verify(topk-hit)  (rank1, else rank2/3)
    MISSING <=>  otherwise (uri not in pool, or not confirmable -> manual)

verify(hit, query):
  * citation-consistency: if the query carries a distinctive citation
    (neutral cite, CanLII id, statute ref) AND the candidate metadata carries
    one, they must overlap. Catches same-title/wrong-jurisdiction
    (#67 Labour Relations Act RSO-vs-RSNL) and same-parties/wrong-instance
    (#47 UFCW 2008 LRB vs 2010 SKQB).
  * exact-title: normalized main title equals the candidate title AND that
    title is UNIQUE in the pool (guards generic titles like
    "Workers' Compensation Act" / "Municipal Government Act" -> manual)
    AND statute jurisdictions agree when both sides cite one (PEI-cited
    query never matches Alberta's act). Citeless title-unique legislation
    still assigns (best guess) but is flagged needs_review.
  * alias: Canadian Charter of Rights and Freedoms -> Constitution Act, 1982.

Usage (from repo root)::

    python uri_request/apply_decision.py
    python uri_request/apply_decision.py --threshold 10 --topk 5

Outputs to ``uri_request/output_decision/``: ``uri_decision.{json,csv}``,
``missing_by_document.json`` and ``decision_summary.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

MISSION_DIR = Path(__file__).resolve().parent
DEFAULT_TOPK = MISSION_DIR / "output_splade" / "splade_topk.json"
DEFAULT_OUT = MISSION_DIR / "output_decision"

_STATUTE_CODES = (
    r"R\.?S\.?N\.?W\.?T\.?|N\.?W\.?T\.?|R\.?S\.?N\.?L\.?|S\.?N\.?L\.?|"
    r"R\.?S\.?N\.?B\.?|S\.?N\.?B\.?|R\.?S\.?B\.?C\.?|S\.?B\.?C\.?|"
    r"R\.?S\.?P\.?E\.?I\.?|P\.?E\.?I\.?|C\.?Q\.?L\.?R\.?|R\.?L\.?R\.?Q\.?|"
    r"L\.?R\.?Q\.?|R\.?S\.?Q\.?|C\.?C\.?S\.?M\.?|R\.?S\.?M\.?|S\.?M\.?|"
    r"R\.?S\.?A\.?|S\.?A\.?|R\.?S\.?O\.?|S\.?O\.?|R\.?S\.?C\.?|L\.?R\.?C\.?|"
    r"S\.?C\.?|R\.?S\.?S\.?|S\.?S\.?|S\.?N\.?S\.?"
)
NEUTRAL_RX = re.compile(r"\b((?:19|20)\d{2})\s+([A-Z]{1,12})\s+(\d{1,5})\b")
CANLII_RX = re.compile(r"\b((?:19|20)\d{2})\s+CanLII\s+(\d{1,6})", re.I)
STATUTE_RX = re.compile(
    r"\b(" + _STATUTE_CODES + r")\s*\.?,?\s*(?:19\d{2}|20\d{2})"
    r"\s*,?\s*(?:c(?:h)?\.?\s*([\w\-\.]+)|([\w\-\.]*\d[\w\-\.]*))|"
    r"\bc\.?\s*([\w\-\.]+)\s+(" + _STATUTE_CODES + r")", re.I)

ALIASES = {
    # norm(query segment) -> {norm(candidate title)}; only used when the
    # query carries no citation (else the citation rule decides).
    "canadiancharterofrightsandfreedoms": {"constitutionact1982"},
}


def norm(s: str) -> str:
    return re.sub(r"\s+", "", re.sub(r"[^a-z0-9 ]", " ", (s or "").lower()))


def title_key(s: str) -> str:
    n = norm(s)
    return n[3:] if n.startswith("the") else n


def query_main_title(query_text: str) -> str:
    parts = [p.strip() for p in (query_text or "").split("|")]
    return parts[1] if len(parts) > 1 else ""


def extract_cites(text: str) -> set[str]:
    out: set[str] = set()
    for rx in (NEUTRAL_RX, CANLII_RX, STATUTE_RX):
        for m in rx.finditer(text or ""):
            out.add(norm(m.group(0)))
    return out


# Statute/jurisdiction codes -> jurisdiction. Matched as code + year (no
# chapter needed), so forms like ``R.S.P.E.I. 1988, W-7.1`` are caught.
# Neutral court cites (``2012 BCSC 1579``) have year BEFORE the code and the
# court codes (BCSC, SKCA, ...) are not listed, so they never match.
_STATUTE_CODES = (
    r"R\.?S\.?N\.?W\.?T\.?|N\.?W\.?T\.?|R\.?S\.?N\.?L\.?|S\.?N\.?L\.?|"
    r"R\.?S\.?N\.?B\.?|S\.?N\.?B\.?|R\.?S\.?B\.?C\.?|S\.?B\.?C\.?|"
    r"R\.?S\.?P\.?E\.?I\.?|P\.?E\.?I\.?|C\.?Q\.?L\.?R\.?|R\.?L\.?R\.?Q\.?|"
    r"L\.?R\.?Q\.?|R\.?S\.?Q\.?|C\.?C\.?S\.?M\.?|R\.?S\.?M\.?|S\.?M\.?|"
    r"R\.?S\.?A\.?|S\.?A\.?|R\.?S\.?O\.?|S\.?O\.?|R\.?S\.?C\.?|L\.?R\.?C\.?|"
    r"S\.?C\.?|R\.?S\.?S\.?|S\.?S\.?|S\.?N\.?S\.?"
)
JURIS_RX = re.compile(
    r"\b(" + _STATUTE_CODES + r")\s*\.?,?\s*((?:19|20)\d{2})(?!\d)", re.I)

# Letters-only code -> jurisdiction (longest first: RSA before SA, ...).
JURIS_CODES = [
    ("RSNWT", "NT"), ("NWT", "NT"), ("RSNL", "NL"), ("SNL", "NL"),
    ("RSNB", "NB"), ("SNB", "NB"), ("RSBC", "BC"), ("SBC", "BC"),
    ("RSPEI", "PE"), ("SPEI", "PE"), ("PEI", "PE"), ("CQLR", "QC"),
    ("RLRQ", "QC"), ("LRQ", "QC"), ("RSQ", "QC"), ("CCSM", "MB"),
    ("RSM", "MB"), ("SM", "MB"), ("RSA", "AB"), ("SA", "AB"),
    ("RSO", "ON"), ("SO", "ON"), ("RSC", "CA"), ("LRC", "CA"),
    ("SC", "CA"), ("RSS", "SK"), ("SS", "SK"), ("SNS", "NS"),
]


def statute_jurisdictions(text: str) -> set[str]:
    """Jurisdictions evidenced by ``<statute code> <year>`` in a text."""
    found: set[str] = set()
    for m in JURIS_RX.finditer(text or ""):
        code = re.sub(r"[^A-Z]", "", m.group(1).upper())
        for prefix, jur in JURIS_CODES:
            if code.startswith(prefix):
                found.add(jur)
                break
    return found


class Verifier:
    def __init__(self, candidates: list[dict]):
        self.meta_by_uri: dict[str, dict] = {}
        counts: dict[str, int] = defaultdict(int)
        for c in candidates:
            for k in ("original_url", "uri"):
                if c.get(k):
                    self.meta_by_uri.setdefault(str(c[k]), c)
            counts[title_key(c.get("docTitle") or "")] += 1
        self.title_counts = counts

    def check(self, hit: dict, query_text: str) -> tuple[bool, str]:
        meta = (self.meta_by_uri.get(str(hit.get("uri")), {}).get("metadata_text")
                or hit.get("metadata_text") or "")
        qcites, mcites = extract_cites(query_text), extract_cites(meta)
        if qcites & mcites:
            return True, "citation"
        if qcites and mcites:
            return False, "cite-mismatch"
        qt = title_key(query_main_title(query_text))
        ct = title_key(hit.get("candidate_title") or "")
        if qt and qt == ct:
            if qcites:  # query cites something the candidate metadata lacks
                return False, "cite-unconfirmable"
            if self.title_counts.get(ct, 0) != 1:
                return False, "title-ambiguous"
            # Same title, different jurisdiction's enactment (e.g. PEI vs
            # Alberta Workers' Compensation Act): the pool holds one, the
            # query cites another -> not the same document.
            qj, cj = statute_jurisdictions(query_text), statute_jurisdictions(meta)
            if qj and cj and not (qj & cj):
                return False, "jurisdiction-mismatch"
            return True, "title-unique"
        if not qcites:
            segs = {norm(p) for p in (query_text or "").split("|")}
            for alias, targets in ALIASES.items():
                if alias in segs and ct in targets:
                    return True, "alias"
        return False, "-"


def main() -> None:
    p = argparse.ArgumentParser(description="Verified URI decision.")
    p.add_argument("--topk-path", type=Path, default=DEFAULT_TOPK)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--candidates", type=Path,
                   default=Path("cache/candidate_pool_index_splade_v1/candidates.json"))
    p.add_argument("--threshold", type=float, default=10.0)
    p.add_argument("--topk", type=int, default=5)
    args = p.parse_args()

    results = json.loads(args.topk_path.read_text(encoding="utf-8"))
    candidates = json.loads(args.candidates.read_text(encoding="utf-8"))
    verifier = Verifier(candidates)

    decisions, missing_by_doc = [], defaultdict(list)
    for r in results:
        accepted, via_rank, why = None, None, "below-threshold"
        if (r["top1_score"] or -1) >= args.threshold:
            why = "unverified"
            for h in r["hits"][: args.topk]:
                ok, reason = verifier.check(h, r["query_text"])
                if ok:
                    accepted, via_rank, why = h, h["rank"], reason
                    break
        # Flag EXISTS rows whose profile title disagrees with the accepted
        # candidate title (typo/reversed-parties/short-form = benign, but a
        # genuinely different instrument means upstream coref merged two
        # works -> human glance required despite the citation match).
        needs_review = False
        if accepted:
            needs_review = (title_key(query_main_title(r["query_text"]))
                            != title_key(accepted["candidate_title"] or ""))
            # Citeless generic-title assignment (e.g. bare "Workers'
            # Compensation Act" in an NB doc -> Alberta's act): unknowable
            # from text alone, flag for the company glance.
            if (r.get("doc_type") == "legislation" and why == "title-unique"
                    and not statute_jurisdictions(r["query_text"])):
                needs_review = True
        decisions.append({
            "profile_key": r["profile_key"], "source_document": r["source_document"],
            "docid": r["docid"], "doc_type": r["doc_type"],
            "n_mentions": r.get("n_mentions"),
            "decision": "EXISTS" if accepted else "MISSING",
            "via_rank": via_rank, "verify_reason": why,
            "score": r["top1_score"],
            "uri": accepted["uri"] if accepted else None,
            "candidate_title": accepted["candidate_title"] if accepted else None,
            "needs_review": needs_review,
            "top1_uri": r["top1_uri"], "top1_score": r["top1_score"],
        })
        if not accepted:
            missing_by_doc[r["source_document"]].append(r["profile_key"])
    missing_by_doc = dict(missing_by_doc)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "uri_decision.json").write_text(
        json.dumps(decisions, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.out_dir / "uri_decision.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["profile_key", "source_document", "docid", "doc_type", "decision",
                    "via_rank", "verify_reason", "score", "uri", "candidate_title",
                    "needs_review"])
        for d in decisions:
            w.writerow([d["profile_key"], d["source_document"], d["docid"], d["doc_type"],
                        d["decision"], d["via_rank"], d["verify_reason"], d["score"],
                        d["uri"], d["candidate_title"], d.get("needs_review", False)])
    (args.out_dir / "missing_by_document.json").write_text(
        json.dumps(missing_by_doc, indent=2, ensure_ascii=False), encoding="utf-8")

    by_type: dict[str, dict] = defaultdict(lambda: {"EXISTS": 0, "MISSING": 0})
    via = defaultdict(int)
    for d in decisions:
        by_type[str(d["doc_type"])][d["decision"]] += 1
        if d["decision"] == "EXISTS":
            via[f"rank{d['via_rank']}/{d['verify_reason']}"] += 1
    summary = {"threshold": args.threshold, "n": len(decisions),
               "exists": sum(1 for d in decisions if d["decision"] == "EXISTS"),
               "needs_review": sum(1 for d in decisions if d.get("needs_review")),
               "by_type": dict(by_type), "via": dict(via),
               "docs_affected": len(missing_by_doc)}
    (args.out_dir / "decision_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"[DONE] -> {args.out_dir}")


if __name__ == "__main__":
    main()
