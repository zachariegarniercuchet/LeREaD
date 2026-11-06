"""Manually resolve the 4 docid-less parent spans left in annotated silver.

These spans have no ``docid`` (bare pinpoints the coref stage never grouped
into a profile), so ``complete_silver_uris.py`` could not touch them. Each was
resolved by reading the surrounding paragraph; in all 4 cases the passage
discusses exactly one source, already carried by the sibling spans:

1. 1983CanLII2475SKQB id=549 "s. 1" — paragraph quotes Charter s.1 ("rights
   guaranteed by the Charter ... demonstrably justified"); siblings resolve
   to the Canadian Charter of Rights and Freedoms (https://canlii.ca/t/ldsx).
2. 2003ABCA256 id=309 "s. 32" — "s. 32 limits Parliament and legislatures
   ..." (Charter application provision); siblings resolve to the Charter.
3. 2011CanLII 91839PEIPC id=507 "section 17" — explicitly "Alberta's section
   17" vs "our section 15" (PEI); sibling id=513 resolves to the Alberta
   FOIPP Act (synthetic G0641).
4. 2025MBKB113 id=109 "para. 25" — ends a sentence about the argument made
   "in that case" (Ed Bulley Ventures, 2014 BCCA 52, G1459), whose holding is
   cited just before (paras. 42-43, 46-48, id=106 -> G1459). A self-cite is
   ruled out: this document's own para. 25 states Cobbe's/Rule 34.14
   principles, unrelated to the funding argument.

Only the parent tag gets docid+uri; the nested fragment child is untouched
(children carry no docid by convention).

Usage (from repo root)::

    python uri_request/resolve_silver_docidless.py --dry-run
    python uri_request/resolve_silver_docidless.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MISSION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MISSION_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bs4 import BeautifulSoup  # noqa: E402

SILVER_DIR = PROJECT_ROOT / "data" / "annotated" / "silver"

# (filename, tag id, docid, uri, rationale short tag)
FIXES: list[tuple[str, str, str, str, str]] = [
    ("1983CanLII2475SKQB.html", "549",
     "Canadian Charter of Rights and Freedoms", "https://canlii.ca/t/ldsx",
     "s.1 Charter limitation clause; siblings -> Charter"),
    ("2003ABCA256.html", "309",
     "Canadian Charter of Rights and Freedoms", "https://canlii.ca/t/ldsx",
     "s.32 limits Parliament/legislatures; siblings -> Charter"),
    ("2011CanLII 91839PEIPC.html", "507",
     "Freedom of Information and Protection of Privacy Act (Alberta)",
     "https://synthetic.leread.ca/silver/G0641",
     "\"Alberta's section 17\" vs PEI s.15; sibling id=513 -> G0641"),
    ("2025MBKB113.html", "109",
     "Ed Bulley Ventures Ltd. v. The Pantry Hospitality Corporation",
     "https://synthetic.leread.ca/silver/G1459",
     "\"in that case\" argument pinpoint; sibling id=106 -> G1459; "
     "self-cite ruled out (own para.25 is Cobbe's principles)"),
]


def main() -> None:
    p = argparse.ArgumentParser(description="Resolve docid-less silver spans.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    for fname, tid, docid, uri, why in FIXES:
        f = SILVER_DIR / fname
        soup = BeautifulSoup(f.read_text(encoding="utf-8"), "html.parser")
        tag = soup.find(attrs={"id": tid})
        assert tag is not None, f"tag {tid} not found in {fname}"
        assert tag.get("docid") is None, f"tag {tid} in {fname} already has docid"
        print(f"[{fname} id={tid}] {tag.get('labelname')} {tag.get_text(strip=True)!r}")
        print(f"  -> docid={docid!r}\n  -> uri={uri}\n  ({why})")
        if not args.dry_run:
            tag["docid"] = docid
            attrs = dict(tag.attrs)
            attrs.pop("uri", None)
            tag.attrs = {"uri": uri, **attrs}  # uri first, like the pipeline
            f.write_text(str(soup), encoding="utf-8")

    print("\n[DONE]" + (" (dry-run, nothing written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
