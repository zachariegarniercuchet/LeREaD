"""Build the missing-only URI request for the company.

Reads the full ``uri_request.json`` registry + the verified
``output_decision/uri_decision.json`` and writes ``uri_request_missing.json``
in the SAME shape as ``uri_request.json`` (same keys), but keeping only the
profiles whose decision is MISSING — i.e. URIs not currently in
``data/candidate_pool.csv``. The company fills ``uri`` per profile and sends
it back with metadata.

Usage (from repo root)::

    python uri_request/build_missing_request.py

Output: ``uri_request/uri_request_missing.json``.
"""

from __future__ import annotations

import datetime
import json
import sys
from pathlib import Path

MISSION_DIR = Path(__file__).resolve().parent
FULL_PATH = MISSION_DIR / "uri_request.json"
DECISION_PATH = MISSION_DIR / "output_decision" / "uri_decision.json"
OUT_PATH = MISSION_DIR / "uri_request_missing.json"


def main() -> None:
    full = json.loads(FULL_PATH.read_text(encoding="utf-8"))
    decisions = json.loads(DECISION_PATH.read_text(encoding="utf-8"))
    missing_keys = {d["profile_key"] for d in decisions if d["decision"] == "MISSING"}
    n_exists = sum(1 for d in decisions if d["decision"] == "EXISTS")
    print(f"[Setup] {len(decisions)} decided: {n_exists} EXISTS, "
          f"{len(missing_keys)} MISSING")

    documents = []
    for doc in full.get("documents", []):
        kept = [p for p in doc.get("profiles", [])
                if p.get("profile_key") in missing_keys]
        if not kept:
            continue
        documents.append({
            "source_document": doc.get("source_document"),
            "coref_file": doc.get("coref_file"),
            "n_mentions": sum(p.get("n_mentions", 0) for p in kept),
            "n_profiles": len(kept),
            "n_mentions_no_docid": doc.get("n_mentions_no_docid", 0),
            "profiles": kept,
        })
    documents.sort(key=lambda d: d["source_document"])

    payload = {
        "meta": {
            "producer": "uri_request/build_missing_request.py",
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "based_on": "uri_request.json + output_decision/uri_decision.json "
                        "(verified SPLADE rule, threshold 10.0)",
            "n_documents": len(documents),
            "n_profiles": sum(d["n_profiles"] for d in documents),
            "n_mentions": sum(d["n_mentions"] for d in documents),
            "n_profiles_exists_excluded": n_exists,
            "uri": "EMPTY - to be filled by the company (one uri per profile)",
        },
        "documents": documents,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"[DONE] {payload['meta']['n_documents']} document(s), "
          f"{payload['meta']['n_profiles']} missing profile(s) -> {OUT_PATH}")


if __name__ == "__main__":
    if not FULL_PATH.exists():
        sys.exit(f"Missing input: {FULL_PATH}")
    if not DECISION_PATH.exists():
        sys.exit(f"Missing input: {DECISION_PATH} (run apply_decision.py first)")
    main()
