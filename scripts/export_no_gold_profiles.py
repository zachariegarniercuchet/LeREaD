"""Export no-gold coref profiles for external URI matching.

Takes the ``*_coref.html`` files of a stage-2 (coref) run folder — typically
the no-gold run, which has no gold ``uri`` attributes — rebuilds the
:class:`ReferenceProfileRegistry` for each document exactly like stage 3
(``src/main_resolution.py::process_document_resolution``) does, but WITHOUT
running any retrieval, and dumps every grouped profile plus its identifying
metadata to a single JSON file that can be sent to the company so they can
match each profile to the right ``uri``.

Rebuild logic mirrors ``main_resolution.py`` on purpose::

    mentions = extract_parent_level_annotations(html_content)
    rpr = ReferenceProfileRegistry()
    for mention in mentions:
        rpr.update_from_mention(ReferenceMention(mention.html_str))

Only profiles with a ``docid`` are exported (profiles without one were never
coref-resolved and have no ``docid`` tag for a ``uri`` to be attached to —
same rule as ``process_document_resolution``).

Usage (from repo root)::

    python scripts/export_no_gold_profiles.py
    python scripts/export_no_gold_profiles.py --input-dir output_coref/<run> --output data/no-gold_uri_request/<run>_profiles.json
    python scripts/export_no_gold_profiles.py --per-doc

Output JSON shape::

    {
      "meta": {"producer": ..., "input_dir": ..., "n_documents": 55,
               "n_profiles": ..., "n_mentions": ..., "n_skipped_no_docid": ...},
      "documents": [
        {
          "source_document": "2011FC18",
          "coref_file": "2011FC18/2011FC18_coref.html",
          "n_mentions": 29,
          "n_profiles": 11,
          "profiles": [
            {
              "profile_key": "2011FC18::Trade-marks Act",
              "docid": "Trade-marks Act",
              "doc_type": "legislation",
              "main_title": "Trade-marks Act",
              "alternative_titles": [...],
              "citations": [...],
              "fragments_mentioned": [...],
              "authors": [...],
              "first_seen_id": "0",
              "n_mentions": 17,
              "mention_ids": ["0", "4", ...],
              "mention_texts": ["s. 45 of the Trade-marks Act, RSC ...", ...],
              "query_text": "legislation | Trade-marks Act | RSC 1985, c T-13",
              "uri": ""
            },
            ...
          ]
        },
        ...
      ]
    }

The ``uri`` field is left empty for the company to fill in. Once filled, the
file can be read back to inject ``uri`` attributes into the ``*_coref.html``
files (see ``apply_uris_to_html`` in ``src/main_resolution.py``).
"""

from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import DATA_DIR  # noqa: E402
from src.ann_extractor import extract_parent_level_annotations  # noqa: E402
from src.htmlLabel import ReferenceMention  # noqa: E402
from src.main_resolution import build_profile_text  # noqa: E402 (same query text as stage 3)
from src.rpr import ReferenceProfileRegistry, normalize_docid  # noqa: E402

DEFAULT_INPUT_DIR = (
    Path(
        "output_coref/no-gold_gemma4_coref_fs6_random_from-no-gold_gemma4_DEC_fs6_greedy_paragraph_long-final"
    )
)
DEFAULT_OUTPUT_DIR = DATA_DIR / "no-gold_uri_request"


def _profile_export_key(source_document: str, docid: str) -> str:
    # docids are only unique within a document, so the round-trip key must
    # be namespaced by document.
    return f"{source_document}::{normalize_docid(docid)}"


def export_document(coref_path: Path, max_mention_texts: int | None = None) -> dict:
    """Rebuild the RPR for one ``*_coref.html`` file and export its profiles."""
    html_content = coref_path.read_text(encoding="utf-8")
    source_document = coref_path.name[: -len("_coref.html")]

    mentions = extract_parent_level_annotations(html_content)

    rpr = ReferenceProfileRegistry()
    for mention in mentions:
        rpr.update_from_mention(ReferenceMention(mention.html_str))

    # Group mention ids/texts per normalized docid for the export.
    mentions_by_docid: dict[str, dict] = {}
    n_skipped_no_docid = 0
    for mention in mentions:
        raw_docid = mention.html_tag.attributes.get("docid")
        norm_docid = normalize_docid(raw_docid) if raw_docid else None
        if not norm_docid:
            n_skipped_no_docid += 1
            continue
        bucket = mentions_by_docid.setdefault(norm_docid, {"ids": [], "texts": []})
        mention_id = mention.html_tag.attributes.get("id")
        bucket["ids"].append(str(mention_id) if mention_id is not None else None)
        bucket["texts"].append(mention.text)

    profiles = []
    for profile in rpr:
        if not profile.docid:
            continue
        norm_docid = normalize_docid(profile.docid)
        bucket = mentions_by_docid.get(norm_docid, {"ids": [], "texts": []})
        mention_texts = bucket["texts"]
        if max_mention_texts is not None:
            mention_texts = mention_texts[:max_mention_texts]
        profiles.append(
            {
                "profile_key": _profile_export_key(source_document, profile.docid),
                "docid": profile.docid,
                "doc_type": profile.doc_type,
                "main_title": profile.main_title,
                "alternative_titles": sorted(profile.alternative_titles.keys()),
                "citations": sorted(profile.citations.keys()),
                "fragments_mentioned": sorted(profile.fragments_mentioned.keys()),
                "authors": sorted(profile.authors.keys()),
                "first_seen_id": (
                    str(profile.first_seen_id)
                    if profile.first_seen_id is not None
                    else None
                ),
                "n_mentions": len(bucket["ids"]),
                "mention_ids": bucket["ids"],
                "mention_texts": mention_texts,
                "query_text": build_profile_text(profile),
                "uri": "",
            }
        )

    # Document order (registry insertion order) is deterministic; keep it so
    # the file reads in the same order as the source document.
    return {
        "source_document": source_document,
        "coref_file": f"{coref_path.parent.name}/{coref_path.name}",
        "n_mentions": len(mentions),
        "n_profiles": len(profiles),
        "n_mentions_no_docid": n_skipped_no_docid,
        "profiles": profiles,
    }


def collect_coref_files(input_dir: Path) -> list[Path]:
    """Find one ``*_coref.html`` per document subfolder, sorted by name."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    files = sorted(input_dir.glob("*/ *_coref.html".replace(" ", "")))
    if not files:
        # Fall back to a recursive search for odd layouts.
        files = sorted(input_dir.rglob("*_coref.html"))
    return files


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild ReferenceProfileRegistries from a coref run folder "
            "(no-gold) and export them to one JSON file for external URI matching."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Stage-2 run folder with per-doc *_coref.html files (default: {DEFAULT_INPUT_DIR}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON path (default: "
            "data/no-gold_uri_request/<input-dir-name>_profiles.json)."
        ),
    )
    parser.add_argument(
        "--per-doc",
        action="store_true",
        help="Also write one JSON file per document next to the combined file.",
    )
    parser.add_argument(
        "--max-mention-texts",
        type=int,
        default=None,
        help="Cap on mention_texts stored per profile (default: all).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file(s).",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output: Path = args.output or (DEFAULT_OUTPUT_DIR / f"{input_dir.name}_profiles.json")

    coref_files = collect_coref_files(input_dir)
    if not coref_files:
        parser.error(f"No *_coref.html files found under {input_dir}")
    print(f"[Setup] Input dir : {input_dir}")
    print(f"[Setup] Found {len(coref_files)} *_coref.html file(s)")
    print(f"[Setup] Output    : {output}\n")

    if output.exists() and not args.overwrite:
        parser.error(f"Output already exists: {output} (use --overwrite)")

    documents = []
    for coref_path in coref_files:
        try:
            documents.append(
                export_document(coref_path, max_mention_texts=args.max_mention_texts)
            )
        except Exception as exc:  # keep going; report at the end
            print(f"  [!] {coref_path.parent.name}: FAILED ({exc})")

    # Deterministic order for diffability.
    documents.sort(key=lambda d: d["source_document"])

    n_profiles = sum(d["n_profiles"] for d in documents)
    n_mentions = sum(d["n_mentions"] for d in documents)
    n_no_docid = sum(d["n_mentions_no_docid"] for d in documents)

    payload = {
        "meta": {
            "producer": "scripts/export_no_gold_profiles.py",
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "input_dir": str(input_dir),
            "n_documents": len(documents),
            "n_profiles": n_profiles,
            "n_mentions": n_mentions,
            "n_mentions_no_docid": n_no_docid,
            "uri": "EMPTY - to be filled by the company (one uri per profile)",
        },
        "documents": documents,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    if args.per_doc:
        per_doc_dir = output.parent / f"{output.stem}_per-doc"
        per_doc_dir.mkdir(parents=True, exist_ok=True)
        for doc in documents:
            doc_path = per_doc_dir / f"{doc['source_document']}_profiles.json"
            if doc_path.exists() and not args.overwrite:
                continue
            with doc_path.open("w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2, ensure_ascii=False)
        print(f"[Done] Wrote {len(documents)} per-doc file(s) to {per_doc_dir}")

    print(
        f"\n[DONE] {len(documents)} document(s), {n_profiles} profile(s), "
        f"{n_mentions} mention(s) ({n_no_docid} without docid) -> {output}"
    )


if __name__ == "__main__":
    main()
