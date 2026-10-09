"""
Convert a "cited_metadata.csv"-style file into the simpler
"uri,metadata" CSV format.

Input columns (per row):
    id, final_url, normalized_final_url, docType, jurisdictionId,
    solr_collection, metadata, tiny_url (optional)

The `metadata` column holds a JSON string shaped like:
    {"response": {"numFound": 1, "docs": [ {...one doc...} ]}}

Output columns (per row):
    uri, metadata

Where:
    - `uri` comes from the input row's `tiny_url`, except when
      `docType == DOCTRINE` (tiny url is not tiny there), in which case
      it comes from `final_url` (falls back to `normalized_final_url`
      if `final_url` is missing/empty). Non-DOCTRINE rows with a
      missing/empty `tiny_url` also fall back to `final_url` then
      `normalized_final_url`.
    - `metadata` is the `response.docs` list from the input's metadata JSON,
      pretty-printed (indent=2) so it reads the same way as the target
      example file.

Usage:
    python -m scripts.change_format_csv data/extra_metadata.csv \
        --output_dir data/extra_metadata_formated.csv
"""

import argparse
import csv
import json
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Reformat a cited_metadata.csv file (id, final_url, "
            "normalized_final_url, docType, jurisdictionId, solr_collection, "
            "metadata, tiny_url) into a uri,metadata CSV file."
        )
    )
    parser.add_argument(
        "input_csv",
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help=(
            "Path of the output CSV file to write. Despite the name, this "
            "should be a full file path (e.g. data/extra_metadata_formated.csv)."
        ),
    )
    return parser.parse_args()


def extract_docs(raw_metadata):
    """Parse the input `metadata` JSON string and return the list of docs."""
    parsed = json.loads(raw_metadata)
    return parsed.get("response", {}).get("docs", [])


def convert(input_csv, output_csv):
    # Some metadata blobs can be long; make sure the csv module can handle them.
    csv.field_size_limit(sys.maxsize)

    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    rows_written = 0
    rows_skipped = 0

    with open(input_csv, "r", encoding="utf-8", newline="") as infile, open(
        output_csv, "w", encoding="utf-8", newline=""
    ) as outfile:
        reader = csv.DictReader(infile)
        writer = csv.writer(outfile)
        writer.writerow(["uri", "metadata"])

        for row in reader:
            doc_type = (row.get("docType") or "").strip().upper()
            tiny_url = (row.get("tiny_url") or "").strip()
            final_url = (row.get("final_url") or "").strip() or (row.get("normalized_final_url") or "").strip()
            if doc_type == "DOCTRINE":
                uri = final_url
            else:
                uri = tiny_url or final_url
            raw_metadata = row.get("metadata", "")

            if not uri or not raw_metadata:
                rows_skipped += 1
                print(
                    f"Warning: skipping row id={row.get('id')!r} "
                    f"(missing url or metadata)",
                    file=sys.stderr,
                )
                continue

            try:
                docs = extract_docs(raw_metadata)
            except json.JSONDecodeError as exc:
                rows_skipped += 1
                print(
                    f"Warning: skipping row id={row.get('id')!r} "
                    f"(could not parse metadata JSON: {exc})",
                    file=sys.stderr,
                )
                continue

            formatted_metadata = json.dumps(docs, indent=2, ensure_ascii=False)
            writer.writerow([uri, formatted_metadata])
            rows_written += 1

    print(f"Done. Wrote {rows_written} row(s) to {output_csv}.", file=sys.stderr)
    if rows_skipped:
        print(f"Skipped {rows_skipped} row(s) due to missing/invalid data.", file=sys.stderr)


def main():
    args = parse_args()
    convert(args.input_csv, args.output_dir)


if __name__ == "__main__":
    main()