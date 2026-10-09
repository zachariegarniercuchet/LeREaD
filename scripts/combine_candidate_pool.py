"""
Combine a "cited_metadata.csv" (true positives — the real reference
repository for the cited document) and an "extra_metadata.csv" (false
positives) into a single, shuffled "candidate_pool.csv".

Both input files are expected to already be in the "uri,metadata"
format (the format produced by scripts/change_format_csv.py):

    uri,metadata

The output file has the same two columns plus a `label` column marking
where each row came from, so downstream consumers can still tell true
positives from false positives after shuffling:

    uri,metadata,label

`label` is "true_positive" for rows from the cited-metadata file and
"false_positive" for rows from the extra-metadata file.

Usage:
    python -m scripts.combine_candidate_pool \
        data/cited_metadata.csv data/extra_metadata_formated.csv \
        --output_dir data/candidate_pool.csv \
        --seed 42
"""

import argparse
import csv
import os
import random
import sys


TRUE_POSITIVE_LABEL = "true_positive"
FALSE_POSITIVE_LABEL = "false_positive"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Combine cited_metadata.csv (true positives) and "
            "extra_metadata.csv (false positives) into a single shuffled "
            "candidate_pool.csv."
        )
    )
    parser.add_argument(
        "cited_metadata_csv",
        help="Path to cited_metadata.csv (true positives / real citations).",
    )
    parser.add_argument(
        "extra_metadata_csv",
        help="Path to extra_metadata.csv (false positives).",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help=(
            "Path of the output CSV file to write. Despite the name, this "
            "should be a full file path (e.g. data/candidate_pool.csv)."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed, for a reproducible shuffle order.",
    )
    return parser.parse_args()


def read_rows(path):
    """Read an uri,metadata CSV and tag each row with a label."""
    csv.field_size_limit(sys.maxsize)
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as infile:
        reader = csv.DictReader(infile)
        missing = {"uri", "metadata"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is missing expected column(s): {sorted(missing)}. "
                f"Found columns: {reader.fieldnames}"
            )
        for row in reader:
            rows.append(
                {
                    "uri": row["uri"],
                    "metadata": row["metadata"],
                }
            )
    return rows


def combine(cited_path, extra_path, output_path, seed=None):
    true_positive_rows = read_rows(cited_path)
    false_positive_rows = read_rows(extra_path)

    combined = true_positive_rows + false_positive_rows

    rng = random.Random(seed)
    rng.shuffle(combined)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as outfile:
        writer = csv.DictWriter(
            outfile, fieldnames=["uri", "metadata"]
        )
        writer.writeheader()
        writer.writerows(combined)

    print(
        f"Done. Wrote {len(combined)} row(s) to {output_path} "
        f"({len(true_positive_rows)} true positives, "
        f"{len(false_positive_rows)} false positives).",
        file=sys.stderr,
    )


def main():
    args = parse_args()
    combine(
        args.cited_metadata_csv,
        args.extra_metadata_csv,
        args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()