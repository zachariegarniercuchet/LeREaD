"""
Add a unique sequential `id` attribute to every <auto_label> and
<manual_label> tag in HTML files ending with `processed.html` or `final.html`.

The script recursively searches a given root directory, regardless of
how deeply the HTML files are nested.

Usage:
    python scripts/add_ids_to_label_pipeline.py /path/to/output

    python scripts/add_ids_to_label_pipeline.py /path/to/output --ignore temp cache old

Examples:
    python scripts/add_ids_to_label_pipeline.py output
    python scripts/add_ids_to_label_pipeline.py output --ignore tmp backup supplementary

For each HTML document:
    - IDs start at 0.
    - IDs are assigned in document order.
    - Nested labels are included.
    - Parent labels appear before their nested children.
    - Existing IDs are overwritten.
    - Files are modified in place.
"""

import argparse
from pathlib import Path

import chardet
from bs4 import BeautifulSoup


LABEL_TAGS = ["auto_label", "manual_label"]
TARGET_SUFFIXES = ("processed.html", "final.html")


def parse_arguments():
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Recursively add sequential IDs to <auto_label> and "
            "<manual_label> tags in *processed.html and *final.html files."
        )
    )

    parser.add_argument(
        "root_dir",
        type=Path,
        help="Root directory to search recursively.",
    )

    parser.add_argument(
        "--ignore",
        nargs="*",
        default=[],
        metavar="FOLDER",
        help=(
            "Folder names to ignore anywhere in the directory tree. "
            "Example: --ignore tmp backup old"
        ),
    )

    return parser.parse_args()


def find_html_files(root_dir, ignored_folders):
    """
    Recursively find all files ending with 'processed.html' or 'final.html'.

    Any directory whose name is present in ignored_folders is skipped.
    """

    ignored_folders = set(ignored_folders)

    for path in root_dir.rglob("*"):
        if not path.is_file():
            continue

        # Skip files that are inside an ignored directory.
        if any(
            parent.name in ignored_folders
            for parent in path.parents
        ):
            continue

        # Only process files whose name ends with a supported suffix.
        if path.name.endswith(TARGET_SUFFIXES):
            yield path


def add_ids_to_labels(html_file):
    """
    Detect encoding, read the HTML, assign sequential IDs to every
    auto_label/manual_label tag, and return the modified HTML.

    Returns:
        str | None:
            Modified HTML, or None if no target labels were found.
    """

    # ------------------------------------------------------------
    # Detect encoding
    # ------------------------------------------------------------

    with open(html_file, "rb") as f:
        raw_data = f.read()

    detected = chardet.detect(raw_data)
    detected_encoding = detected.get("encoding")
    confidence = detected.get("confidence", 0)

    if detected_encoding is None:
        # Fallback if chardet cannot determine the encoding.
        detected_encoding = "utf-8"

    print(
        f"  → Detected encoding: {detected_encoding} "
        f"(confidence: {confidence:.2%})"
    )

    # ------------------------------------------------------------
    # Read HTML
    # ------------------------------------------------------------

    try:
        with open(
            html_file,
            "r",
            encoding=detected_encoding,
        ) as f:
            content = f.read()

    except UnicodeDecodeError:
        print(
            f"  ⚠ Could not decode using {detected_encoding}. "
            f"Trying UTF-8..."
        )

        with open(
            html_file,
            "r",
            encoding="utf-8",
        ) as f:
            content = f.read()

    # ------------------------------------------------------------
    # Parse HTML
    # ------------------------------------------------------------

    soup = BeautifulSoup(content, "html.parser")

    # BeautifulSoup's find_all() returns elements in document order.
    # For nested labels, the parent therefore appears before its child.
    label_tags = soup.find_all(LABEL_TAGS)

    if not label_tags:
        print(
            "  → No <auto_label> or <manual_label> tags found. "
            "Leaving file unchanged."
        )
        return None

    # ------------------------------------------------------------
    # Assign IDs
    # ------------------------------------------------------------

    for position, tag in enumerate(label_tags):
        # Overwrite an existing ID if one is present.
        tag["id"] = str(position)

    print(
        f"  → Assigned IDs 0..{len(label_tags) - 1} "
        f"to {len(label_tags)} label tag(s)."
    )

    # BeautifulSoup serializes the modified document.
    return str(soup)


def process_file(html_file):
    """Process one HTML file and overwrite it in place."""

    print(f"Processing: {html_file}")

    modified_html = add_ids_to_labels(html_file)

    if modified_html is None:
        print()
        return False

    with open(
        html_file,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(modified_html)

    print(f"  ✓ Updated in place: {html_file}\n")

    return True


def main():
    args = parse_arguments()

    root_dir = args.root_dir.resolve()
    ignored_folders = args.ignore

    # ------------------------------------------------------------
    # Validate root directory
    # ------------------------------------------------------------

    if not root_dir.exists():
        print(f"ERROR: Directory does not exist: {root_dir}")
        return

    if not root_dir.is_dir():
        print(f"ERROR: Not a directory: {root_dir}")
        return

    # ------------------------------------------------------------
    # Print configuration
    # ------------------------------------------------------------

    print("=" * 70)
    print("HTML LABEL ID ASSIGNMENT")
    print("=" * 70)

    print(f"Root directory : {root_dir}")
    print(f"Target files   : *processed.html, *final.html")

    if ignored_folders:
        print(
            "Ignored folders: "
            + ", ".join(ignored_folders)
        )
    else:
        print("Ignored folders: None")

    print("=" * 70)
    print()

    # ------------------------------------------------------------
    # Find files recursively
    # ------------------------------------------------------------

    html_files = sorted(
        find_html_files(
            root_dir,
            ignored_folders,
        )
    )

    print(
        f"Found {len(html_files)} "
        f"*processed.html/*final.html file(s).\n"
    )

    if not html_files:
        print("Nothing to process.")
        return

    # ------------------------------------------------------------
    # Process files
    # ------------------------------------------------------------

    processed = 0
    skipped = 0

    for html_file in html_files:
        if process_file(html_file):
            processed += 1
        else:
            skipped += 1

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print("=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)

    print(f"Files found     : {len(html_files)}")
    print(f"Files modified  : {processed}")
    print(f"Files unchanged : {skipped}")
    print("=" * 70)


if __name__ == "__main__":
    main()
