"""
Build the candidate-pool index (stage 3 offline step).

    python -m src.precompute_index --retriever bge-s
    python -m src.precompute_index --retriever splade
    python -m src.precompute_index --retriever bm25
    python -m src.precompute_index --retriever legal-bert
    python -m src.precompute_index --retriever qwen3-8b --device cuda --batch-size 4

    bge-s / qwen3 / legal-bert -> dense vectors     -> cosine similarity
    splade                     -> sparse vectors    -> dot product
    bm25                       -> inverted index    -> BM25 score

Every retriever writes the same three things into its cache directory, so
main_resolution does not care which one produced them:

    candidates.json   candidate metadata, row i matches index row i
    config.json       what was built, from what, and how
    <index artifact>  embeddings.npy or bm25.pkl (+ retriever.json)

"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from src.retrievers import available_retrievers, create_retriever

# Bump this when build_metadata_text() changes, so old caches can't be silently
# mixed with new query representations.
REPRESENTATION_VERSION = "v1"


# ---------------------------------------------------------------------------
# Metadata representation
# ---------------------------------------------------------------------------

def extract_autocompletion_texts(auto_completions) -> List[str]:
    """
    Extract only the textual values from CanLII autoCompletions.

        [{"type": "MAIN_TITLE", "text": "Mills v. The Queen"}]  ->  ["Mills v. The Queen"]

    The completion type is intentionally discarded.
    """

    if not auto_completions:
        return []

    # autoCompletions is itself stored as a JSON string in the CSV.
    if isinstance(auto_completions, str):
        try:
            auto_completions = json.loads(auto_completions)
        except json.JSONDecodeError:
            return []

    if not isinstance(auto_completions, list):
        return []

    texts = []
    for completion in auto_completions:
        if not isinstance(completion, dict):
            continue
        text = completion.get("text")
        if text is not None:
            text = str(text).strip()
            if text:
                texts.append(text)

    return texts


def build_metadata_text(candidate: dict) -> str:
    """
    Build the textual representation used for indexing.

    Fields: docType | docTitle | autoCompletions[].text

    Must stay aligned with build_profile_text() in main_resolution.py — the two
    are the document side and the query side of the same representation.
    """

    parts = []

    doc_type = candidate.get("docType")
    if doc_type:
        parts.append(str(doc_type))

    doc_title = candidate.get("docTitle")
    if doc_title:
        parts.append(str(doc_title))

    parts.extend(extract_autocompletion_texts(candidate.get("autoCompletions")))

    return " | ".join(parts)


# ---------------------------------------------------------------------------
# CSV loading
# ---------------------------------------------------------------------------

def load_candidates(csv_path: Path) -> Tuple[List[dict], List[str]]:
    """
    Load candidate metadata from candidate_pool_metadata.csv.

    Returns (candidates, texts) where texts[i] is the indexable representation
    of candidates[i].
    """

    df = pd.read_csv(csv_path)

    required_columns = {"uri", "metadata"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

    candidates: List[dict] = []
    texts: List[str] = []

    for row_idx, row in df.iterrows():
        uri = row["uri"]
        metadata_raw = row["metadata"]

        try:
            metadata = json.loads(metadata_raw)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ValueError(
                f"Could not parse metadata JSON on row {row_idx}: {exc}"
            ) from exc

        if not isinstance(metadata, list):
            raise ValueError(
                f"Expected metadata to be a list on row {row_idx}, "
                f"got {type(metadata).__name__}"
            )

        for candidate in metadata:
            if not isinstance(candidate, dict):
                continue

            text = build_metadata_text(candidate)

            candidates.append(
                {
                    "uri": uri,
                    "id": candidate.get("id"),
                    "docType": candidate.get("docType"),
                    "docTitle": candidate.get("docTitle"),
                    "path": candidate.get("path"),
                    "jurisdictionId": candidate.get("jurisdictionId"),
                    "citedCount": candidate.get("citedCount"),
                    "metadata_text": text,
                }
            )
            texts.append(text)

    return candidates, texts


def default_cache_dir(retriever: str) -> Path:
    return Path(f"cache/candidate_pool_index_{retriever}_{REPRESENTATION_VERSION}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Precompute the candidate-pool index for CanLII metadata."
    )

    parser.add_argument(
        "--retriever",
        choices=available_retrievers(),
        default="bge-s",
        help="Retrieval model to index with.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/candidate_pool.csv"),
        help="Input candidate metadata CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: cache/candidate_pool_index_<retriever>_<version>).",
    )
    parser.add_argument("--device", default=None, help="PyTorch device, e.g. cuda or cpu.")
    parser.add_argument("--batch-size", type=int, default=None, help="Encoding batch size.")
    parser.add_argument(
        "--model-name",
        default=None,
        help="Override the HuggingFace checkpoint for the chosen retriever.",
    )

    args = parser.parse_args()

    output_dir = args.output_dir or default_cache_dir(args.retriever)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- candidates ---------------------------------------------------------

    print(f"Loading candidate metadata from: {args.input}")
    candidates, texts = load_candidates(args.input)
    print(f"Loaded {len(candidates):,} candidates.")

    if not candidates:
        raise ValueError("No candidates were found.")

    # --- retriever ----------------------------------------------------------
    # Vector retrievers accept device/batch_size/model_name; BM25 swallows them
    # via **_ignored, so no branching is needed here.

    retriever_kwargs = {}
    if args.device is not None:
        retriever_kwargs["device"] = args.device
    if args.batch_size is not None:
        retriever_kwargs["batch_size"] = args.batch_size
    if args.model_name is not None:
        retriever_kwargs["model_name"] = args.model_name

    print(f"Creating retriever: {args.retriever}")
    retriever = create_retriever(args.retriever, **retriever_kwargs)

    print("Indexing candidate metadata...")
    retriever.build(texts)

    retriever.save(output_dir)
    print(f"Saved index to: {output_dir}")

    # --- candidates.json ----------------------------------------------------

    candidates_path = output_dir / "candidates.json"
    with candidates_path.open("w", encoding="utf-8") as f:
        json.dump(candidates, f, ensure_ascii=False, indent=2)
    print(f"Saved candidate metadata to: {candidates_path}")

    # --- config.json --------------------------------------------------------

    config = {
        "version": REPRESENTATION_VERSION,
        "retriever": retriever.describe(),
        "input_file": str(args.input),
        "num_candidates": len(candidates),
        "representation": {
            "fields": ["docType", "docTitle", "autoCompletions.text"],
            "autoCompletions": (
                "Only the text field of each autoCompletion is encoded; "
                "the completion type is discarded."
            ),
            "format": "fields concatenated with ' | '",
        },
    }

    config_path = output_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"Saved configuration to: {config_path}")

    print()
    print("Done.")
    print(f"Output directory: {output_dir}")


if __name__ == "__main__":
    main()