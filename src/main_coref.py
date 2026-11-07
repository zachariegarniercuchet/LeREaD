"""Main processing script for coreference resolution over extracted references.

Mirrors main_extraction.py in structure and CLI so the two pipelines stay easy
to read side by side. This runs the coref baseline (no hyperparameters yet):
for each parent-level mention in a document, it asks the assistant to resolve
it against a running ReferenceProfileRegistry and assigns a docid, then
groups mentions that share a docid into clusters.

Input can come from two places (--input-source):
  - "gold":       read mentions directly from the annotated dataset
                  (DATA_DIR/<variant>/<split>/<filename>.html)
  - "extraction": read mentions from a previous main_extraction.py run's
                  output (for a full extraction -> coref pipeline)
"""

import sys
from pathlib import Path
import json
import argparse
import random
import re
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from collections import defaultdict
from src.ann_extractor import get_mention_upper_context


from bs4 import BeautifulSoup
from tqdm import tqdm

from src.transforme_utils import clean_tokens

# Setup project path
PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.config import CONTEXT_MAX_TOKENS, DATA_DIR, FEWSHOT_CACHE_DIR, PROMPT_DIR, SPLITS


from configs.constants import MODEL_MAPPING_NAME

from src.extractor import LabelTransformConfig, prepare_label_tokens
from src.ann_extractor import extract_parent_level_annotations
from src.htmlLabel import ReferenceMention
from src.tokenizer_utils import tokenize, decode
from src.models import get_messages, AssistantFactory
from src.fewshot.coref_helper import example_to_string, format_profile_for_prompt
from src.rpr import ReferenceProfileRegistry
from src.coref_run_config import CorefRunConfig

DEFAULT_PROFILE_ATTRIBUTES = [
    "docid",
    "alternative_titles",
    "citations",
    "fragments_mentioned",
    "authors",
]

# Matches docid="value" or docid=\"value\" (escaped quotes), non-greedy,
# stops at the first closing quote.
_DOCID_RE = re.compile(r'docid\s*=\s*\\?["\'](.*?)\\?["\']', re.DOTALL)





# =============================================================================
# Few-shot loading
# =============================================================================

def load_coref_fewshot_examples(
    fewshot_method: str,
    nb_examples: int,
    seed: int = 42,
) -> List[Tuple[str, str]]:
    """Load and prepare coref few-shot examples."""
    fewshot_filename = f"examples_coref_{fewshot_method}"
    with open(FEWSHOT_CACHE_DIR / f"{fewshot_filename}.json", "r", encoding="utf-8") as f:
        fewshot_file_content = json.load(f)

    print("fewshot examples from :", fewshot_filename)

    fewshot_examples = [
        (json.loads(example["input"]), example["output"], example["meta"])
        for example in fewshot_file_content["examples"]
    ]


    fewshot_examples = fewshot_examples[:nb_examples]

    final_fewshot = []
    for input_, output, meta in fewshot_examples:

        doctype = input_["input_mention"].split(">")[0][1:]
        final_input = example_to_string(input_, docid=meta["docid"], doctype=doctype)
        final_fewshot.append((final_input, output))

    return final_fewshot


# =============================================================================
# Generation post-processing
# =============================================================================

def extract_docid_from_generation(generated: str) -> Optional[str]:
    """Extract the value of the first `docid="..."` attribute in the LLM output,
    tolerating malformed/unclosed markup."""
    if not generated:
        return None
    match = _DOCID_RE.search(generated)
    if not match:
        return None
    docid = match.group(1).strip()
    return docid or None


def dict_to_clusters(mapping: Dict[str, str]) -> List[List[str]]:
    """mention_id -> predicted docid, grouped into clusters of mention ids."""
    clusters = defaultdict(list)
    for mention_id, docid in mapping.items():
        clusters[docid].append(mention_id)
    return list(clusters.values())

def apply_coref_docids(html_content: str, mapping: Dict[str, str], parser: str = "html.parser") -> str:
    """Reconstruct a full HTML document with predicted docids applied.
 
    Every existing `docid` attribute in the document is stripped first —
    this is what makes the function source-agnostic: it doesn't matter
    whether `html_content` came from an annotated/gold file (which already
    carries ground-truth docids we must not leak into evaluation) or from a
    fresh extraction-pipeline output (which has none). Only mentions present
    in `mapping` (i.e. that were successfully resolved) get a docid set;
    anything that failed resolution is simply left without one.
    """
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
        print(f"  ⚠️  {len(missing)} mention id(s) from the mapping were not found "
              f"in the document while reconstructing HTML: {missing}")
 
    return str(soup)

# =============================================================================
# Checkpointing
# =============================================================================

def get_checkpoint(exp_dir: Path, filename: str) -> Optional[Dict[str, str]]:
    """Return the mention_id -> docid mapping if a checkpoint exists, else None."""
    mapping_path = exp_dir / f"{filename}_coref_mapping.json"
    if mapping_path.exists():
        print(f"  ⏭  Checkpoint found: {mapping_path.name} — skipping")
        with open(mapping_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# =============================================================================
# Core processing
# =============================================================================

def process_document_coref(
    html_content:str,
    mentions: List[Any],
    system_prompt: str,
    fewshot_examples: List[tuple],
    assistant,
    max_fragments: int,
    profile_attributes: List[str],
) -> Tuple[Dict[str, str], int]:
    """Resolve coreference for every mention in a document, sequentially
    growing a ReferenceProfileRegistry as we go."""

    rpr = ReferenceProfileRegistry()

    input_label_config = LabelTransformConfig(
        use_simplified=True,
        switch_type=False,
        keep_attributes=["labelname"],
    )

    result: Dict[str, str] = {}
    failed_count = 0


    for idx, mention in enumerate(tqdm(mentions, desc="Resolving mentions")):
        mention_id = mention.html_tag.attributes["id"]
        prepared_tokens = prepare_label_tokens(tokenize(mention.html_str), input_label_config)

        context = get_mention_upper_context(html = decode(clean_tokens(tokenize(html_content), keep_manual_label=False, keep_auto_label=False, protected_id=mention_id)), 
                                            mention=mention,
                                            max_tokens=CONTEXT_MAX_TOKENS)

        profiles_formatted = [
            format_profile_for_prompt(profile.to_dict(attributes=profile_attributes), max_fragments=max_fragments)
            for profile in rpr
        ]
        lines = ["Reference Profile Registry:"]
        lines += [f"  Profile {i}: {p}" for i, p in enumerate(profiles_formatted)]
        profiles_str = "\n".join(lines)

        user_input = decode(prepared_tokens) + "\n Upper Context : \n" + context + "\n" + profiles_str
        #print(user_input)
        messages = get_messages(
            system_prompt=system_prompt,
            user_input=user_input,
            fewshot_examples=fewshot_examples,
            has_system_role=assistant.has_system_role,
            prefix="Annotate this mention: ",
        )

        generated = assistant.generate(messages=messages)
        generated = str(generated)
        docid_generated = extract_docid_from_generation(generated)

        if docid_generated is None:
            failed_count += 1
            print(f"  ⚠️  Failed to extract docid for mention {idx} (id={mention_id})")
            continue

        mention.html_tag.set_attribute("docid", docid_generated)

        profile = rpr.update_from_mention(ReferenceMention(mention.html_str))

        result[mention_id] = docid_generated


    return result, failed_count


def process_file_coref(filename: str, run: CorefRunConfig, assistant, exp_dir: Path):

    # =========================================================================
    # Step 1: Resolve input path & load HTML
    # =========================================================================
    print("[Step 1] Loading input document...")
    path = run.resolve_input_path(filename)
    if not path.exists():
        print(f"  ✗ Input not found at {path} — skipping {filename}")
        return
    with open(path, encoding="utf-8") as f:
        html_content = f.read()
    print(f"  ✓ Loaded {len(html_content)} characters from {path}\n")

    # =========================================================================
    # Step 2: Extract parent-level mentions to resolve
    # =========================================================================
    print("[Step 2] Extracting mentions...")
    mentions = extract_parent_level_annotations(html_content)
    print(f"  ✓ Found {len(mentions)} mentions\n")

    # =========================================================================
    # Step 3: Load few-shot examples
    # =========================================================================
    print("[Step 3] Loading few-shot examples...")
    fewshot_examples = load_coref_fewshot_examples(
        run.fewshot_method, run.fewshot_examples, seed=run.seed
    )
    print(f"  ✓ Loaded {len(fewshot_examples)} examples\n")

    # =========================================================================
    # Step 4: Load prompt
    # =========================================================================
    print("[Step 4] Loading prompt...")
    prompt_filename = f"coref_{run.prompt_type}.txt"
    with open(PROMPT_DIR / prompt_filename, "r", encoding="utf-8") as f:
        system_prompt = f.read()
    print(f"  ✓ Loaded prompt {prompt_filename} ({len(system_prompt)} characters)\n")

    # =========================================================================
    # Step 5: Resolve coreference (checkpointed)
    # =========================================================================
    cached = get_checkpoint(exp_dir, filename)
    if cached is not None:
        result = cached
        failed_count = 0
    else:
        print("[Step 5] Resolving coreference...")
        result, failed_count = process_document_coref(
            html_content=html_content,
            mentions=mentions,
            system_prompt=system_prompt,
            fewshot_examples=fewshot_examples,
            assistant=assistant,
            max_fragments=run.max_fragments,
            profile_attributes=run.profile_attributes,
        )
        print(f"  ✓ Resolved {len(result)}/{len(mentions)} mentions")
        if failed_count:
            print(f"  ℹ️  {failed_count} mention(s) failed docid extraction\n")
        else:
            print()

        # =====================================================================
        # Step 6: Save mapping + clusters
        # =====================================================================
        print("[Step 6] Saving output...")
        mapping_path = exp_dir / f"{filename}_coref_mapping.json"
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        clusters_path = exp_dir / f"{filename}_coref_clusters.json"
        with open(clusters_path, "w", encoding="utf-8") as f:
            json.dump(dict_to_clusters(result), f, indent=2, ensure_ascii=False)
        print(f"  ✓ Saved to {mapping_path.name} and {clusters_path.name}\n")

    # =========================================================================
    # Step 7: Reconstruct the annotated HTML (always, even on a cache hit —
    # this is deterministic/cheap, no LLM calls, so it stays in sync with
    # `result` regardless of where `result` came from).
    # =========================================================================
    print("[Step 7] Reconstructing annotated HTML...")
    output_html_content = apply_coref_docids(html_content, result)
    html_path = exp_dir / f"{filename}_coref.html"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(output_html_content)
    print(f"  ✓ Saved to {html_path.name}\n")
    


    print("="*80)
    print("✓ COREF PROCESSING COMPLETE")
    print(f"✓ Experiment folder: {exp_dir}")
    print("="*80)


# =============================================================================
# Arg parser
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--filename", type=str)
    mode_group.add_argument("--split", choices=SPLITS)
    parser.add_argument("--single-split", default="dev", choices=SPLITS,
                         help="Split to look up gold input in when using --filename (single mode).")

    parser.add_argument("--input-source", default="gold", choices=["gold", "extraction"],
                         help="Read mentions from gold/annotated data, or from a prior "
                              "main_extraction.py output (for a full extraction+coref pipeline).")

    # gold source
    parser.add_argument("--data-dir", type=Path, default=None,
                         help="Gold dataset root (used when --input-source=gold). Defaults to DATA_DIR.")
    parser.add_argument("--variant", default="annotated", choices=["original", "annotated"],
                         help="Subfolder variant under the gold dataset root.")

    # extraction source
    parser.add_argument("--extraction-dir", type=Path, default=None,
                         help="Path to a main_extraction.py output folder (used when --input-source=extraction).")
    parser.add_argument("--extraction-stage", default="final",
                         choices=["processed", "final", "0", "1", "2", "3"],
                         help="Which extraction checkpoint to read as input.")

    parser.add_argument("--fewshot-method", default="random", choices=["greedy", "random"])
    parser.add_argument("--fewshot-examples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--prompt-type", default="long", choices=["short", "long"])

    parser.add_argument("--max-fragments", type=int, default=10)
    parser.add_argument("--profile-attributes", type=str, default=",".join(DEFAULT_PROFILE_ATTRIBUTES),
                         help="Comma-separated profile attributes to show in the prompt.")

    parser.add_argument("--output-dir", type=Path, default=Path("./output_coref"))
    parser.add_argument("--with-timestamp", action="store_true")

    parser.add_argument("--start", type=int, default=None,
                        help="1-based index of the first document to process (inclusive).")
    parser.add_argument("--end", type=int, default=None,
                        help="1-based index of the last document to process (inclusive).")

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.input_source == "extraction" and args.extraction_dir is None:
        parser.error("--extraction-dir is required when --input-source=extraction")

    run = CorefRunConfig(
        mode="single" if args.filename else "split",
        filename=args.filename,
        split=args.split,
        single_split=args.single_split,
        input_source=args.input_source,
        data_dir=args.data_dir or Path(DATA_DIR),
        variant=args.variant,
        extraction_dir=args.extraction_dir,
        extraction_stage=args.extraction_stage,
        fewshot_method=args.fewshot_method,
        fewshot_examples=args.fewshot_examples,
        seed=args.seed,
        model=args.model,
        temperature=args.temperature,
        prompt_type=args.prompt_type,
        max_fragments=args.max_fragments,
        profile_attributes=[a.strip() for a in args.profile_attributes.split(",") if a.strip()],
        output_dir=args.output_dir,
        with_timestamp=args.with_timestamp,
    )

    output_root = run.output_root()
    filenames = run.get_filenames()

    # ── Optional document range selection (1-based, inclusive) ──────────────
    # Same semantics as main_extraction.py: --start/--end slice the sorted
    # file list so a whole folder can be split across parallel jobs.
    total_files = len(filenames)
    start = args.start if args.start is not None else 1
    end = args.end if args.end is not None else total_files

    if start < 1 or end < start:
        parser.error(f"Invalid range: --start {start} --end {end}")
    if start > total_files:
        parser.error(f"--start {start} is greater than the number of documents ({total_files})")

    end = min(end, total_files)
    filenames = filenames[start - 1:end]
    print(f"[Setup] Selected documents {start} to {end} out of {total_files}")

    # ── Model loaded ONCE here ──────────────────────────────────────────────
    print(f"[Setup] Loading model {run.model}...")
    if run.model == "gpt-5.2":
        assistant = AssistantFactory.create_from_config({
            "type": "openai",
            "model_name": run.model,
            "temperature": run.temperature,
        })
    else:
        assistant = AssistantFactory.create(MODEL_MAPPING_NAME[run.model])

    print(f"[Setup] Mode: {run.mode.upper()} | Source: {run.input_source} | "
          f"Files: {len(filenames)} | Output: {output_root}\n")

    for i, filename in enumerate(filenames, start=start):
        print(f"\n{'='*80}")
        print(f"Processing: {filename}  (#{i}/{total_files})")
        print(f"{'='*80}")

        if run.mode == "split":
            exp_dir = output_root / filename
            exp_dir.mkdir(exist_ok=True)
        else:
            exp_dir = output_root

        process_file_coref(filename, run, assistant, exp_dir)


if __name__ == "__main__":
    main()