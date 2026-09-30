"""Main processing script for document-level annotation extraction."""

import sys
from pathlib import Path
import json
import argparse
import token
from typing import List, Optional, Dict, Any
from datetime import datetime

from configs.constants import MODEL_MAPPING_NAME

from tqdm import tqdm

# Setup project path
PROJECT_ROOT = Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))



from configs.config import CHUNK_CACHE_DIR, DATA_DIR, FEWSHOT_CACHE_DIR, KEEP_ATRIBUTES, PROMPT_DIR, SPLITS, USE_SIMPLIFIED_LABELS
from src.run_config import RunConfig
from src.extractor import (
    LabelTransformConfig, prepare_label_tokens, _parse_parent_annotations,
    build_processing_segments, get_list_of_mention
)
from src.tokenizer_utils import tokenize, decode
from src import clean_tokens
from src.chunkers.cache import cache_exists, load_cache
from src.chunkers import ChunkerFactory
from src.models import get_messages, AssistantFactory
from src.output_control.processor import OutputProcessor
from src.output_control.fallback import FallbackHandler 
from src.output_control.verification import VerificationResult 
from src.post_processing import tokens_to_html
from src.post_processing.main import tokens_to_html_after_decomposed1_3_prompting
from src.htmlLabel import simplified_to_normal_form
from src.prompts.prompt_utils import build_sublabel_definitions
from src.prompts.sublabel_definitions import SUBLABEL_DEFINITIONS_V2
from src.post_processing.token_operations import flatten_token_chunks



def get_method_config(method: str, prompt_type: str = "long") -> Dict[str, Any]:
    """Get configuration for the specified prompting method."""
    configs = {
        "AIO": {
            "parents": ["decision", "legislation", "secondary sources"],
            "already_labeled_labels": [],
            "new_labels": ["decision", "legislation", "secondary sources", "title", "citation", "source", "authors", "fragment"],
            "spans_in_context": True,
            "prompt_filename": f"allInOne_{prompt_type}.txt",
            "use_chunking": True,
            "surface_pattern": 1.0,
            "structural_pattern": 1.0,
        },
        "DEC0": {
            "parents": ["decision", "legislation", "secondary sources"],
            "already_labeled_labels": [],
            "new_labels": ["decision", "legislation", "secondary sources"],
            "spans_in_context": True,
            "prompt_filename": f"decomposed0_{prompt_type}.txt",
            "use_chunking": True,
            "surface_pattern": 0.0,
            "structural_pattern": 1.0,
        },
        "DEC1": {
            "parents": ["decision", "legislation", "secondary sources"],
            "already_labeled_labels": ["decision", "legislation", "secondary sources"],
            "new_labels": ["title", "fragment"],
            "spans_in_context": False,
            "prompt_filename": "decomposed1-3.txt",
            "use_chunking": False,
            "surface_pattern": 1.0,
            "structural_pattern": 0.0,
        },
        "DEC2": {
            "parents": ["secondary sources"],
            "already_labeled_labels": ["decision", "legislation", "secondary sources", "title", "fragment"],
            "new_labels": ["source", "authors"],
            "spans_in_context": False,
            "prompt_filename": "decomposed1-3.txt",
            "use_chunking": False,
            "surface_pattern": 1.0,
            "structural_pattern": 0.0,
        },
        "DEC3": {
            "parents": ["decision", "legislation"],
            "already_labeled_labels": ["decision", "legislation", "secondary sources", "title", "fragment", "source", "authors"],
            "new_labels": ["citation"],
            "spans_in_context": False,
            "prompt_filename": "decomposed1-3.txt",
            "use_chunking": False,
            "surface_pattern": 1.0,
            "structural_pattern": 0.0,
        },
    }
    return configs.get(method, configs["AIO"])


def load_fewshot_examples(
    method: str,
    fewshot_method: str,
    nb_examples: int,
    spans_in_context: bool = True,
    already_labeled_labels: List[str] = None,
    new_labels: List[str] = None,
    parents: List[str] = None,
    source_file: Optional[str] = None,
) -> List[tuple]:
    """Load and prepare few-shot examples based on method configuration.
    if source_file is provided, examples from that file will be excluded."""
    if already_labeled_labels is None:
        already_labeled_labels = []
    if new_labels is None:
        new_labels = []
    if parents is None:
        parents = ["decision", "legislation", "secondary sources"]
    fewshot_filename = f"examples_{fewshot_method}_surf-{get_method_config(method)['surface_pattern']}_struct-{get_method_config(method)['structural_pattern']}"
    with open(FEWSHOT_CACHE_DIR / f"{fewshot_filename}.json", "r", encoding="utf-8") as f:
        fewshot_file_content = json.load(f)

    print("fewshot examples from :", fewshot_filename)

    # Former, 27/08/2026
    #fewshot_examples = [
    #        (example["example"]["input"], example["example"]["output"])
    #        for example in fewshot_file_content["examples"]
    #    ]
    
    fewshot_examples = [
        (example["example"]["input"], example["example"]["output"])
        for example in fewshot_file_content["examples"] if example["source_file"] != source_file
    ]
    
    # Create label configs
    input_label_config = LabelTransformConfig(
        use_simplified=True,
        switch_type=True,
        keep_labels=already_labeled_labels,
        keep_attributes=["labelname"]
    )
    
    output_label_config = LabelTransformConfig(
        use_simplified=True,
        switch_type=True,
        keep_labels=already_labeled_labels + new_labels,
        keep_attributes=["labelname"]
    )
    
    # Transform examples
    final_fewshot = []
    total_output_text = ""
    
    for example in fewshot_examples:
        input_text, output_text = example
        
        input_tokens = tokenize(input_text)
        transformed_input_tokens = prepare_label_tokens(input_tokens, input_label_config)
        
        output_tokens = tokenize(output_text)
        transformed_output_tokens = prepare_label_tokens(output_tokens, output_label_config)
        
        if spans_in_context:
            final_fewshot.append((decode(transformed_input_tokens), decode(transformed_output_tokens)))
        else:
            total_output_text += "|||" + decode(transformed_output_tokens)

    final_fewshot = final_fewshot[:nb_examples]
    
    if not spans_in_context:
        parents_dict = _parse_parent_annotations(total_output_text)
        for parent_name, annotations in parents_dict.items():
            if parent_name not in parents:
                continue
            for annotation in annotations:
                input_text = decode(
                    prepare_label_tokens(
                        simplified_to_normal_form(tokenize(annotation), label_type="manual_label"),
                        input_label_config
                    )
                )
                if input_text != annotation:
                    final_fewshot.append((input_text, annotation))
    
    return final_fewshot


def load_html_document(path: Path) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def get_token_chunks(html_content: str, chunker: str, split: str, filename: str, annotated: bool = False) -> list[list[str]]:
    """Get token chunks, using cache if available."""

    if annotated:
        cache_dir = f"{CHUNK_CACHE_DIR}/annotated"
    else:
        cache_dir = f"{CHUNK_CACHE_DIR}/original"
    
    if not cache_exists(method=chunker, cache_dir=cache_dir, split=split, filename=filename):
        nlp = None
        if chunker == "sentence":
            import spacy
            nlp = spacy.load("en_core_web_trf")
            print("✅ spaCy model loaded.\n")
        
        token_chunks = ChunkerFactory.get_chunks(
            html_content, method=chunker, split=split, filename=filename, nlp=nlp
        )
    else:
        token_chunks = load_cache(method=chunker, cache_dir=cache_dir, split=split, filename=filename)
    
    return token_chunks


def process_output(
    generated: str,
    token_chunk,
    allowed_labels: List[str],
    assistant,
    with_fallback: bool = True,
) -> tuple:
    """Process and verify LLM output."""
    controller = OutputProcessor()
    fallback_handler = FallbackHandler(processor=controller)
    
    corrected_generated_tokens, status = controller.process(
        raw_llm_output=generated,
        original_chunk=token_chunk,
        allowed_labels=allowed_labels
    )

    if status.passed:
        return corrected_generated_tokens, status

    if not with_fallback:
        return token_chunk, status
    
    
    corrected_generated_tokens, status_dict = fallback_handler.handle_failure(
        assistant=assistant,
        corrected_output=corrected_generated_tokens,
        original_chunk=token_chunk,
        initial_status=status,
        allowed_labels=allowed_labels,
        fallback_prompt_filename="fallback.txt"
    )
    # Convert dict to VerificationResult
    status = VerificationResult(
        passed=status_dict.get('passed', False),
        error_type=status_dict.get('error_type'),
        details=status_dict.get('error_details'),
        tokens=corrected_generated_tokens
    )
    
    return corrected_generated_tokens, status


def get_checkpoint(exp_dir: Path, filename: str, step: str) -> Optional[str]:
    """Return HTML content if checkpoint exists, else None.
    
    step examples: 'processed', '0', '1', '2', '3', 'final'
    """
    checkpoint_path = exp_dir / f"{filename}_{step}.html"
    if checkpoint_path.exists():
        print(f"  ⏭  Checkpoint found: {checkpoint_path.name} — skipping")
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            return f.read()
    return None


def process_aio_dec0(
    token_chunks: List,
    system_prompt: str,
    fewshot_examples: List[tuple],
    allowed_labels: List[str],
    assistant,
    disable_fallback: bool = False,
) -> List:
    """Process document using AIO or DEC0 method (chunk-based)."""
    print(f"  Processing {len(token_chunks)} chunks...\n")
    
    processed_chunks = []
    fallback_count = 0

    print(">>> ENTERED process_aio_dec0 v2 <<<", flush=True)
    
    for chunk_tokens in tqdm(token_chunks, desc="Processing chunks"):
        print("This is the main loop", flush=True)
        user_input = decode(chunk_tokens)
        messages = get_messages(
            system_prompt=system_prompt,
            user_input=user_input,
            fewshot_examples=fewshot_examples,
            has_system_role=assistant.has_system_role
        )
        
        generated = assistant.generate(messages=messages)
        print(f"  [chunk] primary generate returned ({len(generated)} chars)", flush=True)
        corrected_generated_tokens, status = process_output(
            generated=generated,
            token_chunk=chunk_tokens,
            allowed_labels=allowed_labels,
            assistant=assistant,
            with_fallback=not disable_fallback,
        )
        print(f"  [chunk] process_output returned (passed={status.passed})", flush=True)
        
        processed_chunks.append(corrected_generated_tokens)
        
        if not status.passed:
            fallback_count += 1
    
    return processed_chunks, fallback_count


def process_dec1_3(
    html_content: str,
    system_prompt: str,
    fewshot_examples: List[tuple],
    allowed_labels: List[str],
    assistant,
    parents: List[str],
    already_labeled_labels: List[str],
    new_labels: List[str],
    disable_fallback: bool = False,
) -> List:
    """Process document using DEC1-3 methods (mention-based)."""
    # Tokenize document
    tokens = tokenize(html_content)
    
    # Get already extracted parent mentions
    parent_mentions = get_list_of_mention(
        tokens=tokens,
        keep_labels=parents,
        label_type="auto_label"
    )
    
    print(f"  Found {len(parent_mentions)} parent mentions to process")
    
    segments = build_processing_segments(tokens, parent_mentions)
    print(f"  Built {len(segments)} token segments ({sum(s['process'] for s in segments)} to process)\n")
    
    # Create input label config for processing
    input_label_config = LabelTransformConfig(
        use_simplified=USE_SIMPLIFIED_LABELS,
        switch_type=False,
        keep_labels=already_labeled_labels,
        keep_attributes=KEEP_ATRIBUTES
    )
    
    config = LabelTransformConfig(
        use_simplified=False,
        switch_type=False,
        keep_labels=already_labeled_labels,
        keep_attributes=KEEP_ATRIBUTES
    )
    
    failed_count = 0
    
    for segment in tqdm(segments, desc="Processing mentions"):
        if not segment["process"]:
            continue
        
        mention = segment["tokens"]
        html_label = segment["meta"]["label"]
        
        prepared_tokens = prepare_label_tokens(mention, input_label_config)
        user_input = decode(prepared_tokens)
        
        # Filter few-shot examples by parent label
        filtered_fewshot = [
            example for example in fewshot_examples
            if example[0].startswith(f"<{html_label.name}>")
        ]
        
        
        messages = get_messages(
            system_prompt=system_prompt,
            user_input=user_input,
            fewshot_examples=filtered_fewshot,
            has_system_role=assistant.has_system_role
        )
        
        generated = assistant.generate(messages=messages)
        corrected_generated_tokens, status = process_output(
            generated=generated,
            token_chunk=prepare_label_tokens(mention, config),
            allowed_labels=allowed_labels,
            assistant=assistant,
            with_fallback=False,
        )
        
        if not status.passed:
            failed_count += 1
        
        segment["tokens"] = corrected_generated_tokens
    
    processed_tokens = [
        token for segment in segments for token in segment["tokens"]
    ]
    
    return processed_tokens, failed_count


def process_file(filename: str, run: RunConfig, assistant, exp_dir: Path):

    # =========================================================================
    # Step 1: Load HTML content
    # =========================================================================
    print("[Step 1] Loading HTML document...")
    path = run.resolve_input_path(filename) # the split is updated here
    html_content = load_html_document(path)
    print(f"  ✓ Loaded {len(html_content)} characters from {path}\n")

    
    
    # =========================================================================
    # AIO Method: All-in-one single pass
    # =========================================================================
    if run.method == "AIO":
        print("[AIO] ALL-IN-ONE METHOD")
        print("="*80)
        
        config = get_method_config("AIO", run.prompt_type)
        
        # Step 3: Chunk the document
        print("[Step 3] Chunking document...")
        token_chunks = get_token_chunks(html_content, run.chunker, run.split, filename)
        print(f"  ✓ Created {len(token_chunks)} chunks\n")
        
        # Step 4: Load few-shot examples
        print("[Step 4] Loading few-shot examples...")
        fewshot_examples = load_fewshot_examples(
            run.method,
            run.fewshot_method,
            run.fewshot_examples,
            spans_in_context=config["spans_in_context"],
            already_labeled_labels=config["already_labeled_labels"],
            new_labels=config["new_labels"],
            parents=config["parents"],
            source_file=filename
        )
        allowed_labels = config["already_labeled_labels"] + config["new_labels"]
        print(f"  ✓ Loaded {len(fewshot_examples)} examples")
        print(f"  ✓ Allowed labels: {', '.join(allowed_labels)}\n")
        
        # Step 5: Load prompt
        print("[Step 5] Loading prompt...")
        with open(PROMPT_DIR / config["prompt_filename"], "r", encoding="utf-8") as f:
            system_prompt = f.read()
        print(f"  ✓ Loaded prompt ({len(system_prompt)} characters)\n")
        
        # Step 6: Process all chunks
        cached = get_checkpoint(exp_dir, filename, "processed")
        if cached is not None:
            output_html_content = cached
        else:
            print("[Step 6] Processing chunks...")
            processed_chunks, fallback_count = process_aio_dec0(
                token_chunks, system_prompt, fewshot_examples,
                allowed_labels, assistant, run.disable_fallback
            )
            print(f"  ✓ Processed all {len(token_chunks)} chunks")
            if fallback_count > 0:
                print(f"  ℹ️  Fallback mechanism invoked {fallback_count} time(s)\n")
            else:
                print()
            
            # Step 7: Post-processing
            print("[Step 7] Post-processing document...")

            processed_tokens_flat = flatten_token_chunks(processed_chunks)

            output_html_content = tokens_to_html(processed_tokens_flat, html_content)
            print(f"  ✓ Post-processing complete\n")
            
            # Step 8: Save output
            print("[Step 8] Saving output...")
            output_filename = exp_dir / f"{filename}_processed.html"
            with open(output_filename, "w", encoding="utf-8") as f:
                f.write(output_html_content)
            print(f"  ✓ Saved to {output_filename}\n")
    
    # =========================================================================
    # DEC Method: Decomposed 4-pass method
    # =========================================================================
    elif run.method == "DEC":
        print("[DEC] DECOMPOSED METHOD (4 PASSES)")
        print("="*80)
        
        dec_methods = ["DEC0", "DEC1", "DEC2", "DEC3"]
        current_html = html_content
        
        for dec_idx, dec_method in enumerate(dec_methods):
            print(f"\n{'='*80}")
            print(f"[{dec_method}] {dec_method} Pass")
            print(f"{'='*80}\n")
            
            config = get_method_config(dec_method, run.prompt_type)

            # ── Checkpoint check ──────────────────────────────────────────
            cached = get_checkpoint(exp_dir, filename, str(dec_idx))
            if cached is not None:
                current_html = cached   # feed into next pass as usual
                continue                # skip everything below
        # ─────────────────────────────────────────────────────────────
            
            # Load few-shot examples
            print(f"[{dec_method}] Step 1: Loading few-shot examples...")
            fewshot_examples = load_fewshot_examples(
                dec_method,
                run.fewshot_method,
                run.fewshot_examples,
                spans_in_context=config["spans_in_context"],
                already_labeled_labels=config["already_labeled_labels"],
                new_labels=config["new_labels"],
                parents=config["parents"],
                source_file=filename,
            )
            allowed_labels = config["already_labeled_labels"] + config["new_labels"]
            print(f"  ✓ Loaded {len(fewshot_examples)} examples")
            print(f"  ✓ Allowed labels: {', '.join(allowed_labels)}\n")
            
            # Load prompt
            print(f"[{dec_method}] Step 2: Loading prompt...")
            with open(PROMPT_DIR / config["prompt_filename"], "r", encoding="utf-8") as f:
                system_prompt = f.read()
            
            # For DEC1-3, format prompt with sublabel definitions
            if dec_method in ["DEC1", "DEC2", "DEC3"]:
                sublabels_str = ", ".join(config["new_labels"])
                sublabels_definition = build_sublabel_definitions(
                    set(config["new_labels"]) - set(config["parents"]),
                    sublabel_definitions=SUBLABEL_DEFINITIONS_V2
                )
                system_prompt = system_prompt.format(
                    sublabels=sublabels_str,
                    sublabels_definition=sublabels_definition,
                )
            
            print(f"  ✓ Loaded prompt ({len(system_prompt)} characters)\n")
            
            # Process based on method
            print(f"[{dec_method}] Step 3: Processing document...")
            
            if config["use_chunking"]:
                # DEC0 uses chunking
                token_chunks = get_token_chunks(current_html, run.chunker, run.split, filename)
                print(f"  ✓ Created {len(token_chunks)} chunks\n")
                
                processed_chunks, fallback_count = process_aio_dec0(
                    token_chunks, system_prompt, fewshot_examples,
                    allowed_labels, assistant, run.disable_fallback
                )
                print(f"  ✓ Processed all {len(token_chunks)} chunks")
                if fallback_count > 0:
                    print(f"  ℹ️  Fallback mechanism invoked {fallback_count} time(s)\n")
                
                # Post-process
                print(f"[{dec_method}] Step 4: Post-processing document...")
                processed_tokens_flat = flatten_token_chunks(processed_chunks)

                current_html = tokens_to_html(processed_tokens_flat, current_html)
                print(f"  ✓ Post-processing complete\n")
            else:
                # DEC1-3 use mention-based processing
                processed_tokens, failed_count = process_dec1_3(
                    current_html, system_prompt, fewshot_examples, allowed_labels,
                    assistant, config["parents"], config["already_labeled_labels"],
                    config["new_labels"], disable_fallback=True
                )
                print(f"  ✓ Processed mentions")
                if failed_count > 0:
                    print(f"  ℹ️  {failed_count} mention(s) failed verification\n")
                
                # Post-process
                print(f"[{dec_method}] Step 4: Post-processing document...")
                current_html = tokens_to_html_after_decomposed1_3_prompting(processed_tokens, current_html)
                print(f"  ✓ Post-processing complete\n")
            
            # Save intermediate result
            print(f"[{dec_method}] Step 5: Saving intermediate result...")
            intermediate_filename = exp_dir / f"{filename}_{dec_idx}.html"
            with open(intermediate_filename, "w", encoding="utf-8") as f:
                f.write(current_html)
            print(f"  ✓ Saved to {intermediate_filename}\n")
        
        # Final output — also checkpointed
        cached_final = get_checkpoint(exp_dir, filename, "final")
        if cached_final is None:
            print(f"\n{'='*80}")
            print("✓ ALL DECOMPOSED PASSES COMPLETE")
            print(f"{'='*80}")
            final_filename = exp_dir / f"{filename}_final.html"
            with open(final_filename, "w", encoding="utf-8") as f:
                f.write(current_html)
            print(f"\n✓ Final output saved to {final_filename}\n")
    
    print("="*80)
    print("✓ PROCESSING COMPLETE")
    print(f"✓ Experiment folder: {exp_dir}")
    print("="*80)

# ── Arg parser ──────────────────────────────────────────────────────────────

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--filename", type=str)
    mode_group.add_argument("--split", choices=SPLITS)

    # --data-dir overrides the dataset root (default: DATA_DIR)
    # --variant selects the subfolder under that root (default: "original")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="Dataset root, e.g. data/incoming. Defaults to DATA_DIR.")
    parser.add_argument("--variant", default="original", choices=["original", "annotated"],
                        help="Subfolder variant under the dataset root.")

    parser.add_argument("--method", default="AIO", choices=["AIO", "DEC"])
    parser.add_argument("--chunker", default="paragraph", choices=["paragraph", "sentence"])
    parser.add_argument("--fewshot-method", default="greedy", choices=["greedy", "random"])
    parser.add_argument("--fewshot-examples", type=int, default=6)
    parser.add_argument("--model", default="gpt-5.2")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=Path("./output"))
    parser.add_argument("--disable-fallback", action="store_true")
    parser.add_argument("--with-timestamp", action="store_true")
    parser.add_argument("--prompt-type", default="long", choices=["short", "long"])

    parser.add_argument("--start", type=int, default=None,
                        help="1-based index of the first document to process (inclusive).")
    parser.add_argument("--end", type=int, default=None,
                        help="1-based index of the last document to process (inclusive).")

    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    run = RunConfig(
        mode="single" if args.filename else "split",
        filename=args.filename,
        split=args.split,
        method=args.method,
        chunker=args.chunker,
        fewshot_method=args.fewshot_method,
        fewshot_examples=args.fewshot_examples,
        model=args.model,
        temperature=args.temperature,
        disable_fallback=args.disable_fallback,
        with_timestamp=args.with_timestamp,
        data_dir=args.data_dir or Path(DATA_DIR),
        variant=args.variant,
        output_dir=args.output_dir,
        prompt_type=args.prompt_type,
    )

    output_root = run.output_root()
    filenames = run.get_filenames()

    # ── Optional document range selection (1-based, inclusive) ──────────────
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

    print(f"[Setup] Mode: {run.mode.upper()} | Files: {len(filenames)} | Output: {output_root}\n")

    for i, filename in enumerate(filenames, start=start):
        print(f"\n{'='*80}")
        print(f"Processing: {filename}  (#{i}/{total_files})")
        print(f"{'='*80}")
        
        # Per-file output subfolder (only in split mode; flat in single mode)
        if run.mode == "split":
            exp_dir = output_root / filename
            exp_dir.mkdir(exist_ok=True)
        else:
            exp_dir = output_root

        split = run.split if run.mode == "split" else "test"  # or make split required always
        
        
        process_file(filename, run, assistant, exp_dir)


    
    


if __name__ == "__main__":
    main()