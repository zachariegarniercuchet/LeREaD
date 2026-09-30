"""Main post-processing orchestration for document-level HTML processing."""

from ..tokenizer_utils import tokenize, decode
from ..html_utils import is_auto_label_tag, is_tag_token

from .token_operations import merge_tokens_general, flatten_token_chunks
from .validation import compare_html_allow_auto_labels, verify_end_to_end_preservation
from .bracket_fixing import correct_tokens_brackets, check_tokens_brackets
from .html_operations import add_attributes_to_auto_labels, clean_html_formatting, fix_labels


def tokens_to_html(processed_tokens, html_content):
    """
    Main orchestration function for document-level post-processing.
    
    This function:
    1. Flattens processed chunks from the model into a single token list
    2. Tokenizes the original HTML content
    3. Merges original tokens with processed tokens, preserving auto_label insertions
    4. Validates the merge against the original HTML
    5. Corrects any bracket nesting issues introduced by merging
    6. Validates bracket coherence
    7. Cleans up HTML formatting
    8. Adds label scheme attributes to auto_labels
    9. Verifies end-to-end preservation: final output must be the original
       HTML plus only the authorized auto_label insertions (exact text
       identity + tag-structure identity), failing loudly otherwise
    
    Args:
        processed_chunks: List of lists of tokens from the model output
        html_content: Original HTML content string
    
    Returns:
        str: Processed HTML with properly annotated auto_label tags
    
    Raises:
        AssertionError: If validation or bracket checking fails
    """
    
    # =====================================================================
    # Step 2: Tokenize original HTML and merge with processed tokens
    # =====================================================================
    original_tokens = tokenize(html_content)
    
    processed_html_content_tokens = merge_tokens_general(
        original_tokens=original_tokens,
        derived_tokens=processed_tokens,
        is_protected_func=lambda tok: is_auto_label_tag(tok) != 0,
        is_opening_protected_func=lambda tok: is_auto_label_tag(tok) == 1,
        is_tag_token_func=lambda tok: is_tag_token(tok),
        log=False
    )
    print(f"   ✓ Merged to {len(processed_html_content_tokens)} tokens")
    
    # =====================================================================
    # Step 4: Validate merge against original HTML
    # =====================================================================
    comparison_result = compare_html_allow_auto_labels(
        decode(processed_html_content_tokens), 
        html_content
    )
    assert comparison_result, (
        "The processed HTML content does not match the original HTML content "
        "when ignoring auto_label tags. Please check the merging and "
        "post-processing steps for errors."
    )
    
    # =====================================================================
    # Step 5: Fix bracket nesting issues
    # =====================================================================
    processed_html_content_tokens_corrected = correct_tokens_brackets(processed_html_content_tokens)
    print(f"   ✓ Corrected {len(processed_html_content_tokens_corrected)} tokens")
    
    # =====================================================================
    # Step 6: Validate bracket coherence
    # =====================================================================
    ok, message, position, context = check_tokens_brackets(processed_html_content_tokens_corrected)
    assert ok, (
        f"The brackets in the merged tokens are not balanced: {message} "
        f"(at position {position}). Please check the merging and bracket "
        f"correction steps for errors.\nContext: {context}"
    )
    print(f"   ✓ {message}")


    # =====================================================================
    # Step 7: Push tags outside auto_labels
    # =====================================================================
    processed_html = decode(processed_html_content_tokens_corrected)
    processed_html_swapped = fix_labels(processed_html)
    
    # =====================================================================
    # Step 8: Clean HTML formatting
    # =====================================================================
    processed_html_cleaned = clean_html_formatting(processed_html_swapped)
    
    # =====================================================================
    # Step 9: Add label scheme attributes
    # =====================================================================
    processed_html_content = add_attributes_to_auto_labels(processed_html_cleaned)

    # =====================================================================
    # Step 10: End-to-end preservation safeguard
    # =====================================================================
    # The checks above run BEFORE fix_labels/cleanup/attribute injection and
    # therefore cannot catch corruption introduced by those stages (e.g. the
    # former fix_labels oscillation that unwrapped nested sublabels). Verify
    # the FINAL output against the original input: identical text, tags
    # identical modulo the authorized auto_label insertions. Raises on any
    # violation so corruption can never flow downstream silently.
    verify_end_to_end_preservation(processed_html_content, html_content)

    # =====================================================================
    # Final result
    # =====================================================================
    print("\n" + "="*80)
    print("✓ POST-PROCESSING COMPLETE")
    print("="*80)
    print(f"Final HTML length: {len(processed_html_content)} characters\n")
    
    return processed_html_content


def tokens_to_html_after_decomposed1_3_prompting(processed_tokens, html_content):
    """
    Orchestration function for converting processed tokens to HTML after the 
    decomposed 1.3 prompting step.
    
    This function:
    1. Validates the processed tokens against the original HTML
    2. Corrects any bracket nesting issues
    3. Validates bracket coherence
    4. Cleans up HTML formatting
    5. Adds label scheme attributes to auto_labels
    
    Args:
        processed_tokens: List of tokens from the model output after decomposed prompting
        html_content: Original HTML content string
    """
    # Useless verification to check if the tokens are the same after processing (except for the auto labels)
    t1 = []
    t2 = []
    for token in processed_tokens:
        if not is_auto_label_tag(token) in [1, 2]:
            t1.append(token)

    original_tokens = tokenize(html_content)
    for token in original_tokens:
        if not is_auto_label_tag(token) in [1, 2]:
            t2.append(token)

    for o, p in zip(t1, t2):
        if o != p:
            print("Original token: ", o)
            print("Processed token: ", p)

    assert t1 == t2, "The tokens are different after processing, which should not happen as we are only adding auto_label tags without changing the original tokens."

    processed_html = decode(processed_tokens)
    comparison_result = compare_html_allow_auto_labels(processed_html, html_content)
    assert comparison_result, "The processed HTML content does not match the original HTML content when ignoring auto_label tags."


    
    processed_html_cleaned = clean_html_formatting(processed_html)

    processed_html_final = add_attributes_to_auto_labels(processed_html_cleaned)


    return processed_html_final



