"""HTML validation and comparison utilities."""

from bs4 import BeautifulSoup

from ..html_utils import strip_auto_labels
from .html_operations import clean_html_formatting


def compare_html_allow_auto_labels(merged_html: str, original_html: str) -> bool:
    """
    Compare two HTML strings character-by-character, considering them equivalent
    if the only differences are the presence or placement of <auto_label ...>
    and </auto_label> tags, empty formatting tags, or redundant tag pairs.

    This function is used to verify that the merging process hasn't introduced
    unwanted changes to the original HTML structure.

    Args:
        merged_html: The merged HTML with auto_labels
        original_html: The original HTML without auto_labels

    Returns:
        bool: True if the HTMLs match after normalization, False otherwise
              Prints detailed diff information on mismatch
    """
    
    # Strip auto_labels and clean formatting artifacts
    a = strip_auto_labels(merged_html)
    b = strip_auto_labels(original_html)
    
    # Clean HTML formatting (removes empty tags and redundant pairs)
    a = clean_html_formatting(a)
    b = clean_html_formatting(b)
    
    if a == b:
        print("   ✓ HTMLs match after normalization (ignoring auto_label tags and formatting artifacts)")
        return True
    
    # Find first index of difference
    min_len = min(len(a), len(b))
    diff_idx = None
    for i in range(min_len):
        if a[i] != b[i]:
            diff_idx = i
            print(f"   ✗ Difference at index {diff_idx}: '{a[i]}' vs '{b[i]}'")
            break
    if diff_idx is None and len(a) != len(b):
        diff_idx = min_len
    
    # Print a small window around the difference
    if diff_idx is not None:
        start = max(0, diff_idx - 50)
        end_a = min(len(a), diff_idx + 50)
        end_b = min(len(b), diff_idx + 50)
        print("   ✗ Difference found (ignoring auto_label):")
        print("--- merged_html (stripped) ---")
        print(a[start:end_a])
        print("--- original_html (stripped) ---")
        print(b[start:end_b])
    else:
        print("   ✗ Difference detected but could not locate index")
    
    return False

# NOT USED FUNCION
def verify_end_to_end_preservation(final_html: str, original_html: str) -> bool:
    """Strict input -> output preservation guarantee for post-processing.

    Contract: ``final_html`` must equal ``original_html`` plus ONLY the
    authorized token insertions (``auto_label`` tags and their attributes).
    Words, punctuation and all other HTML tags must be
    preserved exactly. Every post-processing stage (merge, bracket fixing,
    ``fix_labels``, formatting cleanup, attribute injection) is only allowed
    to move/add tags — never to alter, drop or reorder text or tags.

    Two checks, fail loudly (AssertionError) on any violation:

    1. Exact text identity: all tags stripped, the character streams must be
       identical. Catches any word/punctuation change, however small.
    2. Tag-structure identity modulo authorized insertions: reuses
       :func:`compare_html_allow_auto_labels`, which additionally tolerates
       only the formatting-tag split/merge normalization the pipeline itself
       intentionally performs around label boundaries
       (``<i>ab</i>`` -> ``<i>a</i><label/><i>b</i>``).

    Returns True when both checks pass.
    """
    # --- Check 1: exact text identity (words + punctuation) ---
    final_text = BeautifulSoup(final_html, "html.parser").get_text()
    original_text = BeautifulSoup(original_html, "html.parser").get_text()

    if final_text != original_text:
        min_len = min(len(final_text), len(original_text))
        diff_idx = next(
            (i for i in range(min_len) if final_text[i] != original_text[i]),
            min_len,
        )
        start = max(0, diff_idx - 50)
        raise AssertionError(
            "Post-processing altered the document text: output is not the "
            "input plus label insertions.\n"
            f"First text divergence at character {diff_idx}.\n"
            f"--- output   --- ...{final_text[start:diff_idx + 50]!r}...\n"
            f"--- original --- ...{original_text[start:diff_idx + 50]!r}..."
        )

    # --- Check 2: tag structure identity modulo label insertions ---
    if not compare_html_allow_auto_labels(final_html, original_html):
        raise AssertionError(
            "Post-processing altered HTML tags beyond the authorized "
            "auto_label insertions (see diff above)."
        )

    print("   ✓ End-to-end preservation verified: identical text, tags identical modulo label insertions")
    return True
