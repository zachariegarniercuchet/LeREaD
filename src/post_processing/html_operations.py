import re
import json
from bs4 import BeautifulSoup, NavigableString, Tag

from configs.config import LABEL_SCHEME_PATH
from ..html_utils import is_auto_label_tag, is_manual_label_tag


def is_pure_whitespace(node):
    """Return True if node is a NavigableString containing only whitespace."""
    return isinstance(node, NavigableString) and str(node).strip() == ""

def get_significant_children(tag):
    """Return children that are not pure-whitespace text nodes."""
    return [c for c in tag.children if not is_pure_whitespace(c)]

def fix_labels(html_content, max_iterations=None):
    """Push formatting tags outside auto_label boundaries.

    Only formatting/content tags are ever considered as restructuring
    candidates. Nested ``auto_label``/``manual_label`` tags are NEVER
    unwrapped or wrapped here: treating a nested label as a "wrapping tag"
    both destroys sublabel annotations and can oscillate forever between
    ``parent > child`` and ``child > parent`` states (every pass reports a
    change, so the ``while changed`` loop never exits). That configuration
    is routine in AIO output (a parent mention covered by a single
    sublabel), which is why AIO hung while DEC0 (parent labels only) did not.

    Termination: each applied fix unwraps >= 1 formatting tag from inside a
    label and places the replacement strictly outside it, so tags only ever
    move outward and the loop must converge. ``max_iterations`` is a tripwire
    (fail loudly, never hang silently) for any unforeseen pathology.
    """

    def normalize_attr_value(k, v):
        if k == "style":
            # Remove all spaces around : and ; for consistent comparison
            return ";".join(
                p.strip() for p in v.replace(" ", "").split(";") if p.strip()
            )
        if isinstance(v, list):
            return " ".join(v)
        return str(v)


    changed = True
    soup = BeautifulSoup(html_content, 'html.parser')

    if max_iterations is None:
        n_labels = len(soup.find_all(["auto_label", "manual_label"]))
        max_iterations = max(500, 5 * n_labels + 50)
    iteration = 0
    fixes = 0

    while changed:
        iteration += 1
        if iteration > max_iterations:
            raise RuntimeError(
                f"fix_labels did not converge after {max_iterations} passes "
                f"({fixes} fixes applied). Aborting instead of hanging: inspect "
                f"label/formatting-tag nesting in the input."
            )
        changed = False
        for label in soup.find_all(["auto_label", "manual_label"]):

            # Get the full text of the label (ground truth)
            label_text = label.get_text().strip()
            if not label_text:
                continue

            # For each possible wrapping tag type found inside the label,
            # check if concatenation of all its text == label text.
            # NOTE: nested auto_label/manual_label tags are excluded — they
            # are annotations, not formatting, and must be preserved as-is.
            candidate_tags = {}  # (tag_name, attrs_tuple) -> [list of tag instances]
            for child_tag in label.find_all(True):
                if child_tag.name in ("auto_label", "manual_label"):
                    continue
                key = (child_tag.name, tuple(sorted(
                    (k, normalize_attr_value(k, v))
                    for k, v in child_tag.attrs.items()
                )))
                if key not in candidate_tags:
                    candidate_tags[key] = []
                candidate_tags[key].append(child_tag)

            # Filter out instances that are descendants of another instance with the same key
            filtered_candidate_tags = {}
            for key, instances in candidate_tags.items():
                top_level = []
                for instance in instances:
                    # Check if any ancestor of this instance is also in the same group
                    is_nested = any(
                        ancestor in instances
                        for ancestor in instance.parents
                    )
                    if not is_nested:
                        top_level.append(instance)
                filtered_candidate_tags[key] = top_level

            winner = None
            winner_instances = None
            for (tag_name, attrs_tuple), instances in filtered_candidate_tags.items():
                # Concatenate text of all instances of this tag
                combined_text = "".join(t.get_text() for t in instances).strip()
                if combined_text.replace(" ", "") == label_text.replace(" ", ""):
                    winner = (tag_name, attrs_tuple)
                    winner_instances = instances
                    break

            if winner is None:
                continue

            winning_tag_name, winning_attrs_tuple = winner
            winning_attrs = {
                k: (v.split(" ") if k == "class" else v)  
                for k, v in winning_attrs_tuple
            }

            # Unwrap all instances of the winning tag inside the label
            for instance in winner_instances:
                instance.unwrap()

            # Wrap the label with the winning tag
            outer = soup.new_tag(winning_tag_name, **winner_instances[0].attrs)
            label.wrap(outer)

            changed = True
            fixes += 1
            break

    if fixes:
        print(f"   ✓ fix_labels applied {fixes} formatting-tag swap(s) in {iteration} pass(es)")
    return str(soup)



def clean_html_formatting(html: str, tags_to_clean: set = None, debug: bool = False) -> str:
    """
    Clean HTML by removing useless formatting artifacts WITHOUT changing any text content.
    
    This function performs comprehensive HTML normalization by:
    1. PASS 1: Remove ALL empty tags (tags with NO children)
    2. PASS 2: Merge ALL adjacent identical tags (same name + attributes, no text between)
    3. Repeat until no more changes
    
    CRITICAL: Only merges tags that are truly adjacent with no text nodes between them.
    This preserves ALL text content (including spaces) for character-by-character comparison.
    
    Helps normalize HTML for comparison by removing artifacts like:
    - Empty tags: <i></i>, <span class="..."></span>
    - Adjacent empty + non-empty: <i></i><i>text</i> → <i>text</i>
    - Adjacent identical tags: <b>a</b><b>b</b> → <b>ab</b>
    
    Does NOT merge tags with ANY content between them:
    - <i>a</i> <i>b</i> → stays as is (space preserved)
    - <b>a</b>text<b>b</b> → stays as is
    
    Args:
        html: HTML string to clean
        tags_to_clean: Set of tag names to check. If None, checks common formatting tags.
        debug: If True, print debug information about cleaning operations
    
    Returns:
        Cleaned HTML string with useless formatting removed, all text preserved
    
    Examples:
        >>> clean_html_formatting('<b>text</b><b>more</b>')
        '<b>textmore</b>'
        >>> clean_html_formatting('<i></i><i>text</i>')
        '<i>text</i>'
        >>> clean_html_formatting('<span></span>text')
        'text'
        >>> clean_html_formatting('<i>a</i> <i>b</i>')  # space preserved
        '<i>a</i> <i>b</i>'
    """
    if tags_to_clean is None:
        tags_to_clean = {"span", "i", "b", "strong", "u", "em", "mark", "sup", "sub"}
    
    soup = BeautifulSoup(html, 'html.parser')
    
    max_iterations = 50  # Safety limit
    total_empty_removed = 0
    total_merged = 0
    
    if debug:
        print(f"\n=== Starting clean_html_formatting ===")
        print(f"Tags to clean: {tags_to_clean}")
        print(f"Input length: {len(html)} chars")
    
    # Loop until no more changes can be made
    for iteration in range(max_iterations):
        if debug:
            print(f"\n--- Iteration {iteration + 1} ---")
        
        empty_removed_this_pass = 0
        # PASS 1: Remove ALL empty tags (or whitespace-only tags) in one complete pass
        for tag_name in tags_to_clean:
            while True:
                tags = soup.find_all(tag_name)
                found_empty = False
                
                for tag in tags:
                    children = list(tag.children)
                    has_element_children = any(
                        isinstance(c, Tag) or (isinstance(c, NavigableString) and c.strip() != "")
                        for c in children
                    )
                    
                    # Remove tag if it has no element children (pure text, space, or truly empty)
                    # Always unwrap: keep whatever text is inside, just strip the tag itself
                    if not has_element_children:
                        if debug:
                            print(f"  [PASS 1] Unwrapping <{tag_name}>: {str(tag)[:60]}")
                        tag.unwrap()
                        empty_removed_this_pass += 1
                        found_empty = True
                        break
                
                if not found_empty:
                    break
        
        # PASS 2: Merge ALL adjacent identical tags in one complete pass
        merged_this_pass = 0
        
        for tag_name in tags_to_clean:
            while True:
                tags = soup.find_all(tag_name)
                found_merge = False
                
                for tag in tags:
                    # Look at the next sibling, skipping over pure-whitespace text nodes
                    next_sib = tag.next_sibling
                    whitespace_between = None
                    if (next_sib and
                        isinstance(next_sib, NavigableString) and
                        next_sib.strip() == ""):
                        whitespace_between = next_sib   # remember it so we can remove it
                        next_sib = next_sib.next_sibling

                    # Only merge if next sibling is same tag type with same attributes
                    if (next_sib and
                        hasattr(next_sib, 'name') and
                        next_sib.name == tag_name and
                        dict(tag.attrs) == dict(next_sib.attrs)):

                        if debug:
                            tag_str = str(tag)[:60] + "..." if len(str(tag)) > 60 else str(tag)
                            next_str = str(next_sib)[:60] + "..." if len(str(next_sib)) > 60 else str(next_sib)
                            print(f"  [PASS 2] Merging <{tag_name}> tags:")
                            print(f"           First:  {tag_str}")
                            print(f"           Second: {next_str}")

                        # Merge: move whitespace inside the tag first, then the contents of next_sib
                        if whitespace_between is not None:
                            tag.append(whitespace_between)  # moves the space node inside <b>

                        for child in list(next_sib.children):
                            tag.append(child)

                        next_sib.decompose()
                        merged_this_pass += 1
                        found_merge = True
                        break

                if not found_merge:
                    break
        
        
        total_merged += merged_this_pass
        if debug and merged_this_pass > 0:
            print(f"  [PASS 2] Merged {merged_this_pass} adjacent tag pairs")
        
        # If no changes in this iteration, we're done
        if empty_removed_this_pass == 0 and merged_this_pass == 0:
            if debug:
                print(f"\n=== Cleaning complete after {iteration + 1} iterations ===")
                print(f"Total empty tags removed: {total_empty_removed}")
                print(f"Total tag pairs merged: {total_merged}")
                print(f"Output length: {len(str(soup))} chars")
            break
    
    return str(soup)

def add_attributes_to_auto_labels(html_content: str) -> str:
    """
    Add parent, style, verified, and all label scheme attributes to auto_label tags in HTML content.
    
    The function loads the label scheme from a JSON file to determine:
    - Parent relationships based on nesting context (parent is the immediately enclosing auto_label)
    - Colors for each label (converted to background-color style)
    - All attributes defined in the label scheme for each label
    
    Attributes are initialized as follows:
    - String type: empty string "" (or default value from scheme)
    - Checkbox type: "false" (or default value from scheme)
    - Dropdown type: default value from the label scheme
    
    Also adds verified="false" to all auto_label tags.
    
    Args:
        html_content: HTML string with auto_label tags
    
    Returns:
        Modified HTML string with all attributes added
    """

    with open(LABEL_SCHEME_PATH, 'r', encoding='utf-8') as f:
        label_scheme = json.load(f)
    
    # Build style mapping and attributes mapping from label scheme
    style_map = {}
    attributes_map = {}
    
    # Helper function to determine text color based on background brightness
    def get_text_color(hex_color: str) -> str:
        """Determine if text should be black or white based on background color brightness."""
        # Remove # if present
        hex_color = hex_color.lstrip('#')
        # Convert to RGB
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        # Calculate relative luminance
        luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        return 'black' if luminance > 0.5 else 'white'
    
    # Convert hex color to rgb format
    def hex_to_rgb(hex_color: str) -> str:
        """Convert hex color to rgb() format."""
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        return f"rgb({r}, {g}, {b})"
    
    # Helper to get attribute value based on type
    def get_attribute_value(attr_config: dict) -> str:
        """Get the initial value for an attribute based on its type."""
        attr_type = attr_config.get('type', 'string')
        if attr_type == 'string':
            return attr_config.get('default', '')
        elif attr_type == 'checkbox':
            default = attr_config.get('default', False)
            return 'true' if default else 'false'
        elif attr_type == 'dropdown':
            return attr_config.get('default', '')
        else:
            return ''
    
    # Process label scheme to build style and attributes mappings
    for parent_label, parent_data in label_scheme.items():
        # Set style for top-level label
        if 'color' in parent_data:
            bg_color = hex_to_rgb(parent_data['color'])
            text_color = get_text_color(parent_data['color'])
            style_map[parent_label] = f"background-color: {bg_color}; color: {text_color};"
        
        # Set attributes for top-level label
        if 'attributes' in parent_data:
            attributes_map[parent_label] = parent_data['attributes']
        
        # Process sublabels
        if 'sublabels' in parent_data:
            for sublabel, sublabel_data in parent_data['sublabels'].items():
                # Set style for sublabel
                if 'color' in sublabel_data:
                    bg_color = hex_to_rgb(sublabel_data['color'])
                    text_color = get_text_color(sublabel_data['color'])
                    style_map[sublabel] = f"background-color: {bg_color}; color: {text_color};"
                
                # Set attributes for sublabel
                if 'attributes' in sublabel_data:
                    attributes_map[sublabel] = sublabel_data['attributes']
    
    # Pattern to match auto_label tags (opening and closing)
    tag_pattern = r'<(/?)auto_label([^>]*)>'
    
    # Stack to track currently open auto_labels
    label_stack = []
    result_parts = []
    last_pos = 0
    
    for match in re.finditer(tag_pattern, html_content, flags=re.IGNORECASE):
        # Add text before this tag
        result_parts.append(html_content[last_pos:match.start()])
        
        is_closing = match.group(1) == '/'
        tag_content = match.group(2)
        
        if is_closing:
            # Closing tag - pop from stack
            if label_stack:
                label_stack.pop()
            result_parts.append(match.group(0))
        else:
            # Opening tag - extract labelname and determine parent
            labelname_match = re.search(r'labelname="([^"]*)"', tag_content)
            if labelname_match:
                labelname = labelname_match.group(1)
                
                # Parent is the labelname of the tag at the top of the stack (or "" if stack is empty)
                parent_value = label_stack[-1] if label_stack else ""
                parent_attr = f'parent="{parent_value}"'
                
                # Get style attribute from label scheme
                style = style_map.get(labelname, '')
                style_attr = f'style="{style}"' if style else ''
                
                # Add verified attribute
                verified_attr = 'verified="false"'
                
                # Get all attributes for this label from label scheme
                label_attrs = attributes_map.get(labelname, {})
                scheme_attrs = []
                for attr_name, attr_config in label_attrs.items():
                    attr_value = get_attribute_value(attr_config)
                    scheme_attrs.append(f'{attr_name}="{attr_value}"')
                
                # Remove existing parent/style/verified/scheme attributes if present
                tag_content = re.sub(r'\s*parent="[^"]*"', '', tag_content)
                tag_content = re.sub(r'\s*style="[^"]*"', '', tag_content)
                tag_content = re.sub(r'\s*verified="[^"]*"', '', tag_content)
                # Remove existing scheme attributes
                for attr_name in label_attrs.keys():
                    tag_content = re.sub(rf'\s*{re.escape(attr_name)}="[^"]*"', '', tag_content)
                
                # Build new tag with all attributes
                all_attrs = [parent_attr, style_attr, verified_attr] + scheme_attrs
                attrs_str = ' '.join(filter(None, all_attrs))  # Filter out empty strings
                new_tag = f'<auto_label{tag_content} {attrs_str}>'
                result_parts.append(new_tag)
                
                # Push this label onto the stack
                label_stack.append(labelname)
            else:
                # No labelname found, keep tag as-is
                result_parts.append(match.group(0))
        
        last_pos = match.end()
    
    # Add remaining text after last tag
    result_parts.append(html_content[last_pos:])
    
    return ''.join(result_parts)
