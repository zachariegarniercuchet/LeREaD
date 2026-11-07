"""
At the very begining of each HTML file, add a comment containing the label scheme and meta information.
This process is necessary for the manual annotation process, so that the LeREaD Labelizer tool can read them.
"""

import json
from pathlib import Path
import os
from configs.config import DATA_DIR


with open("./configs/label_scheme.json") as f:
    LABEL_SCHEME = json.load(f)

with open("./configs/meta.json") as f:
    META = json.load(f)

INPUT_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "original" / "no-gold"

from bs4 import BeautifulSoup
import chardet

def add_label_scheme_to_html(html_file):
    """
    For each HTML file in input_dir, detect encoding, read content, parse with BeautifulSoup,
    insert label scheme as a comment before <head>, and save to output_dir as UTF-8.
    """
    # Detect the encoding of the file
    with open(html_file, 'rb') as f:
        raw_data = f.read()
        detected = chardet.detect(raw_data)
        detected_encoding = detected['encoding']
        confidence = detected['confidence']
        print(f"  → Detected encoding: {detected_encoding} (confidence: {confidence:.2%})")
    
    # Read the original HTML content with detected encoding
    with open(html_file, 'r', encoding=detected_encoding) as f:
        content = f.read()
    
    # Parse the HTML
    soup = BeautifulSoup(content, 'html.parser')
    
    # Find the head tag
    head_tag = soup.find('head')
    
    if head_tag:
        # Create a comment node with the label tree
        from bs4 import Comment
        comment_text = f''' HTMLLabelizer
{{
  "labeltree": {json.dumps(LABEL_SCHEME, indent=4)},
  "meta": {json.dumps(META, indent=4)}
}}
'''
        comment = Comment(comment_text)
        
        # Insert the comment before the head tag
        head_tag.insert_before(comment)


        return str(soup)

if __name__ == "__main__":



    html_files = list(INPUT_DIR.glob("*.html")) + list(INPUT_DIR.glob("*.htm"))
    print(f"Found {len(html_files)} HTML/HTM files to process.\n")

    for html_file in html_files:
        print(f"Processing: {html_file.name}")
    
        modified_html = add_label_scheme_to_html(html_file)
        
        # Save to output directory as UTF-8
        output_file = OUTPUT_DIR / html_file.name
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(modified_html)
        
        print(f"✓ Saved to: {output_file.name} (as UTF-8)\n")
   

    print(f"\n✓ Processing complete! All files converted to UTF-8 and saved to {OUTPUT_DIR}")