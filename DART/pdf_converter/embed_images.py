#!/usr/bin/env python3
"""
Image Embedding Utility

Embeds images from a directory (with metadata JSON) into an existing HTML file.
Places images near their figure references in the text.

Usage:
    python embed_images.py <html_file> <images_dir> [-o output.html]
"""

import argparse
import base64
import json
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup


def load_metadata(images_dir: Path) -> list:
    """Load images metadata from JSON file."""
    metadata_path = images_dir / 'images_metadata.json'
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_figure_number(caption: str) -> int | None:
    """Extract figure number from caption text."""
    if not caption:
        return None
    match = re.match(r'^(?:Figure|Fig\.?)\s*(\d+)', caption, re.IGNORECASE)
    return int(match.group(1)) if match else None


def get_mime_type(filename: str) -> str:
    """Get MIME type from filename."""
    ext = Path(filename).suffix.lower()
    mime_map = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp',
    }
    return mime_map.get(ext, 'image/jpeg')


def create_figure_element(soup: BeautifulSoup, img_data: dict, images_dir: Path, fig_id: str) -> BeautifulSoup:
    """Create a figure element with embedded image."""
    # Read image file
    img_path = images_dir / img_data['filename']
    if not img_path.exists():
        print(f"Warning: Image file not found: {img_path}")
        return None

    with open(img_path, 'rb') as f:
        img_bytes = f.read()

    # Create data URI
    mime_type = get_mime_type(img_data['filename'])
    data_uri = f"data:{mime_type};base64,{base64.b64encode(img_bytes).decode()}"

    # Create figure element
    figure = soup.new_tag('figure')
    figure['id'] = fig_id
    figure['class'] = 'embedded-figure'

    # Create img tag
    img_tag = soup.new_tag('img')
    img_tag['src'] = data_uri
    img_tag['alt'] = img_data.get('alt_text', f"Figure from page {img_data.get('page', '?')}")
    img_tag['loading'] = 'lazy'
    if img_data.get('width'):
        img_tag['width'] = img_data['width']
    if img_data.get('height'):
        img_tag['height'] = img_data['height']

    figure.append(img_tag)

    # Create figcaption
    figcaption = soup.new_tag('figcaption')
    caption_text = img_data.get('caption', '') or f"Figure (page {img_data.get('page', '?')})"

    # If we have a long description different from alt text, add expandable details
    long_desc = img_data.get('long_description', '')
    alt_text = img_data.get('alt_text', '')

    if long_desc and long_desc != alt_text and len(long_desc) > len(alt_text):
        figcaption.string = caption_text

        details = soup.new_tag('details')
        summary = soup.new_tag('summary')
        summary.string = 'Image description'
        details.append(summary)

        desc_p = soup.new_tag('p')
        desc_p.string = long_desc
        details.append(desc_p)

        figcaption.append(details)
    else:
        figcaption.string = caption_text

    figure.append(figcaption)

    return figure


def find_figure_references(text: str) -> list[int]:
    """Find all figure references in text."""
    pattern = r'(?:Figure|Fig\.?)\s*(\d+)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return [int(m) for m in matches]


def embed_images(html_path: Path, images_dir: Path, output_path: Path = None) -> None:
    """Embed images into HTML file near their references."""
    # Load HTML
    with open(html_path, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')

    # Load metadata
    metadata = load_metadata(images_dir)
    print(f"Loaded {len(metadata)} images from metadata")

    # Build figure number -> image data map
    figure_map = {}
    unmatched = []

    for img_data in metadata:
        fig_num = extract_figure_number(img_data.get('caption', ''))
        if fig_num is not None:
            figure_map[fig_num] = img_data
        else:
            unmatched.append(img_data)

    print(f"Matched {len(figure_map)} images to figure numbers: {sorted(figure_map.keys())}")
    print(f"Unmatched images (no figure caption): {len(unmatched)}")

    # Track which figures have been inserted
    inserted = set()
    figures_inserted = 0

    # Find all paragraphs and look for figure references
    main = soup.find('main') or soup.find('body')
    if not main:
        print("Error: No <main> or <body> element found")
        return

    # Process paragraphs to find and insert figures after references
    for para in main.find_all(['p', 'li']):
        text = para.get_text()
        refs = find_figure_references(text)

        for fig_num in refs:
            if fig_num in inserted:
                continue
            if fig_num not in figure_map:
                continue

            img_data = figure_map[fig_num]
            fig_id = f"figure-{fig_num}"

            figure = create_figure_element(soup, img_data, images_dir, fig_id)
            if figure:
                para.insert_after(figure)
                inserted.add(fig_num)
                figures_inserted += 1
                print(f"  Inserted Figure {fig_num} after paragraph")

    # Insert any remaining matched figures that weren't referenced in text
    for fig_num, img_data in sorted(figure_map.items()):
        if fig_num in inserted:
            continue

        fig_id = f"figure-{fig_num}"
        figure = create_figure_element(soup, img_data, images_dir, fig_id)
        if figure:
            main.append(figure)
            inserted.add(fig_num)
            figures_inserted += 1
            print(f"  Appended Figure {fig_num} (no text reference found)")

    # Append unmatched images at the end
    for idx, img_data in enumerate(unmatched):
        fig_id = f"figure-unmatched-{idx + 1}"
        figure = create_figure_element(soup, img_data, images_dir, fig_id)
        if figure:
            main.append(figure)
            figures_inserted += 1
            print(f"  Appended unmatched image from page {img_data.get('page', '?')}")

    # Add CSS for figures if not present
    style = soup.find('style')
    if style and '.embedded-figure' not in style.string:
        figure_css = """
        .embedded-figure {
            margin: 2rem auto;
            text-align: center;
            max-width: 100%;
        }
        .embedded-figure img {
            max-width: 100%;
            height: auto;
            border: 1px solid var(--border-color, #e0e0e0);
            border-radius: 4px;
        }
        .embedded-figure figcaption {
            font-style: italic;
            margin-top: 0.5rem;
            font-size: 0.95rem;
            text-align: center;
        }
        .embedded-figure details {
            margin-top: 0.5rem;
            text-align: left;
        }
        .embedded-figure details summary {
            cursor: pointer;
            color: var(--link-color, #0066cc);
        }
        .embedded-figure details p {
            margin: 0.5rem 0;
            font-style: normal;
            text-align: left;
        }
"""
        style.string = style.string + figure_css

    # Write output
    output = output_path or html_path
    with open(output, 'w', encoding='utf-8') as f:
        f.write(str(soup))

    print(f"\nDone! Inserted {figures_inserted} figures into {output}")


def main():
    parser = argparse.ArgumentParser(
        description='Embed images from a directory into an HTML file'
    )
    parser.add_argument('html_file', help='Path to HTML file')
    parser.add_argument('images_dir', help='Path to images directory with metadata.json')
    parser.add_argument('-o', '--output', help='Output file path (default: overwrites input)')

    args = parser.parse_args()

    html_path = Path(args.html_file)
    images_dir = Path(args.images_dir)
    output_path = Path(args.output) if args.output else None

    if not html_path.exists():
        print(f"Error: HTML file not found: {html_path}")
        sys.exit(1)

    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)

    embed_images(html_path, images_dir, output_path)


if __name__ == '__main__':
    main()
