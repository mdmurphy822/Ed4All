#!/usr/bin/env python3
"""
Simple IMSCC Creator

Basic IMSCC package creator for Linear Algebra course.
Creates ZIP package from predefined file list.

Usage:
    python3 simple_imscc_creator.py

Dependencies:
    - zipfile (built-in)
    - os (built-in)
"""

import zipfile
import os
import argparse
from pathlib import Path

# Configurable paths via environment variables
# Defaults to the project root (two directories up from this script)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
COURSEFORGE_PATH = os.environ.get('COURSEFORGE_PATH', str(PROJECT_ROOT))
DEFAULT_EXPORTS_DIR = os.path.join(COURSEFORGE_PATH, 'exports')

def create_simple_imscc(source_dir=None, output_path=None):
    """Create simple IMSCC package with predefined file list

    Args:
        source_dir: Source directory containing content files (or uses env/default)
        output_path: Output path for IMSCC file (or uses env/default)
    """

    source = source_dir or os.environ.get('IMSCC_SOURCE_DIR')
    if not source:
        print("Error: No source directory specified. Use -i flag or set IMSCC_SOURCE_DIR environment variable.")
        return

    output = output_path or os.environ.get('IMSCC_OUTPUT_PATH')
    if not output:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = os.path.join(DEFAULT_EXPORTS_DIR, f'{timestamp}_course.imscc')
    
    files = [
        "imsmanifest.xml",
        # Week 1
        "week_01_overview.html", "week_01_concept_summary_01.html", "week_01_concept_summary_02.html", 
        "week_01_key_concepts.html", "week_01_visual_content.html", "week_01_application_examples.html", "week_01_study_questions.html",
        # Week 2
        "week_02_overview.html", "week_02_concept_summary_01.html", "week_02_concept_summary_02.html", 
        "week_02_key_concepts.html", "week_02_visual_content.html", "week_02_application_examples.html", "week_02_study_questions.html",
        # Week 3
        "week_03_overview.html", "week_03_concept_summary_01.html", "week_03_concept_summary_02.html", 
        "week_03_key_concepts.html", "week_03_visual_content.html", "week_03_application_examples.html", "week_03_study_questions.html",
        # Week 4
        "week_04_overview.html", "week_04_concept_summary_01.html", "week_04_concept_summary_02.html",
        "week_04_key_concepts.html", "week_04_visual_content.html", "week_04_application_examples.html", "week_04_study_questions.html",
        # Assessments
        "assignment_week_01.xml", "assignment_week_02.xml", "assignment_week_03.xml", "assignment_week_04.xml",
        "quiz_week_01.xml", "quiz_week_02.xml", "quiz_week_03.xml", "quiz_week_04.xml",
        "discussion_week_01.xml", "discussion_week_02.xml", "discussion_week_03.xml", "discussion_week_04.xml"
    ]
    
    print("Creating simple IMSCC package...")
    
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for filename in files:
            filepath = os.path.join(source, filename)
            if os.path.exists(filepath):
                zipf.write(filepath, filename)
                print(f"Added: {filename}")
            else:
                print(f"Missing: {filename}")
    
    print(f"Package created: {output}")

def main():
    """Main entry point with CLI argument support"""
    parser = argparse.ArgumentParser(
        description='Create simple IMSCC package from content files'
    )
    parser.add_argument(
        '-i', '--input',
        help='Source directory containing content files'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output path for IMSCC file'
    )

    args = parser.parse_args()

    create_simple_imscc(source_dir=args.input, output_path=args.output)


if __name__ == "__main__":
    main()