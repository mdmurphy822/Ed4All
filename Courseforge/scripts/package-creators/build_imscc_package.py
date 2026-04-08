#!/usr/bin/env python3
"""
IMSCC Package Builder - Advanced Version

Creates functional ZIP archive for Brightspace import with comprehensive validation.
Includes full error checking and package verification.

Usage:
    python3 build_imscc_package.py -i /source/dir -o /output/file.imscc
    python3 build_imscc_package.py  # Uses default/environment paths

Dependencies:
    - zipfile (built-in)
    - pathlib (built-in)
    - os (built-in)
    - argparse (built-in)
    - logging (built-in)
"""

import argparse
import logging
import sys
import zipfile
import io
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configurable paths via environment variables
# Defaults to the project root (two directories up from this script)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
COURSEFORGE_PATH = os.environ.get('COURSEFORGE_PATH', str(PROJECT_ROOT))
DEFAULT_EXPORTS_DIR = os.path.join(COURSEFORGE_PATH, 'exports')


def build_working_imscc(source_dir=None, output_file=None):
    """Build complete IMSCC package with all required files.

    Args:
        source_dir: Path to source directory (or uses env/default)
        output_file: Path for output IMSCC file (or uses env/default)

    Returns:
        bool: True if successful, False otherwise
    """
    logger.info("=" * 60)
    logger.info("ADVANCED IMSCC PACKAGE BUILDER")
    logger.info("=" * 60)

    # Configuration with environment variable support
    if source_dir is None:
        source_dir_env = os.environ.get('IMSCC_SOURCE_DIR')
        if not source_dir_env:
            logger.error("No source directory specified. Use -i flag or set IMSCC_SOURCE_DIR environment variable.")
            return False
        source_dir = Path(source_dir_env)
    else:
        source_dir = Path(source_dir)

    if output_file is None:
        output_file_env = os.environ.get('IMSCC_OUTPUT_PATH')
        if not output_file_env:
            # Default to exports directory with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(DEFAULT_EXPORTS_DIR) / f'{timestamp}_course.imscc'
        else:
            output_file = Path(output_file_env)
    else:
        output_file = Path(output_file)
    
    print(f"Source: {source_dir}")
    print(f"Output: {output_file}")
    print()
    
    # File lists
    required_files = [
        "imsmanifest.xml",
        # Week content files (28 total)
        *[f"week_{w:02d}_{t}.html" for w in range(1,5) for t in [
            "overview", "concept_summary_01", "concept_summary_02", 
            "key_concepts", "visual_content", "application_examples", "study_questions"
        ]],
        # Assessment files (12 total)  
        *[f"{atype}_week_{w:02d}.xml" for w in range(1,5) for atype in ["assignment", "quiz", "discussion"]]
    ]
    
    # Validate source files
    missing_files = []
    for filename in required_files:
        if not (source_dir / filename).exists():
            missing_files.append(filename)
    
    if missing_files:
        print(f"❌ Missing {len(missing_files)} files:")
        for missing in missing_files[:5]:
            print(f"   - {missing}")
        if len(missing_files) > 5:
            print(f"   ... and {len(missing_files) - 5} more")
        return False
    
    print(f"✅ All {len(required_files)} source files validated")
    
    # Create package
    try:
        with zipfile.ZipFile(str(output_file), 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
            total_size = 0
            
            for filename in required_files:
                source_path = source_dir / filename
                file_size = source_path.stat().st_size
                total_size += file_size
                
                zipf.write(str(source_path), filename)
                print(f"  📄 {filename} ({file_size:,} bytes)")
            
            print(f"\n📊 Package completed:")
            print(f"   Files: {len(required_files)}")
            print(f"   Total size: {total_size:,} bytes")
            
        # Verify package
        package_size = output_file.stat().st_size
        compression = (1 - package_size / total_size) * 100
        
        print(f"   Package size: {package_size:,} bytes")
        print(f"   Compression: {compression:.1f}%")
        print(f"\n✅ Package created successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating package: {e}")
        return False

def main():
    """Main entry point with CLI argument support."""
    parser = argparse.ArgumentParser(
        description='Build IMSCC package with validation'
    )
    parser.add_argument(
        '-i', '--input',
        help='Source directory containing content files'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output IMSCC file path'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    success = build_working_imscc(source_dir=args.input, output_file=args.output)

    if success:
        logger.info("Ready for Brightspace import!")
    else:
        logger.error("Package creation failed")

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()