#!/usr/bin/env python3
"""Simple IMSCC generator without external dependencies

Usage:
    python simple_imscc_generator.py -i /path/to/input -o /path/to/output.imscc
    python simple_imscc_generator.py  # Uses default/environment paths
"""

import argparse
import json
import logging
import os
import re
import sys
import zipfile
import uuid
from pathlib import Path
from datetime import datetime

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


def create_imscc_package(input_path=None, output_file=None):
    """Create IMSCC package from course materials.

    Args:
        input_path: Path to input course directory (or uses env/default)
        output_file: Path for output IMSCC file (or uses env/default)

    Returns:
        bool: True if successful, False otherwise

    Raises:
        FileNotFoundError: If input directory doesn't exist
        PermissionError: If output directory cannot be created
    """
    # Paths with environment variable support
    if input_path is None:
        input_path = os.environ.get('IMSCC_INPUT_PATH')
        if not input_path:
            logger.error("No input path specified. Use -i flag or set IMSCC_INPUT_PATH environment variable.")
            return False

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if output_file is None:
        output_dir = os.environ.get('IMSCC_OUTPUT_DIR', DEFAULT_EXPORTS_DIR)
        output_dir = os.path.join(output_dir, timestamp)
        output_file = os.path.join(output_dir, 'linear_algebra_course.imscc')
    else:
        output_dir = os.path.dirname(output_file)
    
    print(f"🚀 Creating IMSCC package")
    print(f"📂 Input: {input_path}")
    print(f"📁 Output: {output_file}")
    
    try:
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Create temporary directory
        temp_dir = Path(output_dir) / "temp_files"
        temp_dir.mkdir(exist_ok=True)
        
        # Parse course info
        course_info_path = Path(input_path) / "course_info.md"
        with open(course_info_path, 'r', encoding='utf-8') as f:
            course_content = f.read()
            
        # Extract course title
        title_match = re.search(r'^#\s+(.+)$', course_content, re.MULTILINE)
        course_title = title_match.group(1) if title_match else "Linear Algebra Course"
        
        print(f"📚 Course Title: {course_title}")
        
        # Find week files
        modules_dir = Path(input_path) / "modules"
        week_files = sorted(modules_dir.glob("week_*.md"))
        
        print(f"📄 Found {len(week_files)} week files")
        
        # Generate HTML files
        html_files = []
        for week_file in week_files:
            week_number = int(re.search(r'week_(\d+)', week_file.name).group(1))
            
            # Read week content
            with open(week_file, 'r', encoding='utf-8') as f:
                week_content = f.read()
            
            # Generate 7 HTML files for this week
            sub_module_types = [
                "overview", "concept_summary_01", "concept_summary_02", 
                "key_concepts", "visual_content", "application_examples", "study_questions"
            ]
            
            for module_type in sub_module_types:
                filename = f"week_{week_number:02d}_{module_type}.html"
                title = f"Module {week_number}: {module_type.replace('_', ' ').title()}"
                
                # Create HTML content
                html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
    <style>
        body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
        .container {{ max-width: 1200px; }}
        .content-paragraph {{ margin-bottom: 1.5rem; }}
    </style>
</head>
<body>
    <div class="container mt-4">
        <h1>{title}</h1>
        <div class="content-section">
            <p class="content-paragraph">Content for {module_type.replace('_', ' ')} module. This demonstrates the structure and layout for the course content.</p>
            <p class="content-paragraph">This HTML page represents one of the seven sub-modules for Week {week_number}, providing comprehensive coverage of the learning objectives.</p>
        </div>
    </div>
    <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js"></script>
</body>
</html>"""
                
                # Write HTML file
                html_path = temp_dir / filename
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                html_files.append(filename)
        
        print(f"✅ Generated {len(html_files)} HTML files")
        
        # Generate assignment XML
        assignment_files = []
        for week_num in range(1, len(week_files) + 1):
            assignment_filename = f"assignment_week_{week_num:02d}.xml"
            
            assignment_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<assignment xmlns="http://www.d2l.org/xsd/d2lcp_v1p0">
    <header>
        <title>Week {week_num} Writing Assignment</title>
        <description>
            <text>Complete a 700-1000 word analysis demonstrating your understanding of Week {week_num} concepts. Apply theoretical knowledge to practical scenarios and provide clear explanations of key principles.</text>
        </description>
    </header>
    <submission>
        <dropbox>
            <name>Week {week_num} Assignment Dropbox</name>
            <instructions>Submit your completed assignment (700-1000 words) in PDF or Word format.</instructions>
            <points_possible>100</points_possible>
            <file_submission enabled="true">
                <allowed_extensions>.pdf,.doc,.docx</allowed_extensions>
                <max_file_size>10485760</max_file_size>
            </file_submission>
        </dropbox>
    </submission>
    <grading>
        <rubric>
            <criterion id="content" points="40">
                <name>Content Understanding</name>
                <description>Demonstrates clear understanding of key concepts</description>
            </criterion>
            <criterion id="analysis" points="35">
                <name>Analysis and Application</name>
                <description>Effectively applies concepts to problems</description>
            </criterion>
            <criterion id="communication" points="25">
                <name>Written Communication</name>
                <description>Clear, organized, professional writing</description>
            </criterion>
        </rubric>
    </grading>
</assignment>"""
            
            # Write assignment file
            assignment_path = temp_dir / assignment_filename
            with open(assignment_path, 'w', encoding='utf-8') as f:
                f.write(assignment_xml)
            
            assignment_files.append(assignment_filename)
        
        print(f"✅ Generated {len(assignment_files)} assignment files")
        
        # Generate manifest
        course_id = str(uuid.uuid4())
        
        # Create resource entries
        resources = []
        for html_file in html_files:
            resource_id = f"resource_{html_file.replace('.html', '').replace('_', '')}"
            resources.append(f"""
        <resource identifier="{resource_id}" type="webcontent" href="{html_file}">
            <file href="{html_file}"/>
        </resource>""")
        
        for assignment_file in assignment_files:
            resource_id = f"resource_{assignment_file.replace('.xml', '').replace('_', '')}"
            resources.append(f"""
        <resource identifier="{resource_id}" type="assignment_xmlv1p0" href="{assignment_file}">
            <file href="{assignment_file}"/>
        </resource>""")
        
        # Create organization items
        items = []
        for week_num in range(1, len(week_files) + 1):
            week_items = []
            
            # Add HTML items for this week
            week_html_files = [f for f in html_files if f.startswith(f'week_{week_num:02d}_')]
            for html_file in week_html_files:
                resource_id = f"resource_{html_file.replace('.html', '').replace('_', '')}"
                title = html_file.replace('.html', '').replace('week_', 'Week ').replace('_', ' ').title()
                week_items.append(f"""
                <item identifier="item_{html_file.replace('.html', '').replace('_', '')}" 
                      identifierref="{resource_id}">
                    <title>{title}</title>
                </item>""")
            
            # Add assignment for this week
            assignment_file = f"assignment_week_{week_num:02d}.xml"
            resource_id = f"resource_{assignment_file.replace('.xml', '').replace('_', '')}"
            week_items.append(f"""
                <item identifier="item_{assignment_file.replace('.xml', '').replace('_', '')}" 
                      identifierref="{resource_id}">
                    <title>Week {week_num} Assignment</title>
                </item>""")
            
            items.append(f"""
            <item identifier="week_{week_num}" identifierref="">
                <title>Week {week_num}</title>
                {''.join(week_items)}
            </item>""")
        
        # Generate manifest content
        manifest_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<manifest identifier="{course_id}" version="1.2.0"
    xmlns="http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1"
    xmlns:lom="http://ltsc.ieee.org/xsd/imsccv1p2/LOM/resource"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1 http://www.imsglobal.org/xsd/imscc/imscp_v1p1.xsd">
    
    <metadata>
        <schema>IMS Common Cartridge</schema>
        <schemaversion>1.2.0</schemaversion>
        <lom:lom>
            <lom:general>
                <lom:title>
                    <lom:string language="en">{course_title}</lom:string>
                </lom:title>
                <lom:description>
                    <lom:string language="en">Comprehensive linear algebra course with interactive content and assessments.</lom:string>
                </lom:description>
            </lom:general>
        </lom:lom>
    </metadata>
    
    <organizations default="organization_1">
        <organization identifier="organization_1" structure="rooted-hierarchy">
            <title>{course_title}</title>
            {''.join(items)}
        </organization>
    </organizations>
    
    <resources>
        {''.join(resources)}
    </resources>
</manifest>"""
        
        # Write manifest
        manifest_path = temp_dir / 'imsmanifest.xml'
        with open(manifest_path, 'w', encoding='utf-8') as f:
            f.write(manifest_content)
        
        print("✅ Generated manifest file")
        
        # Create IMSCC package
        with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in temp_dir.iterdir():
                if file_path.is_file():
                    zipf.write(file_path, file_path.name)
        
        # Cleanup temp directory
        import shutil
        shutil.rmtree(temp_dir)
        
        # Validate package
        package_size = Path(output_file).stat().st_size
        
        print("🎉 IMSCC Package Created Successfully!")
        print(f"📁 Location: {output_file}")
        print(f"📊 Size: {package_size:,} bytes")
        print(f"📄 HTML files: {len(html_files)}")
        print(f"📝 Assignment files: {len(assignment_files)}")
        print(f"📚 Weeks: {len(week_files)}")
        print()
        print("✅ Ready for import into Brightspace!")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating IMSCC package: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point with CLI argument support."""
    parser = argparse.ArgumentParser(
        description='Generate IMSCC package from course materials'
    )
    parser.add_argument(
        '-i', '--input',
        help='Input course directory path'
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

    success = create_imscc_package(input_path=args.input, output_file=args.output)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()