#!/usr/bin/env python3
"""
Pattern 16.2 Prevention Framework
==================================

This script implements comprehensive validation to prevent Pattern 16.2 failures:
- Empty Module Content: Modules import but contain no educational materials
- Content Generation vs Display Gap: Files exist but don't display in Brightspace
- Course Duration Mismatches: Generated content doesn't match requirements

CRITICAL REQUIREMENTS:
- All HTML files must contain substantial educational content (300+ words per concept)
- Learning objectives must include specific, detailed explanations
- Course duration must match user requirements (explicit collection required)
- All content files must be properly linked in manifest for Brightspace display

Usage:
    python3 pattern_16_2_prevention.py --course-dir /path/to/course --weeks 12
"""

import os
import re
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pattern_16_2_prevention.log'),
        logging.StreamHandler()
    ]
)

class Pattern162PreventionError(Exception):
    """Custom exception for Pattern 16.2 prevention failures"""
    pass

class ContentValidator:
    """Validates educational content to prevent Pattern 16.2 failures"""
    
    def __init__(self, course_dir: str, required_weeks: int = 12):
        self.course_dir = Path(course_dir)
        self.required_weeks = required_weeks
        self.validation_results = {}
        
    def validate_course_duration_requirements(self) -> bool:
        """Validate course has required number of weeks"""
        logging.info(f"Validating course duration requirements: {self.required_weeks} weeks")
        
        # Find all week-related HTML files
        week_files = list(self.course_dir.glob("week_*.html"))
        week_numbers = set()
        
        for file in week_files:
            match = re.search(r'week_(\d+)', file.name)
            if match:
                week_numbers.add(int(match.group(1)))
        
        actual_weeks = len(week_numbers)
        
        if actual_weeks < self.required_weeks:
            raise Pattern162PreventionError(
                f"Course duration insufficient: {actual_weeks} weeks generated, "
                f"{self.required_weeks} weeks required"
            )
        
        logging.info(f"✅ Course duration validation passed: {actual_weeks} weeks found")
        return True
    
    def validate_substantial_content(self, file_path: Path) -> bool:
        """Validate HTML file contains substantial educational content"""
        if not file_path.exists():
            raise Pattern162PreventionError(f"Content file missing: {file_path}")
        
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception as e:
            raise Pattern162PreventionError(f"Cannot read content file {file_path}: {e}")
        
        # Remove HTML tags to count actual content
        text_content = re.sub(r'<[^>]+>', '', content)
        text_content = re.sub(r'\s+', ' ', text_content).strip()
        
        # Detect placeholder content
        placeholder_patterns = [
            "content will be developed based on course materials",
            "todo:",
            "placeholder",
            "coming soon",
            "to be completed",
            "under construction"
        ]
        
        for pattern in placeholder_patterns:
            if pattern in text_content.lower():
                raise Pattern162PreventionError(
                    f"Placeholder content detected in {file_path}: '{pattern}'"
                )
        
        # Validate HTML formatting (detect broken markdown artifacts)
        if "**" in content:
            raise Pattern162PreventionError(
                f"Malformed markdown artifacts in {file_path}: '**' characters found"
            )
        
        # Validate minimum content length
        min_content_length = 300  # Minimum 300 characters per concept
        if len(text_content) < min_content_length:
            raise Pattern162PreventionError(
                f"Insufficient content in {file_path}: {len(text_content)} characters "
                f"(minimum {min_content_length} required)"
            )
        
        # Validate learning objectives completeness
        if "by the end of this week, you will be able to:" in content.lower():
            objectives = re.findall(r'\d+\.\s*([^<\n]+)', content)
            if len(objectives) < 3:
                raise Pattern162PreventionError(
                    f"Incomplete learning objectives in {file_path}: "
                    f"{len(objectives)} found (minimum 3 required)"
                )
            
            # Validate each objective has substantial description
            for i, objective in enumerate(objectives, 1):
                if len(objective.strip()) < 50:
                    raise Pattern162PreventionError(
                        f"Learning objective {i} too brief in {file_path}: "
                        f"'{objective.strip()}' (minimum 50 characters)"
                    )
        
        logging.info(f"✅ Content validation passed: {file_path.name} ({len(text_content)} characters)")
        return True
    
    def validate_educational_depth(self, file_path: Path) -> bool:
        """Validate content has appropriate educational depth"""
        content = file_path.read_text(encoding='utf-8')
        
        # Check for mathematical content if it's a math course
        if "linear algebra" in self.course_dir.name.lower() or "math" in self.course_dir.name.lower():
            # Look for mathematical notation, examples, or worked solutions
            math_indicators = [
                r'\$.*\$',  # LaTeX math notation
                r'\\[a-zA-Z]+',  # LaTeX commands
                r'matrix|vector|equation|solution',  # Mathematical terms
                r'example \d+:|step \d+:',  # Worked examples
            ]
            
            has_math_content = any(re.search(pattern, content, re.IGNORECASE) 
                                 for pattern in math_indicators)
            
            if not has_math_content:
                logging.warning(
                    f"⚠️ Mathematical content may be insufficient in {file_path.name}"
                )
        
        # Check for educational structure
        educational_elements = [
            r'learning objective',
            r'key concept',
            r'example',
            r'application',
            r'summary',
            r'definition'
        ]
        
        element_count = sum(1 for pattern in educational_elements 
                           if re.search(pattern, content, re.IGNORECASE))
        
        if element_count < 3:
            raise Pattern162PreventionError(
                f"Insufficient educational structure in {file_path}: "
                f"{element_count} educational elements found (minimum 3 required)"
            )
        
        logging.info(f"✅ Educational depth validation passed: {file_path.name}")
        return True
    
    def validate_manifest_content_links(self, manifest_path: Path) -> bool:
        """Validate all content files are properly linked in manifest"""
        if not manifest_path.exists():
            raise Pattern162PreventionError(f"Manifest file missing: {manifest_path}")
        
        try:
            tree = ET.parse(manifest_path)
            root = tree.getroot()
        except ET.ParseError as e:
            raise Pattern162PreventionError(f"Invalid manifest XML: {e}")
        
        # Find all resource references in manifest
        resources = root.findall('.//{http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1}resource')
        manifest_files = set()
        
        for resource in resources:
            href = resource.get('href')
            if href and href.endswith('.html'):
                manifest_files.add(href)
        
        # Find all actual HTML content files
        content_files = set(f.name for f in self.course_dir.glob("*.html") 
                           if not f.name.startswith('index'))
        
        # Check for missing links
        missing_links = content_files - manifest_files
        if missing_links:
            raise Pattern162PreventionError(
                f"Content files not linked in manifest: {missing_links}"
            )
        
        # Check for broken links
        broken_links = manifest_files - content_files
        if broken_links:
            raise Pattern162PreventionError(
                f"Manifest references non-existent files: {broken_links}"
            )
        
        logging.info(f"✅ Manifest content linking validated: {len(content_files)} files properly linked")
        return True
    
    def validate_assessment_functionality(self) -> bool:
        """Validate assessment files will create functional Brightspace tools"""
        assessment_files = list(self.course_dir.glob("*assignment*.xml")) + \
                          list(self.course_dir.glob("*quiz*.xml")) + \
                          list(self.course_dir.glob("*discussion*.xml"))
        
        if not assessment_files:
            logging.warning("⚠️ No assessment files found for functionality validation")
            return True
        
        for file in assessment_files:
            try:
                tree = ET.parse(file)
                root = tree.getroot()
                
                # Basic XML structure validation
                if "assignment" in file.name:
                    # Check for D2L assignment structure
                    if not any(elem.tag.endswith('assignment') for elem in root.iter()):
                        raise Pattern162PreventionError(
                            f"Invalid assignment XML structure in {file.name}"
                        )
                
                elif "quiz" in file.name:
                    # Check for QTI structure
                    if not any(elem.tag.endswith('questestinterop') for elem in root.iter()):
                        raise Pattern162PreventionError(
                            f"Invalid quiz XML structure in {file.name} "
                            f"(must use QTI 1.2 questestinterop format)"
                        )
                
                elif "discussion" in file.name:
                    # Check for D2L discussion structure
                    if not any(elem.tag.endswith('discussion') for elem in root.iter()):
                        raise Pattern162PreventionError(
                            f"Invalid discussion XML structure in {file.name}"
                        )
                
            except ET.ParseError as e:
                raise Pattern162PreventionError(f"Invalid assessment XML in {file.name}: {e}")
        
        logging.info(f"✅ Assessment functionality validated: {len(assessment_files)} files")
        return True
    
    def run_complete_validation(self) -> Dict[str, bool]:
        """Run complete Pattern 16.2 prevention validation"""
        logging.info("Starting Pattern 16.2 prevention validation")
        
        results = {}
        
        try:
            # 1. Course Duration Validation
            results['duration'] = self.validate_course_duration_requirements()
            
            # 2. Content File Validation
            html_files = list(self.course_dir.glob("*.html"))
            content_results = []
            
            for html_file in html_files:
                if html_file.name.startswith('index'):
                    continue  # Skip index files
                
                self.validate_substantial_content(html_file)
                self.validate_educational_depth(html_file)
                content_results.append(html_file.name)
            
            results['content_files'] = content_results
            
            # 3. Manifest Linking Validation
            manifest_file = self.course_dir / "imsmanifest.xml"
            results['manifest_linking'] = self.validate_manifest_content_links(manifest_file)
            
            # 4. Assessment Functionality Validation
            results['assessment_functionality'] = self.validate_assessment_functionality()
            
            logging.info("✅ Pattern 16.2 prevention validation PASSED")
            return results
            
        except Pattern162PreventionError as e:
            logging.error(f"❌ Pattern 16.2 prevention validation FAILED: {e}")
            raise
        except Exception as e:
            logging.error(f"❌ Validation error: {e}")
            raise Pattern162PreventionError(f"Validation failed: {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Pattern 16.2 Prevention Framework - Validate course content"
    )
    parser.add_argument(
        '--course-dir', 
        required=True,
        help="Path to course content directory"
    )
    parser.add_argument(
        '--weeks', 
        type=int,
        default=12,
        help="Required number of course weeks (default: 12)"
    )
    parser.add_argument(
        '--output',
        help="Output validation results to JSON file"
    )
    
    args = parser.parse_args()
    
    try:
        validator = ContentValidator(args.course_dir, args.weeks)
        results = validator.run_complete_validation()
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            logging.info(f"Validation results written to {args.output}")
        
        print("✅ Pattern 16.2 Prevention: All validations PASSED")
        print(f"Course ready for packaging: {args.course_dir}")
        
    except Pattern162PreventionError as e:
        print(f"❌ Pattern 16.2 Prevention FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"❌ Validation error: {e}")
        exit(1)

if __name__ == "__main__":
    main()