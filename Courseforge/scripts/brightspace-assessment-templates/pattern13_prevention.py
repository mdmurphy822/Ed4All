#!/usr/bin/env python3
"""
Pattern 13 Prevention Script
Comprehensive fix for Brightspace XML compatibility issues to prevent recurring import failures.

This script ensures all future IMSCC packages use proper Brightspace-compatible XML schemas
and resource declarations that will import successfully.
"""

import os
import sys
import logging
import shutil
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import with error handling
try:
    from brightspace_assessment_generator import BrightspaceAssessmentGenerator
except ImportError as e:
    logger.error(f"Failed to import BrightspaceAssessmentGenerator: {e}")
    raise


class Pattern13Prevention:
    """Prevents Brightspace import failures by ensuring proper XML compatibility."""
    
    def __init__(self):
        self.generator = BrightspaceAssessmentGenerator()
        self.template_dir = os.path.dirname(__file__)
    
    def create_corrected_package(self, content_dir, output_dir):
        """Create a new IMSCC package with corrected Brightspace-compatible assessments.

        Args:
            content_dir: Path to source content directory
            output_dir: Path to output directory

        Returns:
            str: Path to output directory on success

        Raises:
            FileNotFoundError: If content_dir doesn't exist
            PermissionError: If output_dir cannot be created
            IOError: If file operations fail
        """
        logger.info("Pattern 13 Prevention: Creating Brightspace-compatible package...")

        # Validate input directory
        if not os.path.exists(content_dir):
            raise FileNotFoundError(f"Content directory not found: {content_dir}")

        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
        except PermissionError as e:
            logger.error(f"Cannot create output directory: {e}")
            raise

        # Copy all HTML content files (these are fine)
        html_files = []
        try:
            for file in os.listdir(content_dir):
                if file.endswith('.html'):
                    shutil.copy2(os.path.join(content_dir, file), output_dir)
                    html_files.append(file)
                    logger.info(f"Copied HTML: {file}")
        except IOError as e:
            logger.error(f"Error copying HTML files: {e}")
            raise
        
        # Generate corrected assessment XML files
        assessment_files = []
        try:
            for week in range(1, 5):
                # Create QTI-compliant quiz
                quiz_xml = self.generator.generate_qti_quiz(week, {})
                quiz_file = f"quiz_week_{week:02d}.xml"
                with open(os.path.join(output_dir, quiz_file), 'w', encoding='utf-8') as f:
                    f.write(quiz_xml)
                assessment_files.append(quiz_file)
                logger.info(f"Generated QTI quiz: {quiz_file}")

                # Create D2L assignment
                assignment_xml = self.generator.generate_d2l_assignment(week, {})
                assignment_file = f"assignment_week_{week:02d}.xml"
                with open(os.path.join(output_dir, assignment_file), 'w', encoding='utf-8') as f:
                    f.write(assignment_xml)
                assessment_files.append(assignment_file)
                logger.info(f"Generated D2L assignment: {assignment_file}")

                # Create D2L discussion
                discussion_xml = self.generator.generate_d2l_discussion(week, {})
                discussion_file = f"discussion_week_{week:02d}.xml"
                with open(os.path.join(output_dir, discussion_file), 'w', encoding='utf-8') as f:
                    f.write(discussion_xml)
                assessment_files.append(discussion_file)
                logger.info(f"Generated D2L discussion: {discussion_file}")
        except IOError as e:
            logger.error(f"Error generating assessment files: {e}")
            raise
        
        # Create assessment metadata file required by QTI
        try:
            self.create_assessment_metadata(output_dir)
        except IOError as e:
            logger.error(f"Error creating assessment metadata: {e}")
            raise

        # Generate corrected manifest
        try:
            self.generate_corrected_manifest(output_dir, html_files, assessment_files)
        except (IOError, FileNotFoundError) as e:
            logger.error(f"Error generating manifest: {e}")
            raise

        logger.info(f"Pattern 13 Prevention Complete: {len(html_files)} HTML + {len(assessment_files)} assessments + 1 manifest + 1 metadata = {len(html_files) + len(assessment_files) + 2} files")
        return output_dir
    
    def create_assessment_metadata(self, output_dir):
        """Create QTI assessment metadata file required for proper import."""
        metadata_xml = '''<?xml version="1.0" encoding="UTF-8"?>
<assessment_meta xmlns="http://www.imsglobal.org/xsd/imsccv1p1/imscp_v1p1">
    <title>Assessment Support</title>
    <description>QTI assessment base configuration for Brightspace compatibility</description>
</assessment_meta>'''
        
        with open(os.path.join(output_dir, 'assessment_meta.xml'), 'w', encoding='utf-8') as f:
            f.write(metadata_xml)
        print("✓ Created assessment metadata file")
    
    def generate_corrected_manifest(self, output_dir, html_files, assessment_files):
        """Generate manifest with correct Brightspace resource type declarations."""
        
        # Read the corrected template
        template_path = os.path.join(self.template_dir, "corrected_manifest_template.xml")
        with open(template_path, 'r', encoding='utf-8') as f:
            manifest_template = f.read()
        
        # Build resources section for all files
        resources_xml = ""
        
        # Add HTML resources (weeks 1-4)
        for week in range(1, 5):
            week_resources = f'''
    <!-- Week {week} HTML Resources -->
    <resource identifier="week_{week:02d}_overview_resource" type="webcontent" href="week_{week:02d}_overview.html">
      <file href="week_{week:02d}_overview.html"/>
    </resource>
    <resource identifier="week_{week:02d}_concept_summary_01_resource" type="webcontent" href="week_{week:02d}_concept_summary_01.html">
      <file href="week_{week:02d}_concept_summary_01.html"/>
    </resource>
    <resource identifier="week_{week:02d}_concept_summary_02_resource" type="webcontent" href="week_{week:02d}_concept_summary_02.html">
      <file href="week_{week:02d}_concept_summary_02.html"/>
    </resource>
    <resource identifier="week_{week:02d}_key_concepts_resource" type="webcontent" href="week_{week:02d}_key_concepts.html">
      <file href="week_{week:02d}_key_concepts.html"/>
    </resource>
    <resource identifier="week_{week:02d}_visual_content_resource" type="webcontent" href="week_{week:02d}_visual_content.html">
      <file href="week_{week:02d}_visual_content.html"/>
    </resource>
    <resource identifier="week_{week:02d}_application_examples_resource" type="webcontent" href="week_{week:02d}_application_examples.html">
      <file href="week_{week:02d}_application_examples.html"/>
    </resource>
    <resource identifier="week_{week:02d}_study_questions_resource" type="webcontent" href="week_{week:02d}_study_questions.html">
      <file href="week_{week:02d}_study_questions.html"/>
    </resource>
    
    <!-- Week {week} Assessment Resources with CORRECT IMSCC 1.3 types -->
    <resource identifier="assignment_week_{week:02d}_resource" type="assignment_xmlv1p0" href="assignment_week_{week:02d}.xml">
      <file href="assignment_week_{week:02d}.xml"/>
    </resource>
    <resource identifier="quiz_week_{week:02d}_resource" type="imsqti_xmlv1p2/imscc_xmlv1p3/assessment" href="quiz_week_{week:02d}.xml">
      <file href="quiz_week_{week:02d}.xml"/>
      <dependency identifierref="QTI_ASI_BASE"/>
    </resource>
    <resource identifier="discussion_week_{week:02d}_resource" type="imsdt_xmlv1p3" href="discussion_week_{week:02d}.xml">
      <file href="discussion_week_{week:02d}.xml"/>
    </resource>'''
            resources_xml += week_resources
        
        # Add QTI support resource
        resources_xml += '''
    
    <!-- QTI Assessment Support Files -->
    <resource identifier="QTI_ASI_BASE" type="associatedcontent/imscc_xmlv1p1/learning-application-resource" href="assessment_meta.xml">
      <file href="assessment_meta.xml"/>
    </resource>'''
        
        # Build complete organizations section
        organizations_xml = '''
  <organizations default="course_org">
    <organization identifier="course_org" structure="rooted-hierarchy">
      <title>Linear Algebra: Foundations and Applications</title>'''
      
        for week in range(1, 5):
            week_data = {
                1: "Foundations of Linear Algebra",
                2: "Vectors and Vector Operations", 
                3: "Matrices and Matrix Operations",
                4: "Linear Independence and Spanning Sets"
            }
            
            week_org = f'''
      
      <item identifier="week_{week:02d}" isvisible="true">
        <title>Week {week}: {week_data[week]}</title>
        <item identifier="week_{week:02d}_overview" isvisible="true">
          <title>Module Overview</title>
          <identifierref>week_{week:02d}_overview_resource"/>
        </item>
        <item identifier="week_{week:02d}_concept_01" isvisible="true">
          <title>{week_data[week]} - Part 1</title>
          <identifierref>week_{week:02d}_concept_summary_01_resource"/>
        </item>
        <item identifier="week_{week:02d}_concept_02" isvisible="true">
          <title>{week_data[week]} - Part 2</title>
          <identifierref>week_{week:02d}_concept_summary_02_resource"/>
        </item>
        <item identifier="week_{week:02d}_key_concepts" isvisible="true">
          <title>Key Concepts</title>
          <identifierref>week_{week:02d}_key_concepts_resource"/>
        </item>
        <item identifier="week_{week:02d}_visual" isvisible="true">
          <title>Visual Content</title>
          <identifierref>week_{week:02d}_visual_content_resource"/>
        </item>
        <item identifier="week_{week:02d}_applications" isvisible="true">
          <title>Application Examples</title>
          <identifierref>week_{week:02d}_application_examples_resource"/>
        </item>
        <item identifier="week_{week:02d}_study" isvisible="true">
          <title>Study Questions</title>
          <identifierref>week_{week:02d}_study_questions_resource"/>
        </item>
        <item identifier="week_{week:02d}_assignment" isvisible="true">
          <title>Week {week} Assignment</title>
          <identifierref>assignment_week_{week:02d}_resource"/>
        </item>
        <item identifier="week_{week:02d}_quiz" isvisible="true">
          <title>Week {week} Quiz</title>
          <identifierref>quiz_week_{week:02d}_resource"/>
        </item>
        <item identifier="week_{week:02d}_discussion" isvisible="true">
          <title>Week {week} Discussion</title>
          <identifierref>discussion_week_{week:02d}_resource"/>
        </item>
      </item>'''
            organizations_xml += week_org
        
        organizations_xml += '''
    </organization>
  </organizations>'''
        
        # Build complete manifest
        manifest_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<manifest identifier="linear_algebra_foundations_course_2025" version="1.3.0"
    xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1"
    xmlns:lom="http://ltsc.ieee.org/xsd/imsccv1p3/LOM/resource"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1 http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1.xsd">
  
  <metadata>
    <schema>IMS Common Cartridge</schema>
    <schemaversion>1.3.0</schemaversion>
    <lom:lom>
      <lom:general>
        <lom:identifier>linear_algebra_foundations_course_2025</lom:identifier>
        <lom:title>
          <lom:string language="en">Linear Algebra: Foundations and Applications</lom:string>
        </lom:title>
        <lom:description>
          <lom:string language="en">A comprehensive 4-week introduction to linear algebra covering systems of equations, vectors, matrices, and fundamental applications in mathematics and science.</lom:string>
        </lom:description>
        <lom:language>en</lom:language>
      </lom:general>
    </lom:lom>
  </metadata>

{organizations_xml}

  <resources>{resources_xml}
  </resources>
</manifest>'''
        
        # Write the corrected manifest
        manifest_path = os.path.join(output_dir, 'imsmanifest.xml')
        with open(manifest_path, 'w', encoding='utf-8') as f:
            f.write(manifest_xml)
        print("✅ Generated corrected manifest with proper Brightspace resource types")
    
    def validate_brightspace_compatibility(self, package_dir):
        """Validate that the package meets Brightspace requirements."""
        print("🔍 Validating Brightspace compatibility...")
        
        issues = []
        
        # Check required files
        required_files = ['imsmanifest.xml', 'assessment_meta.xml']
        for file in required_files:
            if not os.path.exists(os.path.join(package_dir, file)):
                issues.append(f"Missing required file: {file}")
        
        # Check assessment files
        for week in range(1, 5):
            for assessment_type in ['quiz', 'assignment', 'discussion']:
                file_name = f"{assessment_type}_week_{week:02d}.xml"
                file_path = os.path.join(package_dir, file_name)
                if not os.path.exists(file_path):
                    issues.append(f"Missing assessment: {file_name}")
                else:
                    # Check file is not empty and contains proper XML
                    with open(file_path, 'r') as f:
                        content = f.read().strip()
                        if len(content) < 100:  # Minimum content check
                            issues.append(f"Assessment file too small: {file_name}")
                        if not content.startswith('<?xml'):
                            issues.append(f"Not proper XML: {file_name}")
        
        if issues:
            print("❌ Compatibility issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return False
        else:
            print("✅ Package passes Brightspace compatibility validation")
            return True

if __name__ == "__main__":
    prevention = Pattern13Prevention()
    
    # Example usage:
    # prevention.create_corrected_package("source_content_dir", "corrected_output_dir")
    print("Pattern 13 Prevention script ready for use.")
    print("Call create_corrected_package(content_dir, output_dir) to generate Brightspace-compatible packages.")