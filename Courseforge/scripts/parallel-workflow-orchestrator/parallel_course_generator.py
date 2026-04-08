#!/usr/bin/env python3
"""
Parallel Course Generation and IMSCC Packaging Orchestrator

This script coordinates multiple parallel agents for:
1. Weekly content generation (one agent per week)
2. IMSCC file creation (parallel brightspace-packager agents)
3. Final manifest generation (after all content completed)

Implements the updated workflow for reduced project time through parallel processing.
"""

import asyncio
import json
import os
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

class ParallelCourseOrchestrator:
    """Orchestrates parallel agents for course generation and IMSCC packaging"""

    def __init__(self, course_requirements: Dict):
        self.requirements = course_requirements
        self.course_duration = course_requirements.get('duration_weeks', 12)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Use environment variable or derive from script location
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent
        base_dir = Path(os.environ.get('COURSEFORGE_PATH', str(project_root)))

        self.working_dir = base_dir / f"{self.timestamp}_parallel_generation"
        self.export_dir = base_dir / "exports" / self.timestamp
        self.content_files = {}
        self.imscc_files = {}
        
    def setup_directories(self):
        """Create necessary directory structure"""
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Create week subdirectories for parallel processing
        for week in range(1, self.course_duration + 1):
            (self.working_dir / f"week_{week:02d}").mkdir(exist_ok=True)
    
    async def generate_week_content(self, week_number: int) -> Dict:
        """Generate content for a single week using dedicated agent"""
        week_dir = self.working_dir / f"week_{week_number:02d}"
        
        # Agent prompt for week-specific content generation
        agent_prompt = f"""
        Generate comprehensive Linear Algebra course content for Week {week_number} of {self.course_duration}.
        
        Requirements:
        - Create exactly 7 HTML sub-modules following Pattern 19 prevention
        - Implement authentic mathematical examples with comprehensive theoretical context
        - Generate 1 weekly assignment with proper D2L XML formatting
        - Ensure content depth meets Pattern 22 comprehensive educational standards
        - Output files to: {week_dir}
        
        Required files for Week {week_number}:
        1. week_{week_number:02d}_overview.html
        2. week_{week_number:02d}_concept1.html  
        3. week_{week_number:02d}_concept2.html
        4. week_{week_number:02d}_key_concepts.html
        5. week_{week_number:02d}_visual_display.html
        6. week_{week_number:02d}_applications.html
        7. week_{week_number:02d}_study_questions.html
        8. week_{week_number:02d}_assignment.xml (D2L format)
        
        Validation requirements:
        - Each HTML file must contain 600+ words of substantial educational content
        - Mathematical notation must be properly formatted
        - Content must support 3-credit undergraduate course standards
        - Assignment XML must create functional Brightspace dropbox
        """
        
        print(f"Starting Week {week_number} content generation...")
        
        # This would interface with Claude Code's agent system
        # For now, we'll simulate the agent call
        result = await self._simulate_agent_call("general-purpose", agent_prompt)
        
        # Validate generated files
        expected_files = [
            f"week_{week_number:02d}_overview.html",
            f"week_{week_number:02d}_concept1.html", 
            f"week_{week_number:02d}_concept2.html",
            f"week_{week_number:02d}_key_concepts.html",
            f"week_{week_number:02d}_visual_display.html",
            f"week_{week_number:02d}_applications.html",
            f"week_{week_number:02d}_study_questions.html",
            f"week_{week_number:02d}_assignment.xml"
        ]
        
        generated_files = []
        for file_name in expected_files:
            file_path = week_dir / file_name
            if file_path.exists():
                generated_files.append(str(file_path))
            else:
                print(f"WARNING: Expected file not generated: {file_path}")
        
        print(f"Week {week_number} generation completed: {len(generated_files)}/8 files")
        
        return {
            'week': week_number,
            'files': generated_files,
            'status': 'completed' if len(generated_files) == 8 else 'partial'
        }
    
    async def package_week_content(self, week_number: int, week_files: List[str]) -> Dict:
        """Package week content using brightspace-packager agent"""
        
        agent_prompt = f"""
        Create IMSCC-compatible files for Week {week_number} content using brightspace-packager agent.
        
        Input files: {week_files}
        
        Tasks:
        1. Convert HTML content files to IMSCC-compatible format
        2. Generate proper resource entries for manifest (DO NOT create manifest yet)
        3. Create QTI 1.2 compliant quiz files
        4. Generate D2L XML assessment files
        5. Ensure all files follow IMS Common Cartridge 1.2.0 standards
        
        Output requirements:
        - All files must be IMSCC-ready but not zipped
        - Generate resource metadata for later manifest compilation
        - Validate QTI and D2L XML compliance
        - Prepare organization items for hierarchical structure
        
        IMPORTANT: Do not generate imsmanifest.xml - this will be created after all weeks are complete.
        """
        
        print(f"Starting Week {week_number} IMSCC packaging...")
        
        result = await self._simulate_agent_call("brightspace-packager", agent_prompt)
        
        # Collect packaged files for this week
        packaged_files = []
        week_dir = self.working_dir / f"week_{week_number:02d}"
        
        for file_path in week_dir.glob("*"):
            if file_path.is_file() and file_path.suffix in ['.html', '.xml']:
                packaged_files.append(str(file_path))
        
        print(f"Week {week_number} packaging completed: {len(packaged_files)} files ready")
        
        return {
            'week': week_number,
            'packaged_files': packaged_files,
            'resource_metadata': self._generate_resource_metadata(week_number, packaged_files),
            'status': 'completed'
        }
    
    async def generate_final_manifest(self, all_resources: List[Dict]) -> str:
        """Generate imsmanifest.xml after all content and packaging is complete"""
        
        print("Generating final imsmanifest.xml...")
        
        # Collect all resources and organization items
        all_files = []
        organization_items = []
        
        for week_data in all_resources:
            week_num = week_data['week']
            files = week_data['packaged_files']
            
            all_files.extend(files)
            
            # Create organization structure for this week
            organization_items.append({
                'identifier': f'week_{week_num:02d}',
                'title': f'Week {week_num}',
                'items': self._create_week_organization_items(week_num, files)
            })
        
        # Generate manifest content
        manifest_content = self._create_manifest_xml(all_files, organization_items)
        
        manifest_path = self.export_dir / 'imsmanifest.xml'
        with open(manifest_path, 'w', encoding='utf-8') as f:
            f.write(manifest_content)
        
        print(f"Manifest generated: {manifest_path}")
        return str(manifest_path)
    
    async def create_final_imscc_package(self, manifest_path: str, all_files: List[str]) -> str:
        """Create final IMSCC ZIP package"""
        
        package_name = f"linear_algebra_parallel_generated_{self.timestamp}.imscc"
        package_path = self.export_dir / package_name
        
        print(f"Creating final IMSCC package: {package_name}")
        
        with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add manifest
            zipf.write(manifest_path, 'imsmanifest.xml')
            
            # Add all content files
            for file_path in all_files:
                file_obj = Path(file_path)
                archive_path = file_obj.name
                zipf.write(file_path, archive_path)
        
        # Validate package
        package_size = package_path.stat().st_size
        print(f"Package created: {package_path} ({package_size / 1024:.1f} KB)")
        
        if package_size < 100 * 1024:  # Less than 100KB indicates potential issue
            print("WARNING: Package size below expected threshold")
        
        return str(package_path)
    
    async def run_parallel_workflow(self) -> str:
        """Execute the complete parallel workflow"""
        
        print(f"Starting parallel workflow for {self.course_duration}-week course...")
        self.setup_directories()
        
        # Phase 1: Parallel content generation (one agent per week)
        print("\n=== Phase 1: Parallel Content Generation ===")
        content_tasks = []
        
        for week in range(1, self.course_duration + 1):
            task = self.generate_week_content(week)
            content_tasks.append(task)
        
        # Execute all content generation tasks concurrently
        content_results = await asyncio.gather(*content_tasks)
        
        # Validate all content was generated successfully
        successful_weeks = [r for r in content_results if r['status'] == 'completed']
        if len(successful_weeks) != self.course_duration:
            raise Exception(f"Content generation failed: {len(successful_weeks)}/{self.course_duration} weeks completed")
        
        print(f"Content generation completed: {len(successful_weeks)} weeks generated")
        
        # Phase 2: Parallel IMSCC packaging (one agent per week)  
        print("\n=== Phase 2: Parallel IMSCC Packaging ===")
        packaging_tasks = []
        
        for week_result in content_results:
            if week_result['status'] == 'completed':
                task = self.package_week_content(week_result['week'], week_result['files'])
                packaging_tasks.append(task)
        
        # Execute all packaging tasks concurrently
        packaging_results = await asyncio.gather(*packaging_tasks)
        
        # Validate all packaging completed successfully
        successful_packages = [r for r in packaging_results if r['status'] == 'completed']
        if len(successful_packages) != self.course_duration:
            raise Exception(f"Packaging failed: {len(successful_packages)}/{self.course_duration} weeks packaged")
        
        print(f"Packaging completed: {len(successful_packages)} weeks packaged")
        
        # Phase 3: Final manifest generation (after all content complete)
        print("\n=== Phase 3: Final Manifest Generation ===")
        manifest_path = await self.generate_final_manifest(packaging_results)
        
        # Phase 4: Create final IMSCC package
        print("\n=== Phase 4: Final Package Creation ===")
        all_files = []
        for result in packaging_results:
            all_files.extend(result['packaged_files'])
        
        package_path = await self.create_final_imscc_package(manifest_path, all_files)
        
        print(f"\n=== PARALLEL WORKFLOW COMPLETED ===")
        print(f"Final package: {package_path}")
        print(f"Total content files: {len(all_files)}")
        print(f"Processing time reduced through parallel agent execution")
        
        return package_path
    
    async def _simulate_agent_call(self, agent_type: str, prompt: str) -> Dict:
        """Simulate agent call (replace with actual Claude Code agent interface)"""
        # In real implementation, this would use Claude Code's Task tool
        # For now, simulate processing time
        await asyncio.sleep(2)  # Simulate agent processing time
        
        return {
            'agent_type': agent_type,
            'status': 'completed',
            'timestamp': datetime.now().isoformat()
        }
    
    def _generate_resource_metadata(self, week_number: int, files: List[str]) -> Dict:
        """Generate resource metadata for manifest compilation"""
        resources = []
        
        for file_path in files:
            file_obj = Path(file_path)
            resource_type = "webcontent"
            
            if file_obj.suffix == '.xml':
                if 'assignment' in file_obj.name:
                    resource_type = "imsccv1p1/d2l_2p0/assignment"
                elif 'quiz' in file_obj.name:
                    resource_type = "imsqti_xmlv1p2/imscc_xmlv1p1/assessment"
                elif 'discussion' in file_obj.name:
                    resource_type = "imsccv1p1/d2l_2p0/discussion"
            
            resources.append({
                'identifier': f"week_{week_number:02d}_{file_obj.stem}",
                'type': resource_type,
                'href': file_obj.name
            })
        
        return {'resources': resources}
    
    def _create_week_organization_items(self, week_number: int, files: List[str]) -> List[Dict]:
        """Create organization items for a week's content"""
        items = []
        
        for file_path in files:
            file_obj = Path(file_path)
            if file_obj.suffix == '.html':
                items.append({
                    'identifier': f"week_{week_number:02d}_{file_obj.stem}_item",
                    'title': self._format_title_from_filename(file_obj.stem),
                    'identifierref': f"week_{week_number:02d}_{file_obj.stem}"
                })
        
        return items
    
    def _format_title_from_filename(self, filename: str) -> str:
        """Format human-readable title from filename"""
        # Convert week_01_overview to "Week 1: Overview"
        parts = filename.split('_')
        if len(parts) >= 3:
            week_num = int(parts[1])
            content_type = parts[2].replace('_', ' ').title()
            return f"Week {week_num}: {content_type}"
        return filename.replace('_', ' ').title()
    
    def _create_manifest_xml(self, all_files: List[str], organization_items: List[Dict]) -> str:
        """Create complete imsmanifest.xml content"""
        
        manifest_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<manifest xmlns="http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1" 
          xmlns:lomimscc="http://ltsc.ieee.org/xsd/imsccv1p2/LOM/manifest" 
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
          identifier="manifest_{self.timestamp}" 
          xsi:schemaLocation="http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1 http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1.xsd http://ltsc.ieee.org/xsd/imsccv1p2/LOM/manifest http://ltsc.ieee.org/xsd/imsccv1p2/LOM/manifest.xsd">
  
  <metadata>
    <schema>IMS Common Cartridge</schema>
    <schemaversion>1.2.0</schemaversion>
    <lomimscc:lom>
      <lomimscc:general>
        <lomimscc:title>
          <lomimscc:string>Linear Algebra - Parallel Generated Course</lomimscc:string>
        </lomimscc:title>
      </lomimscc:general>
    </lomimscc:lom>
  </metadata>
  
  <organizations default="org_1">
    <organization identifier="org_1">
      <title>Linear Algebra Course Structure</title>
'''
        
        # Add organization items for each week
        for week_data in organization_items:
            manifest_xml += f'      <item identifier="{week_data["identifier"]}" title="{week_data["title"]}">\n'
            for item in week_data['items']:
                manifest_xml += f'        <item identifier="{item["identifier"]}" title="{item["title"]}" identifierref="{item["identifierref"]}"/>\n'
            manifest_xml += '      </item>\n'
        
        manifest_xml += '''    </organization>
  </organizations>
  
  <resources>
'''
        
        # Add resource entries for all files
        for file_path in all_files:
            file_obj = Path(file_path)
            resource_id = file_obj.stem
            resource_type = "webcontent"
            
            if file_obj.suffix == '.xml':
                if 'assignment' in file_obj.name:
                    resource_type = "imsccv1p1/d2l_2p0/assignment"
                elif 'quiz' in file_obj.name:
                    resource_type = "imsqti_xmlv1p2/imscc_xmlv1p1/assessment"
                elif 'discussion' in file_obj.name:
                    resource_type = "imsccv1p1/d2l_2p0/discussion"
            
            manifest_xml += f'    <resource identifier="{resource_id}" type="{resource_type}" href="{file_obj.name}"/>\n'
        
        manifest_xml += '''  </resources>
</manifest>'''
        
        return manifest_xml


async def main():
    """Main execution function"""
    
    # Load course requirements (would typically come from duration_specification.py)
    course_requirements = {
        'duration_weeks': 12,
        'credit_hours': 3,
        'course_level': 'undergraduate',
        'subject': 'Linear Algebra',
        'assessment_types': ['assignments', 'quizzes', 'discussions']
    }
    
    # Create and run parallel orchestrator
    orchestrator = ParallelCourseOrchestrator(course_requirements)
    
    try:
        package_path = await orchestrator.run_parallel_workflow()
        print(f"\nSUCCESS: Parallel workflow completed")
        print(f"Package location: {package_path}")
        return package_path
    
    except Exception as e:
        print(f"\nERROR: Parallel workflow failed: {e}")
        return None


if __name__ == "__main__":
    # Run the parallel workflow
    result = asyncio.run(main())
    
    if result:
        print(f"\nParallel course generation completed successfully: {result}")
        sys.exit(0)
    else:
        print(f"\nParallel course generation failed")
        sys.exit(1)