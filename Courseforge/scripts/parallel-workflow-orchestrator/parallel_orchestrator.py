#!/usr/bin/env python3
"""
Main Parallel Workflow Orchestrator

This is the primary entry point for the parallel course generation workflow.
It coordinates multiple agents to significantly reduce total project time.
"""

import asyncio
import json
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Import our agent coordination modules
from agent_interface import ParallelAgentCoordinator


class ParallelWorkflowOrchestrator:
    """Main orchestrator for the complete parallel workflow"""
    
    def __init__(self, course_requirements: Optional[Dict] = None):
        self.requirements = course_requirements or self._load_default_requirements()
        self.duration_weeks = self.requirements.get('duration_weeks', 12)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Directory structure - uses environment variable or derives from script location
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent.parent
        self.base_dir = Path(os.environ.get('COURSEFORGE_PATH', str(project_root)))
        self.working_dir = self.base_dir / f"{self.timestamp}_parallel_firstdraft"
        self.export_dir = self.base_dir / "exports" / self.timestamp
        
        # Agent coordinator
        self.agent_coordinator = ParallelAgentCoordinator()
        
        # Results tracking
        self.content_results = []
        self.packaging_results = []
        self.final_package_path = None
        
    def _load_default_requirements(self) -> Dict:
        """Load default course requirements"""
        return {
            'duration_weeks': 12,
            'credit_hours': 3,
            'course_level': 'undergraduate',
            'subject': 'Linear Algebra',
            'course_title': 'Introduction to Linear Algebra',
            'assessment_types': ['assignments', 'quizzes', 'discussions'],
            'pattern_prevention': {
                'pattern_19': True,  # Educational structure preservation
                'pattern_21': True,  # Complete content generation  
                'pattern_22': True   # Comprehensive educational content
            }
        }
    
    def setup_workspace(self):
        """Setup directory structure for parallel processing"""
        print(f"Setting up workspace: {self.working_dir}")
        
        # Create main directories
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Create week-specific directories
        for week in range(1, self.duration_weeks + 1):
            week_dir = self.working_dir / f"week_{week:02d}"
            week_dir.mkdir(exist_ok=True)
            
            # Also create export directories for packaging
            export_week_dir = self.export_dir / f"week_{week:02d}"
            export_week_dir.mkdir(exist_ok=True)
        
        print(f"Workspace ready: {self.duration_weeks} week directories created")
    
    async def execute_parallel_workflow(self) -> str:
        """Execute the complete parallel workflow"""
        
        print("="*80)
        print("PARALLEL COURSE GENERATION WORKFLOW STARTING")
        print("="*80)
        print(f"Course: {self.requirements['course_title']}")
        print(f"Duration: {self.duration_weeks} weeks")
        print(f"Timestamp: {self.timestamp}")
        print(f"Working Directory: {self.working_dir}")
        print(f"Export Directory: {self.export_dir}")
        print()
        
        workflow_start = datetime.now()
        
        try:
            # Setup workspace
            self.setup_workspace()
            
            # Phase 1: Parallel Content Generation
            print("PHASE 1: PARALLEL CONTENT GENERATION")
            print("-" * 40)
            
            phase1_start = datetime.now()
            self.content_results = await self.agent_coordinator.run_parallel_content_generation(
                self.requirements, self.working_dir
            )
            phase1_duration = (datetime.now() - phase1_start).total_seconds()
            
            if not self.content_results:
                raise Exception("No content was generated successfully")
            
            print(f"Phase 1 completed in {phase1_duration:.1f} seconds")
            print(f"Successfully generated content for {len(self.content_results)}/{self.duration_weeks} weeks")
            print()
            
            # Validate content generation
            await self._validate_content_generation()
            
            # Phase 2: Parallel IMSCC Packaging  
            print("PHASE 2: PARALLEL IMSCC PACKAGING")
            print("-" * 40)
            
            phase2_start = datetime.now()
            self.packaging_results = await self.agent_coordinator.run_parallel_packaging(
                self.content_results, self.export_dir
            )
            phase2_duration = (datetime.now() - phase2_start).total_seconds()
            
            if not self.packaging_results:
                raise Exception("No content was packaged successfully")
                
            print(f"Phase 2 completed in {phase2_duration:.1f} seconds")
            print(f"Successfully packaged {len(self.packaging_results)}/{len(self.content_results)} weeks")
            print()
            
            # Phase 3: Final Manifest Generation (after all content complete)
            print("PHASE 3: FINAL MANIFEST GENERATION")
            print("-" * 40)
            
            phase3_start = datetime.now()
            manifest_path = await self._generate_final_manifest()
            phase3_duration = (datetime.now() - phase3_start).total_seconds()
            
            print(f"Phase 3 completed in {phase3_duration:.1f} seconds")
            print(f"Manifest generated: {manifest_path}")
            print()
            
            # Phase 4: Final IMSCC Package Creation
            print("PHASE 4: FINAL PACKAGE CREATION")
            print("-" * 40)
            
            phase4_start = datetime.now()
            self.final_package_path = await self._create_final_package(manifest_path)
            phase4_duration = (datetime.now() - phase4_start).total_seconds()
            
            print(f"Phase 4 completed in {phase4_duration:.1f} seconds")
            print(f"Final package: {self.final_package_path}")
            print()
            
            # Workflow completion summary
            total_duration = (datetime.now() - workflow_start).total_seconds()
            
            print("="*80)
            print("PARALLEL WORKFLOW COMPLETED SUCCESSFULLY")
            print("="*80)
            print(f"Total Processing Time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
            print(f"Phase 1 (Content Generation): {phase1_duration:.1f}s")
            print(f"Phase 2 (IMSCC Packaging): {phase2_duration:.1f}s") 
            print(f"Phase 3 (Manifest Generation): {phase3_duration:.1f}s")
            print(f"Phase 4 (Package Creation): {phase4_duration:.1f}s")
            print()
            print(f"Final Package: {self.final_package_path}")
            print(f"Package Size: {self._get_package_size()}")
            print(f"Content Files: {self._count_total_files()}")
            print()
            
            # Performance analysis
            estimated_sequential_time = self.duration_weeks * 75  # ~75 seconds per week sequentially
            time_savings = estimated_sequential_time - total_duration
            efficiency_gain = (time_savings / estimated_sequential_time) * 100
            
            print(f"PERFORMANCE ANALYSIS:")
            print(f"Estimated Sequential Time: {estimated_sequential_time:.1f}s ({estimated_sequential_time/60:.1f}m)")
            print(f"Actual Parallel Time: {total_duration:.1f}s ({total_duration/60:.1f}m)")
            print(f"Time Savings: {time_savings:.1f}s ({time_savings/60:.1f}m)")
            print(f"Efficiency Gain: {efficiency_gain:.1f}%")
            print()
            
            return self.final_package_path
            
        except Exception as e:
            print(f"\nERROR: Parallel workflow failed: {e}")
            await self._cleanup_on_error()
            raise e
    
    async def _validate_content_generation(self):
        """Validate all content was generated successfully"""
        
        print("Validating content generation...")
        
        total_files_expected = self.duration_weeks * 8  # 7 HTML + 1 XML per week
        total_files_found = 0
        
        validation_errors = []
        
        for week_result in self.content_results:
            week_num = week_result.get('week', 0)
            week_dir = self.working_dir / f"week_{week_num:02d}"
            
            # Expected files for each week
            expected_files = [
                f"week_{week_num:02d}_overview.html",
                f"week_{week_num:02d}_concept1.html",
                f"week_{week_num:02d}_concept2.html", 
                f"week_{week_num:02d}_key_concepts.html",
                f"week_{week_num:02d}_visual_display.html",
                f"week_{week_num:02d}_applications.html",
                f"week_{week_num:02d}_study_questions.html",
                f"week_{week_num:02d}_assignment.xml"
            ]
            
            week_files_found = 0
            for expected_file in expected_files:
                file_path = week_dir / expected_file
                if file_path.exists():
                    # Check file size to ensure substantial content
                    file_size = file_path.stat().st_size
                    if file_size < 1000:  # Less than 1KB indicates placeholder content
                        validation_errors.append(f"Week {week_num}: {expected_file} too small ({file_size} bytes)")
                    else:
                        week_files_found += 1
                        total_files_found += 1
                else:
                    validation_errors.append(f"Week {week_num}: Missing {expected_file}")
            
            print(f"Week {week_num}: {week_files_found}/8 files validated")
        
        if validation_errors:
            print(f"\nVALIDATION ERRORS ({len(validation_errors)} issues):")
            for error in validation_errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(validation_errors) > 10:
                print(f"  ... and {len(validation_errors) - 10} more errors")
                
            raise Exception(f"Content validation failed: {len(validation_errors)} issues found")
        
        print(f"Content validation passed: {total_files_found}/{total_files_expected} files validated")
    
    async def _generate_final_manifest(self) -> str:
        """Generate the final imsmanifest.xml using an agent"""
        
        # Collect all resource metadata
        all_resources = []
        for week_result in self.packaging_results:
            all_resources.append(week_result)
        
        # Use agent to generate manifest
        manifest_result = await self.agent_coordinator.agent_interface.launch_manifest_generator(
            all_resources, self.export_dir
        )
        
        manifest_path = self.export_dir / "imsmanifest.xml"
        
        # For now, create a basic manifest since we're simulating agents
        # In real implementation, the agent would create this
        manifest_content = self._create_manifest_content(all_resources)
        
        with open(manifest_path, 'w', encoding='utf-8') as f:
            f.write(manifest_content)
        
        return str(manifest_path)
    
    async def _create_final_package(self, manifest_path: str) -> str:
        """Create the final IMSCC ZIP package"""
        
        package_name = f"linear_algebra_parallel_{self.timestamp}.imscc"
        package_path = self.export_dir / package_name
        
        print(f"Creating final IMSCC package: {package_name}")
        
        # Collect all files for packaging
        all_files = []
        
        # Add all week content files
        for week in range(1, self.duration_weeks + 1):
            week_dir = self.working_dir / f"week_{week:02d}"
            for file_path in week_dir.glob("*"):
                if file_path.is_file():
                    all_files.append(file_path)
        
        # Create ZIP package
        with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Add manifest
            zipf.write(manifest_path, 'imsmanifest.xml')
            
            # Add all content files
            for file_path in all_files:
                # Use just the filename in the archive
                archive_name = file_path.name
                zipf.write(file_path, archive_name)
        
        # Validate package
        package_size = package_path.stat().st_size
        print(f"Package created: {package_size / 1024:.1f} KB")
        
        if package_size < 100 * 1024:  # Less than 100KB
            print("WARNING: Package size below expected threshold")
        
        return str(package_path)
    
    def _create_manifest_content(self, all_resources: List[Dict]) -> str:
        """Create basic manifest content (placeholder for agent-generated content)"""
        
        manifest_xml = f'''<?xml version="1.0" encoding="UTF-8"?>
<manifest xmlns="http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1" 
          xmlns:lomimscc="http://ltsc.ieee.org/xsd/imsccv1p2/LOM/manifest" 
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
          identifier="manifest_{self.timestamp}" 
          xsi:schemaLocation="http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1 http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1.xsd">
  
  <metadata>
    <schema>IMS Common Cartridge</schema>
    <schemaversion>1.2.0</schemaversion>
    <lomimscc:lom>
      <lomimscc:general>
        <lomimscc:title>
          <lomimscc:string>{self.requirements['course_title']} - Parallel Generated</lomimscc:string>
        </lomimscc:title>
      </lomimscc:general>
    </lomimscc:lom>
  </metadata>
  
  <organizations default="org_1">
    <organization identifier="org_1">
      <title>Course Structure</title>
'''
        
        # Add organization items for each week
        for week in range(1, self.duration_weeks + 1):
            manifest_xml += f'      <item identifier="week_{week:02d}" title="Week {week}">\n'
            
            # Add sub-module items
            sub_modules = ['overview', 'concept1', 'concept2', 'key_concepts', 
                          'visual_display', 'applications', 'study_questions']
            
            for sub_module in sub_modules:
                item_id = f"week_{week:02d}_{sub_module}_item"
                resource_id = f"week_{week:02d}_{sub_module}"
                title = f"Week {week}: {sub_module.replace('_', ' ').title()}"
                
                manifest_xml += f'        <item identifier="{item_id}" title="{title}" identifierref="{resource_id}"/>\n'
            
            manifest_xml += '      </item>\n'
        
        manifest_xml += '''    </organization>
  </organizations>
  
  <resources>
'''
        
        # Add resource entries
        for week in range(1, self.duration_weeks + 1):
            # HTML resources
            sub_modules = ['overview', 'concept1', 'concept2', 'key_concepts',
                          'visual_display', 'applications', 'study_questions']
            
            for sub_module in sub_modules:
                resource_id = f"week_{week:02d}_{sub_module}"
                file_name = f"week_{week:02d}_{sub_module}.html"
                manifest_xml += f'    <resource identifier="{resource_id}" type="webcontent" href="{file_name}"/>\n'
            
            # Assignment XML resource
            assignment_id = f"week_{week:02d}_assignment"
            assignment_file = f"week_{week:02d}_assignment.xml"
            manifest_xml += f'    <resource identifier="{assignment_id}" type="imsccv1p1/d2l_2p0/assignment" href="{assignment_file}"/>\n'
        
        manifest_xml += '''  </resources>
</manifest>'''
        
        return manifest_xml
    
    def _get_package_size(self) -> str:
        """Get formatted package size"""
        if not self.final_package_path:
            return "Unknown"
        
        try:
            size_bytes = Path(self.final_package_path).stat().st_size
            if size_bytes > 1024 * 1024:
                return f"{size_bytes / (1024 * 1024):.1f} MB"
            else:
                return f"{size_bytes / 1024:.1f} KB"
        except:
            return "Unknown"
    
    def _count_total_files(self) -> int:
        """Count total files in package"""
        if not self.final_package_path:
            return 0
        
        try:
            with zipfile.ZipFile(self.final_package_path, 'r') as zipf:
                return len(zipf.filelist)
        except:
            return 0
    
    async def _cleanup_on_error(self):
        """Cleanup on workflow error"""
        print("Performing cleanup after error...")
        
        # Could implement cleanup logic here
        # For now, just log the error state
        error_log = {
            'timestamp': self.timestamp,
            'working_dir': str(self.working_dir),
            'export_dir': str(self.export_dir),
            'content_results_count': len(self.content_results),
            'packaging_results_count': len(self.packaging_results)
        }
        
        print(f"Error state logged: {error_log}")


async def main():
    """Main execution function"""

    # Load course requirements from file if available
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    base_dir = Path(os.environ.get('COURSEFORGE_PATH', str(project_root)))
    requirements_file = base_dir / "scripts" / "course-requirements" / "current_requirements.json"
    
    if requirements_file.exists():
        try:
            with open(requirements_file, 'r') as f:
                course_requirements = json.load(f)
            print(f"Loaded requirements from: {requirements_file}")
        except Exception as e:
            print(f"Error loading requirements file: {e}")
            course_requirements = None
    else:
        print("No requirements file found, using defaults")
        course_requirements = None
    
    # Create and run orchestrator
    orchestrator = ParallelWorkflowOrchestrator(course_requirements)
    
    try:
        package_path = await orchestrator.execute_parallel_workflow()
        print(f"\n✅ SUCCESS: Parallel workflow completed")
        print(f"📦 Package: {package_path}")
        return package_path
        
    except Exception as e:
        print(f"\n❌ FAILED: Parallel workflow error: {e}")
        return None


if __name__ == "__main__":
    # Run the parallel workflow
    result = asyncio.run(main())
    
    if result:
        print(f"\n🎉 Parallel course generation completed successfully!")
        print(f"📍 Location: {result}")
        sys.exit(0)
    else:
        print(f"\n💥 Parallel course generation failed")
        sys.exit(1)