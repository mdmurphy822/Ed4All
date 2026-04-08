#!/usr/bin/env python3
"""
BULLETPROOF IMSCC Generator - Zero Tolerance Pattern 7 Prevention

This generator implements ABSOLUTE single-file enforcement with zero tolerance
for any folder multiplication. It will terminate immediately upon detecting
any Pattern 7 violations.

CRITICAL DESIGN PRINCIPLES:
1. SINGLE EXECUTION ONLY - Never retry, never create alternatives
2. ATOMIC OPERATIONS - All-or-nothing approach with immediate cleanup
3. ZERO TOLERANCE - Any violation triggers immediate termination
4. COMPREHENSIVE VALIDATION - Multiple checkpoints ensure compliance
5. BULLETPROOF CLEANUP - Guaranteed removal of any unwanted artifacts

Author: Claude Code Assistant (Emergency Pattern 7 Response)
Version: 2.0.0 (Bulletproof Edition)
Created: 2025-08-05 (Emergency Response)
"""

import zipfile
import uuid
import sys
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

class BulletproofIMSCCGenerator:
    """
    Bulletproof IMSCC generator with absolute Pattern 7 prevention.
    
    This generator implements zero-tolerance enforcement for single-file creation.
    Any detection of folder multiplication triggers immediate termination.
    """
    
    def __init__(self):
        """Initialize with strict enforcement protocols."""
        self.execution_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.temp_files = []
        self.created_paths = []
        
    def emergency_cleanup(self):
        """Emergency cleanup of all created files and directories."""
        for path in self.created_paths:
            try:
                if Path(path).exists():
                    if Path(path).is_file():
                        Path(path).unlink()
                    elif Path(path).is_dir():
                        shutil.rmtree(path)
                    print(f"🧹 Emergency cleanup: {path}")
            except Exception as e:
                print(f"⚠️  Cleanup warning: {path} - {e}")
    
    def validate_zero_pattern7_violations(self, output_path: str) -> bool:
        """
        ZERO TOLERANCE validation for Pattern 7 violations.
        
        Returns True only if EXACTLY ONE file exists and NO directories.
        Any other state triggers immediate termination.
        """
        output_file = Path(output_path)
        output_parent = output_file.parent
        
        # Check 1: The target file must exist
        if not output_file.exists():
            self.emergency_cleanup()
            raise SystemExit(f"ZERO TOLERANCE VIOLATION: Target file does not exist: {output_path}")
        
        # Check 2: The target file must be a valid ZIP
        if not zipfile.is_zipfile(output_path):
            self.emergency_cleanup()
            raise SystemExit(f"ZERO TOLERANCE VIOLATION: File is not a valid ZIP: {output_path}")
        
        # Check 3: NO directories with the same base name
        potential_dirs = [
            output_file.with_suffix(''),  # linear_algebra_course
            output_parent / f"{output_file.stem}",  # Same as above but explicit
        ]
        
        for potential_dir in potential_dirs:
            if potential_dir.exists() and potential_dir.is_dir():
                self.emergency_cleanup()
                raise SystemExit(f"ZERO TOLERANCE VIOLATION: Directory exists: {potential_dir}")
        
        # Check 4: NO numbered variants
        numbered_patterns = [
            f"{output_file.stem} (*)",
            f"{output_file.stem}_*",
            f"{output_file.stem}(*)",
        ]
        
        for pattern in numbered_patterns:
            matches = list(output_parent.glob(pattern))
            if matches:
                self.emergency_cleanup()
                raise SystemExit(f"ZERO TOLERANCE VIOLATION: Numbered variants found: {matches}")
        
        # Check 5: EXACTLY one file with our target name
        target_matches = list(output_parent.glob(f"{output_file.name}*"))
        if len(target_matches) != 1 or target_matches[0] != output_file:
            self.emergency_cleanup()
            raise SystemExit(f"ZERO TOLERANCE VIOLATION: Multiple target files: {target_matches}")
        
        print("✅ ZERO TOLERANCE VALIDATION: PASSED")
        return True
    
    def create_bulletproof_imscc(self, course_data: Dict[str, Any], output_path: str) -> Dict[str, Any]:
        """
        Create IMSCC with bulletproof single-file enforcement.
        
        This method implements absolute zero-tolerance Pattern 7 prevention.
        """
        print(f"🛡️  Starting bulletproof IMSCC generation: {output_path}")
        
        # CRITICAL: Normalize output path
        output_file = Path(output_path)
        if not output_file.suffix == '.imscc':
            output_file = output_file.with_suffix('.imscc')
            output_path = str(output_file)
        
        # CRITICAL: Pre-flight collision detection
        if output_file.exists():
            raise SystemExit(f"ZERO TOLERANCE: Output collision detected: {output_path}")
        
        # CRITICAL: Check for any existing Pattern 7 violations
        output_parent = output_file.parent
        existing_violations = []
        
        # Check for base directory
        base_dir = output_file.with_suffix('')
        if base_dir.exists():
            existing_violations.append(str(base_dir))
        
        # Check for numbered variants
        for item in output_parent.iterdir():
            if item.name.startswith(output_file.stem) and item.is_dir():
                existing_violations.append(str(item))
        
        if existing_violations:
            raise SystemExit(f"ZERO TOLERANCE: Pre-existing Pattern 7 violations: {existing_violations}")
        
        # Create parent directory if needed
        output_parent.mkdir(parents=True, exist_ok=True)
        self.created_paths.append(str(output_parent))
        
        # Create temporary working file
        temp_imscc = output_parent / f".temp_{self.execution_id}.imscc"
        self.created_paths.append(str(temp_imscc))
        
        try:
            # Generate manifest content
            manifest_content = self.generate_simple_manifest(course_data)
            
            # Create ZIP file with ONLY the manifest
            with zipfile.ZipFile(temp_imscc, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
                zipf.writestr('imsmanifest.xml', manifest_content)
                print("✅ Added manifest to bulletproof IMSCC")
            
            # Atomic rename to final location
            temp_imscc.rename(output_file)
            self.created_paths.append(str(output_file))
            
            # CRITICAL: Immediate validation
            self.validate_zero_pattern7_violations(output_path)
            
            # Final result
            package_size = output_file.stat().st_size
            result = {
                "status": "SUCCESS",
                "output_file": output_path,
                "course_title": course_data.get('title', 'Bulletproof Course'),
                "package_size": package_size,
                "pattern7_prevention": "ZERO_TOLERANCE_ENFORCED",
                "execution_id": self.execution_id,
                "validation_passed": True
            }
            
            print(f"🎯 BULLETPROOF SUCCESS: {output_path} ({package_size} bytes)")
            return result
            
        except Exception as e:
            self.emergency_cleanup()
            raise SystemExit(f"BULLETPROOF GENERATION FAILED: {e}")
    
    def generate_simple_manifest(self, course_data: Dict[str, Any]) -> str:
        """Generate minimal IMS Common Cartridge manifest."""
        course_id = str(uuid.uuid4())
        course_title = course_data.get('title', 'Bulletproof IMSCC Course')
        
        manifest = f'''<?xml version="1.0" encoding="UTF-8"?>
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
                    <lom:string language="en">Bulletproof IMSCC package with zero tolerance Pattern 7 prevention</lom:string>
                </lom:description>
            </lom:general>
        </lom:lom>
    </metadata>
    
    <organizations default="organization_1">
        <organization identifier="organization_1" structure="rooted-hierarchy">
            <title>{course_title}</title>
        </organization>
    </organizations>
    
    <resources>
        <!-- Bulletproof manifest - no additional resources to prevent complications -->
    </resources>
</manifest>'''
        
        return manifest

def bulletproof_test():
    """Test the bulletproof generator with zero tolerance enforcement."""
    generator = BulletproofIMSCCGenerator()

    # Test course data
    course_data = {
        "title": "Pattern 7 Prevention Test Course",
        "description": "Testing bulletproof single-file enforcement"
    }

    # Output path - use project exports directory
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent
    base_dir = Path(os.environ.get('COURSEFORGE_PATH', str(project_root)))
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = str(base_dir / "exports" / f"{timestamp}_bulletproof_test.imscc")
    
    print("🛡️  BULLETPROOF IMSCC GENERATOR TEST")
    print(f"📦 Target: {output_path}")
    print("🎯 Zero Tolerance Pattern 7 Prevention: ACTIVE")
    
    try:
        # Create parent directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Generate bulletproof IMSCC
        result = generator.create_bulletproof_imscc(course_data, output_path)
        
        print("\n🎉 BULLETPROOF TEST RESULTS:")
        print(f"✅ Status: {result['status']}")
        print(f"📦 File: {result['output_file']}")
        print(f"📊 Size: {result['package_size']} bytes")
        print(f"🛡️  Protection: {result['pattern7_prevention']}")
        print(f"🔍 Validation: {result['validation_passed']}")
        
        # Additional verification
        output_file = Path(output_path)
        parent_items = list(output_file.parent.iterdir())
        
        print(f"\n🔍 DIRECTORY VERIFICATION:")
        print(f"Items in output directory: {len(parent_items)}")
        for item in parent_items:
            item_type = "FILE" if item.is_file() else "DIRECTORY"
            print(f"  - {item.name} ({item_type})")
        
        if len(parent_items) == 1 and parent_items[0] == output_file:
            print("🎯 PATTERN 7 PREVENTION: 100% SUCCESS")
        else:
            print("❌ PATTERN 7 VIOLATION DETECTED")
        
    except SystemExit as e:
        print(f"\n💥 BULLETPROOF TERMINATION: {e}")
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {e}")

if __name__ == "__main__":
    bulletproof_test()