#!/usr/bin/env python3
"""
Course Content Parser - Extract and structure content from markdown course materials

This script implements atomic operations and comprehensive validation to parse
course content into structured JSON format for IMSCC generation.

Author: Claude Code Assistant
Version: 1.0.0
Created: 2025-08-05
"""

import json
import re
import sys
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, TYPE_CHECKING
import argparse

# Add Ed4All lib to path for decision capture
ED4ALL_ROOT = Path(__file__).resolve().parents[3]  # scripts/course-content-parser/... → Ed4All/
if str(ED4ALL_ROOT) not in sys.path:
    sys.path.insert(0, str(ED4ALL_ROOT))

if TYPE_CHECKING:
    from lib.decision_capture import DecisionCapture

class CourseContentParser:
    """
    Parses markdown course materials into structured JSON format.
    
    Implements atomic operations with comprehensive validation to ensure
    reliable content extraction meeting IMSCC generation requirements.
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        capture: Optional["DecisionCapture"] = None,
    ):
        """
        Initialize parser with configuration and logging.

        Args:
            config_path (str, optional): Path to configuration file
            capture: Optional DecisionCapture for logging parsing decisions
        """
        self.setup_logging()
        self.config = self.load_config(config_path)
        self.temp_files = []
        self.execution_lock = None
        self.capture = capture
        
    def setup_logging(self):
        """Configure comprehensive logging for all operations."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('course_parser.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load parser configuration with defaults.
        
        Args:
            config_path (str, optional): Path to config file
            
        Returns:
            dict: Configuration parameters
        """
        default_config = {
            "min_word_counts": {
                "overview": 200,
                "concept_summary": 300,
                "key_concept": 50,
                "application_example": 250,
                "study_questions": 150
            },
            "required_sub_modules": 7,
            "max_weeks": 16,
            "validation_strict": True
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
                
        return default_config
    
    def validate_execution_environment(self, input_path: str, output_path: str):
        """
        Comprehensive pre-flight validation before any processing.
        
        Args:
            input_path (str): Input course directory path
            output_path (str): Output JSON file path
            
        Raises:
            SystemExit: If validation fails
        """
        self.logger.info("Starting execution environment validation")
        
        # Validate input directory exists and has required structure
        input_dir = Path(input_path)
        if not input_dir.exists():
            raise SystemExit(f"CRITICAL ERROR: Input directory does not exist: {input_path}")
            
        # Check for required files
        required_files = ['course_info.md', 'syllabus.md', 'assessment_guide.md']
        for file_name in required_files:
            if not (input_dir / file_name).exists():
                raise SystemExit(f"CRITICAL ERROR: Required file missing: {file_name}")
        
        # Validate modules directory exists
        modules_dir = input_dir / 'modules'
        if not modules_dir.exists():
            raise SystemExit(f"CRITICAL ERROR: Modules directory missing: {modules_dir}")
            
        # Check for week files
        week_files = list(modules_dir.glob('week_*.md'))
        if not week_files:
            raise SystemExit("CRITICAL ERROR: No week_*.md files found in modules directory")
            
        if len(week_files) > self.config['max_weeks']:
            raise SystemExit(f"CRITICAL ERROR: Too many weeks ({len(week_files)}) exceeds maximum ({self.config['max_weeks']})")
        
        # Validate output path doesn't exist
        output_file = Path(output_path)
        if output_file.exists():
            raise SystemExit(f"COLLISION DETECTED: Output file already exists: {output_path}")
            
        # Create execution lock
        self.execution_lock = input_dir / '.parser_execution_lock'
        if self.execution_lock.exists():
            raise SystemExit("EXECUTION LOCK ERROR: Another parser process is running")
            
        self.execution_lock.touch()
        self.logger.info("Execution environment validation passed")
    
    def parse_course_info(self, course_info_path: Path) -> Dict[str, Any]:
        """
        Parse course_info.md file for basic course metadata.
        
        Args:
            course_info_path (Path): Path to course_info.md file
            
        Returns:
            dict: Course information structure
        """
        self.logger.info(f"Parsing course info from: {course_info_path}")
        
        with open(course_info_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract course title (first heading)
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        title = title_match.group(1) if title_match else "Unknown Course"
        
        # Extract description (content after title, before next heading)
        desc_pattern = r'^#\s+.+?\n\n(.*?)(?=\n#+|\n\*\*|\Z)'
        desc_match = re.search(desc_pattern, content, re.DOTALL | re.MULTILINE)
        description = desc_match.group(1).strip() if desc_match else ""
        
        # Extract learning objectives
        objectives = []
        obj_pattern = r'(?:Learning Objectives?|Objectives?|Goals?):\s*\n((?:\s*[-\*]\s*.+\n?)*)'
        obj_match = re.search(obj_pattern, content, re.IGNORECASE | re.MULTILINE)
        if obj_match:
            obj_content = obj_match.group(1)
            objectives = re.findall(r'[-\*]\s*(.+)', obj_content)
        
        # Extract credits and duration
        credits_match = re.search(r'(\d+)\s*credits?', content, re.IGNORECASE)
        credits = int(credits_match.group(1)) if credits_match else 3
        
        weeks_match = re.search(r'(\d+)\s*weeks?', content, re.IGNORECASE)
        duration_weeks = int(weeks_match.group(1)) if weeks_match else 4
        
        course_info = {
            "title": self.clean_text(title),
            "description": self.clean_text(description),
            "learning_objectives": [self.clean_text(obj) for obj in objectives],
            "credits": credits,
            "duration_weeks": duration_weeks
        }
        
        self.logger.info(f"Parsed course info: {course_info['title']}")
        return course_info
    
    def parse_syllabus(self, syllabus_path: Path) -> Dict[str, Any]:
        """
        Parse syllabus.md file for course policies and schedule.
        
        Args:
            syllabus_path (Path): Path to syllabus.md file
            
        Returns:
            dict: Syllabus information structure
        """
        self.logger.info(f"Parsing syllabus from: {syllabus_path}")
        
        with open(syllabus_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Extract major sections
        policies_match = re.search(r'(?:Policies?|Rules?)[\s:]*\n(.*?)(?=\n#+|\Z)', content, re.DOTALL | re.IGNORECASE)
        policies = policies_match.group(1).strip() if policies_match else ""
        
        schedule_match = re.search(r'(?:Schedule|Calendar)[\s:]*\n(.*?)(?=\n#+|\Z)', content, re.DOTALL | re.IGNORECASE)
        schedule = schedule_match.group(1).strip() if schedule_match else ""
        
        requirements_match = re.search(r'(?:Technical Requirements?|Requirements?)[\s:]*\n(.*?)(?=\n#+|\Z)', content, re.DOTALL | re.IGNORECASE)
        requirements = requirements_match.group(1).strip() if requirements_match else ""
        
        syllabus_info = {
            "policies": self.clean_text(policies),
            "schedule": self.clean_text(schedule),
            "requirements": self.clean_text(requirements),
            "full_content": self.clean_text(content)
        }
        
        self.logger.info("Parsed syllabus information")
        return syllabus_info
    
    def parse_week_content(self, week_file: Path) -> Dict[str, Any]:
        """
        Parse individual week markdown file to extract sub-modules.

        Args:
            week_file (Path): Path to week_XX.md file

        Returns:
            dict: Week structure with sub-modules (count varies by content)
        """
        week_number = self.extract_week_number(week_file.name)
        self.logger.info(f"Parsing week {week_number} content from: {week_file}")
        
        with open(week_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Validate template variables are resolved
        unresolved_vars = re.findall(r'\{[^}]+\}', content)
        if unresolved_vars:
            raise SystemExit(f"CRITICAL ERROR: Unresolved template variables in {week_file}: {unresolved_vars}")
        
        # Extract week title
        title_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        week_title = title_match.group(1) if title_match else f"Week {week_number}"
        
        # Parse the 7 required sub-modules
        sub_modules = []
        
        # 1. Module Overview
        overview = self.extract_sub_module(content, "overview", "Module Overview|Week.*Overview|Introduction")
        if overview:
            sub_modules.append(overview)
        
        # 2-3. Concept Summary Pages (2-3 per week)
        concept_summaries = self.extract_concept_summaries(content)
        sub_modules.extend(concept_summaries)
        
        # 4. Key Concepts Accordion
        key_concepts = self.extract_sub_module(content, "key_concepts", "Key Concepts|Important Terms|Vocabulary")
        if key_concepts:
            sub_modules.append(key_concepts)
        
        # 5. Visual/Graphical/Math Display
        visual_content = self.extract_sub_module(content, "visual_content", "Visual|Graphical|Math|Diagrams|Charts")
        if visual_content:
            sub_modules.append(visual_content)
        
        # 6. Application Examples
        application_examples = self.extract_sub_module(content, "application_examples", "Application|Examples|Practice")
        if application_examples:
            sub_modules.append(application_examples)
        
        # 7. Real World Applications
        real_world = self.extract_sub_module(content, "real_world", "Real World|Industry|Professional")
        if real_world:
            sub_modules.append(real_world)
        
        # 8. Study Questions
        study_questions = self.extract_sub_module(content, "study_questions", "Study Questions|Questions|Reflection")
        if study_questions:
            sub_modules.append(study_questions)
        
        # Validate minimum sub-modules (dynamic count based on content)
        min_modules = self.config.get('min_sub_modules', 1)
        if len(sub_modules) < min_modules:
            self.logger.warning(f"Week {week_number} has {len(sub_modules)} sub-modules, minimum {min_modules}")
            # Add additional modules from content if needed
            sub_modules = self.ensure_minimum_modules(sub_modules, content, week_number, min_modules)
        
        week_data = {
            "week_number": week_number,
            "title": self.clean_text(week_title),
            "sub_modules": sub_modules,
            "full_content": self.clean_text(content)
        }

        # Log content structure decision
        if self.capture:
            module_types = [m.get("type", "unknown") for m in sub_modules]
            self.capture.log_decision(
                decision_type="content_structure",
                decision=f"Parsed week {week_number} into {len(sub_modules)} sub-modules",
                rationale=f"Module types: {module_types}, Title: {week_title}",
            )

        self.logger.info(f"Parsed week {week_number} with {len(sub_modules)} sub-modules")
        return week_data
    
    def extract_sub_module(self, content: str, module_type: str, header_pattern: str) -> Optional[Dict[str, Any]]:
        """
        Extract a specific sub-module from week content.
        
        Args:
            content (str): Full week content
            module_type (str): Type of sub-module
            header_pattern (str): Regex pattern for section headers
            
        Returns:
            dict: Sub-module structure or None if not found
        """
        # Find section with matching header
        pattern = rf'^#+\s*({header_pattern}).*?\n(.*?)(?=\n#+|\Z)'
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL | re.MULTILINE)
        
        if not match:
            return None
        
        title = match.group(1).strip()
        module_content = match.group(2).strip()
        
        # Validate content quality
        word_count = len(module_content.split())
        min_words = self.config['min_word_counts'].get(module_type, 100)
        
        if word_count < min_words:
            self.logger.warning(f"Sub-module '{title}' has only {word_count} words, minimum is {min_words}")
        
        # Extract learning objectives if present
        objectives = []
        obj_pattern = r'(?:Learning Objectives?|Objectives?):\s*\n((?:\s*[-\*]\s*.+\n?)*)'
        obj_match = re.search(obj_pattern, module_content, re.IGNORECASE | re.MULTILINE)
        if obj_match:
            obj_content = obj_match.group(1)
            objectives = re.findall(r'[-\*]\s*(.+)', obj_content)
        
        # Extract key concepts for accordion modules
        key_concepts = []
        if module_type == "key_concepts":
            key_concepts = self.extract_key_concept_definitions(module_content)
        
        return {
            "type": module_type,
            "title": self.clean_text(title),
            "content": self.clean_text(module_content),
            "learning_objectives": [self.clean_text(obj) for obj in objectives],
            "key_concepts": key_concepts,
            "word_count": word_count
        }
    
    def extract_concept_summaries(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract 2-3 concept summary pages from week content.
        
        Args:
            content (str): Full week content
            
        Returns:
            list: List of concept summary sub-modules
        """
        summaries = []
        
        # Look for concept or summary sections
        concept_patterns = [
            r'^#+\s*(Concept.*Summary|Summary.*Concept).*?\n(.*?)(?=\n#+|\Z)',
            r'^#+\s*(Concept\s+\d+|Chapter\s+\d+).*?\n(.*?)(?=\n#+|\Z)',
            r'^#+\s*(.*Concepts?.*|.*Theory.*|.*Principles?.*).*?\n(.*?)(?=\n#+|\Z)'
        ]
        
        for pattern in concept_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE | re.DOTALL | re.MULTILINE)
            for i, (title, module_content) in enumerate(matches[:3]):  # Max 3 concept summaries
                word_count = len(module_content.split())
                min_words = self.config['min_word_counts'].get('concept_summary', 300)
                
                if word_count >= min_words:
                    summaries.append({
                        "type": "concept_summary",
                        "title": self.clean_text(title),
                        "content": self.clean_text(module_content),
                        "learning_objectives": [],
                        "key_concepts": [],
                        "word_count": word_count
                    })
                    
                if len(summaries) >= 3:
                    break
            
            if len(summaries) >= 2:
                break
        
        # If we didn't find enough concept summaries, create them from content
        if len(summaries) < 2:
            summaries = self.create_concept_summaries_from_content(content, summaries)
        
        return summaries[:3]  # Return maximum 3
    
    def create_concept_summaries_from_content(self, content: str, existing_summaries: List[Dict]) -> List[Dict]:
        """
        Create concept summaries from general content when not explicitly found.
        
        Args:
            content (str): Full week content
            existing_summaries (list): Already found summaries
            
        Returns:
            list: Updated list with created summaries
        """
        # Split content into logical sections
        sections = re.split(r'\n#+\s+', content)
        sections = [s.strip() for s in sections if len(s.split()) > 200]  # Only substantial sections
        
        summaries = existing_summaries.copy()
        
        for i, section in enumerate(sections[:3]):  # Max 3 sections
            if len(summaries) >= 3:
                break
                
            # Extract title from first line
            lines = section.split('\n')
            title = lines[0] if lines else f"Concept Summary {i+1}"
            section_content = '\n'.join(lines[1:]) if len(lines) > 1 else section
            
            word_count = len(section_content.split())
            if word_count >= 200:  # Minimum for generated summaries
                summaries.append({
                    "type": "concept_summary",
                    "title": self.clean_text(title),
                    "content": self.clean_text(section_content),
                    "learning_objectives": [],
                    "key_concepts": [],
                    "word_count": word_count
                })
        
        return summaries
    
    def extract_key_concept_definitions(self, content: str) -> List[Dict[str, str]]:
        """
        Extract key concepts and their definitions for accordion display.
        
        Args:
            content (str): Key concepts section content
            
        Returns:
            list: List of key concept dictionaries with term and definition
        """
        key_concepts = []
        
        # Pattern 1: Term: Definition format
        definition_pattern = r'^([^:\n]+):\s*(.+)$'
        matches = re.findall(definition_pattern, content, re.MULTILINE)
        
        for term, definition in matches:
            term = term.strip()
            definition = definition.strip()
            word_count = len(definition.split())
            
            if word_count >= self.config['min_word_counts'].get('key_concept', 50):
                key_concepts.append({
                    "term": self.clean_text(term),
                    "definition": self.clean_text(definition),
                    "word_count": word_count
                })
        
        # Pattern 2: Bulleted list with definitions
        if not key_concepts:
            bullet_pattern = r'^\s*[-\*]\s*([^:\n]+):\s*(.+)$'
            matches = re.findall(bullet_pattern, content, re.MULTILINE)
            
            for term, definition in matches:
                term = term.strip()
                definition = definition.strip()
                word_count = len(definition.split())
                
                if word_count >= 30:  # Lower threshold for bulleted format
                    key_concepts.append({
                        "term": self.clean_text(term),
                        "definition": self.clean_text(definition),
                        "word_count": word_count
                    })
        
        return key_concepts[:10]  # Maximum 10 key concepts
    
    def ensure_minimum_modules(self, existing_modules: List[Dict], content: str, week_number: int, min_count: int = 1) -> List[Dict]:
        """
        Ensure minimum sub-modules by filling gaps with content sections if needed.

        Args:
            existing_modules (list): Already parsed modules
            content (str): Full week content
            week_number (int): Week number for reference
            min_count (int): Minimum number of modules required

        Returns:
            list: List with at least min_count sub-modules
        """
        common_types = [
            "overview", "concept_summary", "key_concepts",
            "application_examples", "study_questions"
        ]

        modules = existing_modules.copy()
        existing_types = [m['type'] for m in modules]

        # Fill missing types up to minimum count
        for module_type in common_types:
            if len(modules) >= min_count:
                break
            if module_type not in existing_types:
                placeholder = self.create_placeholder_module(module_type, content, week_number)
                modules.append(placeholder)
                existing_types.append(module_type)

        return modules
    
    def create_placeholder_module(self, module_type: str, content: str, week_number: int) -> Dict[str, Any]:
        """
        Create a placeholder module when content is not explicitly found.
        
        Args:
            module_type (str): Type of module to create
            content (str): Source content to extract from
            week_number (int): Week number for reference
            
        Returns:
            dict: Placeholder module structure
        """
        type_titles = {
            "overview": f"Week {week_number} Overview",
            "concept_summary": f"Concept Summary",
            "key_concepts": f"Key Concepts",
            "visual_content": f"Visual Content",
            "application_examples": f"Application Examples",
            "real_world": f"Real World Applications",
            "study_questions": f"Study Questions"
        }
        
        # Extract a reasonable portion of content
        sentences = re.split(r'[.!?]+', content)
        sentences = [s.strip() for s in sentences if len(s.split()) > 10]
        
        # Take first few sentences for placeholder content
        placeholder_content = '. '.join(sentences[:3]) + '.' if sentences else "Content to be developed."
        
        return {
            "type": module_type,
            "title": type_titles.get(module_type, f"Module {module_type.title()}"),
            "content": self.clean_text(placeholder_content),
            "learning_objectives": [],
            "key_concepts": [],
            "word_count": len(placeholder_content.split())
        }
    
    def parse_assessments(self, assessment_guide_path: Path) -> List[Dict[str, Any]]:
        """
        Parse assessment_guide.md file for assignment details.
        
        Args:
            assessment_guide_path (Path): Path to assessment_guide.md file
            
        Returns:
            list: List of assessment structures
        """
        self.logger.info(f"Parsing assessments from: {assessment_guide_path}")
        
        with open(assessment_guide_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        assessments = []
        
        # Extract assignments by week
        assignment_pattern = r'Week\s+(\d+)[^:]*:\s*([^\n]+)\n(.*?)(?=Week\s+\d+|Assignment\s+\d+|\Z)'
        matches = re.findall(assignment_pattern, content, re.IGNORECASE | re.DOTALL)
        
        for week, title, description in matches:
            week_num = int(week)
            
            # Extract word limit
            word_limit_match = re.search(r'(\d+)[-\s]*(\d+)?\s*words?', description, re.IGNORECASE)
            word_limit = "700-1000 words"
            if word_limit_match:
                if word_limit_match.group(2):
                    word_limit = f"{word_limit_match.group(1)}-{word_limit_match.group(2)} words"
                else:
                    word_limit = f"{word_limit_match.group(1)} words"
            
            # Extract points
            points_match = re.search(r'(\d+)\s*points?', description, re.IGNORECASE)
            points = int(points_match.group(1)) if points_match else 100
            
            # Extract rubric information
            rubric_match = re.search(r'(?:Rubric|Criteria|Grading):(.*?)(?=\n\n|\Z)', description, re.IGNORECASE | re.DOTALL)
            rubric = rubric_match.group(1).strip() if rubric_match else "Standard rubric applies"
            
            assessments.append({
                "week": week_num,
                "type": "assignment",
                "title": self.clean_text(title),
                "description": self.clean_text(description),
                "word_limit": word_limit,
                "points": points,
                "rubric": self.clean_text(rubric)
            })
        
        self.logger.info(f"Parsed {len(assessments)} assessments")
        return assessments
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Remove markdown formatting
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)      # Italic
        text = re.sub(r'`(.*?)`', r'\1', text)        # Code
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        return text
    
    def extract_week_number(self, filename: str) -> int:
        """
        Extract week number from filename.
        
        Args:
            filename (str): Week file name
            
        Returns:
            int: Week number
        """
        match = re.search(r'week_(\d+)', filename, re.IGNORECASE)
        return int(match.group(1)) if match else 1
    
    def atomic_execution(self, input_path: str, output_path: str) -> Dict[str, Any]:
        """
        Execute complete parsing operation with atomic behavior.
        
        Args:
            input_path (str): Path to course directory
            output_path (str): Path for output JSON file
            
        Returns:
            dict: Complete structured course data
        """
        try:
            # Pre-flight validation
            self.validate_execution_environment(input_path, output_path)
            
            input_dir = Path(input_path)
            
            # Parse all components
            course_info = self.parse_course_info(input_dir / 'course_info.md')
            syllabus = self.parse_syllabus(input_dir / 'syllabus.md')
            assessments = self.parse_assessments(input_dir / 'assessment_guide.md')
            
            # Parse all week files
            modules_dir = input_dir / 'modules'
            week_files = sorted(modules_dir.glob('week_*.md'))
            weeks_data = []
            
            for week_file in week_files:
                week_data = self.parse_week_content(week_file)
                weeks_data.append(week_data)
            
            # Assemble complete structure
            structured_data = {
                "course_info": course_info,
                "syllabus": syllabus,
                "weeks": weeks_data,
                "assessments": assessments,
                "metadata": {
                    "parser_version": "1.0.0",
                    "parsed_at": "2025-08-05",
                    "total_weeks": len(weeks_data),
                    "total_sub_modules": sum(len(week['sub_modules']) for week in weeks_data)
                }
            }
            
            # Final validation
            self.validate_structured_data(structured_data)
            
            # Write output atomically
            temp_output = Path(output_path).with_suffix('.tmp')
            with open(temp_output, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)
            
            temp_output.rename(output_path)
            
            self.logger.info(f"Successfully parsed course to: {output_path}")
            return structured_data
            
        except Exception as e:
            self.cleanup_temps()
            raise SystemExit(f"ATOMIC EXECUTION FAILED: {e}")
        
        finally:
            if self.execution_lock and self.execution_lock.exists():
                self.execution_lock.unlink()
    
    def validate_structured_data(self, data: Dict[str, Any]):
        """
        Validate the complete structured data meets requirements.
        
        Args:
            data (dict): Complete structured course data
            
        Raises:
            SystemExit: If validation fails
        """
        self.logger.info("Validating structured data")
        
        # Check required sections exist
        required_sections = ['course_info', 'syllabus', 'weeks', 'assessments']
        for section in required_sections:
            if section not in data:
                raise SystemExit(f"VALIDATION ERROR: Missing required section: {section}")
        
        # Validate each week has minimum sub-modules (dynamic count)
        min_modules = self.config.get('min_sub_modules', 1)
        max_modules = self.config.get('max_sub_modules', 20)
        for week in data['weeks']:
            module_count = len(week['sub_modules'])
            if module_count < min_modules:
                raise SystemExit(f"VALIDATION ERROR: Week {week['week_number']} has {module_count} sub-modules, minimum {min_modules}")
            if module_count > max_modules:
                self.logger.warning(f"Week {week['week_number']} has {module_count} sub-modules, exceeds recommended max {max_modules}")
        
        # Check for unresolved template variables
        data_str = json.dumps(data)
        unresolved_vars = re.findall(r'\{[^}]+\}', data_str)
        if unresolved_vars:
            raise SystemExit(f"VALIDATION ERROR: Unresolved template variables: {unresolved_vars}")
        
        self.logger.info("Structured data validation passed")
    
    def cleanup_temps(self):
        """Clean up all temporary files."""
        for temp_file in self.temp_files:
            try:
                Path(temp_file).unlink()
            except FileNotFoundError:
                pass
    
    def save_structured_content(self, data: Dict[str, Any], output_path: str):
        """
        Save structured content to JSON file.
        
        Args:
            data (dict): Structured course data
            output_path (str): Output file path
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved structured content to: {output_path}")

def main():
    """Command line interface for course content parser."""
    parser = argparse.ArgumentParser(description='Parse course content into structured JSON')
    parser.add_argument('-i', '--input', required=True, help='Input course directory path')
    parser.add_argument('-o', '--output', required=True, help='Output JSON file path')
    parser.add_argument('-c', '--config', help='Configuration file path (optional)')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                       help='Verbose output (-vv for debug)')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0.0')

    args = parser.parse_args()

    # Configure logging based on verbosity
    if args.verbose >= 2:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.verbose >= 1:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.WARNING)

    try:
        content_parser = CourseContentParser(args.config)
        result = content_parser.atomic_execution(args.input, args.output)

        print(f"Successfully parsed course with {result['metadata']['total_sub_modules']} sub-modules across {result['metadata']['total_weeks']} weeks")
        sys.exit(0)
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        sys.exit(2)
    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON: {e}")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()