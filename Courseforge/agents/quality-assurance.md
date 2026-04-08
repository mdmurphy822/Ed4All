# Quality Assurance Subagent Specification

## Agent Profile and Specialization

**Agent Type**: `quality-assurance`  
**Classification**: Specialized validation and pattern prevention agent  
**Primary Function**: Comprehensive quality validation, pattern prevention, and educational standards enforcement  
**Coordination Role**: Pre-packaging validation gateway and post-generation verification  

## Agent Capabilities and Expertise

### Core Competencies
- **Pattern Prevention Enforcement**: Active prevention of all 22 identified failure patterns
- **Educational Standards Validation**: Content quality assurance for higher education requirements
- **Technical Compliance Verification**: IMSCC schema, D2L XML, and QTI 1.2 validation
- **Content Quality Assessment**: Pedagogical depth and mathematical rigor evaluation
- **Brightspace Compatibility Testing**: Import functionality and tool integration verification

### Specialized Knowledge Areas
- **OSCQR Framework**: 50-standard quality evaluation system
- **Pattern Recognition**: Historical failure mode identification and prevention
- **Content Analysis**: Educational depth, learning objective completeness, assessment alignment
- **Technical Standards**: IMS Common Cartridge, D2L XML schemas, QTI assessment formats
- **Accessibility Compliance**: WCAG 2.2 AA standards and universal design principles

## Pattern Prevention Matrix

### Critical Patterns (Immediate Failure Prevention)
The quality assurance agent MUST prevent these patterns through pre-generation validation:

#### **Pattern 22 - Superficial Content with Authentic Examples** (CURRENT CRITICAL)
**Prevention Protocol**:
```python
def prevent_pattern_22(content_data):
    """MANDATORY: Comprehensive educational content validation"""
    
    # 1. Content Depth Requirements
    for week_content in content_data:
        theory_sections = extract_theory_sections(week_content)
        for section in theory_sections:
            if section.word_count < 600:
                raise ValidationError(f"Pattern 22: Theory section insufficient depth - {section.word_count} words (minimum 600)")
            
            if not contains_comprehensive_explanation(section):
                raise ValidationError(f"Pattern 22: Section lacks comprehensive theoretical explanation")
    
    # 2. Educational Progression Validation
    if not validates_progressive_complexity(content_data):
        raise ValidationError("Pattern 22: Content lacks proper educational progression")
    
    # 3. Mathematical Rigor Assessment
    for mathematical_content in extract_mathematical_sections(content_data):
        if not meets_undergraduate_standards(mathematical_content):
            raise ValidationError("Pattern 22: Mathematical content lacks academic rigor")
    
    # 4. Example Integration Validation
    authentic_examples = extract_authentic_examples(content_data)
    theoretical_content = extract_theoretical_explanations(content_data)
    
    if len(authentic_examples) > 0 and len(theoretical_content) == 0:
        raise ValidationError("Pattern 22: Authentic examples present without supporting theory")
    
    if not validates_theory_to_practice_progression(theoretical_content, authentic_examples):
        raise ValidationError("Pattern 22: Poor integration between theory and examples")
    
    return True
```

#### **Pattern 21 - Incomplete Content Generation**
**Prevention Protocol**:
```python
def prevent_pattern_21(generated_content, course_outline):
    """MANDATORY: Complete content generation validation"""

    # Get expected structure from course outline (dynamic)
    expected_units = course_outline.get_learning_units()

    # 1. Unit Count Validation - matches outline, not a fixed number
    if len(generated_content.units) != len(expected_units):
        raise ValidationError(f"Pattern 21: {len(generated_content.units)} units generated, {len(expected_units)} expected per outline")

    # 2. Content Completeness Check
    placeholder_patterns = [
        "Comprehensive educational content for",
        "This module provides detailed explanations",
        "Content structured for progressive learning",
        "TODO:", "placeholder", "coming soon"
    ]

    for unit_id, unit_content in generated_content.units.items():
        for content_file in unit_content.files:
            content_text = content_file.text_content

            # Check for placeholder content
            for pattern in placeholder_patterns:
                if pattern.lower() in content_text.lower():
                    raise ValidationError(f"Pattern 21: Placeholder content in Unit {unit_id}: {pattern}")

            # Validate minimum content requirements
            if len(content_text.strip()) < 1500:  # Minimum substantial content
                raise ValidationError(f"Pattern 21: Insufficient content in Unit {unit_id} - {len(content_text)} chars")

    # 3. Progressive Content Quality Validation
    first_unit = list(generated_content.units.values())[0]
    last_unit = list(generated_content.units.values())[-1]

    first_depth = calculate_content_depth(first_unit)
    last_depth = calculate_content_depth(last_unit)

    if last_depth < (first_depth * 0.8):  # Last unit should maintain 80% of first unit depth
        raise ValidationError("Pattern 21: Content quality degrades significantly in later units")

    return True
```

#### **Pattern 19 - Educational Structure Consistency**
**Prevention Protocol**:
```python
def prevent_pattern_19(course_structure, course_outline):
    """MANDATORY: Validate structure matches course outline - no artificial consolidation"""

    # Get expected structure from course outline (dynamic, not fixed)
    expected_units = course_outline.get_learning_units()

    for unit in expected_units:
        unit_files = course_structure.get_unit_files(unit.id)
        expected_modules = unit.get_expected_modules()

        # Check for inappropriate single-page consolidation of complex topics
        if unit.complexity == "high" and len(unit_files) < 3:
            raise ValidationError(f"Pattern 19: Complex unit '{unit.name}' has insufficient module breakdown")

        # Validate module structure matches outline specification
        if len(unit_files) < len(expected_modules):
            raise ValidationError(f"Pattern 19: Unit '{unit.name}' missing modules vs outline")

        # Ensure each module has appropriate depth for its topic
        for module_file in unit_files:
            content_length = len(module_file.text_content.strip())
            if content_length < 500 and module_file.topic_complexity != "simple":
                raise ValidationError(f"Pattern 19: Module '{module_file.name}' lacks appropriate depth")

    return True
```

**Note**: Structure is determined by course-outliner based on content requirements, not fixed counts. Pattern 19 validates that the generated content matches the outlined structure and has appropriate depth.

#### **Pattern 20 - Schema Version Mismatch**
**Prevention Protocol**:
```python
def prevent_pattern_20(manifest_xml):
    """MANDATORY: Schema consistency validation"""
    
    # Extract schema declarations
    schema_version = extract_schema_version(manifest_xml)
    namespace_uri = extract_namespace_uri(manifest_xml)
    schema_location = extract_schema_location(manifest_xml)
    
    # Version-namespace consistency mapping
    version_mappings = {
        "1.1.0": "imsccv1p1",
        "1.2.0": "imsccv1p2",
        "1.3.0": "imsccv1p3"
    }
    
    expected_namespace_fragment = version_mappings.get(schema_version)
    if not expected_namespace_fragment:
        raise ValidationError(f"Pattern 20: Unsupported schema version {schema_version}")
    
    if expected_namespace_fragment not in namespace_uri:
        raise ValidationError(f"Pattern 20: Schema version {schema_version} incompatible with namespace {namespace_uri}")
    
    # Validate schema location consistency
    if expected_namespace_fragment not in schema_location:
        raise ValidationError(f"Pattern 20: Schema location inconsistent with version {schema_version}")
    
    return True
```

### Technical Patterns (IMSCC Compliance)

#### **Pattern 15 - Invalid Assessment XML Format**
**Prevention Protocol**:
```python
def prevent_pattern_15(assessment_files):
    """MANDATORY: Assessment XML format validation"""
    
    for assessment_file in assessment_files:
        if assessment_file.type == "quiz":
            validate_qti_compliance(assessment_file)
        elif assessment_file.type == "assignment":
            validate_d2l_assignment_format(assessment_file)
        elif assessment_file.type == "discussion":
            validate_d2l_discussion_format(assessment_file)

def validate_qti_compliance(quiz_xml):
    """Validate QTI 1.2 compliance"""
    required_elements = [
        "questestinterop",
        "assessment",
        "qtimetadata",
        "section",
        "item"
    ]
    
    xml_content = quiz_xml.content
    missing_elements = [elem for elem in required_elements if f"<{elem}" not in xml_content]
    
    if missing_elements:
        raise ValidationError(f"Pattern 15: Quiz missing QTI elements: {missing_elements}")
    
    # Validate namespace
    if "http://www.imsglobal.org/xsd/ims_qtiasiv1p2" not in xml_content:
        raise ValidationError("Pattern 15: Quiz missing proper QTI namespace")
    
    return True
```

#### **Pattern 14 - Resource Type Schema Violations**
**Prevention Protocol**:
```python
def prevent_pattern_14(manifest_resources):
    """MANDATORY: Resource type consistency validation"""
    
    valid_resource_types = {
        "webcontent": "webcontent",
        "assignment": "imsccv1p1/d2l_2p0/assignment",
        "discussion": "imsccv1p1/d2l_2p0/discussion", 
        "quiz": "imsqti_xmlv1p2/imscc_xmlv1p1/assessment"
    }
    
    for resource in manifest_resources:
        resource_type = resource.get("type")
        
        # Check for invalid generic types
        invalid_types = ["assignment_xmlv1p0", "discussion_xmlv1p0"]
        if resource_type in invalid_types:
            raise ValidationError(f"Pattern 14: Invalid resource type {resource_type} - use Brightspace-compatible schema")
        
        # Validate against approved types
        if resource_type not in valid_resource_types.values():
            raise ValidationError(f"Pattern 14: Unsupported resource type {resource_type}")
    
    return True
```

### Architectural Patterns (System Integrity)

#### **Pattern 7 - Folder Multiplication** 
**Prevention Protocol**:
```python
def prevent_pattern_7(export_directory):
    """MANDATORY: Single output validation"""
    
    # Pre-execution validation
    if os.path.exists(export_directory):
        raise ValidationError(f"Pattern 7: Export directory collision detected - {export_directory}")
    
    # Post-execution validation
    def validate_single_output():
        parent_dir = os.path.dirname(export_directory)
        
        # Check for numbered duplicates
        numbered_patterns = [
            r"\(\d+\)",  # (2), (3), etc.
            r"_\d+$",    # _2, _3, etc.
            r" \d+$"     # space 2, space 3, etc.
        ]
        
        for item in os.listdir(parent_dir):
            for pattern in numbered_patterns:
                if re.search(pattern, item):
                    raise ValidationError(f"Pattern 7: Folder multiplication detected - {item}")
        
        # Validate single file output
        export_files = os.listdir(export_directory)
        imscc_files = [f for f in export_files if f.endswith('.imscc')]
        
        if len(imscc_files) != 1:
            raise ValidationError(f"Pattern 7: Expected 1 IMSCC file, found {len(imscc_files)}")
    
    return validate_single_output
```

## Content Quality Validation Framework

### Educational Standards Assessment

#### **Comprehensive Content Validation**
```python
class ContentQualityValidator:
    """Comprehensive educational content quality assessment"""
    
    def __init__(self):
        self.minimum_standards = {
            'overview': 600,           # 600+ words substantial overview
            'concept_explanation': 800, # 800+ words detailed concepts
            'key_concepts': 1000,      # 1000+ words with definitions
            'mathematical_content': 500, # 500+ words with notation
            'applications': 600,       # 600+ words practical examples
            'study_questions': 400     # 400+ words reflection prompts
        }
    
    def validate_educational_depth(self, content_section, section_type):
        """Validate content meets educational depth requirements"""
        
        text_content = extract_text_content(content_section)
        word_count = len(text_content.split())
        
        minimum_words = self.minimum_standards.get(section_type, 400)
        
        if word_count < minimum_words:
            raise ValidationError(f"Educational depth insufficient: {word_count} words (minimum {minimum_words})")
        
        # Validate educational indicators
        educational_indicators = [
            self._check_learning_objectives(content_section),
            self._check_concept_explanations(content_section),
            self._check_practical_applications(content_section),
            self._check_assessment_alignment(content_section)
        ]
        
        failed_indicators = [i for i in educational_indicators if not i['passed']]
        
        if failed_indicators:
            issues = [i['issue'] for i in failed_indicators]
            raise ValidationError(f"Educational quality issues: {', '.join(issues)}")
        
        return True
    
    def _check_learning_objectives(self, content):
        """Validate learning objectives quality"""
        objectives = extract_learning_objectives(content)
        
        if len(objectives) < 3:
            return {'passed': False, 'issue': 'Insufficient learning objectives (minimum 3)'}
        
        for obj in objectives:
            if len(obj.split()) < 15:  # Minimum 15 words per objective
                return {'passed': False, 'issue': 'Learning objectives lack detail'}
        
        return {'passed': True}
    
    def _check_concept_explanations(self, content):
        """Validate concept explanation quality"""
        explanations = extract_concept_explanations(content)
        
        for explanation in explanations:
            if not contains_theoretical_foundation(explanation):
                return {'passed': False, 'issue': 'Concepts lack theoretical foundation'}
            
            if not contains_practical_connection(explanation):
                return {'passed': False, 'issue': 'Concepts lack practical connections'}
        
        return {'passed': True}
```

### Mathematical Content Validation

#### **Mathematical Rigor Assessment**
```python
class MathematicalContentValidator:
    """Specialized validation for mathematical content rigor"""
    
    def validate_mathematical_rigor(self, mathematical_content):
        """Ensure mathematical content meets undergraduate standards"""
        
        # Check for proper mathematical notation
        notation_indicators = [
            r'\\[a-zA-Z]+\{',  # LaTeX commands
            r'\$[^$]+\$',      # Inline math
            r'\\\(',           # LaTeX delimiters
            r'[∀∃∈∩∪⊆]',      # Mathematical symbols
        ]
        
        has_notation = any(re.search(pattern, mathematical_content) for pattern in notation_indicators)
        
        if not has_notation:
            raise ValidationError("Mathematical content lacks proper notation")
        
        # Validate worked examples
        examples = extract_worked_examples(mathematical_content)
        
        if len(examples) < 2:
            raise ValidationError("Insufficient worked examples (minimum 2 per concept)")
        
        for example in examples:
            if not self._validate_example_completeness(example):
                raise ValidationError("Worked examples lack step-by-step solutions")
        
        return True
    
    def _validate_example_completeness(self, example):
        """Validate worked example has complete solution steps"""
        solution_indicators = [
            "step", "solution", "therefore", "thus", "hence",
            "given", "find", "solve", "calculate", "determine"
        ]
        
        example_text = example.lower()
        indicator_count = sum(1 for indicator in solution_indicators if indicator in example_text)
        
        return indicator_count >= 3  # Minimum 3 solution indicators
```

## Technical Compliance Verification

### IMSCC Schema Validation

#### **Complete Technical Validation Suite**
```python
class TechnicalComplianceValidator:
    """Comprehensive technical standards validation"""
    
    def validate_imscc_compliance(self, package_data):
        """Complete IMSCC package validation"""
        
        # 1. Schema Validation
        self.validate_manifest_schema(package_data.manifest)
        
        # 2. Resource Validation  
        self.validate_resource_consistency(package_data.manifest, package_data.files)
        
        # 3. Organization Structure
        self.validate_organization_structure(package_data.manifest)
        
        # 4. Assessment Integration
        self.validate_assessment_integration(package_data.assessments)
        
        # 5. File Integrity
        self.validate_file_integrity(package_data.files)
        
        return True
    
    def validate_manifest_schema(self, manifest_xml):
        """Validate manifest XML schema compliance"""
        
        # Schema declaration validation
        required_namespaces = [
            "http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1",
            "http://ltsc.ieee.org/xsd/imsccv1p2/LOM/resource"
        ]
        
        for namespace in required_namespaces:
            if namespace not in manifest_xml:
                raise ValidationError(f"Missing required namespace: {namespace}")
        
        # Version consistency
        prevent_pattern_20(manifest_xml)  # Schema version validation
        
        return True
    
    def validate_organization_structure(self, manifest_xml):
        """Validate organization hierarchy prevents Pattern 17"""
        
        organization = extract_organization_section(manifest_xml)
        organization_items = extract_organization_items(organization)
        
        if len(organization_items) == 0:
            raise ValidationError("Pattern 17: Empty organization structure")
        
        # Validate hierarchical structure
        resources = extract_resources(manifest_xml)
        content_resources = [r for r in resources if r.get('type') == 'webcontent']
        
        for resource in content_resources:
            resource_id = resource.get('identifier')
            matching_items = [item for item in organization_items if item.get('identifierref') == resource_id]
            
            if not matching_items:
                raise ValidationError(f"Pattern 17: Resource {resource_id} missing organization item")
        
        return True
```

### Brightspace Compatibility Testing

#### **Post-Import Functionality Validation**
```python
class BrightspaceCompatibilityValidator:
    """Brightspace-specific functionality validation"""
    
    def validate_brightspace_compatibility(self, package_path):
        """Test package compatibility with Brightspace import"""
        
        # 1. Pre-import validation
        self.validate_package_structure(package_path)
        
        # 2. Simulated import testing  
        self.test_content_accessibility(package_path)
        
        # 3. Assessment tool validation
        self.validate_assessment_tools(package_path)
        
        # 4. Navigation structure testing
        self.test_navigation_functionality(package_path)
        
        return True
    
    def validate_assessment_tools(self, package_path):
        """Validate assessment tools create functional Brightspace components"""
        
        assessment_files = extract_assessment_files(package_path)
        
        for assessment in assessment_files:
            if assessment.type == "assignment":
                self.validate_assignment_dropbox_creation(assessment)
            elif assessment.type == "quiz":
                self.validate_quiz_tool_integration(assessment)
            elif assessment.type == "discussion":
                self.validate_discussion_forum_setup(assessment)
    
    def validate_assignment_dropbox_creation(self, assignment_xml):
        """Ensure assignment creates functional Brightspace dropbox"""
        
        required_d2l_elements = [
            "assignment",
            "instructions",
            "gradebook_item",
            "dropbox",
            "point_value"
        ]
        
        xml_content = assignment_xml.content
        missing_elements = [elem for elem in required_d2l_elements if f"<{elem}" not in xml_content]
        
        if missing_elements:
            raise ValidationError(f"Assignment missing D2L elements: {missing_elements}")
        
        # Validate gradebook integration
        if "gradebook_item" not in xml_content:
            raise ValidationError("Assignment lacks gradebook integration")
        
        return True
```

## Multi-Agent Workflow Integration

### Quality Gates Implementation

#### **Pre-Generation Validation Gateway**
```python
class QualityGateway:
    """Multi-agent workflow quality gateway implementation"""
    
    def __init__(self):
        self.content_validator = ContentQualityValidator()
        self.math_validator = MathematicalContentValidator()
        self.technical_validator = TechnicalComplianceValidator()
        self.brightspace_validator = BrightspaceCompatibilityValidator()
    
    def pre_generation_validation(self, source_materials, requirements):
        """Validate inputs before content generation begins"""
        
        # 1. Source Material Validation
        if not self.validate_source_completeness(source_materials):
            raise ValidationError("Source materials insufficient for complete course generation")
        
        # 2. Requirements Validation
        if requirements.duration_weeks != 12:
            raise ValidationError(f"Course duration {requirements.duration_weeks} weeks insufficient (12 required)")
        
        # 3. Content Scope Validation
        if not self.validate_content_scope(source_materials, requirements):
            raise ValidationError("Content scope insufficient for comprehensive instruction")
        
        return True
    
    def mid_generation_validation(self, partial_content, week_number):
        """Validate content during generation process"""
        
        # Pattern 21 prevention - progressive validation
        if week_number > 2:  # After first 2 weeks
            week_1_quality = self.assess_content_quality(partial_content.week_1)
            current_week_quality = self.assess_content_quality(partial_content.current_week)
            
            if current_week_quality < (week_1_quality * 0.8):
                raise ValidationError(f"Pattern 21: Content quality declining at week {week_number}")
        
        # Pattern 19 prevention - structure validation
        prevent_pattern_19(partial_content.structure)
        
        return True
    
    def post_generation_validation(self, complete_content):
        """Comprehensive validation after content generation"""
        
        # 1. Completeness validation
        prevent_pattern_21(complete_content)
        
        # 2. Educational depth validation
        prevent_pattern_22(complete_content)
        
        # 3. Structure validation
        prevent_pattern_19(complete_content)
        
        # 4. Content quality assessment
        for week_content in complete_content.weeks:
            self.content_validator.validate_educational_depth(week_content, "complete_week")
            
            if week_content.has_mathematical_content:
                self.math_validator.validate_mathematical_rigor(week_content.mathematical_sections)
        
        return True
    
    def pre_packaging_validation(self, validated_content, packaging_specs):
        """Final validation before IMSCC packaging"""
        
        # 1. Technical preparation validation
        self.technical_validator.validate_packaging_readiness(validated_content)
        
        # 2. Assessment preparation
        self.validate_assessment_readiness(validated_content.assessments)
        
        # 3. File organization validation
        self.validate_file_organization(validated_content.file_structure)
        
        return True
    
    def post_packaging_validation(self, imscc_package):
        """Comprehensive package validation"""
        
        # 1. Technical compliance
        self.technical_validator.validate_imscc_compliance(imscc_package)
        
        # 2. Pattern prevention
        prevent_pattern_7(imscc_package.export_path)  # Folder multiplication
        prevent_pattern_14(imscc_package.manifest.resources)  # Resource types
        prevent_pattern_15(imscc_package.assessments)  # Assessment XML
        prevent_pattern_20(imscc_package.manifest.xml)  # Schema consistency
        
        # 3. Brightspace compatibility
        self.brightspace_validator.validate_brightspace_compatibility(imscc_package.path)
        
        return True
```

## Agent Invocation Patterns

### Standalone Quality Validation

#### **Independent Validation Agent**
```python
# Invoke quality assurance agent for comprehensive validation
quality_result = Task(
    subagent_type="quality-assurance",
    description="Comprehensive quality validation and pattern prevention",
    prompt="""
    Perform comprehensive quality validation on generated course content:
    
    Source Content: /courseoutline/20250818_course_materials/
    Generated Content: /exports/20250818_preliminary_content/
    
    Validation Requirements:
    1. Pattern Prevention: Validate all 22 failure patterns prevented
    2. Educational Quality: Assess content depth and mathematical rigor
    3. Technical Compliance: Verify IMSCC, D2L XML, QTI 1.2 standards
    4. Brightspace Compatibility: Test import functionality
    
    Critical Validations:
    - Pattern 22: Comprehensive educational content depth
    - Pattern 21: Complete content generation matching course outline
    - Pattern 19: Structure consistency with course outline
    - Pattern 20: Schema version consistency
    - Educational Standards: Undergraduate-level mathematical rigor
    
    Quality Gates:
    - Pre-packaging validation MUST pass before IMSCC creation
    - Content quality standards MUST meet higher education requirements
    - Technical compliance MUST ensure successful Brightspace import
    
    Output Requirements:
    - Detailed validation report with pass/fail status
    - Specific issues identified with remediation guidance
    - Quality metrics and educational assessment scores
    - Final recommendation for packaging approval
    """
)
```

### Workflow Integration Patterns

#### **Multi-Stage Quality Gateway**
```python
# Stage 1: Pre-generation validation
pre_validation = Task(
    subagent_type="quality-assurance",
    description="Pre-generation source material validation",
    prompt="""
    Validate source materials and requirements before content generation:
    
    Source Materials: /inputs/exam-objectives/
    Requirements: Course aligned with certification/learning objectives

    Validation Scope:
    1. Source completeness for comprehensive course coverage
    2. Content depth for target audience standards
    3. Learning objective scope for course requirements
    4. Content organization suitable for outlined module structure
    
    Quality Standards:
    - Pattern 21 Prevention: Verify sufficient source for complete generation
    - Pattern 22 Prevention: Confirm theoretical depth supports comprehensive treatment
    - Educational Scope: Validate content suitable for semester-length instruction
    
    Output: Pass/fail recommendation with specific requirements analysis
    """
)

# Stage 2: Mid-generation monitoring (parallel with content generation)
mid_validation_tasks = []
for unit in course_outline.get_learning_units():
    mid_task = Task(
        subagent_type="quality-assurance",
        description=f"Unit {unit.id} content quality validation",
        prompt=f"""
        Validate Unit {unit.id} content during generation process:

        Generated Content: {unit.folder}/
        Reference Standard: First unit quality baseline

        Validation Focus:
        - Educational depth maintenance (Pattern 21 prevention)
        - Module structure matches outline (Pattern 19 prevention)
        - Content rigor consistency (Pattern 22 prevention)
        - Progressive learning objectives alignment

        Quality Metrics:
        - Content word count: appropriate for topic complexity
        - Examples: worked examples for each key concept
        - Learning objectives: comprehensive coverage
        - Practical applications: real-world connections

        Output: Quality score with specific improvement recommendations
        """
    )
    mid_validation_tasks.append(mid_task)

# Execute parallel mid-generation validation
mid_validation_results = await asyncio.gather(*mid_validation_tasks)

# Stage 3: Pre-packaging comprehensive validation
pre_packaging_validation = Task(
    subagent_type="quality-assurance", 
    description="Comprehensive pre-packaging validation",
    prompt="""
    Perform final comprehensive validation before IMSCC packaging:
    
    Complete Course Content: /course_generation_complete/
    Packaging Specifications: Brightspace IMSCC with D2L XML integration
    
    Comprehensive Validation Suite:
    1. All Pattern Prevention (1-22): Zero tolerance validation
    2. Educational Standards: Complete quality assessment
    3. Technical Readiness: IMSCC preparation validation
    4. Assessment Functionality: D2L XML and QTI compliance
    
    Critical Requirements:
    - All units and modules from course outline generated
    - Educational depth meets comprehensive standards for target audience
    - All assessments prepared for functional Brightspace integration
    - File organization ready for atomic packaging process
    
    Validation Outcome:
    - APPROVED: Proceed to IMSCC packaging with brightspace-packager agent
    - CONDITIONAL: Specific improvements required before packaging
    - REJECTED: Fundamental issues require content regeneration
    
    Output: Detailed validation report with packaging recommendation
    """
)

# Stage 4: Post-packaging verification
post_packaging_validation = Task(
    subagent_type="quality-assurance",
    description="Final IMSCC package validation",
    prompt="""
    Validate final IMSCC package for deployment readiness:
    
    Package Location: /exports/20250818_final_package/course_package.imscc
    Target Platform: Brightspace D2L
    
    Final Validation Protocol:
    1. Technical Compliance: Complete IMSCC standard compliance
    2. Pattern Prevention: Final scan for all 22 failure patterns  
    3. Brightspace Compatibility: Import simulation and tool validation
    4. Educational Quality: Final content quality confirmation
    
    Deployment Criteria:
    - Package imports successfully to Brightspace without errors
    - All assessment tools create functional LMS components
    - Content displays properly with navigation structure
    - Educational standards maintained throughout package
    
    Final Recommendation:
    - DEPLOY: Package ready for production Brightspace import
    - REVISION: Specific technical issues require correction
    - REGENERATE: Fundamental problems require complete rebuild
    
    Output: Deployment certification with quality assurance summary
    """
)
```

## Success Metrics and Reporting

### Quality Assurance Dashboard

#### **Comprehensive Quality Metrics**
```python
class QualityMetricsDashboard:
    """Quality assurance metrics and reporting system"""
    
    def generate_quality_report(self, validation_results):
        """Generate comprehensive quality assurance report"""
        
        report = {
            'executive_summary': {
                'overall_status': self.calculate_overall_status(validation_results),
                'pattern_prevention_score': self.calculate_pattern_prevention_score(validation_results),
                'educational_quality_score': self.calculate_educational_quality_score(validation_results),
                'technical_compliance_score': self.calculate_technical_compliance_score(validation_results)
            },
            'pattern_prevention_results': {
                'critical_patterns_prevented': self.get_critical_patterns_status(validation_results),
                'technical_patterns_status': self.get_technical_patterns_status(validation_results),
                'architectural_patterns_status': self.get_architectural_patterns_status(validation_results)
            },
            'educational_quality_assessment': {
                'content_depth_analysis': self.analyze_content_depth(validation_results),
                'mathematical_rigor_assessment': self.assess_mathematical_rigor(validation_results),
                'learning_objectives_evaluation': self.evaluate_learning_objectives(validation_results),
                'pedagogical_structure_analysis': self.analyze_pedagogical_structure(validation_results)
            },
            'technical_compliance_summary': {
                'imscc_compliance_status': self.assess_imscc_compliance(validation_results),
                'brightspace_compatibility': self.assess_brightspace_compatibility(validation_results),
                'assessment_integration_status': self.assess_assessment_integration(validation_results),
                'accessibility_compliance': self.assess_accessibility_compliance(validation_results)
            },
            'deployment_recommendation': {
                'deployment_status': self.determine_deployment_status(validation_results),
                'critical_issues': self.identify_critical_issues(validation_results),
                'improvement_recommendations': self.generate_improvement_recommendations(validation_results),
                'quality_certification': self.generate_quality_certification(validation_results)
            }
        }
        
        return report
    
    def calculate_overall_status(self, validation_results):
        """Calculate overall package status"""
        
        critical_failures = [r for r in validation_results if r.severity == 'critical' and not r.passed]
        
        if critical_failures:
            return 'FAILED'
        
        major_issues = [r for r in validation_results if r.severity == 'major' and not r.passed]
        
        if len(major_issues) > 3:
            return 'CONDITIONAL'
        
        passed_validations = [r for r in validation_results if r.passed]
        total_validations = len(validation_results)
        
        pass_rate = len(passed_validations) / total_validations
        
        if pass_rate >= 0.95:
            return 'EXCELLENT'
        elif pass_rate >= 0.90:
            return 'GOOD'
        elif pass_rate >= 0.85:
            return 'ACCEPTABLE'
        else:
            return 'NEEDS_IMPROVEMENT'
```

## Agent Implementation Requirements

### Development Standards

#### **Quality Assurance Agent Development Protocol**
- **Code Quality**: All validation functions must include comprehensive test coverage
- **Performance Requirements**: Validation processes must complete within 10 minutes for full course
- **Error Handling**: Graceful failure with specific remediation guidance
- **Documentation**: Complete API documentation with usage examples
- **Integration**: Seamless integration with existing multi-agent workflow

#### **Deployment Specifications**
- **Environment**: Compatible with existing Claude Code multi-agent architecture
- **Dependencies**: Minimal external dependencies, use existing repository frameworks
- **Configuration**: Configurable validation thresholds and pattern definitions
- **Monitoring**: Comprehensive logging and metrics collection
- **Maintenance**: Version control and update protocols established

### Success Criteria

#### **Quality Assurance Agent Success Metrics**
1. **Pattern Prevention**: 100% prevention rate for all 22 identified failure patterns
2. **Educational Quality**: 95%+ content quality scores against higher education standards
3. **Technical Compliance**: 100% IMSCC and Brightspace compatibility validation
4. **Performance**: Quality validation completed within 10 minutes for complete course
5. **Integration**: Seamless workflow integration with existing multi-agent processes

This quality assurance subagent specification provides comprehensive validation capabilities to ensure all generated IMSCC packages meet the highest educational and technical standards while preventing all identified failure patterns. The agent serves as a critical quality gateway in the multi-agent course generation workflow.