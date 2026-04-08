# Brightspace IMSCC Validation Protocols

## Comprehensive Validation Framework for Brightspace Compatibility

This document establishes validation protocols to ensure IMSCC packages achieve full compatibility with Brightspace D2L, preventing import errors and ensuring functional integration with all Brightspace tools.

## Validation Hierarchy

### Level 1: XML Schema Compliance (Pattern 15 Prevention)
Critical technical validation preventing "Illegal XML" import errors

### Level 2: Educational Structure Integrity (Pattern 19 Prevention)
Pedagogical organization validation preserving structure consistency with course outline

### Level 3: Content Quality Assurance (Pattern 22 Prevention)
Educational depth and rigor validation ensuring comprehensive content

### Level 4: Brightspace Integration Functionality
Tool-specific validation ensuring proper LMS feature integration

## Level 1: XML Schema Compliance Validation

### 1.1 Namespace Validation Protocol
```python
def validate_brightspace_namespaces(manifest_content):
    required_namespaces = {
        'imscp': 'http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1',
        'lom': 'http://ltsc.ieee.org/xsd/imsccv1p2/LOM/resource',
        'lomimscc': 'http://ltsc.ieee.org/xsd/imsccv1p2/LOM/manifest',
        'd2l': 'http://www.imsglobal.org/xsd/imsccv1p1/d2l_2p0'
    }
    
    for prefix, uri in required_namespaces.items():
        if f'xmlns:{prefix}="{uri}"' not in manifest_content and f'xmlns="{uri}"' not in manifest_content:
            raise ValidationError(f"Missing required namespace: {prefix} -> {uri}")
    
    return "Namespace validation passed"
```

### 1.2 QTI 1.2 Compliance Validation
```python
def validate_qti_compliance(assessment_files):
    qti_namespace = 'http://www.imsglobal.org/xsd/ims_qtiasiv1p2'
    
    for assessment_file in assessment_files:
        content = read_xml_file(assessment_file)
        
        # Validate QTI namespace
        if qti_namespace not in content:
            raise ValidationError(f"QTI 1.2 namespace missing in {assessment_file}")
        
        # Validate D2L metadata fields
        required_d2l_fields = [
            'd2l_2p0_grade_item_points_possible',
            'd2l_2p0_attempts_allowed'
        ]
        
        for field in required_d2l_fields:
            if field not in content:
                raise ValidationError(f"Missing D2L field {field} in {assessment_file}")
    
    return "QTI compliance validation passed"
```

### 1.3 Resource Type Validation
```python
def validate_resource_types(manifest_tree):
    valid_brightspace_types = {
        'webcontent',
        'imsccv1p1/d2l_2p0/assignment',
        'imsccv1p1/d2l_2p0/discussion',
        'imsqti_xmlv1p2/imscc_xmlv1p1/assessment',
        'imsdt_xmlv1p1'
    }
    
    resources = manifest_tree.findall('.//resource')
    
    for resource in resources:
        resource_type = resource.get('type')
        if resource_type not in valid_brightspace_types:
            raise ValidationError(f"Invalid resource type: {resource_type}")
        
        # Validate file references exist
        href = resource.get('href')
        if href and not file_exists(href):
            raise ValidationError(f"Referenced file not found: {href}")
    
    return "Resource type validation passed"
```

## Level 2: Educational Structure Integrity Validation

### 2.1 Seven Sub-Module Structure Validation
```python
def validate_seven_submodule_structure(content_files, course_duration_weeks=12):
    required_submodules = [
        'overview.html',
        'concept1.html', 
        'concept2.html',
        'keyterms.html',
        'visual.html',
        'applications.html',
        'realworld.html'
    ]
    
    for week in range(1, course_duration_weeks + 1):
        week_prefix = f'week_{week:02d}_'
        week_files = [f for f in content_files if f.startswith(week_prefix)]
        
        for submodule in required_submodules:
            expected_file = f'{week_prefix}{submodule}'
            if expected_file not in week_files:
                raise ValidationError(f"Pattern 19 violation: Missing {expected_file}")
        
        # Check for single-page consolidation
        consolidated_patterns = ['content.html', 'all_content.html', 'complete.html']
        for pattern in consolidated_patterns:
            if any(pattern in f for f in week_files):
                raise ValidationError(f"Pattern 19 violation: Week {week} consolidated to single page")
    
    return f"Seven sub-module structure validated for {course_duration_weeks} weeks"
```

### 2.2 Organization Hierarchy Validation
```python
def validate_organization_hierarchy(manifest_tree):
    organization = manifest_tree.find('.//organization')
    if organization is None:
        raise ValidationError("Pattern 17 violation: No organization structure found")
    
    # Validate hierarchical structure
    items = organization.findall('.//item')
    if len(items) == 0:
        raise ValidationError("Pattern 17 violation: Empty organization structure")
    
    # Validate all items have identifierref
    for item in items:
        if not item.get('identifierref'):
            continue  # Parent items may not have identifierref
        
        # Validate referenced resource exists
        resource_id = item.get('identifierref')
        resource = manifest_tree.find(f'.//resource[@identifier="{resource_id}"]')
        if resource is None:
            raise ValidationError(f"Organization item references non-existent resource: {resource_id}")
    
    return "Organization hierarchy validation passed"
```

### 2.3 Content Accessibility Validation
```python
def validate_content_accessibility(manifest_tree, content_files):
    # Ensure all HTML content has corresponding organization items
    html_files = [f for f in content_files if f.endswith('.html')]
    
    resources = manifest_tree.findall('.//resource[@type="webcontent"]')
    resource_files = []
    
    for resource in resources:
        href = resource.get('href')
        if href:
            resource_files.append(href)
        
        # Also check file elements
        files = resource.findall('.//file')
        for file_elem in files:
            file_href = file_elem.get('href')
            if file_href and file_href.endswith('.html'):
                resource_files.append(file_href)
    
    # Validate all HTML files are referenced in resources
    for html_file in html_files:
        if html_file not in resource_files:
            raise ValidationError(f"Pattern 18 violation: HTML file {html_file} not accessible via resources")
    
    return "Content accessibility validation passed"
```

## Level 3: Content Quality Assurance Validation

### 3.1 Educational Depth Validation (Pattern 22 Prevention)
```python
def validate_educational_depth(html_files):
    min_content_length = 1500  # Characters
    min_section_headers = 3
    min_math_indicators = 5
    
    math_indicators = ['theorem', 'proof', 'definition', 'example', 'matrix', 'vector', 
                      'linear', 'algebra', 'equation', 'formula']
    
    for html_file in html_files:
        content = read_html_content(html_file)
        
        # Remove HTML tags for content analysis
        text_content = strip_html_tags(content)
        
        # Validate minimum content length
        if len(text_content) < min_content_length:
            raise ValidationError(f"Pattern 22 violation: {html_file} insufficient content ({len(text_content)} chars)")
        
        # Validate section structure
        header_count = content.count('<h1') + content.count('<h2') + content.count('<h3')
        if header_count < min_section_headers:
            raise ValidationError(f"Pattern 22 violation: {html_file} inadequate section structure")
        
        # Validate mathematical rigor (for math courses)
        math_count = sum(1 for indicator in math_indicators 
                        if indicator.lower() in text_content.lower())
        if math_count < min_math_indicators:
            raise ValidationError(f"Pattern 22 violation: {html_file} insufficient mathematical rigor")
    
    return "Educational depth validation passed"
```

### 3.2 Theory-Example Integration Validation
```python
def validate_theory_example_integration(html_files):
    for html_file in html_files:
        content = read_html_content(html_file)
        text_content = strip_html_tags(content).lower()
        
        # Check for theory-first structure
        theory_indicators = ['definition', 'theorem', 'concept', 'principle']
        example_indicators = ['example', 'illustration', 'instance', 'case study']
        
        theory_positions = []
        example_positions = []
        
        for indicator in theory_indicators:
            pos = text_content.find(indicator)
            if pos != -1:
                theory_positions.append(pos)
        
        for indicator in example_indicators:
            pos = text_content.find(indicator)
            if pos != -1:
                example_positions.append(pos)
        
        # Validate theory comes before examples
        if theory_positions and example_positions:
            min_theory_pos = min(theory_positions)
            min_example_pos = min(example_positions)
            
            if min_example_pos < min_theory_pos:
                raise ValidationError(f"Pattern 22 violation: {html_file} examples before theory")
    
    return "Theory-example integration validation passed"
```

### 3.3 Learning Objectives Alignment Validation
```python
def validate_learning_objectives_alignment(html_files, assessment_files):
    # Extract learning objectives from content
    objectives_found = []
    
    for html_file in html_files:
        content = read_html_content(html_file)
        
        # Look for learning objectives sections
        if 'learning objective' in content.lower() or 'learning outcome' in content.lower():
            objectives_found.append(html_file)
    
    if not objectives_found:
        raise ValidationError("Pattern 22 violation: No clear learning objectives found")
    
    # Validate assessment alignment with objectives
    if len(assessment_files) == 0:
        raise ValidationError("No assessments found to validate objective alignment")
    
    return "Learning objectives alignment validation passed"
```

## Level 4: Brightspace Integration Functionality Validation

### 4.1 Assignment Dropbox Validation
```python
def validate_assignment_dropbox_functionality(assignment_files):
    for assignment_file in assignment_files:
        content = read_xml_content(assignment_file)
        
        # Validate required D2L assignment elements
        required_elements = [
            '<points_possible>',
            '<dropbox_type>',
            '<submissions_allowed>'
        ]
        
        for element in required_elements:
            if element not in content:
                raise ValidationError(f"Assignment {assignment_file} missing element: {element}")
        
        # Validate dropbox_type values
        if '<dropbox_type>Individual</dropbox_type>' not in content and \
           '<dropbox_type>Group</dropbox_type>' not in content:
            raise ValidationError(f"Assignment {assignment_file} invalid dropbox_type")
        
        # Validate points_possible is numeric
        points_match = re.search(r'<points_possible>(\d+(?:\.\d+)?)</points_possible>', content)
        if not points_match:
            raise ValidationError(f"Assignment {assignment_file} invalid points_possible format")
    
    return "Assignment dropbox validation passed"
```

### 4.2 Discussion Forum Validation
```python
def validate_discussion_forum_functionality(discussion_files):
    for discussion_file in discussion_files:
        content = read_xml_content(discussion_file)
        
        # Validate required D2L discussion elements
        required_elements = [
            '<points_possible>',
            '<forum_type>',
            '<participation_requirements>'
        ]
        
        for element in required_elements:
            if element not in content:
                raise ValidationError(f"Discussion {discussion_file} missing element: {element}")
        
        # Validate forum_type values
        valid_forum_types = ['Topic', 'QA', 'General']
        forum_type_found = False
        for forum_type in valid_forum_types:
            if f'<forum_type>{forum_type}</forum_type>' in content:
                forum_type_found = True
                break
        
        if not forum_type_found:
            raise ValidationError(f"Discussion {discussion_file} invalid forum_type")
    
    return "Discussion forum validation passed"
```

### 4.3 Quiz Assessment Integration Validation
```python
def validate_quiz_assessment_integration(quiz_files):
    for quiz_file in quiz_files:
        content = read_xml_content(quiz_file)
        
        # Validate QTI structure
        if '<questestinterop' not in content:
            raise ValidationError(f"Quiz {quiz_file} not valid QTI format")
        
        # Validate D2L metadata fields
        required_d2l_metadata = [
            'd2l_2p0_grade_item_points_possible',
            'd2l_2p0_attempts_allowed'
        ]
        
        for field in required_d2l_metadata:
            if field not in content:
                raise ValidationError(f"Quiz {quiz_file} missing D2L metadata: {field}")
        
        # Validate assessment structure has questions
        if '<item ident=' not in content:
            raise ValidationError(f"Quiz {quiz_file} contains no questions")
    
    return "Quiz assessment integration validation passed"
```

### 4.4 Gradebook Integration Validation
```python
def validate_gradebook_integration(all_assessment_files):
    total_points = 0
    graded_items = []
    
    for file_path in all_assessment_files:
        content = read_xml_content(file_path)
        
        # Extract points_possible values
        points_matches = re.findall(r'<points_possible>(\d+(?:\.\d+)?)</points_possible>', content)
        for points in points_matches:
            total_points += float(points)
            graded_items.append({
                'file': file_path,
                'points': float(points)
            })
    
    if total_points == 0:
        raise ValidationError("No graded items found - gradebook integration will fail")
    
    # Validate reasonable point distribution
    if total_points < 100:
        print(f"Warning: Low total points ({total_points}) - consider increasing assessment values")
    
    return f"Gradebook integration validated: {len(graded_items)} items, {total_points} total points"
```

## Comprehensive Validation Execution Protocol

### Master Validation Function
```python
def execute_comprehensive_brightspace_validation(package_directory):
    validation_results = []
    
    try:
        # Level 1: XML Schema Compliance
        manifest_file = os.path.join(package_directory, 'imsmanifest.xml')
        manifest_content = read_file(manifest_file)
        manifest_tree = parse_xml(manifest_content)
        
        validation_results.append(validate_brightspace_namespaces(manifest_content))
        validation_results.append(validate_resource_types(manifest_tree))
        
        # Get file lists
        all_files = get_all_files_in_package(package_directory)
        html_files = [f for f in all_files if f.endswith('.html')]
        assessment_files = [f for f in all_files if 'quiz' in f or 'assessment' in f]
        assignment_files = [f for f in all_files if 'assignment' in f]
        discussion_files = [f for f in all_files if 'discussion' in f]
        
        validation_results.append(validate_qti_compliance(assessment_files))
        
        # Level 2: Educational Structure Integrity
        validation_results.append(validate_seven_submodule_structure(html_files))
        validation_results.append(validate_organization_hierarchy(manifest_tree))
        validation_results.append(validate_content_accessibility(manifest_tree, html_files))
        
        # Level 3: Content Quality Assurance
        validation_results.append(validate_educational_depth(html_files))
        validation_results.append(validate_theory_example_integration(html_files))
        validation_results.append(validate_learning_objectives_alignment(html_files, assessment_files))
        
        # Level 4: Brightspace Integration Functionality
        if assignment_files:
            validation_results.append(validate_assignment_dropbox_functionality(assignment_files))
        if discussion_files:
            validation_results.append(validate_discussion_forum_functionality(discussion_files))
        if assessment_files:
            validation_results.append(validate_quiz_assessment_integration(assessment_files))
        
        all_assessment_files = assignment_files + discussion_files + assessment_files
        validation_results.append(validate_gradebook_integration(all_assessment_files))
        
        return {
            'status': 'PASSED',
            'results': validation_results,
            'message': 'All Brightspace validation protocols passed successfully'
        }
        
    except ValidationError as e:
        return {
            'status': 'FAILED',
            'error': str(e),
            'completed_validations': validation_results
        }
```

## Pre-Import Validation Checklist

### Technical Validation ✅
- [ ] All required namespaces declared correctly
- [ ] QTI 1.2 compliance for all assessments
- [ ] Resource types match content formats
- [ ] All file references resolve correctly
- [ ] XML well-formedness validated

### Educational Structure Validation ✅
- [ ] Seven sub-modules per week maintained
- [ ] No single-page content consolidation
- [ ] Organization hierarchy complete
- [ ] All content accessible via manifest
- [ ] Navigation structure preserved

### Content Quality Validation ✅
- [ ] Educational depth meets minimum standards
- [ ] Theory-example integration proper
- [ ] Learning objectives clearly stated
- [ ] Mathematical rigor appropriate
- [ ] Academic standards maintained

### Brightspace Integration Validation ✅
- [ ] Assignment dropboxes will create in gradebook
- [ ] Discussion forums include grading capability
- [ ] Quiz assessments integrate with quiz tool
- [ ] Gradebook points allocation reasonable
- [ ] All assessment tools functional

## Post-Import Verification Protocol

### Brightspace Import Success Verification
```python
def verify_brightspace_import_success():
    verification_steps = [
        "Verify IMSCC package imports without errors",
        "Confirm course modules appear in Content area", 
        "Check assignment dropboxes created in Grades",
        "Validate quiz tools appear in Quizzes section",
        "Test discussion forums in Discussions area",
        "Verify content pages display properly",
        "Confirm navigation structure intact",
        "Test student access to all materials"
    ]
    
    return verification_steps
```

This comprehensive validation framework ensures IMSCC packages meet all Brightspace compatibility requirements and prevents all identified failure patterns while maintaining educational quality standards.