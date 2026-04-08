# Brightspace Packager Agent Specialization Guide

## Agent Purpose and Specialization

The `brightspace-packager` agent is specifically designed to convert completed course materials into Brightspace-compatible IMSCC packages. This agent specializes in:

1. **Brightspace D2L XML Schema Implementation** (Updated with actual export format)
2. **IMSCC 1.3 Compliance with Brightspace Extensions** (Now using 1.3 not 1.1/1.2)
3. **Native Brightspace Tool Integration**
4. **Pattern Prevention Protocols (15, 19, 22)**
5. **Educational Content Quality Assurance**

**CRITICAL UPDATE**: All schemas updated based on actual Brightspace export analysis (D2LCCExport package).

## Core Brightspace IMSCC Expertise

### Required Brightspace IMSCC 1.3 Namespaces (From Actual Export)
```xml
<manifest identifier="i53063987-612a-477d-9c3a-86d2d8471636" 
          xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1" 
          xmlns:lomr="http://ltsc.ieee.org/xsd/imsccv1p3/LOM/resource" 
          xmlns:lomm="http://ltsc.ieee.org/xsd/imsccv1p3/LOM/manifest" 
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
          xsi:schemaLocation="http://ltsc.ieee.org/xsd/imsccv1p3/LOM/resource http://www.imsglobal.org/profile/cc/ccv1p3/LOM/ccv1p3_lomresource_v1p0.xsd 
                              http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1 http://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_imscp_v1p2_v1p0.xsd 
                              http://ltsc.ieee.org/xsd/imsccv1p3/LOM/manifest http://www.imsglobal.org/profile/cc/ccv1p3/LOM/ccv1p3_lommanifest_v1p0.xsd">
```

### Brightspace Resource Types Specialization (UPDATED with actual export types)
- `assignment_xmlv1p0`: Assignment with dropbox integration (NOT D2L 2.0 format)
- `imsdt_xmlv1p3`: Discussion forum with IMSCC 1.3 format
- `imsqti_xmlv1p2/imscc_xmlv1p3/assessment`: QTI 1.2 with IMSCC 1.3 wrapper
- `webcontent`: HTML content with Brightspace module structure

**CRITICAL**: Brightspace uses standard IMSCC extensions, not D2L 2.0 format as previously documented.

### Brightspace-Specific XML Schema Implementation

#### Brightspace Assignment Schema (ACTUAL FORMAT from Export)
```xml
<assignment xmlns="http://www.imsglobal.org/xsd/imscc_extensions/assignment" 
            xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
            xsi:schemaLocation="http://www.imsglobal.org/xsd/imscc_extensions/assignment http://www.imsglobal.org/profile/cc/cc_extensions/cc_extresource_assignmentv1p0_v1p0.xsd" 
            identifier="i2c503fa7-d900-46ae-b7e9-417c42801686">
  <title>Assignment Title</title>
  <instructor_text texttype="text/html">&lt;p&gt;Assignment instructions here&lt;/p&gt;</instructor_text>
  <submission_formats>
    <format type="file" />
  </submission_formats>
  <gradable points_possible="100.000000000">true</gradable>
</assignment>
```

**Key Differences from D2L 2.0 Format**:
- Uses `imscc_extensions/assignment` namespace (not D2L 2.0)
- Simplified structure compared to D2L 2.0 schema
- HTML content encoded as character entities
- Points possible as attribute in gradable element
- No complex dropbox configuration (handled by Brightspace import process)

#### Brightspace Discussion Schema (ACTUAL FORMAT from Export)
```xml
<topic xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3" 
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
       xsi:schemaLocation="http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3 http://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_imsdt_v1p3.xsd">
  <title>Discussion Topic Title</title>
  <text texttype="text/html">&lt;p&gt;Discussion topic description here&lt;/p&gt;</text>
</topic>
```

**Key Differences from D2L 2.0 Format**:
- Uses `imsccv1p3/imsdt_v1p3` namespace (IMSCC 1.3 discussion topics)
- Element name is `<topic>` not `<discussion>`
- Simplified structure without complex grading configuration
- Grading handled through separate gradebook integration
- HTML content encoded as character entities

#### Brightspace QTI Assessment Schema (ACTUAL FORMAT from Export)
```xml
<questestinterop xmlns="http://www.imsglobal.org/xsd/ims_qtiasiv1p2" 
                 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
                 xsi:schemaLocation="http://www.imsglobal.org/xsd/ims_qtiasiv1p2 http://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_qtiasiv1p2p1_v1p0.xsd">
  <assessment ident="i358a135b-2158-4406-8a14-85ec19d2e64f" title="Quiz Title">
    <qtimetadata>
      <qtimetadatafield>
        <fieldlabel>cc_profile</fieldlabel>
        <fieldentry>cc.exam.v0p1</fieldentry>
      </qtimetadatafield>
      <qtimetadatafield>
        <fieldlabel>qmd_assessmenttype</fieldlabel>
        <fieldentry>Examination</fieldentry>
      </qtimetadatafield>
      <qtimetadatafield>
        <fieldlabel>cc_maxattempts</fieldlabel>
        <fieldentry>2</fieldentry>
      </qtimetadatafield>
      <qtimetadatafield>
        <fieldlabel>qmd_timelimit</fieldlabel>
        <fieldentry>0</fieldentry>
      </qtimetadatafield>
    </qtimetadata>
    <section ident="section_001">
      <!-- Questions here -->
    </section>
  </assessment>
</questestinterop>
```

**Key Differences from Previous Documentation**:
- Uses standard QTI metadata fields (cc_profile, qmd_assessmenttype)
- No D2L 2.0 specific metadata fields in actual export
- Standard QTI 1.2 format with IMSCC 1.3 schema location
- Brightspace-specific features handled during import process
    <fieldentry>3</fieldentry>
  </qtimetadatafield>
  <qtimetadatafield>
    <fieldlabel>d2l_2p0_time_limit</fieldlabel>
    <fieldentry>60</fieldentry>
  </qtimetadatafield>
  <qtimetadatafield>
    <fieldlabel>d2l_2p0_can_exceed_max_points</fieldlabel>
    <fieldentry>false</fieldentry>
  </qtimetadatafield>
</qtimetadata>
```

## Pattern Prevention Specialization

### Pattern 15 Prevention (XML Compliance) - RESOLVED
The agent MUST implement QTI 1.2 and D2L XML standards to prevent "Illegal XML" import errors:

**Required Implementation**:
1. QTI 1.2 namespace compliance: `http://www.imsglobal.org/xsd/ims_qtiasiv1p2`
2. D2L 2.0 extensions namespace: `http://www.imsglobal.org/xsd/imsccv1p1/d2l_2p0`
3. Proper resource type declarations matching content formats
4. Assessment functionality validation (dropbox creation testing)

### Pattern 19 Prevention (Educational Structure) - CRITICAL
The agent MUST preserve educational structure consistency with course outline:

**Required Structure Validation**:
```python
def validate_educational_structure(content_files, course_outline):
    """Validate module structure matches course outline - no artificial consolidation"""
    for unit in course_outline.get_learning_units():
        unit_files = [f for f in content_files if f.startswith(unit.folder_prefix)]
        expected_modules = unit.get_expected_modules()

        for expected_file in expected_modules:
            if expected_file not in unit_files:
                raise ValueError(f"Pattern 19 violation: Missing {expected_file}")

        # Prevent single-page consolidation of complex topics
        if unit.complexity == "high" and len(unit_files) < 3:
            raise ValueError(f"Pattern 19 violation: Complex unit '{unit.name}' has insufficient module breakdown")
```

### Pattern 22 Prevention (Comprehensive Educational Content) - ENHANCED
The agent MUST validate educational content depth before packaging:

**Content Quality Gates**:
```python
def validate_pattern_22_compliance(html_content):
    # Comprehensive educational content requirements
    if len(html_content) < 1500:  # Minimum educational substance
        raise ValueError("Pattern 22 violation: Insufficient educational content")
    
    # Theory-first validation (examples must support theory)
    if html_content.count('<h') < 3:  # Minimum section headers
        raise ValueError("Pattern 22 violation: Inadequate pedagogical structure")
    
    # Mathematical rigor validation for Linear Algebra content
    math_indicators = ['theorem', 'proof', 'definition', 'example', 'matrix', 'vector']
    if sum(1 for indicator in math_indicators if indicator.lower() in html_content.lower()) < 5:
        raise ValueError("Pattern 22 violation: Insufficient mathematical rigor")
```

## Brightspace Manifest Structure Specialization

### Required Manifest Template
```xml
<?xml version="1.0" encoding="UTF-8"?>
<manifest identifier="COURSE_ID_TIMESTAMP" 
          xmlns="http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1"
          xmlns:lom="http://ltsc.ieee.org/xsd/imsccv1p2/LOM/resource"
          xmlns:lomimscc="http://ltsc.ieee.org/xsd/imsccv1p2/LOM/manifest"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  
  <metadata>
    <schema>IMS Common Cartridge</schema>
    <schemaversion>1.2.0</schemaversion>
    <lomimscc:lom>
      <lomimscc:general>
        <lomimscc:title>
          <lomimscc:string language="en">{{COURSE_TITLE}}</lomimscc:string>
        </lomimscc:title>
        <lomimscc:description>
          <lomimscc:string language="en">{{COURSE_DESCRIPTION}}</lomimscc:string>
        </lomimscc:description>
      </lomimscc:general>
      <lomimscc:educational>
        <lomimscc:intendedEndUserRole>
          <lomimscc:source>LOMv1.0</lomimscc:source>
          <lomimscc:value>learner</lomimscc:value>
        </lomimscc:intendedEndUserRole>
        <lomimscc:context>
          <lomimscc:source>LOMv1.0</lomimscc:source>
          <lomimscc:value>higher education</lomimscc:value>
        </lomimscc:context>
        <lomimscc:typicalLearningTime>
          <lomimscc:duration>PT{{DURATION_WEEKS}}W</lomimscc:duration>
        </lomimscc:typicalLearningTime>
      </lomimscc:educational>
    </lomimscc:lom>
  </metadata>
  
  <organizations default="ORG001">
    <organization identifier="ORG001" structure="rooted-hierarchy">
      <title>{{COURSE_TITLE}}</title>
      <!-- Dynamic structure based on course outline -->
    </organization>
  </organizations>
  
  <resources>
    <!-- 84+ HTML files + XML assessments -->
  </resources>
</manifest>
```

## Agent Processing Workflow

### Phase 1: Content Analysis and Validation
1. **Content Quality Validation**: Verify Pattern 22 compliance before processing
2. **Structure Validation**: Confirm module count matches course outline (Pattern 19 prevention)
3. **Educational Depth Assessment**: Validate comprehensive content standards
4. **Requirements Compliance**: Verify course structure and academic specifications

### Phase 2: Brightspace Schema Implementation
1. **Manifest Generation**: Create IMSCC 1.2 compliant manifest with D2L namespaces
2. **Resource Type Mapping**: Convert content to appropriate Brightspace resource types
3. **Assessment XML Creation**: Generate QTI 1.2 + D2L metadata for quizzes/assignments
4. **Organization Structure**: Build hierarchical navigation per course outline structure

### Phase 3: D2L Integration Specialization
1. **Dropbox Assignment XML**: Create functional assignment dropboxes with grading
2. **Discussion Forum XML**: Implement graded discussion forums with participation requirements
3. **Quiz Assessment XML**: Generate QTI 1.2 compliant quizzes with D2L gradebook integration
4. **Gradebook Metadata**: Ensure points_possible and grading schemas are properly defined

### Phase 4: Package Assembly and Validation
1. **Atomic Generation**: Single .imscc file creation (Pattern 7 prevention)
2. **Schema Validation**: Verify XML compliance for all generated files
3. **Import Testing**: Validate Brightspace compatibility before finalization
4. **Quality Assurance**: Final educational content and technical validation

## Critical Brightspace Implementation Requirements

### Dropbox Creation Success Criteria
```xml
<!-- Must create functional dropbox in Brightspace -->
<resource identifier="week01_assignment" type="imsccv1p1/d2l_2p0/assignment" href="assignments/week01_assignment.xml">
  <file href="assignments/week01_assignment.xml"/>
</resource>
```

### Gradebook Integration Requirements
- All assessments MUST include `points_possible` metadata
- Assignment dropboxes MUST specify `dropbox_type` (Individual/Group)
- Discussions MUST include `participation_requirements` for grading
- Quizzes MUST use QTI 1.2 with D2L extensions for gradebook sync

### Module Content Display Requirements
- HTML content MUST be accessible within Brightspace modules
- Organization items MUST link to actual content resources
- Content navigation MUST preserve pedagogical structure per course outline
- Educational materials MUST display properly (not empty modules)

## Error Handling and Recovery

### XML Validation Errors
```python
def handle_xml_validation_error(error, file_path):
    if "illegal XML" in str(error).lower():
        # Pattern 15 prevention - fix schema compliance
        return fix_qti_d2l_compliance(file_path)
    elif "namespace" in str(error).lower():
        # Fix namespace declarations
        return fix_namespace_compliance(file_path)
    else:
        raise SystemExit(f"Critical XML error in {file_path}: {error}")
```

### Educational Structure Validation
```python
def handle_structure_violation(week_num, missing_files):
    # Pattern 19 prevention - regenerate missing sub-modules
    for missing_file in missing_files:
        regenerate_submodule_content(week_num, missing_file)
    
    # Validate regenerated content meets Pattern 22 standards
    validate_educational_depth(week_num)
```

## Success Metrics and Validation

### Technical Import Success
- ✅ IMSCC package imports to Brightspace without errors
- ✅ All content resources convert successfully
- ✅ Assignment dropboxes appear in gradebook
- ✅ Quiz assessments integrate with Brightspace quiz tool
- ✅ Discussion forums function with grading capability

### Educational Content Success
- ✅ All learning units contain modules per course outline
- ✅ Content maintains comprehensive educational depth
- ✅ Subject rigor appropriate for course level
- ✅ Theory-example integration preserves pedagogical flow
- ✅ Students can navigate through structured learning progression

### Pattern Prevention Success
- ✅ Zero Pattern 15 violations (XML compliance achieved)
- ✅ Zero Pattern 19 violations (educational structure preserved)
- ✅ Zero Pattern 22 violations (comprehensive content maintained)
- ✅ Zero Pattern 7 violations (atomic generation successful)

## Agent Deployment Protocol

### Pre-Processing Requirements
1. **Content Validation**: Verify source materials meet Pattern 22 standards
2. **Structure Analysis**: Confirm module structure matches course outline
3. **Requirements Collection**: Validate course structure and academic specifications
4. **Quality Gates**: All educational content quality checks must pass

### Processing Execution
1. **Schema Implementation**: Apply Brightspace D2L XML schemas
2. **Content Transformation**: Convert to IMSCC format preserving structure
3. **Assessment Integration**: Create functional D2L assessment tools
4. **Manifest Assembly**: Generate compliant IMSCC 1.2 manifest

### Post-Processing Validation
1. **Technical Validation**: XML schema compliance verification
2. **Import Testing**: Brightspace compatibility confirmation
3. **Educational Validation**: Content quality and structure verification
4. **Pattern Prevention**: Final check against all identified patterns

This specialization guide ensures the `brightspace-packager` agent has comprehensive expertise in Brightspace D2L IMSCC standards and can successfully create production-ready packages that import and function correctly in Brightspace environments.