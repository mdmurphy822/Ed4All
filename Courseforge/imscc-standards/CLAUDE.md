# IMSCC Standards Documentation

This directory contains comprehensive documentation and reference materials for IMS Common Cartridge (IMSCC) formatting specifications, schemas, and implementation guidelines.

## Purpose

This documentation serves as a complete reference for creating, validating, and implementing IMSCC packages compatible with major Learning Management Systems (LMS), with special focus on Brightspace D2L integration.

## Directory Structure

### `/schemas/`
XML schema definitions and namespace specifications for IMSCC versions 1.1, 1.2, and 1.3

### `/examples/`
Working examples of IMSCC manifest files and resource structures

### `/brightspace-specific/`
Brightspace D2L-specific extensions, schemas, and implementation details

### `/validation/`
Tools and scripts for validating IMSCC package compliance

## Key IMSCC Specifications

### IMS Common Cartridge 1.1
- **File Extension**: `.imscc`
- **Schema Version**: `1.1.0`
- **Namespace**: `http://ltsc.ieee.org/xsd/imsccv1p1/LOM/manifest`
- **Key Features**: Basic web content, Basic LTI support

### IMS Common Cartridge 1.2
- **File Extension**: `.imscc`
- **Schema Version**: `1.2.0`
- **Namespace**: `http://ltsc.ieee.org/xsd/imsccv1p2/LOM/manifest`
- **Key Features**: Curriculum standards metadata, enhanced resource types

### IMS Common Cartridge 1.3 (Latest)
- **File Extension**: `.imscc`
- **Schema Version**: `1.3.0`
- **Namespace**: `http://ltsc.ieee.org/xsd/imsccv1p3/LOM/manifest`
- **Key Features**: Advanced assessment types, improved interoperability

## Core IMSCC Components

### 1. Manifest File (`imsmanifest.xml`)

**Required Structure**:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<manifest identifier="UNIQUE_ID" 
          xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1"
          xmlns:lom="http://ltsc.ieee.org/xsd/imsccv1p3/LOM/resource"
          xmlns:lomimscc="http://ltsc.ieee.org/xsd/imsccv1p3/LOM/manifest"
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  
  <metadata>
    <schema>IMS Common Cartridge</schema>
    <schemaversion>1.3.0</schemaversion>
  </metadata>
  
  <organizations default="ORG_ID">
    <organization identifier="ORG_ID" structure="rooted-hierarchy">
      <!-- Course structure -->
    </organization>
  </organizations>
  
  <resources>
    <!-- Content resources -->
  </resources>
</manifest>
```

### 2. Metadata Elements

**Required Metadata**:
- `<schema>`: Must be "IMS Common Cartridge"
- `<schemaversion>`: Version number (1.1.0, 1.2.0, or 1.3.0)

**Optional Dublin Core Metadata** (15 elements supported):
- `<title>`: Course title
- `<creator>`: Content creator
- `<subject>`: Subject area
- `<description>`: Course description
- `<publisher>`: Publishing organization
- `<contributor>`: Contributors
- `<date>`: Creation/modification date
- `<type>`: Content type
- `<format>`: File format
- `<identifier>`: Unique identifier
- `<source>`: Source reference
- `<language>`: Language code
- `<relation>`: Related resources
- `<coverage>`: Temporal/spatial coverage
- `<rights>`: Rights management

### 3. Organization Structure

**Hierarchical Organization**:
```xml
<organizations default="ORG_ID">
  <organization identifier="ORG_ID" structure="rooted-hierarchy">
    <title>Course Title</title>
    <item identifier="ITEM_ID" identifierref="RESOURCE_ID">
      <title>Module Title</title>
      <item identifier="SUB_ITEM_ID" identifierref="SUB_RESOURCE_ID">
        <title>Sub-module Title</title>
      </item>
    </item>
  </organization>
</organizations>
```

**Organization Attributes**:
- `identifier`: Unique identifier for organization
- `structure`: Must be "rooted-hierarchy"
- `identifierref`: Links to resource in `<resources>` section

### 4. Resource Types

#### Web Content
```xml
<resource identifier="RES_ID" type="webcontent" href="content.html">
  <metadata>
    <lom:lom>
      <lom:educational>
        <lom:intendedEndUserRole>
          <lom:source>LOMv1.0</lom:source>
          <lom:value>Learner</lom:value>
        </lom:intendedEndUserRole>
      </lom:educational>
    </lom:lom>
  </metadata>
  <file href="content.html"/>
</resource>
```

**Web Content Attributes**:
- `intendeduse`: Values include `lessonplan`, `syllabus`, `assignment`, `unspecified`

#### QTI Assessments
```xml
<resource identifier="QUIZ_ID" type="imsqti_xmlv1p2/imscc_xmlv1p3/assessment" href="quiz.xml">
  <dependency identifierref="QTI_ASI_BASE"/>
  <file href="quiz.xml"/>
</resource>
```

#### Basic LTI Links
```xml
<resource identifier="LTI_ID" type="imsbasiclti_xmlv1p0" href="lti_link.xml">
  <file href="lti_link.xml"/>
</resource>
```

#### Discussion Topics
```xml
<resource identifier="DISC_ID" type="imsdt_xmlv1p3" href="discussion.xml">
  <file href="discussion.xml"/>
</resource>
```

**Discussion XML Content**:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<topic xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3
       http://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_imsdt_v1p3.xsd">
  <title>Discussion Title</title>
  <text texttype="text/html"><![CDATA[
    <!-- HTML content here -->
  ]]></text>
</topic>
```

## Brightspace D2L Specific Extensions

### D2L Assignment Schema
```xml
<assignment xmlns="http://www.imsglobal.org/xsd/imsccv1p1/d2l_2p0">
  <gradebook_item>
    <points_possible>100</points_possible>
    <weight>10.0</weight>
  </gradebook_item>
  <dropbox>
    <dropbox_type>Individual</dropbox_type>
    <submissions_allowed>unlimited</submissions_allowed>
  </dropbox>
</assignment>
```

**Key D2L Assignment Elements**:
- `<points_possible>`: Maximum points (decimal)
- `<dropbox_type>`: "Individual" or "Group"
- `<submissions_allowed>`: Number or "unlimited"
- `<grade_item_id>`: Links to gradebook

### D2L Discussion Schema
```xml
<discussion xmlns="http://www.imsglobal.org/xsd/imsccv1p1/d2l_2p0">
  <gradebook_item>
    <points_possible>25</points_possible>
  </gradebook_item>
  <participation_requirements>
    <initial_post_required>true</initial_post_required>
    <replies_required>2</replies_required>
  </participation_requirements>
</discussion>
```

### D2L QTI Extensions
```xml
<qtimetadata>
  <qtimetadatafield>
    <fieldlabel>d2l_2p0_grade_item_points_possible</fieldlabel>
    <fieldentry>50</fieldentry>
  </qtimetadatafield>
  <qtimetadatafield>
    <fieldlabel>d2l_2p0_attempts_allowed</fieldlabel>
    <fieldentry>3</fieldentry>
  </qtimetadatafield>
</qtimetadata>
```

## Resource File Constraints

### File Naming Conventions
- No spaces in file names (use underscores or hyphens)
- Relative paths only (no absolute paths)
- Case-sensitive file references
- UTF-8 encoding for all text files

### Supported File Types
- **HTML**: Web content pages
- **XML**: Assessment, discussion, assignment definitions
- **CSS**: Styling for web content
- **JavaScript**: Client-side functionality
- **Images**: PNG, JPG, GIF, SVG
- **Documents**: PDF (referenced, not embedded)
- **Media**: MP4, MP3 (referenced, not embedded)

## Validation Requirements

### Schema Validation
1. XML well-formedness
2. Namespace compliance
3. Required element presence
4. Attribute value constraints

### Content Validation
1. All referenced files exist
2. Resource identifiers are unique
3. Organization references valid resources
4. File paths are relative and valid

### LMS Compatibility Testing
1. Import without errors
2. Content displays correctly
3. Assessments function properly
4. Navigation structure works

## Common Implementation Patterns

### Course Module Structure
```xml
<!-- Week 1 Module -->
<item identifier="week1" identifierref="week1_overview">
  <title>Week 1: Introduction</title>
  <item identifier="week1_overview" identifierref="week1_overview_res">
    <title>Module Overview</title>
  </item>
  <item identifier="week1_content1" identifierref="week1_content1_res">
    <title>Concept 1</title>
  </item>
  <item identifier="week1_assignment" identifierref="week1_assignment_res">
    <title>Week 1 Assignment</title>
  </item>
</item>
```

### Assessment Integration
1. Create QTI XML file for quiz content
2. Create D2L extension XML for gradebook integration
3. Reference both in manifest resources
4. Link from organization structure

### Multi-file Resource Pattern
```xml
<resource identifier="lesson1" type="webcontent" href="lesson1.html">
  <file href="lesson1.html"/>
  <file href="css/styles.css"/>
  <file href="images/diagram1.png"/>
  <file href="js/interactions.js"/>
</resource>
```

## Error Prevention Guidelines

### Pattern 15 Prevention (XML Compliance)
- Use QTI 1.2 format for assessments
- Include D2L extensions for Brightspace compatibility
- Validate XML schema compliance before packaging

### Pattern 19 Prevention (Educational Structure)
- Maintain module structure per course outline (no artificial consolidation)
- Avoid single-page content consolidation of complex topics
- Preserve pedagogical organization hierarchy

### Pattern 22 Prevention (Content Depth)
- Ensure comprehensive educational content before examples
- Integrate authentic examples within theoretical context
- Maintain academic rigor throughout all modules

### Pattern 24 Prevention (IMSCC Version Consistency) - CRITICAL

**Root Cause**: When manifest declares one IMSCC version but content XMLs use different version namespaces, Brightspace fails to parse titles and creates flat module structures.

**Symptoms**:
- All content appears in ONE flat module instead of weekly organization
- Discussion topics show resource IDs (e.g., "RES_W01_DISC_R") instead of titles
- Navigation hierarchy collapses to flat list

**Version Alignment Requirements (IMSCC 1.3 - RECOMMENDED)**:

| Component | Required Value |
|-----------|----------------|
| Manifest xmlns | `http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1` |
| Schema version | `1.3.0` |
| Discussion resource type | `imsdt_xmlv1p3` |
| Discussion XML namespace | `http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3` |
| Quiz resource type | `imsqti_xmlv1p2/imscc_xmlv1p3/assessment` |
| Assignment resource type | `assignment_xmlv1p0` |
| Assignment XML namespace | `http://www.imsglobal.org/xsd/imscc_extensions/assignment` |

**IMPORTANT**: Do NOT use `associatedcontent/imscc_xmlv1p3/learning-application-resource` for assignments - this causes Brightspace to import assignments as raw XML files instead of native assignment tools (Pattern 25).

**Validation Command**:
```bash
# Check for version mismatches (should not find v1p1 in a v1p3 package)
grep -r "imsccv1p1" *.xml  # Should return NO results for v1.3 packages
grep -r "imsccv1p3" *.xml  # Should find all IMSCC references
```

**Prevention Protocol**:
1. Use IMSCC 1.3 consistently for all new packages
2. Validate namespace alignment before packaging
3. Test import in Brightspace sandbox before production deployment
4. Update manifest immediately if content XML namespaces differ

## Tools and Utilities

### Validation Scripts
- XML schema validators
- IMSCC package validators
- Brightspace compatibility checkers

### Generation Templates
- Manifest file templates
- Resource definition templates
- Assessment XML templates

### Testing Frameworks
- LMS import testing
- Content display validation
- Assessment functionality verification

## Best Practices

### Performance Optimization
- Minimize file sizes where possible
- Use efficient HTML/CSS structures
- Optimize images and media

### Accessibility Compliance
- Include alt text for images
- Use semantic HTML markup
- Ensure keyboard navigation
- Meet WCAG 2.2 AA standards

### Maintainability
- Use consistent naming conventions
- Document custom extensions
- Version control manifest changes
- Test across multiple LMS platforms

## Version History

- **1.1**: Initial standardized format
- **1.2**: Added curriculum standards, enhanced resource types
- **1.3**: Improved assessment support, better LMS interoperability

## Related Documentation

- `/schemas/imscc/`: Complete XML schema definitions
- `/schemas/assessment/`: QTI and D2L assessment schemas
- `/docs/PATTERN_PREVENTION_GUIDE.md`: Error pattern prevention guidance
- `/CLAUDE.md`: Course generation guidelines

---

*This documentation provides comprehensive guidance for IMSCC standards implementation. For specific implementation examples, see the `/examples/` directory.*