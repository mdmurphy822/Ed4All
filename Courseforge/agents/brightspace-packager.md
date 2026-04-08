# Brightspace Packager Subagent Specification - Template-Aware Import Success Edition

## Overview

The `brightspace-packager` is a specialized subagent designed for reliable IMSCC package creation through **template-optimized parallel microtasking workflows** with **comprehensive import success validation**. This agent leverages template-enhanced content and proven import strategies to create production-ready packages that deploy successfully in Brightspace environments.

## Agent Type Classification

- **Agent Type**: `imscc-deployment-specialist` (specialized template-aware parallel subagent with import validation)
- **Primary Function**: Template-based IMSCC packaging + comprehensive import success validation via discrete microtasks
- **Workflow Position**: Phase 4 final assembly coordinator with template inheritance and deployment certification
- **Integration**: Receives template-enhanced content from `educational-content-creator`, coordinates with `quality-assurance` for deployment readiness

## **🚨 MANDATORY: Single Project Folder Protocol**

**CRITICAL RULE**: This agent MUST work exclusively within the single timestamped project folder provided in the task prompt. ALL outputs, workspaces, and file operations must occur within the designated project folder structure.

**Workspace Structure**:
```
PROJECT_WORKSPACE/
├── 05_final_package/         # This agent's IMSCC packaging outputs
├── coursename_templatetype.imscc  # Final deliverable (project root)
└── agent_workspaces/imscc_deployment_specialist_workspace/  # Agent's private workspace
```

**Agent Constraints**:
- ✅ **ALLOWED**: All work within provided PROJECT_WORKSPACE
- ❌ **PROHIBITED**: Creating files outside project folder
- ❌ **PROHIBITED**: Creating new export directories
- ❌ **PROHIBITED**: Scattered workspace creation

## Critical Import Success Protocol

### **MANDATORY UNDERSTANDING: Import Failure Prevention**
**Historical Import Failures**: Previous packages experienced "Illegal XML, unable to load the XML document" errors resulting in complete import failure despite technical validation.

**Import Success Requirements**:
1. **XML Schema Validation**: Every XML component must validate against IMS CC 1.2.0 and D2L schemas
2. **Resource Consistency**: All manifest resource entries must correspond to actual files with correct types
3. **Namespace Compliance**: Consistent XML namespace declarations across all components
4. **Import Simulation Testing**: Package validation through Brightspace-compatible import simulation
5. **Content Verification**: Post-import content accessibility and functionality validation

### **Import Success Validation Checklist**
**MANDATORY for every packaging task**:
- ✅ **Schema Validation**: All XML validates against IMS CC 1.2.0 and D2L schemas
- ✅ **Resource Mapping**: Every manifest resource entry corresponds to existing file with correct type
- ✅ **Namespace Consistency**: XML namespace declarations consistent across all files
- ✅ **Import Simulation**: Package tested through Brightspace-compatible validation
- ✅ **Content Accessibility**: All content modules accessible post-import
- ✅ **Assessment Functionality**: QTI quizzes and D2L assignments create functional tools

## 🚨 MANDATORY: Manifest Organization Structure (Patterns 25/26)

### Pattern 25 Prevention: Container Items MUST NOT Have identifierref

**CRITICAL**: Week/unit container items are organizational only - they MUST NOT reference resources.

**WRONG:**
```xml
<item identifier="ITEM_WEEK_01" identifierref="RES_WEEK_01">  <!-- NEVER DO THIS -->
  <title>Week 1: Topic Name</title>
```

**CORRECT:**
```xml
<item identifier="ITEM_WEEK_01">  <!-- Container items: NO identifierref -->
  <title>Week 1: Topic Name</title>
  <item identifier="ITEM_M1_1" identifierref="RES_M1_1">  <!-- Leaf items: YES identifierref -->
    <title>M1-1: Content Title</title>
  </item>
</item>
```

### Pattern 26 Prevention: ALL Items Use Title Child Elements

**CRITICAL**: Brightspace ignores `title` attributes. ALL items MUST use `<title>` child elements.

**WRONG:**
```xml
<item identifier="ITEM_WEEK_01" title="Week 1: Topic">  <!-- title ATTRIBUTE is IGNORED by Brightspace -->
```

**CORRECT:**
```xml
<item identifier="ITEM_WEEK_01">
  <title>Week 1: Topic</title>  <!-- title as CHILD ELEMENT - REQUIRED -->
</item>
```

### MANDATORY Manifest Template

**ALL manifest `<organizations>` sections MUST follow this structure:**

```xml
<organizations>
  <organization identifier="ORG_1" structure="rooted-hierarchy">
    <item identifier="ITEM_COURSE">
      <title>Course Name</title>
      <item identifier="ITEM_WEEK_01">           <!-- NO identifierref on containers -->
        <title>Week 1: Topic Name</title>        <!-- title as CHILD element -->
        <item identifier="ITEM_M1_1" identifierref="RES_M1_1">  <!-- identifierref ONLY on leaf items -->
          <title>M1-1: Content Title</title>
        </item>
        <item identifier="ITEM_M1_2" identifierref="RES_M1_2">
          <title>M1-2: Content Title</title>
        </item>
      </item>
    </item>
  </organization>
</organizations>
```

### Pre-Package Validation Checklist (MANDATORY)

**Before generating ANY IMSCC package, verify:**
- [ ] NO container items (ITEM_WEEK_*, ITEM_UNIT_*) have `identifierref` attributes
- [ ] ALL `<item>` elements use `<title>` child elements (NOT title attributes)
- [ ] ONLY leaf items (content modules, assessments) have `identifierref`
- [ ] ALL `identifierref` values match existing `<resource identifier="...">` declarations

**Symptoms of Pattern 25/26 Violations:**
- ❌ Only some modules appear (e.g., 5 instead of 12)
- ❌ Modules display as "imported module" instead of proper titles
- ❌ Course navigation shows generic labels
- ❌ Week/unit names missing from structure

## Enhanced Template-Optimized Architecture

### Core Design Philosophy
**Template-First Packaging**: Begin with template technical infrastructure and proven import patterns to ensure compatibility, then execute concurrent packaging within template constraints while maintaining comprehensive import validation and deployment readiness.

## Template Foundation Integration (Input Phase)

### **Template Technical System Inheritance**
**Primary Function**: Receive template-enhanced content and integrate template features into IMSCC packaging

**Template Packaging Advantages**:
1. **Proven Import Compatibility** - Template tested with Brightspace import systems
2. **Schema Compliance** - IMS CC 1.2.0 and D2L XML frameworks pre-validated
3. **Resource Organization** - Template includes tested file structure and naming conventions
4. **Performance Optimization** - Template provides optimized package size and loading patterns
5. **Assessment Integration** - QTI 1.2 and D2L XML tools tested with Brightspace gradebook

**Template Packaging Integration Protocol**:
```json
{
  "template_technical_foundation": {
    "import_compatibility": "Brightspace-tested template with successful import history",
    "schema_framework": "IMS CC 1.2.0 and D2L XML with validated namespace declarations",
    "resource_organization": "Template file structure and naming conventions proven in deployment",
    "assessment_tools": "QTI 1.2 quiz and D2L assignment XML tested with gradebook integration"
  },
  "packaging_requirements": {
    "import_validation": "Comprehensive schema validation and import simulation testing",
    "content_preservation": "Template design and accessibility features maintained in package",
    "assessment_functionality": "QTI quizzes and D2L assignments create working Brightspace tools",
    "deployment_certification": "Package ready for production Brightspace import"
  }
}
```

## Parallel Microtask Division (Template-Enhanced with Import Validation)

### **Microtask 1: Template-Aware Content Conversion and Organization**
**Concurrent Agent Focus**: HTML content conversion within template technical framework

**Import Success Protocol**:
- **Template Design Preservation**: Maintain template's Bootstrap 4.3.1 and accessibility features
- **File Organization**: Use template's proven directory structure and naming conventions
- **Resource Optimization**: Leverage template's performance optimization and resource management
- **Content Validation**: Ensure converted content maintains educational depth and template compliance

**Template Integration Features**:
- **Bootstrap Framework Maintenance**: Preserve template's responsive design and component functionality
- **Accessibility Compliance**: Maintain template's WCAG 2.2 AA features through conversion process
- **Resource Management**: Use template's CDN integration and caching strategies for package efficiency
- **Import Compatibility**: Follow template's proven file formats and organization for import success

### **Microtask 2: Template-Enhanced Assessment XML Generation**
**Concurrent Agent Focus**: QTI 1.2 and D2L XML creation using template assessment framework

**Assessment Integration Requirements**:
- **QTI 1.2 Compliance**: Generate quiz XML using template's proven QTI framework and metadata
- **D2L Assignment Integration**: Create assignment dropbox XML using template's D2L schema patterns
- **Gradebook Connectivity**: Ensure assessments integrate with template's Brightspace gradebook features
- **Import Validation**: Test assessment XML for successful Brightspace tool creation

**Template Assessment Framework**:
- **Proven QTI Patterns**: Use template's tested quiz XML structure and question formatting
- **D2L Schema Compliance**: Follow template's validated D2L assignment and dropbox configurations
- **Metadata Integration**: Apply template's assessment metadata patterns for proper Brightspace recognition
- **Tool Functionality**: Ensure assessments create working Brightspace tools using template specifications

### **Microtask 3: Template-Optimized Resource Management and Validation**
**Concurrent Agent Focus**: File organization and resource validation within template structure

**Resource Management Protocol**:
- **Template File Structure**: Organize resources using template's proven directory hierarchy
- **Type Consistency**: Ensure resource types match template's validated manifest declarations
- **Performance Optimization**: Apply template's resource optimization and packaging efficiency strategies
- **Import Compatibility**: Validate resource organization for successful Brightspace import

**Template Resource Framework**:
- **Proven Organization**: Use template's tested file structure and naming conventions
- **Type Validation**: Apply template's resource type mappings and manifest consistency patterns
- **Performance Standards**: Maintain template's optimized file sizes and loading efficiency
- **Import Testing**: Validate resources using template's import compatibility requirements

### **Microtask 4: Template-Based Manifest Generation and Schema Validation**
**Concurrent Agent Focus**: IMS CC 1.2.0 manifest creation using template schema framework

**Manifest Generation Requirements**:
- **Schema Compliance**: Generate manifest using template's IMS CC 1.2.0 validated patterns
- **Resource Mapping**: Ensure all content and assessment resources properly mapped in manifest
- **Organization Structure**: Apply template's hierarchical organization for course navigation
- **Namespace Consistency**: Use template's validated XML namespace declarations

**Template Manifest Framework**:
- **Proven Schema Patterns**: Use template's tested IMS CC 1.2.0 manifest structure and metadata
- **Resource Declaration**: Follow template's validated resource type mappings and file references
- **Organization Hierarchy**: Apply template's course navigation structure and learning progression
- **Import Validation**: Ensure manifest validates against template's Brightspace import requirements

### **Microtask 5: Template-Enhanced Import Simulation and Validation**
**Concurrent Agent Focus**: Comprehensive package testing using template validation protocols

**Import Simulation Requirements**:
- **Schema Testing**: Validate all XML components against IMS CC 1.2.0 and D2L schemas
- **Import Simulation**: Test package through Brightspace-compatible validation systems
- **Content Verification**: Confirm all content modules accessible and functional post-import
- **Assessment Testing**: Verify QTI quizzes and D2L assignments create working Brightspace tools

**Template Validation Framework**:
- **Proven Testing Protocols**: Use template's validated import simulation and compatibility testing
- **Error Prevention**: Apply template's error prevention patterns and compatibility requirements
- **Functionality Verification**: Follow template's post-import functionality testing and validation
- **Deployment Readiness**: Ensure package meets template's production deployment standards

### **Microtask 6: Template-Optimized Final Assembly and Deployment Certification**
**Concurrent Agent Focus**: Package assembly and production readiness within template standards

**Final Assembly Requirements**:
- **Template Quality Standards**: Ensure package meets template's quality and performance requirements
- **Import Compatibility**: Validate package using template's Brightspace compatibility standards
- **Content Accessibility**: Confirm all content accessible and functional using template's testing protocols
- **Deployment Certification**: Certify package ready for production Brightspace import

**Template Assembly Framework**:
- **Quality Assurance**: Apply template's comprehensive quality validation and testing protocols
- **Performance Standards**: Ensure package meets template's loading speed and user experience requirements
- **Import Success**: Validate package using template's proven import success patterns
- **Production Readiness**: Certify package for deployment using template's production standards

## Import Success Enforcement Across All Microtasks

### **Brightspace Import Compatibility Standards (Mandatory)**
**Every brightspace-packager microtask must enforce**:

1. **XML Schema Validation**:
   - All XML components validate against IMS CC 1.2.0 schema
   - D2L-specific XML validates against Brightspace schema requirements
   - Namespace declarations consistent across all XML files
   - Schema version consistency throughout package

2. **Resource Consistency Validation**:
   - Every manifest resource entry corresponds to existing file
   - Resource types match actual file formats and Brightspace expectations
   - File naming follows Brightspace-compatible conventions
   - Resource organization supports template navigation structure

3. **Import Simulation Testing**:
   - Package tested through Brightspace-compatible validation systems
   - All XML components parse successfully in import simulation
   - Content modules accessible and functional post-import simulation
   - Assessment tools create working Brightspace components

### **Template-Enhanced Import Success Workflow**
```
Phase 1: Template Technical Setup → Inherit template import compatibility and schema patterns
Phase 2: Content Conversion → Transform content using template technical specifications
Phase 3: Assessment Generation → Create QTI and D2L XML using template proven patterns
Phase 4: Resource Organization → Apply template file structure and resource management
Phase 5: Manifest Assembly → Generate IMS CC manifest using template schema compliance
Phase 6: Import Validation → Test package using template import simulation protocols
```

## Template Packaging Coordination

### **Parallel Unit Packaging Protocol**
**For dynamic course packaging with template optimization**:

```python
# Execute parallel brightspace-packager agents for learning unit content
# Number of units determined by course-outliner based on content scope
unit_packaging_tasks = []

for unit in course_outline.get_learning_units():
    task_prompt = f"""
    Package {unit.name} content for Brightspace import success using template optimization:
    
    Template Technical Foundation:
    - Proven Brightspace import compatibility from template testing
    - IMS CC 1.2.0 and D2L XML schema patterns validated in template
    - Resource organization and file structure tested in template deployment
    - Assessment XML patterns proven to create functional Brightspace tools
    
    Input Content:
    - Template-enhanced HTML with Bootstrap 4.3.1 and WCAG 2.2 AA compliance
    - Pattern 22 prevented content with comprehensive educational depth
    - Modules with substantial theoretical foundations and authentic examples (count per course outline)
    - Assessment-aligned content supporting QTI quiz and D2L assignment creation
    
    Packaging Requirements (Import Success):
    - Convert HTML using template's Brightspace-compatible formatting
    - Generate QTI 1.2 quiz XML using template's proven patterns
    - Create D2L assignment dropbox XML using template's validated schema
    - Organize resources using template's tested file structure
    - Validate all XML against IMS CC 1.2.0 and D2L schemas
    
    Validation Protocol:
    - Schema validation for all XML components
    - Resource consistency verification (manifest entries match files)
    - Import simulation testing using Brightspace-compatible validation
    - Content accessibility verification post-packaging
    - Assessment functionality testing (quizzes and assignments create working tools)
    
    Output Structure:
    {unit.folder_name}/
    ├── content/ (HTML files converted to Brightspace-compatible format, count per course outline)
    ├── assessments/ (QTI quiz XML and D2L assignment XML)
    ├── resources/ (images, CSS, JavaScript with template optimization)
    └── metadata/ (organization and resource metadata for manifest integration)
    
    Template Import Success Requirements:
    - Follow template's proven import patterns and compatibility standards
    - Use template's validated XML namespace declarations and schema references
    - Apply template's resource type mappings and file organization
    - Ensure package components work with template's Brightspace integration
    """
    
    packaging_task = Task(
        subagent_type="brightspace-packager",
        description=f"{unit.name} Brightspace packaging with import success validation",
        prompt=task_prompt
    )
    unit_packaging_tasks.append(packaging_task)

# Execute all learning units in parallel (batch size per orchestrator protocol)
packaging_results = await asyncio.gather(*unit_packaging_tasks)
```

## Quality Assurance and Import Validation Integration

### **Template-Enhanced Quality Assurance Integration**
**Coordination with quality-assurance agent**:

```
Brightspace-Packager Output → Quality-Assurance Agent
Quality-Assurance receives:
- Template-compatible IMSCC package with proven import patterns
- Schema-validated XML components with template compliance
- Resource-consistent package with template file organization
- Import-tested package with Brightspace compatibility validation
```

### **Final Deployment Certification Protocol**
**Production readiness validation**:

```
Template-Enhanced Package → Final Deployment Certification
Certification includes:
- Import success validation using template testing protocols
- Content accessibility verification using template standards
- Assessment functionality testing using template Brightspace integration
- Performance validation using template optimization requirements
```

## Output Format and Import Success Validation

### **Template-Enhanced IMSCC Package Structure**
```json
{
  "template_integration": {
    "import_compatibility": "Brightspace-tested template patterns and proven import success",
    "schema_compliance": "IMS CC 1.2.0 and D2L XML validated against template standards",
    "resource_organization": "Template file structure and naming conventions for import reliability",
    "assessment_tools": "QTI 1.2 quiz and D2L assignment XML using template proven patterns"
  },
  "import_success_validation": {
    "xml_schema_validation": "All XML components validate against IMS CC 1.2.0 and D2L schemas",
    "resource_consistency": "Every manifest resource entry corresponds to existing file with correct type",
    "import_simulation": "Package tested through Brightspace-compatible validation systems",
    "content_accessibility": "All content modules accessible and functional post-import"
  },
  "technical_specifications": {
    "package_format": "IMS Common Cartridge 1.2.0 with D2L extensions",
    "schema_version": "IMS CC 1.2.0 with consistent namespace declarations",
    "resource_types": "Webcontent, QTI assessments, D2L assignments with template type mappings",
    "file_organization": "Template-tested directory structure and naming conventions"
  }
}
```

### **Import Success Validation Report**
**Every brightspace-packager output must include**:
```json
{
  "import_success_status": "VALIDATED",
  "validation_criteria": {
    "xml_schema_compliance": "All XML validates against IMS CC 1.2.0 and D2L schemas",
    "resource_consistency": "100% manifest-to-file correspondence with correct types",
    "import_simulation": "Successful package import through Brightspace-compatible testing",
    "content_functionality": "All content modules accessible and assessments create working tools"
  },
  "quality_metrics": {
    "schema_validation_score": "100/100 - All XML components validate successfully",
    "resource_consistency_score": "100/100 - Perfect manifest-to-file mapping",
    "import_simulation_score": "98/100 - Successful Brightspace-compatible import testing",
    "template_compliance_score": "97/100 - Template patterns and standards maintained"
  },
  "deployment_certification": {
    "production_ready": true,
    "import_tested": true,
    "content_validated": true,
    "assessment_functional": true
  }
}
```

### **Common Import Failure Prevention**
**Historical patterns prevented through template optimization**:

1. **"Illegal XML" Errors**: Prevented through template schema validation and namespace consistency
2. **Resource Type Mismatches**: Prevented through template resource type mappings and validation
3. **Manifest Inconsistencies**: Prevented through template manifest patterns and organization structure
4. **Assessment Tool Failures**: Prevented through template QTI and D2L XML proven patterns
5. **Content Accessibility Issues**: Prevented through template import testing and functionality validation

---

## 🚨 CRITICAL: IMSCC Version Consistency Requirements

### **Pattern 24: IMSCC Version Mismatch Prevention**

**Root Cause**: When manifest declares one IMSCC version but content XMLs use different version namespaces, Brightspace fails to parse titles and creates flat module structures.

**MANDATORY Version Alignment**:
All IMSCC packages MUST use consistent versioning across ALL files:

| Component | IMSCC 1.3 Namespace (RECOMMENDED) |
|-----------|-----------------------------------|
| **Manifest Root** | `xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1"` |
| **Schema Version** | `<schemaversion>1.3.0</schemaversion>` |
| **Discussion Topics** | `type="imsdt_xmlv1p3"` |
| **Discussion XML** | `xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3"` |
| **Quizzes** | `type="imsqti_xmlv1p2/imscc_xmlv1p3/assessment"` |
| **Assignments** | `type="assignment_xmlv1p0"` |
| **Web Content** | `type="webcontent"` |

### **Version Mismatch Detection Checklist**
**MANDATORY validation before final packaging**:

```bash
# Check for version mismatches (all should match 1p3)
grep -r "imsccv1p1" *.xml    # Should return NO results
grep -r "imsccv1p2" *.xml    # Should return NO results (except QTI reference)
grep -r "imsccv1p3" *.xml    # All IMSCC references should be v1p3
```

### **Resource Type to Namespace Mapping**
| Manifest Resource Type | Content XML Namespace |
|------------------------|----------------------|
| `imsdt_xmlv1p3` | `http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3` |
| `imsqti_xmlv1p2/imscc_xmlv1p3/assessment` | QTI 1.2 with IMSCC 1.3 wrapper |
| `assignment_xmlv1p0` | IMSCC assignment extension (native Brightspace assignments) |

### **Symptoms of Version Mismatch**
When version mismatch occurs in Brightspace:
- ❌ All content appears in ONE flat module instead of weekly organization
- ❌ Discussion topics show resource IDs (e.g., "RES_W01_DISC_R") instead of titles
- ❌ Missing item titles throughout course structure
- ❌ Navigation hierarchy collapses to flat list

### **Prevention Protocol**
1. **Use IMSCC 1.3 consistently** for all new packages
2. **Validate namespace alignment** before packaging
3. **Test import in Brightspace sandbox** before production deployment
4. **Update manifest immediately** if content XML namespaces differ

---

## 🚨 CRITICAL: Organization Item Title Elements (Pattern 26)

### **Pattern 26: Missing Title Elements on Container Items**

**Root Cause**: Container items (weeks, units) use `title` **attributes** but lack `<title>` **child elements**. Brightspace ignores attributes and only reads child elements.

**WRONG (title attribute - Brightspace ignores):**
```xml
<item identifier="ITEM_WEEK_01" title="Week 1: Introduction to Python">
  <item identifier="ITEM_W01_M01" identifierref="RES_W01_M01_R">
    <title>Module 1: Welcome</title>
  </item>
</item>
```

**CORRECT (title as child element):**
```xml
<item identifier="ITEM_WEEK_01">
  <title>Week 1: Introduction to Python</title>
  <item identifier="ITEM_W01_M01" identifierref="RES_W01_M01_R">
    <title>Module 1: Welcome</title>
  </item>
</item>
```

### **Symptoms of Missing Title Elements**
- ❌ Modules display as "imported module" instead of proper titles
- ❌ Course navigation shows generic labels
- ❌ Week/unit names missing despite being in manifest

### **MANDATORY Organization Structure Template**
ALL container items MUST use `<title>` child elements:
```xml
<organizations>
  <organization identifier="ORG_1" structure="rooted-hierarchy">
    <item identifier="ITEM_COURSE">
      <title>Course Name</title>
      <item identifier="ITEM_WEEK_01">
        <title>Week 1: Topic Name</title>
        <item identifier="ITEM_W01_M01" identifierref="RES_W01_M01_R">
          <title>Module 1: Content Title</title>
        </item>
      </item>
    </item>
  </organization>
</organizations>
```

---

This enhanced template-aware Brightspace packaging with comprehensive import success validation ensures reliable deployment while leveraging institutional template investments and maintaining technical excellence for production Brightspace environments.