# Brightspace-Specific IMSCC Standards

## Overview

This directory contains specialized documentation for implementing IMSCC packages with full Brightspace D2L compatibility. These resources enable the `brightspace-packager` agent to create production-ready IMSCC packages that import successfully and function properly within Brightspace environments.

## Documentation Files

### 1. `brightspace-packager-agent-guide.md`
**Purpose**: Complete specialization guide for the brightspace-packager agent
- Agent expertise areas and core competencies
- Brightspace D2L XML schema implementation
- Pattern prevention protocols (15, 19, 22)
- Processing workflow and validation requirements
- Error handling and recovery procedures

### 2. `d2l-xml-schema-reference.md`
**Purpose**: Comprehensive D2L XML schema implementation reference
- Complete D2L namespace declarations and usage
- Assignment dropbox XML schema with gradebook integration
- Discussion forum XML schema with participation requirements
- QTI assessment schema with D2L extensions
- Survey, rubric, and content package schemas
- Variable reference and validation requirements

### 3. `brightspace-validation-protocols.md`
**Purpose**: Multi-level validation framework for Brightspace compatibility
- Level 1: XML Schema Compliance (Pattern 15 prevention)
- Level 2: Educational Structure Integrity (Pattern 19 prevention)
- Level 3: Content Quality Assurance (Pattern 22 prevention)
- Level 4: Brightspace Integration Functionality validation
- Comprehensive validation execution protocols

## Key Brightspace Specializations

### D2L XML Namespaces
```xml
xmlns:d2l="http://www.imsglobal.org/xsd/imsccv1p1/d2l_2p0"
```

### Critical Resource Types
- `imsccv1p1/d2l_2p0/assignment`: Assignment dropboxes with gradebook
- `imsccv1p1/d2l_2p0/discussion`: Discussion forums with grading
- `imsqti_xmlv1p2/imscc_xmlv1p1/assessment`: QTI 1.2 + D2L extensions

### Pattern Prevention Focus
- **Pattern 15**: XML compliance for successful import
- **Pattern 19**: Educational structure consistency (per course outline)
- **Pattern 22**: Comprehensive educational content with authentic examples

## Implementation Requirements

### For the Brightspace-Packager Agent
1. **Must implement** all D2L XML schemas correctly
2. **Must validate** all four levels of compliance before packaging
3. **Must prevent** all identified failure patterns (15, 19, 22)
4. **Must ensure** functional integration with Brightspace tools

### For IMSCC Package Generation
1. **XML Schema Compliance**: QTI 1.2 + D2L extensions required
2. **Educational Structure**: Module count per course outline (dynamic)
3. **Content Quality**: Comprehensive educational depth required
4. **Tool Integration**: Assignment dropboxes, quizzes, discussions must function

## Validation Hierarchy

### Pre-Processing Validation
- Content quality meets Pattern 22 standards
- Educational structure matches course outline (Pattern 19)
- Source materials provide comprehensive theoretical foundation

### Processing Validation
- XML schema compliance throughout generation
- Brightspace namespace implementation
- Resource type accuracy and file reference integrity

### Post-Processing Validation
- Complete 4-level validation protocol execution
- Import compatibility verification
- Tool functionality confirmation

## Success Criteria

### Technical Success ✅
- IMSCC imports to Brightspace without errors
- All XML schemas validate correctly
- File references resolve properly
- Resource types match content formats

### Educational Success ✅
- Module structure matches course outline
- Comprehensive educational content depth achieved
- Theory-example integration proper
- Learning objectives alignment confirmed

### Functional Success ✅
- Assignment dropboxes create in gradebook
- Quiz assessments integrate with quiz tool
- Discussion forums enable graded participation
- Content displays properly in modules

## Usage for Brightspace-Packager Agent

The brightspace-packager agent should reference these documents to:

1. **Implement D2L schemas** using the XML schema reference
2. **Follow validation protocols** to ensure compliance
3. **Prevent failure patterns** using the agent guide
4. **Validate functionality** before package finalization

## Integration with Main IMSCC Standards

These Brightspace-specific standards extend the main IMSCC documentation in `/imscc-standards/` with:
- Brightspace-specific implementation details
- D2L extension schemas and usage
- Validation protocols for LMS compatibility
- Agent specialization guidance

## Related Documentation

- `/imscc-standards/CLAUDE.md`: General IMSCC standards overview
- `/imscc-standards/imscc-variables-reference.md`: Complete variable reference
- `/docs/PATTERN_PREVENTION_GUIDE.md`: Pattern prevention guidance
- `/CLAUDE.md`: Course generation standards

---

*This directory provides specialized knowledge for the brightspace-packager agent to create fully functional IMSCC packages compatible with Brightspace D2L environments.*