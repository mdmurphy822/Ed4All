# Pattern Prevention Guide

Comprehensive documentation of all 22 identified failure patterns in IMSCC package generation and course development, with prevention strategies and validation procedures.

## Overview

This guide documents patterns that can cause IMSCC import failures, accessibility issues, or educational quality problems. Each pattern includes detection methods, prevention strategies, and remediation steps.

---

## Pattern Categories

| Category | Patterns | Impact |
|----------|----------|--------|
| Content Quality | 1, 4, 5, 7, 16, 22 | Educational effectiveness |
| Technical/Manifest | 2, 3, 6, 8, 9, 10, 13 | Import failures |
| Accessibility | 14, 15, 17, 18 | WCAG 2.2 compliance |
| Assessment | 11, 12, 19, 20 | Grading functionality |
| Structure | 21 | Navigation/organization |

---

## Critical Patterns (Import Failures)

### Pattern 1: Placeholder Content
**Severity**: CRITICAL
**Impact**: Imports successfully but provides no educational value

**Detection**:
- Text matching: "Lorem ipsum", "TODO", "placeholder", "[insert content]"
- Minimum word count violations (< 200 words per content file)
- Empty or near-empty HTML body elements

**Prevention**:
- Content generator must produce minimum 500 words per learning unit
- Quality gate validates word count before packaging
- No template variables allowed in final output

**Validation**:
```python
def check_placeholder_content(html_content):
    placeholders = ['lorem ipsum', 'todo', 'placeholder', '[insert']
    text = html_content.lower()
    return not any(p in text for p in placeholders)
```

---

### Pattern 2: Missing Manifest
**Severity**: CRITICAL
**Impact**: Package cannot be imported

**Detection**:
- File `imsmanifest.xml` not present at package root
- Empty or malformed XML declaration

**Prevention**:
- Package assembly validates manifest existence
- Schema validation before ZIP creation

---

### Pattern 3: Invalid XML Schema
**Severity**: CRITICAL
**Impact**: LMS rejects package during import

**Detection**:
- Missing namespace declarations
- Incorrect schema version
- Malformed XML structure

**Prevention**:
- Use IMS CC 1.2.0 namespace: `http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1`
- Validate against official XSD before packaging

**Correct Manifest Header**:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<manifest identifier="course_id" version="1.2.0"
    xmlns="http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1"
    xmlns:lom="http://ltsc.ieee.org/xsd/imsccv1p2/LOM/resource"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
```

---

### Pattern 6: Missing Resource Files
**Severity**: CRITICAL
**Impact**: Broken content links after import

**Detection**:
- Manifest references files not included in package
- File paths in manifest don't match actual file locations

**Prevention**:
- Pre-packaging validation of all manifest resource references
- Automated file existence checking

---

### Pattern 8: Duplicate Resource Identifiers
**Severity**: CRITICAL
**Impact**: Unpredictable import behavior

**Detection**:
- Multiple `<resource>` elements with same `identifier` attribute
- Multiple `<item>` elements with same `identifier`

**Prevention**:
- Unique identifier generation using UUID or hash-based approach
- Pre-packaging duplicate detection

---

### Pattern 9: Circular Dependencies
**Severity**: HIGH
**Impact**: Import timeout or failure

**Detection**:
- Resource A depends on B, B depends on A
- Chain dependencies forming cycles

**Prevention**:
- Dependency graph validation before packaging
- Topological sort verification

---

### Pattern 10: Invalid File Paths
**Severity**: CRITICAL
**Impact**: Resources not accessible after import

**Detection**:
- Absolute paths in manifest or HTML
- Paths with special characters
- Case sensitivity mismatches

**Prevention**:
- All paths must be relative
- Use only alphanumeric, underscore, hyphen in filenames
- Consistent lowercase naming

---

### Pattern 13: Brightspace XML Compatibility
**Severity**: CRITICAL
**Impact**: "Illegal XML" import errors

**Detection**:
- Non-standard assessment XML schemas
- Missing QTI namespace declarations
- Invalid D2L resource types

**Prevention**:
- Use `imsqti_xmlv1p2` for quizzes
- Use `imsccv1p1/d2l_2p0/assignment` for assignments
- Use `imsccv1p1/d2l_2p0/discussion` for discussions

**Correct Resource Types**:
```xml
<!-- Quiz -->
<resource identifier="quiz_01" type="imsqti_xmlv1p2/imscc_xmlv1p1/assessment" href="quiz_01.xml">

<!-- Assignment -->
<resource identifier="assign_01" type="imsccv1p1/d2l_2p0/assignment" href="assign_01.xml">

<!-- Discussion -->
<resource identifier="disc_01" type="imsccv1p1/d2l_2p0/discussion" href="disc_01.xml">
```

---

## Content Quality Patterns

### Pattern 4: Missing Learning Objectives
**Severity**: HIGH
**Impact**: Poor educational alignment

**Detection**:
- Content files without explicit learning objectives
- Objectives not using action verbs (Bloom's taxonomy)

**Prevention**:
- Every module requires 3-5 measurable objectives
- Objectives must start with action verbs

---

### Pattern 5: Inconsistent Terminology
**Severity**: MEDIUM
**Impact**: Student confusion

**Detection**:
- Same concept referred to by different names
- Undefined acronyms

**Prevention**:
- Maintain terminology glossary
- Acronym first-use expansion rule

---

### Pattern 7: Insufficient Depth
**Severity**: HIGH
**Impact**: Superficial learning

**Detection**:
- Content below minimum word counts
- Missing examples or applications
- No practice opportunities

**Prevention**:
- Minimum 500 words per learning unit
- Required sections: concept, example, application, practice

---

### Pattern 16: Surface-Level Content (Pattern 16.2)
**Severity**: HIGH
**Impact**: Fails educational quality review

**Detection**:
- Generic descriptions without specifics
- Missing industry context
- No real-world examples

**Prevention**:
- Content must include specific examples
- Domain-specific terminology required
- Real-world application scenarios

---

### Pattern 22: Educational Depth Deficiency
**Severity**: CRITICAL
**Impact**: Course fails quality review

**Detection**:
- Content reads like outline rather than instruction
- Missing explanations of "why" and "how"
- No authentic learning scenarios

**Prevention**:
- Minimum educational density metrics
- Required pedagogical elements per module
- Subject matter accuracy validation

---

## Accessibility Patterns

### Pattern 14: Missing Alt Text
**Severity**: HIGH
**Impact**: WCAG 2.2 AA failure

**Detection**:
- `<img>` elements without `alt` attribute
- Empty alt text on non-decorative images
- Generic alt text ("image", "photo")

**Prevention**:
- All images require descriptive alt text
- Decorative images use `alt=""`
- Alt text describes image purpose, not just content

---

### Pattern 15: Heading Hierarchy Violations
**Severity**: MEDIUM
**Impact**: Screen reader navigation issues

**Detection**:
- Skipped heading levels (h1 → h3)
- Multiple h1 elements per page
- Headings used for styling only

**Prevention**:
- Sequential heading levels only
- Single h1 per page
- Use CSS for styling, headings for structure

---

### Pattern 17: Insufficient Color Contrast
**Severity**: HIGH
**Impact**: WCAG 2.2 AA failure

**Detection**:
- Text/background ratio below 4.5:1
- Large text ratio below 3:1
- Information conveyed by color alone

**Prevention**:
- Use approved color palette
- Validate contrast ratios
- Add non-color indicators (icons, patterns)

---

### Pattern 18: Missing Form Labels
**Severity**: HIGH
**Impact**: Forms inaccessible to screen readers

**Detection**:
- `<input>` without associated `<label>`
- `<select>` without label
- Missing fieldset/legend for groups

**Prevention**:
- Every form input requires explicit label
- Use `for` attribute matching input `id`
- Group related inputs with fieldset

---

## Assessment Patterns

### Pattern 11: Incomplete Assessment Definitions
**Severity**: HIGH
**Impact**: Grading functionality broken

**Detection**:
- Missing point values
- No rubric definitions
- Undefined submission types

**Prevention**:
- All assessments require point values
- Rubrics for written assignments
- Clear submission instructions

---

### Pattern 12: Invalid QTI Structure
**Severity**: CRITICAL
**Impact**: Quizzes don't function

**Detection**:
- Missing `<assessment>` element
- Invalid `<item>` structure
- Missing response processing

**Prevention**:
- Use QTI 1.2 schema validation
- Include all required metadata fields
- Test quiz import before packaging

---

### Pattern 19: Missing Answer Keys
**Severity**: MEDIUM
**Impact**: Auto-grading unavailable

**Detection**:
- Quiz questions without correct answer identification
- Missing feedback for answers

**Prevention**:
- All objective questions require correct answer
- Include answer feedback

---

### Pattern 20: Point Value Mismatches
**Severity**: MEDIUM
**Impact**: Gradebook calculation errors

**Detection**:
- Assessment points don't sum to course total
- Weight percentages exceed 100%

**Prevention**:
- Validate point totals across course
- Weight validation before packaging

---

## Structure Pattern

### Pattern 21: Navigation Inconsistency
**Severity**: MEDIUM
**Impact**: Student confusion, poor UX

**Detection**:
- Inconsistent module naming
- Missing next/previous links
- Orphaned content pages

**Prevention**:
- Standardized naming conventions
- Navigation validation
- All content reachable from TOC

---

## Validation Checklist

### Pre-Packaging Validation
- [ ] All content files meet minimum word counts
- [ ] No placeholder text detected
- [ ] All manifest references resolve to files
- [ ] No duplicate identifiers
- [ ] All images have alt text
- [ ] Heading hierarchy is correct
- [ ] Color contrast meets WCAG 2.2 requirements
- [ ] All assessments have point values
- [ ] QTI schema validation passes
- [ ] No circular dependencies

### Post-Import Validation
- [ ] All content displays correctly
- [ ] Navigation functions properly
- [ ] Assessments accept submissions
- [ ] Grades calculate correctly
- [ ] Mobile responsive rendering

---

## Integration with Quality Assurance Agent

The quality-assurance agent validates against all 22 patterns before packaging. It should be invoked:

1. After content generation (patterns 1, 4, 5, 7, 16, 22)
2. Before manifest generation (patterns 2, 3, 6, 8, 9, 10)
3. After assessment creation (patterns 11, 12, 13, 19, 20)
4. During final packaging (all patterns)

---

## References

- [WCAG 2.2 AA Guidelines](https://www.w3.org/WAI/WCAG22/quickref/)
- [IMS Common Cartridge 1.2 Specification](https://www.imsglobal.org/cc/index.html)
- [QTI 1.2 Specification](https://www.imsglobal.org/question/qtiv1p2/imsqti_asi_bindv1p2.html)
- [Brightspace Import Requirements](https://community.d2l.com/)
