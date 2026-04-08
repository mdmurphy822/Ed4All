# Courseforge Troubleshooting Guide

Condensed lessons from 24 failure patterns identified during IMSCC package development.

---

## Critical Prevention Protocols

### Pre-Generation Validation (MANDATORY)

Before generating ANY IMSCC package:

- [ ] Schema/namespace consistency verified (IMSCC 1.3 recommended)
- [ ] All learning units contain substantial content (1500+ chars/overview)
- [ ] Zero placeholder content detected
- [ ] Module structure matches course outline (no artificial consolidation)
- [ ] Assessment XML uses QTI 1.2 / D2L formats
- [ ] Organization structure includes item hierarchy
- [ ] Assessments linked in organization (not just resources)
- [ ] Resource identifiers use `_R` suffix
- [ ] Content items sorted in pedagogical order
- [ ] **IMSCC version consistent across manifest AND all XML files (Pattern 24)**

---

## Pattern Quick Reference

| Pattern | Issue | Prevention |
|---------|-------|------------|
| 1 | Schema/namespace mismatch | Use IMSCC 1.3 consistently |
| 7 | Folder multiplication | Atomic single-file generation |
| 10 | Empty ZIP files | Validate package size >100KB |
| 14 | Mixed resource types | Standardize all resource declarations |
| 15 | Invalid assessment XML / Quiz questions not loading | Use QTI 1.2 with cc_profile metadata (see Pattern 15 section) |
| 17 | Empty organization | Include full item hierarchy |
| 19 | Single-page consolidation | Maintain structure per course outline |
| 20 | Version/namespace mismatch | Align schemaversion with namespace |
| 21 | Incomplete content | Validate all units before packaging |
| 22 | Educational Depth Deficiency | Ensure 600+ words, authentic examples with pedagogical context |
| 23 | Wrong resource identifier format | Use `_R` suffix for resource identifiers |
| 24 | IMSCC Version Mismatch | Match manifest version with content XML namespaces |
| 25 | Manifest title attributes | Use `<title>` child elements, not `title` attributes |
| 26 | Missing week index files | Week containers should NOT have identifierref |
| 27 | Non-standard CSS colors | Use official Courseforge palette only (see below) |

---

## Schema Consistency (Patterns 1, 14, 20, 24)

**Correct IMS CC 1.3 manifest (RECOMMENDED):**
```xml
<manifest xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imscp_v1p1"
          xmlns:lom="http://ltsc.ieee.org/xsd/imsccv1p3/LOM/resource"
          identifier="course_manifest">
  <metadata>
    <schema>IMS Common Cartridge</schema>
    <schemaversion>1.3.0</schemaversion>
  </metadata>
</manifest>
```

**Resource type standards (IMSCC 1.3):**
- Content: `type="webcontent"`
- Quizzes: `type="imsqti_xmlv1p2/imscc_xmlv1p3/assessment"`
- Assignments: `type="associatedcontent/imscc_xmlv1p3/learning-application-resource"`
- Discussions: `type="imsdt_xmlv1p3"`

---

## Assessment XML Format (Pattern 15)

**Wrong (custom format):**
```xml
<quiz identifier="quiz_week_01">
  <questions>...</questions>
</quiz>
```

**Correct (QTI 1.2 with CC metadata):**
```xml
<questestinterop xmlns="http://www.imsglobal.org/xsd/ims_qtiasiv1p2"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://www.imsglobal.org/xsd/ims_qtiasiv1p2
    http://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_qtiasiv1p2p1_v1p0.xsd">
  <assessment ident="quiz_week_01" title="Week 1 Quiz">
    <qtimetadata>
      <!-- REQUIRED for Brightspace to recognize as quiz -->
      <qtimetadatafield>
        <fieldlabel>cc_profile</fieldlabel>
        <fieldentry>cc.exam.v0p1</fieldentry>
      </qtimetadatafield>
      <qtimetadatafield>
        <fieldlabel>qmd_assessmenttype</fieldlabel>
        <fieldentry>Examination</fieldentry>
      </qtimetadatafield>
    </qtimetadata>
    <section ident="root_section">
      <item ident="question_1">
        <!-- REQUIRED for Brightspace to load questions -->
        <itemmetadata>
          <qtimetadata>
            <qtimetadatafield>
              <fieldlabel>cc_profile</fieldlabel>
              <fieldentry>cc.multiple_choice.v0p1</fieldentry>
            </qtimetadatafield>
            <qtimetadatafield>
              <fieldlabel>cc_weighting</fieldlabel>
              <fieldentry>5</fieldentry>
            </qtimetadatafield>
          </qtimetadata>
        </itemmetadata>
        <presentation>
          <material><mattext texttype="text/html">Question text</mattext></material>
          <response_lid ident="response1" rcardinality="Single">
            <render_choice>
              <response_label ident="A">
                <material><mattext texttype="text/html">Option A</mattext></material>
              </response_label>
            </render_choice>
          </response_lid>
        </presentation>
        <resprocessing>
          <outcomes>
            <decvar varname="SCORE" vartype="Integer" minvalue="0" maxvalue="5"/>
          </outcomes>
          <respcondition>
            <conditionvar>
              <varequal respident="response1">A</varequal>
            </conditionvar>
            <setvar action="Set" varname="SCORE">5</setvar>
          </respcondition>
        </resprocessing>
      </item>
    </section>
  </assessment>
</questestinterop>
```

**Critical CC Metadata Requirements:**
- `xmlns:xsi` and `xsi:schemaLocation` on root element
- `cc_profile` at assessment level (cc.exam.v0p1 or cc.quiz.v0p1)
- `itemmetadata` with `cc_profile` on each question item
- Question types: `cc.multiple_choice.v0p1`, `cc.true_false.v0p1`, `cc.fib.v0p1`, `cc.essay.v0p1`

**Fix Script**: `scripts/fix_quiz_metadata.py` can add missing metadata to existing packages

---

## Organization Structure (Patterns 17, 18)

**Wrong (empty organization):**
```xml
<organization identifier="ORG" structure="rooted-hierarchy">
</organization>
```

**Correct (full hierarchy with assessments):**
```xml
<organization identifier="ORG" structure="rooted-hierarchy">
  <item identifier="module_1_item">
    <title>Module 1: Introduction</title>
    <item identifier="module_1_overview_item" identifierref="module_1_overview_R">
      <title>Module Overview</title>
    </item>
    <item identifier="module_1_content_item" identifierref="module_1_content_R">
      <title>Content</title>
    </item>
    <item identifier="module_1_discussion_item" identifierref="discussion_module_1_R">
      <title>Module 1 Discussion</title>
    </item>
    <item identifier="module_1_assignment_item" identifierref="assignment_module_1_R">
      <title>Module 1 Assignment</title>
    </item>
    <item identifier="module_1_quiz_item" identifierref="quiz_module_1_R">
      <title>Module 1 Quiz</title>
    </item>
  </item>
</organization>
```

---

## Assessment Packaging (CRITICAL)

### Common Mistake: Assessments Missing from Navigation

Assessments must be included in BOTH the `<resources>` AND `<organization>` sections.

**Wrong (assessments only in resources):**
```xml
<resources>
  <resource identifier="quiz_1" type="imsqti_xmlv1p2/..." href="quiz_1.xml"/>
</resources>
<organization>
  <!-- Quiz not linked - won't appear in Brightspace navigation! -->
</organization>
```

**Correct (assessments in both sections):**
```xml
<resources>
  <resource identifier="quiz_module_1_R" type="imsqti_xmlv1p2/..." href="quiz_module_1.xml"/>
</resources>
<organization>
  <item identifier="module_1_item">
    <title>Module 1</title>
    <item identifier="quiz_module_1_item" identifierref="quiz_module_1_R">
      <title>Module 1 Quiz</title>
    </item>
  </item>
</organization>
```

### Resource Identifier Naming Convention

Brightspace expects resource identifiers to end with `_R` suffix:
- Organization item: `identifier="module_1_overview_item"`
- Resource: `identifier="module_1_overview_R"`
- Link: `identifierref="module_1_overview_R"`

### Content Ordering Within Modules

Items should be sorted in pedagogical order:
1. Overview
2. Learning Objectives
3. Content sections (numbered)
4. Summary/Self-check
5. Discussion
6. Assignment
7. Quiz

### Assessment-Module Association

Assessments are mapped to modules by:
1. Parsing week/module number from ID (e.g., `quiz_module_3_` → Module 3)
2. Checking course structure for explicit assignment
3. Defaulting to last module if no pattern detected

---

## Content Quality (Patterns 5, 16, 19, 21)

### Minimum Content Requirements

| Content Type | Minimum Words |
|--------------|---------------|
| Week overview | 400+ |
| Concept explanation | 600+ |
| Key concepts section | 300+ |
| Applications | 400+ |
| Study questions | 200+ |

### Placeholder Detection

Reject content containing:
- "Content will be developed"
- "TODO:" or "placeholder"
- "Coming soon"
- Generic text without subject matter

### Structure Validation

Each learning unit should contain modules appropriate to its complexity:
- Overview module (required)
- Content modules (1+ based on topic complexity)
- Key concepts/summary module
- Applications/examples module
- Assessment module (aligned with learning objectives)

**Note**: Module count is determined by course outline based on content scope - not a fixed number.

---

## Atomic Package Generation (Pattern 7)

**Critical constraints:**
- Generate exactly ONE .imscc file per execution
- No intermediate working directories
- Pre-flight validation: check export path doesn't exist
- Post-execution validation: confirm single output file
- Immediate cleanup on any error

```python
def validate_single_output(export_dir):
    files = os.listdir(export_dir)
    if len(files) != 1 or not files[0].endswith('.imscc'):
        raise SystemExit("Multiple files detected - aborting")
```

---

## IMSCC Version Mismatch (Pattern 24) - CRITICAL

### Symptoms
When Pattern 24 occurs in Brightspace:
- All content appears in ONE flat module instead of organized by week
- Discussion topics show resource IDs (e.g., "RES_W01_DISC_R") instead of titles
- Missing item titles throughout course structure
- Navigation hierarchy collapses to flat list

### Root Cause
Manifest declares one IMSCC version but content XMLs use different version namespaces.

**WRONG (version mismatch):**
```xml
<!-- Manifest says v1.1 -->
<schemaversion>1.1.0</schemaversion>
<resource type="imsdt_xmlv1p1" ...>

<!-- But discussion XML uses v1.3 namespace -->
<topic xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3">
```

**CORRECT (consistent v1.3):**
```xml
<!-- Manifest declares v1.3 -->
<schemaversion>1.3.0</schemaversion>
<resource type="imsdt_xmlv1p3" ...>

<!-- Discussion XML uses matching v1.3 namespace -->
<topic xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3">
```

### Version Alignment Requirements

| Component | IMSCC 1.3 (Recommended) |
|-----------|------------------------|
| Manifest xmlns | `imsccv1p3/imscp_v1p1` |
| Schema version | `1.3.0` |
| Discussion type | `imsdt_xmlv1p3` |
| Discussion XML | `imsccv1p3/imsdt_v1p3` |
| Quiz type | `imsqti_xmlv1p2/imscc_xmlv1p3/assessment` |
| Assignment type | `assignment_xmlv1p0` |

### Validation Command
```bash
# Check for version mismatches (should not find v1p1 in a v1p3 package)
grep -r "imsccv1p1" *.xml  # Should return NO results for v1.3 packages
grep -r "imsccv1p3" *.xml  # Should find all IMSCC references
```

---

## Wrong Assignment Resource Type (Pattern 25) - CRITICAL

### Symptoms
When Pattern 25 occurs in Brightspace:
- Assignments import as raw .xml files instead of native Brightspace assignments
- Assignment dropboxes not created in Brightspace
- No assignment submission functionality available to students
- Module contains "assignment_week_XX.xml" file link instead of assignment tool

### Root Cause
Manifest uses wrong resource type for assignments. The type `associatedcontent/imscc_xmlv1p3/learning-application-resource` tells Brightspace to treat assignments as generic associated content files rather than native assignments.

**WRONG (imports as XML file):**
```xml
<resource identifier="RES_W01_ASSIGN_R"
    type="associatedcontent/imscc_xmlv1p3/learning-application-resource"
    href="week_01/assignment_week_01.xml">
```

**CORRECT (creates native Brightspace assignment):**
```xml
<resource identifier="RES_W01_ASSIGN_R"
    type="assignment_xmlv1p0"
    href="week_01/assignment_week_01.xml">
```

### Assignment XML Namespace
Assignment XML files MUST use the correct IMSCC extension namespace:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<assignment xmlns="http://www.imsglobal.org/xsd/imscc_extensions/assignment"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    identifier="assignment_week_01"
    xsi:schemaLocation="http://www.imsglobal.org/xsd/imscc_extensions/assignment
    http://www.imsglobal.org/profile/cc/cc_extensions/cc_extresource_assignmentv1p0_v1p0.xsd">
  <title>Assignment Title</title>
  <text texttype="text/html"><![CDATA[...]]></text>
  <gradable>true</gradable>
  <submission_types>
    <submission_type>file_upload</submission_type>
  </submission_types>
  <points_possible>100</points_possible>
</assignment>
```

### IMPORTANT: Do NOT include D2L namespace references
Avoid d2l: prefixed elements like `<d2l:dropbox_type>` or `<d2l:rubric>` in assignment XMLs - these cause parsing errors when the d2l namespace is not declared.

### Validation Command
```bash
# Check for wrong assignment resource types
grep -r "learning-application-resource" imsmanifest.xml  # Should return NO results
grep -r "assignment_xmlv1p0" imsmanifest.xml  # Should find all assignments
```

---

## Missing Title Elements on Container Items (Pattern 26) - CRITICAL

### Symptoms
When Pattern 26 occurs in Brightspace:
- Modules display as "imported module" instead of proper names like "Week 1: Introduction..."
- Course structure imports but navigation shows generic labels
- Week/unit titles are missing despite being in the manifest

### Root Cause
Container items (weeks, units, modules) use `title` **attributes** but lack `<title>` **child elements**. Brightspace ignores `title` attributes and only reads `<title>` elements.

**WRONG (title attribute only - Brightspace ignores it):**
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

### IMS CC Specification
The IMS Common Cartridge specification requires `<title>` child elements for organization items, not `title` attributes. While both may be valid XML, Brightspace only processes the child element.

### Fix
Convert all container items to use `<title>` child elements:
1. Remove `title` attribute from `<item>` tag
2. Add `<title>` child element as first child inside `<item>`
3. Ensure ALL organizational levels have `<title>` elements (course, weeks, modules)

### Validation Command
```bash
# Check for items with title attributes (potential issues)
grep -E '<item[^>]+title=' imsmanifest.xml

# Verify title elements exist
grep -E '<item[^>]*>\s*<title>' imsmanifest.xml  # Should find container items
```

---

## Non-Standard CSS Colors (Pattern 27)

### Symptoms
When Pattern 27 occurs:
- Visual inconsistency with Courseforge branding
- Hot pink, purple, or other non-standard colors in generated content
- Practice activity sections using wrong color gradients
- Module headers don't match official brand colors

### Root Cause
Content-generator creates CSS with non-standard colors that aren't from the official Courseforge palette. This typically happens when:
- LLM invents arbitrary colors during generation
- Old templates use purple gradients instead of official blue
- Bootstrap default colors used instead of Courseforge palette

### Official Courseforge Color Palette

**MANDATORY - Use ONLY these colors:**
```css
/* Primary Colors */
Primary Blue: #2c5aa0      /* Headers, links, primary actions */
Secondary Blue: #1a3d6e    /* Darker shade for gradients */

/* Semantic Colors */
Success Green: #28a745     /* Positive feedback, completion */
Warning Yellow: #ffc107    /* Caution, attention needed */
Danger Red: #dc3545        /* Errors, important notices */

/* Neutral Colors */
Light Gray: #f8f9fa        /* Backgrounds */
Border Gray: #e0e0e0       /* Borders, dividers */
```

### Forbidden Colors

**NEVER USE these colors:**
- `#667eea`, `#764ba2` - Purple gradients (from old templates)
- `#f093fb`, `#f5576c` - Hot pink/magenta (model-invented)
- `#007bff`, `#17a2b8` - Bootstrap defaults (use Courseforge blue)

### Correct CSS Gradient Pattern
```css
/* CORRECT - Official Courseforge gradient */
.module-header, .practice-activity, .summary-box {
    background: linear-gradient(135deg, #2c5aa0 0%, #1a3d6e 100%);
}

/* WRONG - Non-standard purple gradient */
.module-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); /* DO NOT USE */
}

/* WRONG - Hot pink gradient */
.practice-activity {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); /* DO NOT USE */
}
```

### Validation Command
```bash
# Check for forbidden colors in HTML/CSS files
grep -r "#667eea\|#764ba2\|#f093fb\|#f5576c\|#007bff" *.html
# Should return NO results - all colors should be from official palette
```

---

## Common Import Errors

### "Illegal XML, unable to load"
- Check schema/namespace consistency
- Verify all resource types are valid
- Validate assessment XML format
- **Check for IMSCC version mismatch (Pattern 24)**

### "Schema version does not match XML Namespace"
- Align `<schemaversion>` with xmlns namespace
- Use 1.3.0 with imsccv1p3 (recommended)

### "Number of content items converted: 0"
- Check organization structure has item hierarchy
- Verify resources are properly linked
- Ensure content files exist and are referenced

### "Plugin not found"
- Use Brightspace-compatible resource types
- Include proper D2L metadata for assessments

---

## Validation Checklist

### Before Packaging
- [ ] All HTML files contain substantial content
- [ ] No placeholder text in any file
- [ ] All learning units have appropriate depth for their topics
- [ ] Assessment XML validates against QTI schema
- [ ] Manifest organization has full hierarchy
- [ ] All file references resolve correctly

### After Packaging
- [ ] Package size >100KB (substantial content)
- [ ] Single .imscc file generated
- [ ] ZIP structure valid
- [ ] Test import in Brightspace dev environment

---

## Success Criteria

A properly generated IMSCC package will:
1. Import without XML errors
2. Create navigation structure with all weeks
3. Display content within modules
4. Create native assessment tools (quizzes, assignments)
5. Integrate with Brightspace gradebook
6. Render properly on mobile devices

---

*For complete pattern documentation, see [PATTERN_PREVENTION_GUIDE.md](PATTERN_PREVENTION_GUIDE.md)*
