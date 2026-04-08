# Accessibility Remediation Agent Specification

## Overview

The `accessibility-remediation` is a specialized subagent that automatically fixes accessibility issues in HTML content to achieve WCAG 2.2 AA compliance. It transforms course content into fully accessible educational materials without manual intervention.

## Agent Type Classification

- **Agent Type**: `accessibility-remediation` (specialized auto-fix subagent)
- **Primary Function**: Automatic WCAG 2.2 AA compliance remediation
- **Workflow Position**: Post-analysis phase (after content-analyzer and dart-automation-coordinator)
- **Integration**: Receives issue manifests, outputs compliant HTML, feeds to intelligent-design-mapper

## Core Capabilities

### 1. Automatic Alt Text Generation
AI-powered image description generation:

| Image Type | Alt Text Strategy |
|------------|-------------------|
| **Diagrams** | Describe structure, relationships, and key information |
| **Charts/Graphs** | Summarize data trends, include key values |
| **Photos** | Describe relevant content and context |
| **Icons** | Describe function or meaning |
| **Decorative** | Apply `alt=""` and `role="presentation"` |
| **Complex** | Generate brief alt + long description |

### 2. Heading Structure Correction
Automatic heading hierarchy repair:

```html
<!-- Before -->
<h1>Course Title</h1>
<h4>Week 1</h4>  <!-- Skipped h2, h3 -->
<h2>Topic A</h2>

<!-- After -->
<h1>Course Title</h1>
<h2>Week 1</h2>  <!-- Corrected level -->
<h3>Topic A</h3> <!-- Adjusted to maintain hierarchy -->
```

### 3. ARIA Enhancement
Strategic ARIA attribute application:

| Element Type | ARIA Enhancement |
|--------------|------------------|
| **Navigation** | `role="navigation"`, `aria-label` |
| **Main content** | `role="main"`, landmark identification |
| **Regions** | `role="region"`, `aria-labelledby` |
| **Interactive** | `aria-expanded`, `aria-controls` |
| **Live regions** | `aria-live`, `aria-atomic` |

### 4. Color Contrast Remediation
Automatic color adjustment for WCAG compliance:

```css
/* Before: 2.5:1 contrast ratio */
.text { color: #999; background: #fff; }

/* After: 4.5:1+ contrast ratio */
.text { color: #595959; background: #fff; }
```

### 5. Form Accessibility
Comprehensive form remediation:

```html
<!-- Before -->
<input type="text" placeholder="Name">

<!-- After -->
<label for="name-input">Full Name <span class="required">(required)</span></label>
<input type="text" id="name-input" name="name" required
       aria-describedby="name-help" autocomplete="name">
<span id="name-help" class="help-text">Enter your first and last name</span>
```

### 6. Table Accessibility
Table structure enhancement:

```html
<!-- Before -->
<table>
  <tr><td>Header1</td><td>Header2</td></tr>
  <tr><td>Data1</td><td>Data2</td></tr>
</table>

<!-- After -->
<table>
  <caption>Description of table contents</caption>
  <thead>
    <tr><th scope="col">Header1</th><th scope="col">Header2</th></tr>
  </thead>
  <tbody>
    <tr><td>Data1</td><td>Data2</td></tr>
  </tbody>
</table>
```

### 7. Skip Links & Keyboard Navigation
Navigation enhancement:

```html
<!-- Added at document start -->
<a href="#main-content" class="skip-link">Skip to main content</a>
<a href="#nav" class="skip-link">Skip to navigation</a>

<!-- Focus management -->
<style>
  :focus { outline: 2px solid #2c5aa0; outline-offset: 2px; }
  .skip-link:not(:focus) {
    position: absolute;
    width: 1px; height: 1px;
    clip: rect(0,0,0,0);
  }
</style>
```

### 8. Language Declaration
Document and content language markup:

```html
<!-- Document language -->
<html lang="en">

<!-- Content in different language -->
<p>The French term <span lang="fr">joie de vivre</span> means joy of living.</p>
```

## Workflow Protocol

### Phase 1: Issue Intake
```
Input: Remediation manifest from content-analyzer
Process:
  1. Parse accessibility issues by file
  2. Categorize by fix type
  3. Determine fix order (dependencies)
  4. Create fix queue
Output: Ordered fix task list
```

### Phase 2: Automated Fixes
```
Input: Fix task list + HTML files
Process:
  1. Load HTML file into DOM
  2. Apply fixes in order:
     a. Structure fixes (headings, landmarks)
     b. Content fixes (alt text, labels)
     c. Style fixes (contrast, focus)
     d. Enhancement fixes (ARIA, skip links)
  3. Validate fixes don't break content
  4. Save remediated HTML
Output: Fixed HTML files
```

### Phase 3: Validation
```
Input: Remediated HTML files
Process:
  1. Run WCAG 2.2 AA compliance check
  2. Verify all reported issues fixed
  3. Check for regression issues
  4. Validate HTML structure
Output: Compliance report with any remaining issues
```

### Phase 4: Documentation
```
Input: Fix history + validation results
Process:
  1. Generate change log per file
  2. Create accessibility statement
  3. Document any manual review needs
  4. Update course metadata
Output: Remediation report + accessibility documentation
```

## Fix Algorithms

### Alt Text Generation
```python
def generate_alt_text(image_path, context):
    """Generate descriptive alt text using AI analysis"""

    # Analyze image content
    image_analysis = analyze_image(image_path)

    # Consider surrounding context
    context_keywords = extract_keywords(context)

    # Determine image purpose
    purpose = determine_purpose(image_analysis, context_keywords)

    if purpose == "decorative":
        return ""
    elif purpose == "functional":
        return describe_function(image_analysis)
    elif purpose == "informative":
        return describe_content(image_analysis, context)
    elif purpose == "complex":
        return {
            "alt": brief_description(image_analysis),
            "longdesc": detailed_description(image_analysis)
        }
```

### Heading Hierarchy Repair
```python
def fix_heading_hierarchy(html_content):
    """Correct heading level sequence"""

    headings = extract_headings(html_content)
    current_level = 0

    for heading in headings:
        if heading.level == 1:
            current_level = 1
        elif heading.level > current_level + 1:
            # Fix: adjust to one level deeper than current
            new_level = current_level + 1
            html_content = replace_heading_level(
                html_content, heading, new_level
            )
            current_level = new_level
        else:
            current_level = heading.level

    return html_content
```

### Contrast Ratio Fix
```python
def fix_contrast(element, background_color):
    """Adjust foreground color to meet WCAG AA contrast"""

    current_color = get_foreground_color(element)
    current_ratio = calculate_contrast(current_color, background_color)

    # Target ratio: 4.5:1 for normal text, 3:1 for large
    target_ratio = 3.0 if is_large_text(element) else 4.5

    if current_ratio >= target_ratio:
        return current_color

    # Darken or lighten to achieve contrast
    new_color = adjust_color_for_contrast(
        current_color, background_color, target_ratio
    )

    return new_color
```

## Output Format

### Remediation Report (JSON)
```json
{
  "remediation_summary": {
    "files_processed": 156,
    "issues_fixed": 234,
    "issues_remaining": 3,
    "manual_review_needed": 5
  },
  "fixes_applied": {
    "alt_text_added": 67,
    "headings_corrected": 45,
    "aria_enhanced": 89,
    "contrast_fixed": 23,
    "forms_labeled": 12,
    "tables_fixed": 8,
    "skip_links_added": 156,
    "language_declared": 156
  },
  "file_changes": [
    {
      "file": "week1/overview.html",
      "fixes": [
        {"type": "alt_text", "element": "img#diagram1", "value": "Network topology diagram showing..."},
        {"type": "heading_level", "from": "h4", "to": "h2"}
      ]
    }
  ],
  "manual_review": [
    {
      "file": "week3/complex_table.html",
      "issue": "complex_table_structure",
      "reason": "Table requires human judgment for header relationships"
    }
  ],
  "wcag_compliance": {
    "level_a": "100%",
    "level_aa": "100%",
    "remaining_issues": []
  }
}
```

## Agent Invocation

### From Orchestrator
```python
Task(
    subagent_type="accessibility-remediation",
    description="Fix accessibility issues",
    prompt="""
    Apply automatic accessibility fixes to all HTML content.

    Input:
    - Remediation manifest: /workspace/remediation_queue.json
    - HTML content: /workspace/content/

    Output: /workspace/remediated_content/

    Requirements:
    1. Fix all alt text issues (generate AI descriptions)
    2. Correct heading hierarchy
    3. Add ARIA landmarks and roles
    4. Fix color contrast issues
    5. Enhance form accessibility
    6. Add skip navigation links
    7. Validate WCAG 2.2 AA compliance
    8. Generate remediation report

    Return: Compliance summary with fix counts
    """
)
```

## Quality Gates

### Pre-Remediation
- [ ] All source files accessible
- [ ] Remediation manifest valid
- [ ] Image analysis capability available
- [ ] Contrast calculation accurate

### Post-Remediation
- [ ] WCAG 2.2 AA compliance: 100%
- [ ] No new issues introduced
- [ ] HTML validation passed
- [ ] Visual appearance preserved
- [ ] Functionality maintained

## WCAG Fixes Implemented

### Perceivable
- **1.1.1**: Auto-generate alt text for images
- **1.3.1**: Add semantic structure and ARIA
- **1.3.2**: Ensure logical reading order
- **1.4.3**: Fix color contrast (4.5:1 minimum)
- **1.4.4**: Support text resize without loss
- **1.4.11**: Fix non-text contrast

### Operable
- **2.1.1**: Enable full keyboard access
- **2.4.1**: Add skip navigation links
- **2.4.2**: Add descriptive page titles
- **2.4.4**: Improve link purpose clarity
- **2.4.6**: Add descriptive headings

### Understandable
- **3.1.1**: Declare page language
- **3.1.2**: Declare content language changes
- **3.2.1**: Predictable on focus behavior
- **3.3.2**: Add form labels and instructions

### Robust
- **4.1.1**: Fix HTML parsing errors
- **4.1.2**: Add name/role/value for controls

## Performance Targets

| Metric | Target |
|--------|--------|
| Fix application speed | <2 seconds per file |
| WCAG compliance rate | 100% after remediation |
| Fix accuracy | 98%+ appropriate fixes |
| Regression rate | <1% |
| Total remediation | <10 minutes for 200-file course |

---

*This agent ensures 100% WCAG 2.2 AA compliance for all course content through intelligent automated remediation, supporting Courseforge's accessibility-first mission.*
