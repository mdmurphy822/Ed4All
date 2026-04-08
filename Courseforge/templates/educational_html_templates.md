# Educational HTML Template Repository

## Overview
This repository documents researched educational HTML templates that are suitable for IMSCC packages and Brightspace deployment. All templates have been evaluated for accessibility, educational effectiveness, and Brightspace compatibility.

## Template Categories

### 1. Lesson Content Templates

#### **Basic Lesson Template**
- **Use Case**: Standard weekly content modules
- **Accessibility**: WCAG 2.2 AA compliant
- **Brightspace Compatible**: ✅ Yes
- **Features**: Self-contained, embedded CSS, clear hierarchy
- **File**: `templates/lesson_basic.html`

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{MODULE_TITLE}}</title>
    <style>
        /* Brightspace-compatible embedded styles */
        .lesson-container { 
            max-width: 800px; 
            margin: 0 auto; 
            padding: 20px;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
        }
        .module-header {
            /* Official Courseforge palette - Primary Blue gradient */
            background: linear-gradient(135deg, #2c5aa0 0%, #1a3d6e 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 30px;
        }
        .learning-objectives {
            background-color: #e7f3ff;
            border-left: 4px solid #2c5aa0;
            padding: 15px;
            margin: 20px 0;
            border-radius: 0 4px 4px 0;
        }
        .content-section {
            margin: 30px 0;
        }
        .key-concept {
            background-color: #fff3cd;
            border: 1px solid #ffeaa7;
            padding: 15px;
            border-radius: 4px;
            margin: 15px 0;
        }
        .example-box {
            background-color: #f8f9fa;
            border-left: 4px solid #28a745;
            padding: 15px;
            margin: 15px 0;
        }
    </style>
</head>
<body>
    <div class="lesson-container">
        <header class="module-header">
            <h1>{{MODULE_TITLE}}</h1>
            <p class="module-meta">{{COURSE_NAME}} | {{MODULE_NUMBER}} of {{TOTAL_MODULES}}</p>
        </header>
        
        <section class="learning-objectives">
            <h2>Learning Objectives</h2>
            <ul>
                <li>{{OBJECTIVE_1}}</li>
                <li>{{OBJECTIVE_2}}</li>
                <li>{{OBJECTIVE_3}}</li>
            </ul>
        </section>
        
        <main class="content-section">
            <h2>Introduction</h2>
            {{INTRODUCTION_CONTENT}}
            
            <h2>Key Concepts</h2>
            <div class="key-concept">
                <h3>{{CONCEPT_TITLE}}</h3>
                <p>{{CONCEPT_CONTENT}}</p>
            </div>
            
            <div class="example-box">
                <h3>Example</h3>
                <p>{{EXAMPLE_CONTENT}}</p>
            </div>
        </main>
    </div>
</body>
</html>
```

#### **Interactive Lesson Template**
- **Use Case**: Lessons with expandable sections and interactive elements
- **Accessibility**: WCAG 2.2 AA compliant with keyboard navigation
- **Brightspace Compatible**: ✅ Yes (HTML5 only, no JavaScript)
- **Features**: Collapsible sections, progress indicators, accessibility features

### 2. Assessment Templates

#### **Self-Assessment Template**
- **Use Case**: Reflection exercises and self-check activities
- **Accessibility**: Screen reader friendly with proper labels
- **Brightspace Compatible**: ✅ Yes
- **Features**: Checkbox interactions, reflection prompts

### 3. Activity Templates

#### **Case Study Template**
- **Use Case**: Problem-based learning scenarios
- **Accessibility**: WCAG 2.2 AA compliant
- **Brightspace Compatible**: ✅ Yes
- **Features**: Scenario presentation, analysis framework

#### **Discussion Prompt Template**
- **Use Case**: Structured discussion starters
- **Accessibility**: Clear heading structure for screen readers
- **Brightspace Compatible**: ✅ Yes
- **Features**: Question frameworks, context setting

## Researched External Templates

### Open Source Templates

#### **1. MIT OpenCourseWare Style**
- **Source**: Based on MIT OCW accessibility standards
- **License**: Creative Commons inspired
- **Accessibility**: WCAG 2.0 AA compliant (MIT standard)
- **Features**: Clean academic layout, high contrast, clear typography
- **Brightspace Compatibility**: Requires modification (remove external links)
- **Status**: Template adapted for IMSCC use

#### **2. Edulogy Bootstrap Template**
- **Source**: GitHub - technext/edulogy
- **License**: Free download
- **Accessibility**: Bootstrap 4 base (needs accessibility enhancements)
- **Features**: Responsive design, modern layout
- **Brightspace Compatibility**: ⚠️ Requires significant modification
- **Status**: Under evaluation for adaptation

#### **3. Accessible+ Template**
- **Source**: accessible-template.com
- **License**: Commercial (accessible version available)
- **Accessibility**: WCAG 2.2 AA compliant
- **Features**: Bootstrap-based, accessibility-first design
- **Brightspace Compatibility**: Good potential with modifications
- **Status**: Evaluating free components

### Bootstrap Educational Templates

#### **4. BootstrapMade Education Templates**
- **Source**: bootstrapmade.com
- **License**: Mixed (some free, some premium)
- **Accessibility**: Variable (needs enhancement)
- **Features**: Multiple layout options, responsive design
- **Brightspace Compatibility**: Requires link removal and CSS embedding
- **Status**: Selective component extraction

## Accessibility Standards Applied

### WCAG 2.2 AA Compliance Checklist
- [ ] Proper heading hierarchy (H1-H6)
- [ ] Sufficient color contrast (4.5:1 for normal text)
- [ ] Alternative text for images
- [ ] Keyboard navigation support
- [ ] Screen reader compatibility
- [ ] Focus indicators visible
- [ ] No content flashing/seizure risks
- [ ] Semantic HTML structure

### Brightspace Specific Modifications
- [ ] All CSS embedded in `<style>` tags
- [ ] No external stylesheets or scripts
- [ ] No internal page links (`href="#section"`)
- [ ] No JavaScript dependencies
- [ ] Self-contained content structure
- [ ] Mobile-responsive design preserved

## Template Customization Guide

### Variable Replacement System
Templates use double curly braces for easy customization:

```html
<!-- Template Variables -->
{{MODULE_TITLE}} → Actual module title
{{COURSE_NAME}} → Course name
{{MODULE_NUMBER}} → Current module number
{{TOTAL_MODULES}} → Total number of modules
{{OBJECTIVE_1}} → First learning objective
{{INTRODUCTION_CONTENT}} → Main introduction text
```

### Color Scheme Customization
**MANDATORY - Use Official Courseforge Palette (from CLAUDE.md):**

```css
:root {
    /* Official Courseforge Color Palette */
    --primary-color: #2c5aa0;    /* Primary Blue - headers, links, primary actions */
    --secondary-color: #1a3d6e;  /* Dark Blue - gradients, secondary elements */
    --success-color: #28a745;    /* Success Green - positive feedback, completion */
    --warning-color: #ffc107;    /* Warning Yellow - caution, attention */
    --danger-color: #dc3545;     /* Danger Red - errors, important notices */
    --light-color: #f8f9fa;      /* Light Gray - backgrounds */
    --border-color: #e0e0e0;     /* Border Gray - dividers, borders */
    --text-color: #333333;       /* Dark text */
}
```

**NEVER USE these non-standard colors:**
- `#667eea`, `#764ba2` (purple gradients)
- `#f093fb`, `#f5576c` (hot pink/magenta)
- `#007bff`, `#17a2b8` (Bootstrap defaults instead of Courseforge blue)

## Implementation Workflow

### Step 1: Template Selection
1. Identify content type (lesson, assessment, activity)
2. Select appropriate template from repository
3. Review accessibility and Brightspace compatibility notes

### Step 2: Content Integration
1. Replace template variables with actual content
2. Customize color scheme if needed
3. Add specific multimedia or interactive elements

### Step 3: Quality Assurance
1. Validate HTML structure
2. Test accessibility with screen reader simulation
3. Verify mobile responsiveness
4. Check Brightspace compatibility

### Step 4: IMSCC Integration
1. Place completed HTML file in appropriate directory
2. Reference in manifest.xml
3. Test import in Brightspace sandbox environment

## Template Maintenance

### Regular Updates
- **Monthly**: Review new open source templates
- **Quarterly**: Update accessibility standards compliance
- **Annually**: Comprehensive Brightspace compatibility testing

### Version Control
- Template version numbers in HTML comments
- Change log documentation
- Backward compatibility maintenance

### Community Contributions
- GitHub repository for template contributions
- Peer review process for new templates
- User feedback integration system

## Expanded Template Library (New)

### Directory Structure
```
templates/
├── _base/                    # Shared CSS variables and reset
│   └── variables.css         # Unified CSS custom properties
├── component/                # Reusable UI components
│   ├── callout_template.html
│   ├── accordion_template.html
│   ├── tabs_template.html
│   ├── card_layout_template.html
│   ├── flip_card_template.html
│   ├── timeline_template.html
│   └── progress_indicator_template.html
├── interactive/              # User interaction elements
│   ├── self_check_template.html
│   ├── reveal_content_template.html
│   └── inline_quiz_template.html
├── layout/                   # Page structure patterns
├── theme/                    # Visual themes
│   ├── color_schemes/
│   │   └── high_contrast.css
│   └── typography/
│       └── dyslexia_friendly.css
├── accessibility/            # Accessibility patterns
├── activity/                 # Learning activities
├── assessment/               # Evaluation templates
└── lesson/                   # Lesson content
```

### New Component Templates

| Template | File | Description |
|----------|------|-------------|
| **Callout** | `component/callout_template.html` | Info/warning/success/danger alert boxes with icons |
| **Accordion** | `component/accordion_template.html` | HTML5 details/summary expandable sections |
| **Tabs** | `component/tabs_template.html` | ARIA-compliant tabbed content panels |
| **Card Layout** | `component/card_layout_template.html` | Responsive grid of content cards |
| **Flip Card** | `component/flip_card_template.html` | CSS flip animation for term/definition pairs |
| **Timeline** | `component/timeline_template.html` | Vertical timeline for chronological content |
| **Progress** | `component/progress_indicator_template.html` | Progress bars and step indicators |

### New Interactive Templates

| Template | File | Description |
|----------|------|-------------|
| **Self-Check** | `interactive/self_check_template.html` | Single-question formative assessment with feedback |
| **Reveal Content** | `interactive/reveal_content_template.html` | Click-to-reveal hidden answers |
| **Inline Quiz** | `interactive/inline_quiz_template.html` | Multi-question embedded quiz with scoring |

### Theme Files

| File | Description |
|------|-------------|
| `_base/variables.css` | Unified CSS custom properties with official Courseforge palette |
| `theme/color_schemes/high_contrast.css` | WCAG AAA (7:1+) high contrast theme |
| `theme/typography/dyslexia_friendly.css` | Optimized typography for dyslexia |

### Key Features of New Templates

1. **WCAG 2.2 AA Compliant**
   - Focus indicators (2px+ outline, 3:1 contrast)
   - Target size minimum 24x24 CSS pixels
   - Scroll margins for focus visibility
   - Color + icon (never color alone)

2. **Brightspace Compatible**
   - Embedded CSS (no external stylesheets)
   - Minimal inline JavaScript (essential only)
   - No external dependencies
   - Self-contained structure

3. **Responsive Design**
   - Mobile-first approach
   - Breakpoints at 600px and 768px
   - Print-optimized styles

4. **Variable System**
   - Uses `{{VARIABLE_NAME}}` substitution pattern
   - Consistent across all templates
   - Documented in each template header

## Future Development

### Planned Template Additions
1. **Video Lecture Template** - Structured video content presentation
2. **Simulation Activity Template** - Interactive learning scenarios
3. **Group Project Template** - Collaborative learning frameworks
4. **Assessment Review Template** - Post-assessment learning reinforcement
5. **Two-Column Layout** - Content with sidebar navigation
6. **Stepper Template** - Step-by-step process visualization

### Technology Integration
- **Progressive Enhancement** - Advanced features for modern browsers
- **Responsive Images** - Optimized media delivery
- **Print Optimization** - Printer-friendly versions

### Research Initiatives
- **User Experience Studies** - Template effectiveness research
- **Accessibility Testing** - Ongoing compliance verification
- **Learning Outcome Analysis** - Educational effectiveness measurement