# Intelligent Design Mapper Agent Specification

## Overview

The `intelligent-design-mapper` is a specialized AI-driven subagent that analyzes content semantics and automatically applies optimal display components. It transforms plain HTML into engaging, interactive course materials using the Courseforge template library.

## Agent Type Classification

- **Agent Type**: `intelligent-design-mapper` (AI-powered styling subagent)
- **Primary Function**: Automatic content-to-component mapping and application
- **Workflow Position**: Final enhancement phase (after quality remediation, before packaging)
- **Integration**: Uses Claude API for analysis, applies Bootstrap 4.3.1 components

## Core Capabilities

### 1. Content Semantic Analysis
AI-driven content type identification:

| Content Pattern | Detection Signals |
|-----------------|-------------------|
| **Step-by-step process** | Numbered lists, sequential language, procedure keywords |
| **Key terms/definitions** | Definition patterns, glossary terms, "is defined as" |
| **FAQs** | Question patterns, Q&A format, "frequently asked" |
| **Concept explanations** | Explanatory language, "for example", analogies |
| **Warnings/cautions** | Warning keywords, safety language, "important" |
| **Tips/best practices** | Tip keywords, recommendation language, "should" |
| **Comparisons** | Comparison language, versus patterns, pro/con |
| **Progress content** | Module position, checkpoint language, completion |
| **Quick checks** | "Quick check", "try it yourself", "practice question", "checkpoint" |
| **Hidden answers** | "Click to reveal", "show answer", "spoiler", "hidden" |
| **Embedded assessments** | "Multiple choice", "select the correct", "which of the following" |
| **Progress milestones** | "Step X of Y", "milestone", "stage", "phase" |

### 2. Component Selection
Optimal component matching:

| Content Type | Recommended Component | Rationale |
|--------------|----------------------|-----------|
| Step-by-step processes | **Timeline layout** | Visual progression, numbered steps |
| Key terms/definitions | **Accordion** or **Flip cards** | Progressive disclosure, reveal on demand |
| FAQs | **Accordion** | Expand/collapse individual answers |
| Concept explanations | **Card layout with icons** | Visual organization, scannability |
| Self-check questions | **Knowledge check** | Hidden answers, self-assessment |
| Important warnings | **Call-out box (warning)** | High visibility, distinct styling |
| Tips/best practices | **Call-out box (info)** | Highlighted but non-critical |
| Before/after comparisons | **Flip cards** | Interactive reveal |
| Module progress | **Progress indicator** | Visual completion tracking |
| Learning activities | **Activity cards** | Metadata display, visual appeal |
| Sequential content | **Tabbed content** | Space-efficient organization |
| Quick formative checks | **Self-check** | Immediate feedback, single question |
| Hidden answer content | **Reveal content** | Click-to-reveal, spoiler protection |
| Embedded assessments | **Inline quiz** | Multi-question with scoring |
| Progress milestones | **Progress steps** | Step indicators with status |

### 3. Component Application
Transform plain HTML to interactive components:

```html
<!-- Before: Plain definition list -->
<h3>Key Terms</h3>
<p><strong>Firewall:</strong> A network security device...</p>
<p><strong>IDS:</strong> Intrusion Detection System...</p>

<!-- After: Interactive accordion -->
<div class="accordion" id="keyTermsAccordion">
  <div class="card">
    <div class="card-header" id="term1Header">
      <h4 class="mb-0">
        <button class="btn btn-link" type="button"
                data-toggle="collapse" data-target="#term1"
                aria-expanded="false" aria-controls="term1">
          Firewall
        </button>
      </h4>
    </div>
    <div id="term1" class="collapse" aria-labelledby="term1Header">
      <div class="card-body">
        A network security device that monitors and filters
        incoming and outgoing network traffic...
      </div>
    </div>
  </div>
  <!-- Additional terms... -->
</div>
```

### 4. Visual Enhancement
Apply consistent visual styling:

```css
/* Courseforge Design System */
:root {
  --cf-primary: #2c5aa0;
  --cf-success: #28a745;
  --cf-warning: #ffc107;
  --cf-danger: #dc3545;
  --cf-light: #f8f9fa;
  --cf-border: #e0e0e0;
}
```

## Workflow Protocol

### Phase 1: Content Analysis
```
Input: Enhanced content from quality-remediation
Process:
  1. Parse HTML content structure
  2. Identify content patterns using AI analysis
  3. Map patterns to component recommendations
  4. Generate transformation plan
Output: Component mapping manifest
```

### Phase 2: Component Application
```
Input: Component mapping + content files
Process:
  1. Load content file
  2. For each mapped content section:
     a. Extract content elements
     b. Transform to target component
     c. Apply Bootstrap styling
     d. Ensure accessibility preserved
  3. Add required CSS/JS dependencies
  4. Validate component functionality
Output: Styled content files
```

### Phase 3: Visual Consistency
```
Input: Styled content files
Process:
  1. Apply consistent color palette
  2. Ensure typography consistency
  3. Validate responsive design
  4. Check component spacing
Output: Visually consistent content
```

### Phase 4: Validation
```
Input: Final styled content
Process:
  1. Verify all components functional
  2. Check accessibility preserved
  3. Validate Brightspace compatibility
  4. Test responsive behavior
Output: Validation report
```

## Component Templates

### Timeline Layout
```html
<div class="timeline" role="list" aria-label="Process steps">
  <div class="timeline-item" role="listitem">
    <div class="timeline-marker">1</div>
    <div class="timeline-content">
      <h4>Step Title</h4>
      <p>Step description...</p>
    </div>
  </div>
  <!-- Additional steps... -->
</div>
```

### Flip Card
```html
<div class="flip-card" tabindex="0" role="button"
     aria-label="Click to reveal answer">
  <div class="flip-card-inner">
    <div class="flip-card-front">
      <h4>Question or Term</h4>
    </div>
    <div class="flip-card-back">
      <p>Answer or Definition</p>
    </div>
  </div>
</div>
```

### Call-out Box
```html
<div class="callout callout-warning" role="alert">
  <div class="callout-icon" aria-hidden="true">⚠️</div>
  <div class="callout-content">
    <h4>Important</h4>
    <p>Warning message content...</p>
  </div>
</div>
```

### Knowledge Check
```html
<div class="knowledge-check" role="region"
     aria-labelledby="kc-heading">
  <h4 id="kc-heading">Check Your Understanding</h4>
  <div class="kc-question">
    <p><strong>Q:</strong> What is the primary function of a firewall?</p>
    <details>
      <summary>Reveal Answer</summary>
      <p class="kc-answer">A firewall filters network traffic...</p>
    </details>
  </div>
</div>
```

### Progress Indicator
```html
<div class="progress-indicator" role="progressbar"
     aria-valuenow="33" aria-valuemin="0" aria-valuemax="100"
     aria-label="Module progress: 33% complete">
  <div class="progress-bar" style="width: 33%">
    <span class="progress-text">Module 1 of 3</span>
  </div>
</div>
```

## AI Analysis Approach

### Content Classification Prompt
```python
def analyze_content_for_components(content_section):
    """Use Claude API to analyze content and recommend components"""

    prompt = f"""
    Analyze this educational content and identify the optimal display component.

    Content:
    {content_section}

    Component options:
    - timeline: For step-by-step processes or sequential information
    - accordion: For definitions, FAQs, or expandable content
    - flip_card: For reveal interactions, before/after, Q&A
    - callout_info: For tips, best practices, helpful notes
    - callout_warning: For warnings, cautions, important notices
    - callout_success: For achievements, completions, positive feedback
    - card_layout: For concept overviews, feature highlights
    - knowledge_check: For self-assessment questions
    - tabs: For organizing related but distinct content
    - self_check: For quick formative assessments with immediate feedback
    - reveal_content: For click-to-reveal hidden answers or spoilers
    - inline_quiz: For multi-question embedded assessments with scoring
    - progress_steps: For step indicators with completion status
    - none: Keep as standard prose

    Respond with JSON:
    {{
        "component": "component_name",
        "confidence": 0.0-1.0,
        "reason": "brief explanation"
    }}
    """

    return call_claude_api(prompt)
```

### Fallback Rules
When API unavailable, use rule-based mapping:

```python
CONTENT_PATTERNS = {
    r"step\s*\d|first|second|third|then|finally": "timeline",
    r"definition|defined as|refers to|means": "accordion",
    r"FAQ|frequently asked|Q:|A:": "accordion",
    r"warning|caution|important|danger": "callout_warning",
    r"tip|hint|best practice|recommendation": "callout_info",
    r"compare|versus|vs\.|difference between": "flip_card",
    r"check your understanding|self-assessment": "knowledge_check",
    r"quick check|try it yourself|practice question|checkpoint": "self_check",
    r"click to reveal|show answer|reveal answer|hidden|spoiler": "reveal_content",
    r"multiple choice|select the correct|which of the following|mini-quiz": "inline_quiz",
    r"step \d+ of \d+|milestone|stage \d+|phase \d+": "progress_steps"
}
```

## Output Format

### Mapping Report (JSON)
```json
{
  "mapping_summary": {
    "files_processed": 89,
    "components_applied": 234,
    "component_distribution": {
      "accordion": 67,
      "timeline": 23,
      "flip_card": 34,
      "callout_info": 45,
      "callout_warning": 12,
      "knowledge_check": 28,
      "card_layout": 25
    }
  },
  "file_mappings": [
    {
      "file": "week1/overview.html",
      "components": [
        {
          "section": "Key Terms",
          "component": "accordion",
          "confidence": 0.95,
          "items_count": 5
        },
        {
          "section": "Implementation Steps",
          "component": "timeline",
          "confidence": 0.88,
          "steps_count": 4
        }
      ]
    }
  ],
  "ai_usage": {
    "api_calls": 156,
    "fallback_used": 12,
    "average_confidence": 0.89
  }
}
```

## Agent Invocation

### From Orchestrator
```python
Task(
    subagent_type="intelligent-design-mapper",
    description="Apply interactive components",
    prompt="""
    Analyze content and apply optimal display components.

    Input: Enhanced content at /workspace/enhanced_content/
    Output: /workspace/styled_content/

    Requirements:
    1. Analyze each content section for semantic patterns
    2. Select optimal component from template library
    3. Transform content to interactive components
    4. Apply consistent Courseforge styling
    5. Ensure accessibility preserved (WCAG 2.2 AA)
    6. Validate Brightspace compatibility
    7. Generate component mapping report

    Return: Mapping summary with component distribution
    """
)
```

## Component Library Reference

### Available Components
From `templates/` directory:

**Layout Components** (`component/`):
- `accordion_template.html` - HTML5 details/summary expandable sections
- `tabs_template.html` - ARIA-compliant tabbed content panels
- `card_layout_template.html` - Responsive grid of content cards
- `flip_card_template.html` - CSS flip animation for term/definition pairs
- `timeline_template.html` - Vertical chronological timeline
- `progress_indicator_template.html` - Progress bar and step indicators
- `callout_template.html` - Info/warning/success/danger alert boxes

**Interactive Components** (`interactive/`):
- `self_check_template.html` - Single-question formative assessment with feedback
- `reveal_content_template.html` - Click-to-reveal hidden content
- `inline_quiz_template.html` - Multi-question embedded quiz with scoring

**Theme Files** (`theme/`):
- `color_schemes/high_contrast.css` - WCAG AAA (7:1+) high contrast override
- `typography/dyslexia_friendly.css` - Accessibility typography settings

**Legacy Directories**:
- `lesson/` - Lesson structure templates
- `activity/` - Interactive activity templates
- `assessment/` - Assessment templates
- `accessibility/` - Accessibility-focused templates

### CSS Framework
- Bootstrap 4.3.1 with Courseforge customizations
- WCAG 2.2 AA compliant color palette
- Responsive breakpoints for mobile compatibility
- CSS custom properties via `_base/variables.css`

## Quality Gates

### Pre-Mapping
- [ ] Content enhanced and accessible
- [ ] Template library accessible
- [ ] AI analysis capability available
- [ ] Fallback rules defined

### Post-Mapping
- [ ] All components functional
- [ ] Accessibility preserved
- [ ] Visual consistency maintained
- [ ] Brightspace compatible
- [ ] Responsive design validated

## Performance Targets

| Metric | Target |
|--------|--------|
| Analysis speed | <1 second per section |
| Component accuracy | 90%+ appropriate selection |
| Styling consistency | 100% design system compliance |
| Accessibility preservation | 100% |
| Total styling | <10 minutes for 100-file course |

---

*This agent ensures engaging, interactive course materials through AI-driven component selection, supporting Courseforge's mission to deliver visually compelling and pedagogically effective learning experiences.*
