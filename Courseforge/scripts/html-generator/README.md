# HTML Generator

**Version**: v1.0.0  
**Created**: 2025-08-05  
**Purpose**: Convert structured course content JSON into HTML pages with Bootstrap framework

## Overview

The HTML Generator transforms structured course content from the Course Content Parser into individual HTML pages optimized for IMSCC package generation. Each sub-module becomes a standalone HTML page with Bootstrap 4.3.1 framework, accessibility compliance, and interactive elements.

## Features

- **Bootstrap 4.3.1 Framework**: Professional responsive design with CDN fallbacks
- **Interactive Accordion Elements**: Key concepts with expand/collapse functionality
- **WCAG 2.2 AA Compliance**: Full accessibility with keyboard navigation and screen readers
- **Atomic Operations**: Single execution with complete success or failure validation
- **Content Quality Assurance**: Validates substantial content in each generated page
- **Professional Typography**: Consistent visual hierarchy and styling

## Input Requirements

### Structured JSON Format
Expected input from Course Content Parser:
```json
{
  "course_info": {
    "title": "Course Title",
    "description": "Course description"
  },
  "weeks": [
    {
      "week_number": 1,
      "title": "Week Title", 
      "sub_modules": [
        {
          "type": "overview",
          "title": "Module Overview",
          "content": "Substantial content text",
          "word_count": 250
        }
      ]
    }
  ]
}
```

## Output Structure

### Generated HTML Files
For each week and sub-module combination:
- `week_01_overview.html` - Week introduction and learning objectives
- `week_01_concept_summary_01.html` - First concept summary page
- `week_01_concept_summary_02.html` - Second concept summary page  
- `week_01_key_concepts.html` - Interactive accordion with key terms
- `week_01_visual_content.html` - Visual displays, charts, diagrams
- `week_01_application_examples.html` - Learning concepts in practice
- `week_01_real_world.html` - Real world applications and industry examples
- `week_01_study_questions.html` - Reflection and knowledge testing

### HTML Template Structure
Each generated HTML page includes:
- Bootstrap 4.3.1 CSS/JS framework with CDN fallbacks
- Responsive design optimized for mobile and desktop
- WCAG 2.2 AA accessibility features (ARIA labels, keyboard navigation)
- Professional typography with consistent styling
- Interactive elements with smooth animations

## Key Features

### Bootstrap Accordion Implementation
Key concepts pages include interactive accordion containers:
- Expandable/collapsible sections for each key term
- Smooth CSS animations and transitions
- Keyboard navigation support (Tab, Enter, Space, Arrow keys)
- Screen reader compatible with proper ARIA attributes
- Font Awesome icons for visual state indication

### Content Display Standards
- **Page Titles**: Formatted as "Module {number}: {title}"
- **Paragraph Structure**: 50-300 words with 1.6 line height
- **Heading Hierarchy**: Proper H1-H6 structure for accessibility
- **Visual Elements**: Charts, graphs, equations with alt-text descriptions

### Accessibility Compliance
- **WCAG 2.2 AA Standards**: Full compliance across all generated content
- **Keyboard Navigation**: Complete keyboard access for interactive elements
- **Screen Reader Support**: Semantic markup and descriptive ARIA labels
- **Focus Management**: Clear focus indicators and logical tab order
- **Skip Links**: Navigation shortcuts for complex content sections

## Usage

### Command Line Interface
```bash
python html_generator.py --input structured_course.json --output html_output/
```

### Python API
```python
from html_generator import HTMLGenerator

generator = HTMLGenerator()
result = generator.generate_html_files('structured_course.json', 'output_directory/')
```

## Configuration

### Config File: `config/html_config.json`
```json
{
  "bootstrap_version": "4.3.1",
  "css_framework": {
    "cdn_primary": "https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css",
    "cdn_fallback": "https://cdnjs.cloudflare.com/ajax/libs/bootstrap/4.3.1/css/bootstrap.min.css"
  },
  "javascript_framework": {
    "bootstrap_js": "https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js",
    "jquery": "https://code.jquery.com/jquery-3.3.1.slim.min.js",
    "font_awesome": "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css"
  },
  "accessibility": {
    "wcag_level": "AA",
    "keyboard_navigation": true,
    "screen_reader_support": true,
    "high_contrast_mode": true
  },
  "styling": {
    "line_height": 1.6,
    "font_family": "Arial, sans-serif",
    "container_max_width": "1200px",
    "accordion_animation_duration": "0.3s"
  }
}
```

## HTML Template Examples

### Overview Page Template
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Module 1: Introduction to Linear Algebra</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css">
</head>
<body>
    <div class="container mt-4">
        <h1>Module 1: Introduction to Linear Algebra</h1>
        <div class="content-section">
            <p class="content-paragraph">Course content with proper formatting...</p>
        </div>
    </div>
    <script src="https://code.jquery.com/jquery-3.3.1.slim.min.js"></script>
    <script src="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js"></script>
</body>
</html>
```

### Accordion Key Concepts Template
```html
<div class="accordion" id="keyConceptsAccordion">
    <div class="card">
        <div class="card-header" id="concept1">
            <h2 class="mb-0">
                <button class="btn btn-link" type="button" data-toggle="collapse" 
                        data-target="#collapse1" aria-expanded="true" aria-controls="collapse1">
                    <i class="fas fa-chevron-right rotate-icon"></i>
                    Vector Space
                </button>
            </h2>
        </div>
        <div id="collapse1" class="collapse" aria-labelledby="concept1" 
             data-parent="#keyConceptsAccordion">
            <div class="card-body">
                <p>A vector space is a mathematical structure formed by a collection of vectors...</p>
            </div>
        </div>
    </div>
</div>
```

## Validation Requirements

### Content Quality Standards
- **Minimum Content Length**: Each HTML page must contain substantial academic content
- **Template Variable Resolution**: Zero unresolved `{variable}` patterns in output
- **Bootstrap Integration**: All interactive elements properly implement Bootstrap components
- **Accessibility Validation**: Full WCAG 2.2 AA compliance verification

### File Generation Standards
- **Exactly 7 HTML Files per Week**: One for each required sub-module type
- **Consistent Naming Convention**: `week_XX_[type].html` format
- **Professional Formatting**: Consistent typography and visual hierarchy
- **Cross-Device Compatibility**: Responsive design across all screen sizes

## Error Handling

### Critical Errors (System Exit)
- Invalid or missing structured JSON input
- Template processing failures
- File system write permissions issues
- Bootstrap framework loading failures

### Validation Warnings
- Content below recommended word counts
- Missing optional elements (images, examples)
- Accessibility concerns requiring attention

## Dependencies

### Required Packages
```
python >= 3.8
jinja2 >= 3.0.0
pathlib >= 1.0.1
json >= 2.0.9
```

### External Resources
- Bootstrap 4.3.1 CSS/JS (CDN)
- Font Awesome 5.15.4 (CDN)
- jQuery 3.3.1 Slim (CDN)

## Testing

### Unit Tests
```bash
python -m pytest tests/
```

### Integration Testing
- HTML validation against W3C standards
- Accessibility testing with screen readers
- Cross-browser compatibility verification
- Responsive design testing across devices

### Quality Assurance Checklist
- [ ] All HTML files validate without errors
- [ ] Bootstrap components function correctly
- [ ] Accordion interactions work smoothly
- [ ] Keyboard navigation operates properly
- [ ] Screen readers announce content appropriately
- [ ] Mobile and desktop displays render correctly

## Performance Optimization

- **CDN Usage**: External resources loaded from reliable CDNs
- **Minified Assets**: Compressed CSS/JS for faster loading
- **Caching Strategy**: Proper cache headers for static resources
- **Image Optimization**: Alt-text and proper sizing for all visual elements

## Changelog

### v1.0.0 (2025-08-05)
- Initial implementation with Bootstrap 4.3.1 framework
- Interactive accordion functionality for key concepts
- WCAG 2.2 AA accessibility compliance
- Responsive design with mobile optimization
- Comprehensive error handling and validation

## License

MIT License - See LICENSE file in project root.