# Courseforge Schemas (tool-local)

This directory contains Courseforge-specific schemas for UI components, layouts, templates, and framework migration — content that is tightly coupled to the Courseforge rendering pipeline and not intended for cross-project reuse.

## Schemas that moved

The following schemas previously lived here and have migrated to the unified project-root `/schemas/` tree for cross-project reuse:

| Former location | New canonical path |
|---|---|
| `academic-metadata/course_metadata_schema.json` | `/schemas/academic/course_metadata.schema.json` |
| `learning-objectives/learning_objectives_schema.json` | `/schemas/academic/learning_objectives.schema.json` |
| `learning-objectives/textbook_structure_schema.json` | `/schemas/academic/textbook_structure.schema.json` |
| `accessibility/wcag22_compliance_schema.json` | `/schemas/compliance/wcag22_compliance.schema.json` |

See `/schemas/README.md` for the unified schema index and conventions.

## What remains here (Courseforge-only)

```
schemas/
├── README.md                           # This file
├── Claude.md                           # Historical schema overview
├── template-integration/               # Template integration patterns
│   └── educational_template_schema.json
├── content-display/                    # Content display standards
│   ├── content-display-schema.json     # Original schema
│   ├── enhanced-content-display-schema.json  # Enhanced with research findings
│   ├── accordion-schema.json
│   └── page-title-standards.json
├── framework-migration/                # Framework migration guides
│   └── bootstrap5_migration_schema.json
├── layouts/                            # Layout-specific schemas
│   └── course_card_schema.json
├── assessment/                         # Assessment integration schemas (placeholder)
└── imscc/                              # IMSCC-specific schema references (placeholder)
```

### 1. Educational Template Integration Schema
**File**: `template-integration/educational_template_schema.json`
**Purpose**: Defines integration patterns for educational templates based on research findings
**Key Features**:
- Template metadata from MIT OCW, Stanford Online, Bootstrap themes
- Course card layouts with hover effects and responsive design
- Hero sections with educational messaging
- Navigation context (replaces internal linking for Brightspace compatibility)
- Accessibility enhancements from academic templates
- Performance optimization patterns

### 2. Enhanced Content Display Schema
**File**: `content-display/enhanced-content-display-schema.json`
**Purpose**: Updated content display standards incorporating research findings
**Key Features**:
- Enhanced visual elements (callout boxes, course cards, progress indicators)
- Template type definitions (basic_lesson, hero_lesson, card_layout, academic_module)
- Navigation context display for Brightspace compatibility
- Improved typography and responsive design patterns
- Self-assessment integration capabilities

### 3. Bootstrap 5 Migration Schema
**File**: `framework-migration/bootstrap5_migration_schema.json`
**Purpose**: Comprehensive migration guide from Bootstrap 4.3.1 to Bootstrap 5
**Key Features**:
- Breaking changes analysis with educational template impact
- Phase-based migration strategy (CSS → Components → Enhancements)
- Brightspace compatibility testing requirements
- Educational template-specific improvements in Bootstrap 5
- Rollback plan for migration issues

### 4. Course Card Layout Schema
**File**: `layouts/course_card_schema.json`
**Purpose**: Detailed course card layouts from Bootstrap educational themes
**Key Features**:
- Responsive grid systems for different screen sizes
- Card structure (header, body, footer) with educational metadata
- Interactive features (hover effects, progress tracking)
- Accessibility compliance with ARIA labels and semantic structure
- Brightspace compatibility (embedded CSS, no external links)

## Integration with Existing Systems

### Content Generator Agent Integration
```javascript
// Example integration with content-generator agent
{
  "templateType": "enhanced_lesson",
  "accessibilityLevel": "WCAG_2.2_AA",
  "frameworkVersion": "Bootstrap_4.3.1",
  "educationalFeatures": {
    "courseCards": true,
    "progressTracking": true,
    "navigationContext": true
  }
}
```

### Brightspace Packaging Integration
All schemas include Brightspace-specific requirements:
- Embedded CSS only (no external stylesheets)
- No internal page linking (navigation context instead)
- Self-contained HTML files
- Mobile responsiveness within Brightspace app
- Print-friendly formatting

### Quality Assurance Integration
Schemas provide validation criteria for:
- WCAG 2.2 AA accessibility compliance (see `/schemas/compliance/wcag22_compliance.schema.json`)
- Educational content depth and engagement
- Cross-browser compatibility
- Mobile responsiveness
- Performance optimization

## Schema Validation Examples

### Template Integration Validation
```json
{
  "templateMetadata": {
    "templateName": "Advanced Course Module",
    "source": "MIT_OCW",
    "frameworkVersion": "Bootstrap_4.3.1",
    "accessibilityLevel": "WCAG_2.2_AA",
    "brighspaceCompatible": true
  },
  "courseCardLayout": {
    "enabled": true,
    "gridSystem": "col-md-6",
    "hoverEffects": {
      "enabled": true,
      "animation": "lift",
      "duration": "0.2s"
    }
  }
}
```

## Research Sources Integration

### MIT OpenCourseWare Patterns
- Modular course content architecture
- Metadata-driven content organization
- Academic-quality accessibility standards
- Multi-media content integration

### Stanford Online Patterns
- Clean academic presentation styles
- Structured learning progression
- Professional course information display

### Bootstrap Educational Themes
- Modern responsive card layouts
- Interactive hover effects and animations
- Professional color schemes and typography
- Mobile-first responsive design

### Accessibility Research
- WCAG 2.2 AA compliance requirements (schema at `/schemas/compliance/wcag22_compliance.schema.json`)
- Cognitive accessibility for learning differences
- Screen reader optimization
- Color contrast validation

## Quality Assurance Checklist

When implementing these schemas, validate:
- [ ] WCAG 2.2 AA accessibility compliance
- [ ] Brightspace/D2L compatibility
- [ ] Mobile responsiveness across devices
- [ ] Cross-browser compatibility
- [ ] Print-friendly formatting
- [ ] Educational content depth and engagement
- [ ] Performance optimization
- [ ] Schema validation against JSON Schema standards

## Support and Documentation

For implementation support:
- See `/templates/` directory for working examples
- Check `/docs/PATTERN_PREVENTION_GUIDE.md` for error prevention
- Reference `/imscc-standards/brightspace-specific/` for LMS specifications
- Reference `/schemas/README.md` (project root) for the unified schema index

These schemas provide a Courseforge-local foundation for UI components, layouts, and template patterns. Cross-project schemas (academic metadata, learning objectives, textbook structure, WCAG compliance) now live under the project-root `/schemas/` tree.
