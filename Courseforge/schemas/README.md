# Educational Template Schemas - Updated with Research Findings

This directory contains comprehensive schemas for educational HTML templates, updated based on research findings from MIT OCW, Stanford Online, Bootstrap educational themes, and accessibility best practices.

## Schema Directory Structure

```
schemas/
├── README.md                           # This file - schema overview
├── template-integration/               # Template integration patterns
│   └── educational_template_schema.json
├── content-display/                    # Content display standards  
│   ├── content-display-schema.json     # Original schema
│   ├── enhanced-content-display-schema.json # Enhanced with research findings
│   ├── accordion-schema.json
│   └── page-title-standards.json
├── framework-migration/                # Framework migration guides
│   └── bootstrap5_migration_schema.json
├── academic-metadata/                  # Course metadata structures
│   └── course_metadata_schema.json
├── accessibility/                      # Accessibility compliance
│   └── wcag22_compliance_schema.json
├── layouts/                           # Layout-specific schemas
│   └── course_card_schema.json
├── assessment/                        # Assessment schemas
└── imscc/                            # IMSCC-specific schemas
```

## New Schemas Created Based on Research

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

### 4. Academic Course Metadata Schema
**File**: `academic-metadata/course_metadata_schema.json`
**Purpose**: Structured course metadata based on MIT OCW course.json patterns
**Key Features**:
- Complete course identification (number, title, department, institution)
- Learning outcomes with Bloom's taxonomy alignment
- Instructional team information with roles and expertise
- Course structure with modules and assessment framework
- Resource management (textbooks, technology, supplementary materials)
- Universal Design for Learning (UDL) integration

### 5. WCAG 2.2 AA Compliance Schema
**File**: `accessibility/wcag22_compliance_schema.json`
**Purpose**: Comprehensive accessibility compliance based on WCAG 2.2 AA standards
**Key Features**:
- All four WCAG principles (Perceivable, Operable, Understandable, Robust)
- Educational-specific accessibility considerations
- Cognitive accessibility support for learning differences  
- Assistive technology compatibility (screen readers, voice control)
- Testing protocols (automated, manual, user testing)

### 6. Course Card Layout Schema
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
- WCAG 2.2 AA accessibility compliance
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

### Accessibility Validation
```json
{
  "complianceLevel": {
    "standard": "WCAG_2.2_AA",
    "testing": {
      "automated": ["axe-core", "WAVE", "Lighthouse"],
      "manual": ["NVDA", "keyboard_testing"]
    }
  },
  "colorContrast": {
    "minimumRatio": 4.5,
    "largeTextRatio": 3.0,
    "nonTextRatio": 3.0
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
- WCAG 2.2 AA compliance requirements
- Cognitive accessibility for learning differences
- Screen reader optimization
- Color contrast validation

## Implementation Priorities

### High Priority (Immediate Implementation)
1. **Course Card Layouts**: Implement responsive card grids for better course presentation
2. **Accessibility Enhancements**: Apply WCAG 2.2 AA standards across all templates  
3. **Navigation Context**: Replace internal links with context displays
4. **Enhanced Visual Elements**: Add callout boxes and progress indicators

### Medium Priority (Phase 2)
1. **Bootstrap 5 Migration**: Plan and execute framework upgrade
2. **Academic Metadata**: Implement comprehensive course information structure
3. **Hero Sections**: Add engaging course introduction layouts
4. **Self-Assessment Integration**: Include interactive self-check capabilities

### Low Priority (Future Enhancement)
1. **Advanced Interactive Elements**: Modal windows, complex animations
2. **Performance Optimization**: Advanced caching and loading strategies  
3. **Multi-language Support**: Internationalization capabilities
4. **Advanced Analytics**: Learning analytics integration

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

## Maintenance and Updates

### Regular Review Schedule
- **Monthly**: Accessibility standards updates
- **Quarterly**: Framework compatibility testing
- **Annually**: Comprehensive educational effectiveness review

### Update Process
1. Research new educational design patterns
2. Test compatibility with latest Brightspace versions  
3. Validate accessibility standards compliance
4. Update schemas with community feedback
5. Document breaking changes and migration paths

## Support and Documentation

For implementation support:
- See `/templates/` directory for working examples
- Check `/docs/PATTERN_PREVENTION_GUIDE.md` for error prevention
- Reference `/imscc-standards/brightspace-specific/` for LMS specifications

These schemas provide a comprehensive foundation for creating accessible, engaging, and technically sound educational HTML templates based on proven academic and industry patterns.