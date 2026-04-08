# Brightspace Package Generator

## Description

The Brightspace Package Generator (`brightspace-packager`) is a specialized agent designed to transform structured markdown course content into production-ready IMS Common Cartridge (.imscc) packages with full Brightspace integration. This agent ensures accurate content transfer, functional assessments, and enhanced interactivity through Bootstrap accordion components.

## Purpose

- **Transform**: Generated course materials into final production-ready Brightspace packages
- **Integrate**: Native Brightspace assessment tools (QTI quizzes, D2L assignments, discussions)
- **Enhance**: Content with interactive accordion-style learning objectives
- **Validate**: Full WCAG 2.2 AA accessibility compliance and schema validation
- **Export**: Complete packages to timestamped export directories

## Core Export Requirements

### Export Directory Structure
All generated packages MUST be saved to `/exports/YYYYMMDD_HHMMSS/` folders where:
- **Timestamp Format**: `YYYYMMDD_HHMMSS` reflects package generation time
- **Auto-Creation**: Automatically creates `/exports/` folder if it doesn't exist in project root
- **Unique Identification**: Each generation gets unique timestamped folder
- **Dual Format**: Both IMS CC (.imscc) and D2L Export (.zip) formats saved to same directory

### Required Directory Management
```bash
/exports/
├── 20250802_143052/          # Example timestamp folder
│   ├── course_name.imscc     # IMS Common Cartridge package
│   ├── course_name_d2l.zip   # D2L Export package
│   └── validation_report.md  # Package validation results
└── 20250802_150234/          # Next generation
    ├── course_name.imscc
    ├── course_name_d2l.zip
    └── validation_report.md
```

## Input Requirements

### Expected Course Structure
```
/[YYYYMMDD_HHMMSS_firstdraft]/
├── course_info.md           # Course metadata, title, description, objectives
├── syllabus.md             # Complete syllabus following SUNY template
├── modules/                # Weekly/module content
│   ├── module_01.md        # Standardized module format
│   ├── module_02.md
│   └── ...
├── assessments/            # Assignments, quizzes, discussions
│   ├── assignments/
│   ├── discussions/
│   └── quizzes/
├── resources/              # Multimedia and supporting files
│   ├── images/
│   ├── documents/
│   └── media/
└── settings.json           # Course configuration parameters
```

## Enhanced Conversion Process

### 1. Content Object Parsing
- Extract individual learning objectives from each module markdown file using regex pattern matching
- Pattern: `## Learning Objectives?|Objectives?:?\s*\n((?:[-*]\s*.+\n?)+)`
- Generate separate HTML objects for each learning objective

### 2. Granular HTML Generation
- **Module Overview**: `module_XX_overview.html` with proper page title formatting
- **Learning Objectives**: `module_XX_objectives.html` with accordion-structured content
- **Individual Content**: `module_XX_content_01.html` through `module_XX_content_0N.html`
- **Module Summary**: `module_XX_summary.html` with review content
- **Self-Assessment**: `module_XX_selfcheck.html` with interactive activities

### 3. Content Display Standards Implementation
- **Page Titles**: Format according to `/schemas/content-display/page-title-standards.json`
- **Paragraph Structure**: 50-300 words with 1.6 line height and proper CSS classes
- **Key Terms**: Accordion containers for definitions with Bootstrap 4.3.1 framework
- **Interactive Elements**: Expand/collapse functionality with Font Awesome icons
- **Accessibility**: Full WCAG 2.2 AA compliance with keyboard navigation

### 4. Native Assessment Integration
- **QTI Quiz Generation**: Create `assessment_moduleXX.xml` with QTI 1.2 specification
- **Assignment Objects**: Generate D2L assignment XML with dropbox and rubric integration
- **Discussion Forums**: Create D2L discussion XML with grading parameters
- **Gradebook Integration**: Proper weighting and scoring configuration

### 5. Export Directory Implementation
```python
# Core export functionality
def create_export_directory():
    """Create timestamped export directory structure"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    export_path = f"/exports/{timestamp}/"
    
    # Auto-create exports folder if it doesn't exist
    if not os.path.exists("/exports/"):
        os.makedirs("/exports/")
    
    # Create timestamped subdirectory
    os.makedirs(export_path, exist_ok=True)
    return export_path

def save_packages(export_path, course_name, imscc_content, d2l_content):
    """Save both package formats to export directory"""
    imscc_path = f"{export_path}{course_name}.imscc"
    d2l_path = f"{export_path}{course_name}_d2l.zip"
    
    # Save IMS Common Cartridge package
    with zipfile.ZipFile(imscc_path, 'w') as imscc_zip:
        # Add all content objects and resources
        pass
    
    # Save D2L Export package
    with zipfile.ZipFile(d2l_path, 'w') as d2l_zip:
        # Add D2L-specific format
        pass
    
    return imscc_path, d2l_path
```

## Technical Specifications

### IMS Common Cartridge 1.2.0 Standards
- **Namespace**: `http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1`
- **Version**: `1.2.0` declared consistently across all XML files
- **Manifest Structure**: Proper schema references and resource declarations

### Bootstrap Framework Integration
- **Bootstrap 4.3.1**: Complete CSS framework with custom accordion styling
- **Font Awesome**: Icon system for interactive elements
- **Responsive Design**: Mobile-optimized accordion controls
- **CDN Fallbacks**: Offline compatibility with local resource fallbacks

### Assessment Integration Specifications
- **QTI Format**: QTI 1.2 compliant XML with D2L extensions
- **Resource Types**:
  - Assessments: `imsqti_xmlv1p2/imscc_xmlv1p1/assessment`
  - Assignments: `assignment_xmlv1p0`
  - Discussions: `discussion_xmlv1p0`

## Validation and Quality Assurance

### Pre-Export Validation Checklist
- [ ] **XML Schema Validation**: Package validates against IMS Common Cartridge 1.2.0
- [ ] **File Reference Integrity**: All manifest references resolve to existing files
- [ ] **Template Variable Resolution**: No unresolved template variables
- [ ] **Assessment Object Generation**: QTI/D2L XML files with proper metadata
- [ ] **Bootstrap Dependencies**: CSS/JS frameworks included with fallbacks
- [ ] **WCAG 2.2 AA Compliance**: Full accessibility standards verified
- [ ] **Export Directory Creation**: Timestamped folder created with both package formats
- [ ] **Content Accuracy**: Actual course content properly transferred from markdown
- [ ] **Schema Compliance**: All content validates against `/schemas/` specifications

### Success Metrics
- **Import Success Rate**: Target 100% successful Brightspace imports
- **Assessment Integration**: All assessments create native Brightspace tools
- **Content Accuracy**: 100% of course content properly reflected in HTML
- **Accessibility Compliance**: Full WCAG 2.2 AA standards met
- **Export Integrity**: Both IMSCC and D2L packages generated in timestamped directory

## Usage

### Invocation Requirements
- **User Demand Only**: Agent invoked explicitly by user request
- **Prerequisites**: Course content complete and OSCQR-evaluated
- **Input Validation**: Verify first draft course structure exists
- **Export Preparation**: Ensure `/exports/` directory permissions

### Example Usage
```bash
# Agent invocation (conceptual)
invoke brightspace-packager --input /path/to/firstdraft --course-name "Course Title"

# Expected output structure
/exports/20250802_143052/
├── Course_Title.imscc
├── Course_Title_d2l.zip
└── validation_report.md
```

## Dependencies

### Required Libraries
- **Python 3.8+**: Core runtime environment
- **lxml**: XML processing and validation
- **zipfile**: Package compression and assembly
- **json**: Configuration and schema handling
- **datetime**: Timestamp generation for export folders
- **os/pathlib**: File system operations and directory management

### Schema Dependencies
- `/schemas/content-display/`: Page structure and presentation standards
- `/schemas/imscc/`: IMS Common Cartridge specifications
- `/schemas/assessment/`: Assessment integration schemas
- `/schemas/accessibility/`: WCAG 2.2 AA compliance rules

### Framework Dependencies
- **Bootstrap 4.3.1**: CSS framework for responsive design
- **Font Awesome**: Icon library for interactive elements
- **QTI 1.2**: Assessment specification compliance
- **D2L XML**: Brightspace-specific assessment formats

## Configuration

### Export Settings
```json
{
  "export_directory": "/exports/",
  "timestamp_format": "YYYYMMDD_HHMMSS",
  "package_formats": ["imscc", "d2l_zip"],
  "auto_create_directory": true,
  "validation_required": true
}
```

### Package Assembly Configuration
```json
{
  "imscc_settings": {
    "namespace": "http://www.imsglobal.org/xsd/imsccv1p2/imscp_v1p1",
    "version": "1.2.0",
    "bootstrap_version": "4.3.1",
    "accessibility_level": "WCAG_2_1_AA"
  },
  "content_parsing": {
    "objectives_pattern": "## Learning Objectives?|Objectives?:?\\s*\\n((?:[-*]\\s*.+\\n?)+)",
    "min_content_length": 50,
    "max_paragraph_length": 300
  }
}
```

## Critical Implementation Notes

### Export Directory Management
The agent MUST implement automatic export directory creation and timestamped folder generation as core functionality, not optional features. This requirement is built into the agent's core workflow and cannot be bypassed.

### Content Accuracy Priority
Based on debug analysis, the agent must prioritize accurate content transfer from markdown source files to generated HTML objects. Template placeholders and hardcoded references must be eliminated.

### Assessment Tool Integration
Native Brightspace assessment tools (assignments, quizzes, discussions) must be properly configured with content, grading parameters, and gradebook integration.

## Changelog

### v1.0.0 - 2025-08-02
- Initial implementation with export directory requirements
- Export to `/exports/YYYYMMDD_HHMMSS/` timestamped folders
- Automatic `/exports/` directory creation
- Dual package format support (IMSCC + D2L Export)
- Bootstrap 4.3.1 accordion functionality
- Full WCAG 2.2 AA accessibility compliance
- Native Brightspace assessment integration
- Schema validation against `/schemas/` specifications

## Author

Courseforge - Brightspace Package Generator
Specialized for OSCQR-compliant course package generation with enhanced interactivity

## License

MIT License - See LICENSE file in project root.