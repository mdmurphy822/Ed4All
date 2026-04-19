# Schemas Directory

This directory contains XML schemas, content structure definitions, and display format specifications for course generation and Brightspace package creation.

## Directory Structure

### `/content-display/`
Schemas and specifications for how course content should be displayed on web pages, including:
- Paragraph structure guidelines
- Key term highlighting and accordion container specifications
- Page title conventions
- Interactive element definitions

### `/imscc/`
IMS Common Cartridge XML schema definitions and validation files:
- IMS Common Cartridge 1.2.0 schema files
- Brightspace-specific extensions and customizations
- Manifest structure templates

### `/assessment/`
Assessment integration schemas for native Brightspace tools:
- QTI 1.2 quiz schema definitions
- D2L assignment XML structure
- Discussion forum configuration schemas
- Gradebook integration specifications

### Accessibility (migrated)
WCAG 2.2 AA compliance schemas now live at `/schemas/compliance/wcag22_compliance.schema.json` in the unified project-root schema tree. Accessibility markup requirements, screen reader compatibility specifications, and keyboard navigation standards are validated via that canonical location.

## Usage Guidelines

1. **Schema Validation**: All generated content must validate against appropriate schemas
2. **Version Control**: Maintain schema versions with backward compatibility
3. **Documentation**: Each schema file includes comprehensive comments and examples
4. **Testing**: Schema compliance validated during pre-deployment checks

## Schema Files

- `content-display-schema.json` - Defines content presentation standards
- `imscc-manifest-schema.xsd` - IMS Common Cartridge manifest validation
- `brightspace-extensions.xsd` - Brightspace-specific XML extensions
- `assessment-integration.json` - Assessment tool configuration standards