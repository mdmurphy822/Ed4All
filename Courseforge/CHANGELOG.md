# Changelog

All notable changes to Courseforge will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-12-11

### Added

#### Core Features
- AI-powered course generation from exam objectives and textbooks
- Universal IMSCC package intake and remediation pipeline
- Support for multiple LMS sources (Brightspace, Canvas, Blackboard, Moodle, Sakai)
- WCAG 2.2 AA accessibility compliance throughout

#### Agent System
- 18 specialized AI agents for course creation and remediation
- Parallel execution with batch size management (5-10 agents per batch)
- Individual file protocol (one agent = one file)

#### Course Creation Agents
- `exam-research` - Certification objective analysis
- `requirements-collector` - Course specification gathering
- `course-outliner` - Structure and learning objectives
- `content-generator` - Educational content creation
- `educational-standards` - Pedagogical framework compliance
- `quality-assurance` - Pattern prevention and validation
- `oscqr-course-evaluator` - OSCQR quality assessment
- `brightspace-packager` - IMSCC package generation

#### Intake & Remediation Agents
- `imscc-intake-parser` - Universal IMSCC parsing
- `content-analyzer` - Accessibility/quality gap detection
- `dart-automation-coordinator` - PDF/Office to accessible HTML
- `accessibility-remediation` - Automatic WCAG 2.2 fixes
- `content-quality-remediation` - Educational depth enhancement

#### Templates & Components
- Bootstrap 4.3.1 educational templates
- Interactive components (flip cards, accordions, tabs, knowledge checks)
- WCAG 2.2 AA compliant styling
- Responsive design support

#### IMSCC Standards
- Full IMS Common Cartridge 1.3 support
- QTI 1.2 quiz format
- Native Brightspace assignment integration
- Discussion topic support

#### Documentation
- Comprehensive troubleshooting guide (26 patterns)
- Workflow reference documentation
- Getting started guide
- Pattern prevention guide

### Technical Details

#### IMSCC Patterns Documented
- Pattern 24: Version mismatch prevention
- Pattern 25: Correct assignment resource type (`assignment_xmlv1p0`)
- Pattern 26: Title elements on container items

#### Resource Types
| Content Type | Resource Type |
|--------------|---------------|
| Web Content | `webcontent` |
| Discussion | `imsdt_xmlv1p3` |
| Quiz | `imsqti_xmlv1p2/imscc_xmlv1p3/assessment` |
| Assignment | `assignment_xmlv1p0` |

## [1.1.0] - 2025-12-12

### Changed

#### WCAG 2.2 AA Upgrade
- Upgraded accessibility standard from WCAG 2.1 AA to WCAG 2.2 AA
- Added validation for 9 new WCAG 2.2 success criteria:
  - 2.4.11 Focus Not Obscured (Minimum)
  - 2.4.12 Focus Not Obscured (Enhanced)
  - 2.4.13 Focus Appearance (2px minimum outline, 3:1 contrast)
  - 2.5.7 Dragging Movements (single pointer alternatives)
  - 2.5.8 Target Size (Minimum) (24x24 CSS pixels)
  - 3.2.6 Consistent Help (help in consistent locations)
  - 3.3.7 Redundant Entry (auto-populate repeated fields)
  - 3.3.8 Accessible Authentication (Minimum) (no cognitive tests)
  - 3.3.9 Accessible Authentication (Enhanced)
- Updated all templates with WCAG 2.2 compliant focus indicators and target sizes
- Enhanced accessibility validators with WCAG 2.2 checks
- Updated all agent specifications to reference WCAG 2.2 AA

## [Unreleased]

### Planned
- Canvas-specific export optimization
- Moodle backup format support
- Additional interactive component templates
- Enhanced rubric generation
