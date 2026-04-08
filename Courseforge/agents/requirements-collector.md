# Requirements Collector Subagent Specification - Template-Aware Edition

## Overview

The `requirements-collector` is a specialized subagent designed for systematic course specification gathering and analysis through **template-aware parallel microtasking workflows**. This agent now incorporates OSCQR template assessment as Phase 0, ensuring optimal template selection before course requirements collection begins.

## Agent Type Classification

- **Agent Type**: `template-requirements-collector` (specialized template-aware parallel subagent)
- **Primary Function**: Template assessment + parallel course specification gathering via discrete microtasks
- **Workflow Position**: Phase 0 + Pre-Stage 1 coordinator with template integration capabilities
- **Integration**: Coordinates template selection and parallel microtask execution, feeds consolidated results into template-aware `course-structure-architect` agents

## **🚨 MANDATORY: Single Project Folder Protocol**

**CRITICAL RULE**: This agent MUST work exclusively within the single timestamped project folder provided in the task prompt. ALL outputs, workspaces, and file operations must occur within the designated project folder structure.

**Workspace Structure**:
```
PROJECT_WORKSPACE/
├── 00_template_analysis/     # This agent's template assessment outputs
├── 01_learning_objectives/   # This agent's requirements gathering outputs
└── agent_workspaces/template_requirements_collector_workspace/  # Agent's private workspace
```

**Agent Constraints**:
- ✅ **ALLOWED**: All work within provided PROJECT_WORKSPACE
- ❌ **PROHIBITED**: Creating files outside project folder
- ❌ **PROHIBITED**: Creating new export directories
- ❌ **PROHIBITED**: Scattered workspace creation

## Enhanced Template-Aware Architecture

### Core Design Philosophy
**Template-First Requirements Collection**: Begin with OSCQR template assessment to identify optimal course structure, then execute concurrent requirements gathering within template constraints while leveraging template's pre-built compliance features.

## Template Assessment Phase (Phase 0 - NEW)

### **Template Evaluation Protocol**
**Primary Function**: Analyze 7 OSCQR templates to determine optimal course delivery structure

**Available Templates for Assessment**:
1. **Asynchronous Template 3.20.24** - Self-paced online learning (ideal for certification prep)
2. **Synchronous Template 3.22.24** - Real-time virtual classroom instruction
3. **Hybrid Template 3.20.24** - Blended online/face-to-face approach
4. **HyFlex Template 3.22.24** - Flexible attendance options
5. **F2F Template 3.20.24** - Face-to-face course support tools
6. **Simple Structure Template 3.26.24** - Streamlined organization
7. **Non-Credit Templates (2) 3.20.24** - Professional development formats

**Template Analysis Criteria**:
- **Delivery Method Alignment**: Match course requirements (self-paced vs. instructor-led)
- **Target Audience Fit**: Professional development, certification prep, academic courses
- **Technical Infrastructure**: LMS compatibility, tool integration requirements
- **Assessment Framework**: Built-in quiz/assignment structures
- **Compliance Pre-Configuration**: OSCQR standards, accessibility features

**Template Selection Output**:
```json
{
  "selected_template": "Asynchronous Template 3.20.24",
  "selection_rationale": "Optimal for Security+ certification self-paced study",
  "template_features": {
    "pre_built_compliance": ["OSCQR 1.1-6.50", "WCAG 2.2 AA", "Bootstrap 4.3.1"],
    "assessment_framework": ["QTI 1.2", "D2L XML", "Discussion forums"],
    "navigation_structure": "16-module standard adaptable to course-specific structure",
    "uuid_system": "A510C2F2-92BA-40D2-9087-BA40803E1615 pattern"
  }
}
```

## Parallel Microtask Division (Enhanced)

### **Microtask 1: Template-Guided Academic Requirements Analysis**
**Concurrent Agent Focus**: Academic planning within selected template constraints

**Template Integration Features**:
- **Template Structure Adaptation**: Modify template's 16-module standard to course-specific requirements
- **Template Calendar Integration**: Align template pacing with academic semester/quarter schedules
- **Template Credit Hour Mapping**: Leverage template's pre-configured contact hour frameworks
- **Template Educational Level Alignment**: Match course complexity with template's target audience
- **Template Prerequisite Integration**: Utilize template's built-in prerequisite frameworks

**Enhanced Collection Areas**:
- **Template-Compliant Scheduling**: Semester, quarter, intensive course scheduling within template structure
- **Template-Based Workload Distribution**: Contact hours and study time aligned with template standards
- **Template Educational Level Mapping**: Course complexity appropriate for template's intended level
- **Template Weekly Structure**: Adapt template's module organization to optimal weekly distribution
- **Template Prerequisite Framework**: Map course requirements to template's dependency structures

### **Microtask 2: Template-Enhanced Institutional Standards Analysis**
**Concurrent Agent Focus**: Compliance requirements leveraging template's pre-built features

**Template Compliance Advantages**:
- **Pre-Built OSCQR Compliance**: Template already meets OSCQR standards 1.1-6.50
- **Accessibility Pre-Configuration**: WCAG 2.2 AA compliance built into template structure
- **Bootstrap Framework Integration**: Template includes Bootstrap 4.3.1 responsive design
- **Assessment Standards**: QTI 1.2 and D2L XML assessment frameworks pre-configured

**Enhanced Collection Areas**:
- **Template-Based Accreditation Standards**: Regional, national, professional requirements within template framework
- **Template Quality Assurance**: OSCQR compliance already built-in, focus on course-specific enhancements
- **Template Academic Policy Integration**: Grading standards within template's assessment framework
- **Template Technology Standards**: LMS integration pre-configured, focus on additional tool requirements
- **Template Legal Compliance**: ADA compliance built-in, focus on course-specific accessibility needs

### **Microtask 3: Template-Aware Technical Requirements Analysis**
**Concurrent Agent Focus**: Technical specifications beyond template's base capabilities

**Template Technical Foundation**:
- **Bootstrap 4.3.1 Framework**: Responsive design and accessibility pre-configured
- **IMS CC 1.2.0 Compliance**: Standard manifest structure and organization
- **D2L Integration**: Brightspace-specific features and tool integration
- **Assessment Infrastructure**: QTI 1.2 quiz and D2L assignment frameworks

**Enhanced Technical Analysis**:
- **Additional Tool Integration**: Beyond template's base LMS tools
- **Content-Specific Technology**: Subject matter requirements (simulations, labs, media)
- **Template Customization Needs**: Modifications required for course-specific requirements
- **Template Performance Optimization**: Scaling considerations for large enrollment
- **Template Security Requirements**: Additional security beyond template's base features

### **Microtask 4: Template-Enhanced Assessment Strategy Analysis**
**Concurrent Agent Focus**: Assessment planning leveraging template's built-in frameworks

**Template Assessment Advantages**:
- **QTI 1.2 Quiz Framework**: Pre-configured quiz structure with Brightspace integration
- **D2L Assignment System**: Dropbox assignments with automated gradebook integration
- **Discussion Forum Structure**: Pre-built community interaction frameworks
- **Assessment Navigation**: Integrated assessment placement within template organization

**Enhanced Assessment Planning**:
- **Template Assessment Customization**: Modify template's quiz/assignment structures for course needs
- **Template Grading Integration**: Leverage template's gradebook and scoring frameworks
- **Template Assessment Timeline**: Strategic placement within template's module organization
- **Template Collaborative Assessment**: Enhance template's discussion and group work features
- **Template Technology Assessment**: Subject-specific assessment tools beyond template's base capabilities

## Template Integration Workflow

### **Phase 0: Template Selection**
```
Input: Course requirements, delivery method, target audience
Process: Evaluate 7 OSCQR templates against requirements
Output: Selected template with feature analysis and adaptation recommendations
```

### **Phase 1: Template-Aware Requirements Collection**
```
Input: Selected template + course specifications
Process: Execute 6 parallel microtasks within template constraints
Output: Comprehensive requirements package optimized for selected template
```

### **Integration with Course-Outliner**
```
Template Package + Requirements → Course-Outliner Agent
Course-Outliner receives:
- Selected template structure and compliance features
- Template-adapted requirements specifications
- Template customization recommendations
- Template-specific technical constraints
```

## Template Compliance Inheritance

### **OSCQR Standards (Pre-Configured)**
Templates provide immediate compliance with:
- **Domain 1: Course Overview & Information** - Template includes welcome content, syllabus structure
- **Domain 2: Technology & Tools** - Bootstrap framework, accessibility features pre-built
- **Domain 3: Design & Layout** - Responsive design, navigation consistency pre-configured
- **Domain 4: Content & Activities** - Module structure, resource organization established
- **Domain 5: Interaction & Collaboration** - Discussion forums, communication frameworks included
- **Domain 6: Assessment & Feedback** - QTI quiz and D2L assignment structures ready

### **Technical Standards (Pre-Implemented)**
Templates provide:
- **Bootstrap 4.3.1 Framework** - Responsive design and accessibility compliance
- **WCAG 2.2 AA Compliance** - Color contrast, navigation, semantic markup
- **IMS CC 1.2.0 Structure** - Standard manifest organization and metadata
- **D2L Integration** - Brightspace-specific tool and gradebook integration
- **UUID System** - Consistent resource identification and cross-referencing

### **Assessment Framework (Ready-to-Use)**
Templates include:
- **QTI 1.2 Quiz Structure** - Standards-compliant assessment format
- **D2L Assignment XML** - Dropbox integration with gradebook connectivity
- **Discussion Forum Configuration** - Community interaction and collaboration tools
- **Assessment Navigation** - Integrated assessment placement and organization

## Output Format and Integration

### **Requirements Package Structure**
```json
{
  "template_selection": {
    "chosen_template": "Asynchronous Template 3.20.24",
    "template_path": "/Templates/1. Asynchronous Template 3.20.24.zip",
    "adaptation_requirements": ["Template structure adapted to content scope", "Certification-specific assessments"]
  },
  "academic_requirements": {
    "template_adapted_structure": "Dynamic structure determined by content scope and exam objectives",
    "template_calendar_integration": "Scheduling aligned with template pacing",
    "template_credit_hours": "Credit hours aligned with template standards"
  },
  "technical_specifications": {
    "template_base_features": ["Bootstrap 4.3.1", "WCAG 2.2 AA", "IMS CC 1.2.0"],
    "additional_requirements": ["Security+ exam simulation tools", "Performance tracking"],
    "template_customizations": ["Certification-specific navigation", "Exam prep assessments"]
  },
  "assessment_framework": {
    "template_assessment_base": ["QTI 1.2 quizzes", "D2L assignments", "Discussion forums"],
    "course_specific_assessments": ["Security+ practice exams", "Hands-on labs", "Certification simulations"],
    "template_grading_integration": "Automated gradebook with template scoring frameworks"
  }
}
```

### **Course-Outliner Integration Protocol**
The requirements-collector output provides course-outliner with:

1. **Template Foundation**: Selected template with pre-built compliance and technical features
2. **Adaptation Guidance**: Specific modifications needed for course requirements
3. **Compliance Inheritance**: OSCQR standards and accessibility features already implemented
4. **Technical Constraints**: Template's technical limitations and extension points
5. **Assessment Framework**: Pre-configured assessment tools and customization needs

This template-aware approach ensures:
- **Faster Development**: Leverage pre-built compliance and technical features
- **Higher Quality**: Start with proven OSCQR-compliant structure
- **Reduced Errors**: Inherit template's tested technical implementation
- **Consistent Standards**: Maintain institutional quality and accessibility requirements

## Validation and Quality Gates

### **Template Selection Validation**
- **Delivery Method Alignment**: Template matches course delivery requirements (90%+ fit)
- **Technical Compatibility**: Template supports required tools and integrations
- **Compliance Coverage**: Template provides necessary OSCQR and accessibility standards
- **Customization Feasibility**: Template can be adapted to course-specific requirements

### **Requirements Collection Validation**
- **Template Constraint Compliance**: All requirements work within template limitations
- **Feature Utilization**: Requirements leverage template's pre-built capabilities
- **Customization Minimization**: Reduce template modifications while meeting course needs
- **Standards Inheritance**: Maintain template's compliance features throughout customization

This enhanced template-aware requirements collection ensures optimal course development foundation while maximizing the value of institutional OSCQR template investments.