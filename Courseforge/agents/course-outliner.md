# Course Outliner Subagent Specification - Template-Aware Edition

## Overview

The `course-outliner` is a specialized subagent designed for comprehensive course structure planning and OSCQR improvement implementation through **template-optimized parallel microtasking workflows**. This agent now leverages pre-selected OSCQR templates to dramatically reduce development time while ensuring educational quality and compliance standards.

## Agent Type Classification

- **Agent Type**: `course-structure-architect` (specialized template-optimized parallel subagent)
- **Primary Function**: Template-based course structure creation + OSCQR improvement implementation via discrete microtasks
- **Workflow Position**: Phase 1 + Phase 3 coordinator with template inheritance capabilities
- **Integration**: Receives template packages from `template-requirements-collector`, feeds template-enhanced structures into `educational-content-creator` agents

## **🚨 MANDATORY: Single Project Folder Protocol**

**CRITICAL RULE**: This agent MUST work exclusively within the single timestamped project folder provided in the task prompt. ALL outputs, workspaces, and file operations must occur within the designated project folder structure.

**Workspace Structure**:
```
PROJECT_WORKSPACE/
├── 01_learning_objectives/   # Learning objectives development
├── 02_course_planning/       # This agent's course structure outputs
└── agent_workspaces/course_structure_architect_workspace/  # Agent's private workspace
```

**Agent Constraints**:
- ✅ **ALLOWED**: All work within provided PROJECT_WORKSPACE
- ❌ **PROHIBITED**: Creating files outside project folder
- ❌ **PROHIBITED**: Creating new export directories
- ❌ **PROHIBITED**: Scattered workspace creation

## Enhanced Template-Optimized Architecture

### Core Design Philosophy
**Template-First Course Planning**: Begin with pre-selected OSCQR template foundations to inherit compliance features, then execute concurrent course planning within template constraints while leveraging template's pre-built educational frameworks.

## Template Inheritance Phase (Phase 1 - ENHANCED)

### **Template Foundation Integration**
**Primary Function**: Receive template selection from requirements-collector and integrate template features into course structure planning

**Template Inheritance Features**:
1. **Pre-Built OSCQR Compliance** - Template already meets OSCQR standards 1.1-6.50
2. **Bootstrap Framework Integration** - Template includes Bootstrap 4.3.1 responsive design  
3. **Assessment Infrastructure** - QTI 1.2 and D2L XML assessment frameworks pre-configured
4. **Navigation Structure** - Template provides 16-module standard adaptable to course needs
5. **Accessibility Features** - WCAG 2.2 AA compliance built into template structure

**Template Adaptation Protocol**:
```json
{
  "template_received": {
    "template_type": "Asynchronous Template 3.20.24",
    "pre_built_features": ["OSCQR 1.1-6.50", "Bootstrap 4.3.1", "WCAG 2.2 AA"],
    "assessment_framework": ["QTI 1.2", "D2L XML", "Discussion forums"],
    "structure_base": "16-module standard"
  },
  "adaptation_requirements": {
    "target_structure": "Dynamic - determined by content scope and exam objectives",
    "specialization": "Certification course aligned to exam domains",
    "customizations": ["Exam simulation", "Hands-on labs", "Performance tracking"]
  }
}
```

## Parallel Microtask Division (Template-Enhanced)

### **Microtask 1: Template-Aware Learning Objectives Development**
**Concurrent Agent Focus**: Course and weekly learning objectives within template educational framework

**Template Educational Advantages**:
- **Pre-Built Learning Scaffolds**: Template includes progression frameworks and assessment alignment
- **OSCQR Compliance**: Learning objectives structure already meets standards 1.1-1.10 and 4.29-4.37
- **Assessment Integration**: Objectives designed to work with template's QTI and D2L assessment tools

**Enhanced Learning Objectives Development**:
- **Template-Guided Course Objectives**: Adapt template's educational framework to course-specific content
- **Template Assessment Alignment**: Leverage template's pre-configured assessment-objective mapping
- **Template Progressive Learning**: Utilize template's scaffolding structure for learning progression
- **Template Compliance Integration**: Ensure objectives work with template's OSCQR pre-compliance

### **Microtask 2: Template-Enhanced Content Structure Planning**
**Concurrent Agent Focus**: Course structure leveraging template's organizational framework

**Template Structure Advantages**:
- **Flexible Module Adaptation**: Systematic adaptation of template's proven structure to content requirements
- **Navigation Consistency**: Template provides tested navigation patterns and user experience
- **Responsive Design**: Template includes Bootstrap 4.3.1 mobile-friendly layouts
- **Accessibility Framework**: WCAG 2.2 AA compliance built into template structure

**Enhanced Structure Planning**:
- **Template Navigation Adaptation**: Modify template navigation for course-specific module structure
- **Template Visual Consistency**: Leverage template's design system for course branding
- **Template Mobile Optimization**: Utilize template's responsive breakpoints and touch interfaces
- **Template Accessibility**: Inherit template's screen reader compatibility and keyboard navigation

### **Microtask 3: Template-Optimized Assessment Framework Design**
**Concurrent Agent Focus**: Assessment strategy leveraging template's pre-configured tools

**Template Assessment Advantages**:
- **QTI 1.2 Compliance**: Template includes working quiz framework with Brightspace integration
- **D2L XML Integration**: Template provides functional assignment dropbox configurations
- **Discussion Framework**: Template includes community interaction and collaboration tools
- **Gradebook Connectivity**: Template ensures assessment results integrate with Brightspace gradebook

**Enhanced Assessment Planning**:
- **Template Quiz Customization**: Adapt template's QTI framework for Security+ practice exams
- **Template Assignment Integration**: Leverage template's D2L dropbox system for hands-on labs
- **Template Discussion Enhancement**: Customize template's forums for certification study groups
- **Template Grading Workflow**: Utilize template's automated scoring and feedback systems

### **Microtask 4: Template-Aware Resource Integration Planning**
**Concurrent Agent Focus**: Multimedia and resource planning within template capabilities

**Template Resource Framework**:
- **CSS/JavaScript Integration**: Template provides optimized resource loading and organization
- **Image and Media Support**: Template includes responsive image handling and accessibility
- **External Tool Integration**: Template supports LTI and third-party tool embedding
- **Performance Optimization**: Template includes CDN integration and caching strategies

**Enhanced Resource Planning**:
- **Template Media Framework**: Utilize template's responsive image and video embedding systems
- **Template Tool Integration**: Leverage template's LTI support for Security+ simulation tools
- **Template Performance**: Use template's optimized resource loading for large certification content
- **Template Accessibility**: Ensure all resources work with template's assistive technology support

### **Microtask 5: Template-Enhanced UDL and Accessibility Implementation**
**Concurrent Agent Focus**: Universal design and accessibility within template compliance

**Template UDL Advantages**:
- **WCAG 2.2 AA Pre-Compliance**: Template already meets accessibility standards
- **Multiple Engagement Pathways**: Template includes varied interaction methods and content formats
- **Assistive Technology Support**: Template tested with screen readers and keyboard navigation
- **Responsive Design**: Template supports diverse device types and display preferences

**Enhanced UDL Implementation**:
- **Template Accessibility Enhancement**: Build upon template's WCAG compliance for certification content
- **Template Engagement Variety**: Use template's interaction frameworks for diverse learning styles
- **Template Assistive Integration**: Leverage template's screen reader optimization for complex content
- **Template Device Adaptation**: Utilize template's responsive design for mobile certification study

### **Microtask 6: Template-Optimized Quality Assurance Framework**
**Concurrent Agent Focus**: Quality validation and improvement processes within template standards

**Template Quality Framework**:
- **OSCQR Pre-Compliance**: Template already meets educational quality standards
- **Technical Validation**: Template includes tested import and functionality protocols
- **User Experience**: Template provides proven navigation and interaction patterns
- **Performance Standards**: Template includes optimized loading and accessibility performance

**Enhanced Quality Assurance**:
- **Template Compliance Verification**: Ensure course adaptations maintain template's OSCQR compliance
- **Template Technical Validation**: Verify customizations work with template's import systems
- **Template User Testing**: Leverage template's proven user experience for certification content
- **Template Performance Monitoring**: Use template's optimization features for large course content

## Template Integration Workflow Enhancement

### **Phase 1: Template-Based Course Planning**
```
Input: Template package + requirements specifications
Process: Execute 6 parallel microtasks within template framework
Output: Template-enhanced course structure with inherited compliance
```

### **Integration with Content-Generator**
```
Template-Enhanced Structure + Course Outline → Content-Generator Agents
Content-Generator receives:
- Template design system and component library
- Template-compliant navigation structure
- Template assessment framework and integration points
- Template accessibility features and requirements
```

### **Phase 3: Template-Aware OSCQR Improvement Implementation**
**Enhanced Function**: Apply OSCQR feedback while preserving template advantages

**Template Preservation Protocol**:
- **Maintain Template Compliance**: Ensure improvements don't break template's OSCQR pre-compliance
- **Leverage Template Features**: Use template's built-in capabilities to address feedback
- **Template-Consistent Enhancement**: Apply improvements using template's design patterns
- **Template Accessibility**: Ensure changes maintain template's WCAG 2.2 AA compliance

**OSCQR Improvement Microtasks**:
```
Microtask 1: Critical issue resolution using template frameworks
Microtask 2: Important quality enhancement through template features
Microtask 3: Recommended improvement implementation with template patterns
Microtask 4: Template compliance verification and quality validation
Microtask 5: User experience optimization using template best practices
Microtask 6: Final integration testing and template functionality validation
```

## Template Compliance Inheritance Benefits

### **Pre-Built Educational Standards**
Templates provide immediate compliance with:
- **OSCQR Domain 1**: Course overview and information structure pre-built
- **OSCQR Domain 2**: Technology tools and accessibility features included
- **OSCQR Domain 3**: Design and layout consistency with responsive frameworks
- **OSCQR Domain 4**: Content and activity frameworks with engagement features
- **OSCQR Domain 5**: Interaction and collaboration tools pre-configured
- **OSCQR Domain 6**: Assessment and feedback systems with gradebook integration

### **Technical Infrastructure Advantages**
Templates include:
- **Bootstrap 4.3.1 Framework**: Responsive design and component library
- **WCAG 2.2 AA Compliance**: Color contrast, navigation, and semantic markup
- **IMS CC 1.2.0 Structure**: Standard manifest organization and metadata
- **Assessment Integration**: QTI 1.2 quiz and D2L assignment frameworks ready
- **Performance Optimization**: CDN integration, caching, and resource optimization

### **Development Efficiency Gains**
Template-aware course outlining provides:
- **75%+ Time Reduction**: Leverage pre-built compliance and technical features
- **Quality Assurance**: Start with proven educational and technical frameworks
- **Risk Reduction**: Use tested template implementation and import compatibility
- **Consistency**: Maintain institutional branding and user experience standards

## Output Format and Integration

### **Template-Enhanced Course Structure Package**
```json
{
  "template_integration": {
    "base_template": "Asynchronous Template 3.20.24",
    "template_features_preserved": ["OSCQR compliance", "Bootstrap framework", "Assessment tools"],
    "template_customizations": ["Content-driven structure", "Certification-aligned content", "Assessment tools"]
  },
  "course_structure": {
    "template_adapted_navigation": "Dynamic structure within template framework based on content requirements",
    "template_learning_objectives": "Course and weekly objectives aligned with template assessment tools",
    "template_assessment_framework": "QTI quizzes and D2L assignments using template infrastructure"
  },
  "oscqr_compliance": {
    "inherited_standards": "Template provides OSCQR 1.1-6.50 compliance",
    "course_specific_validation": "Verified compatibility with Security+ content requirements",
    "quality_gates": "Template compliance maintained through all customizations"
  },
  "content_generator_handoff": {
    "template_design_system": "Bootstrap 4.3.1 components and styling framework",
    "template_technical_specs": "IMS CC 1.2.0 structure with D2L XML integration",
    "template_quality_requirements": "WCAG 2.2 AA compliance and educational depth standards"
  }
}
```

### **Content-Generator Integration Protocol**
The template-enhanced course-outliner output provides content-generator with:

1. **Template Design Foundation**: Bootstrap component library and styling framework
2. **Compliance Framework**: OSCQR and accessibility standards already implemented
3. **Assessment Infrastructure**: Pre-configured QTI and D2L assessment templates
4. **Technical Constraints**: Template's proven import compatibility and performance features
5. **Quality Standards**: Educational depth requirements aligned with template capabilities

This template-aware approach ensures:
- **Faster Development**: Leverage pre-built compliance and technical infrastructure
- **Higher Quality**: Start with proven educational and technical frameworks  
- **Reduced Risk**: Use tested template implementation and import compatibility
- **Institutional Consistency**: Maintain branding and user experience standards

## Textbook Objectives Integration

### **Multi-Source Learning Objectives**

The course-outliner can now receive learning objectives from multiple sources:

1. **Exam Research** (existing) - from `exam-research` agent
2. **Textbook Analysis** (new) - from `objective-synthesizer` agent
3. **Explicit Requirements** - from user-provided specifications

### **Input Sources**

#### Exam Research Input
```json
{
  "source": "exam-research",
  "format": "certification_objectives",
  "data": {
    "domain_analysis": { ... },
    "subject_research": { ... },
    "learning_pathway": { ... }
  }
}
```

#### Textbook Objectives Input (NEW)
```json
{
  "source": "objective-synthesizer",
  "format": "textbook_objectives",
  "schema": "schemas/learning-objectives/learning_objectives_schema.json",
  "data": {
    "documentMetadata": { ... },
    "courseObjectives": [ ... ],
    "chapters": [
      {
        "chapterId": "ch1",
        "chapterTitle": "Chapter Title",
        "chapterObjectives": [ ... ],
        "sections": [ ... ]
      }
    ],
    "objectivesSummary": { ... }
  }
}
```

### **Objectives Merge Strategy**

When both exam research and textbook objectives are available, the course-outliner performs intelligent merging:

#### Merge Protocol
```python
def merge_learning_objectives(exam_objectives, textbook_objectives):
    """
    Merge objectives from exam research and textbook analysis.

    Strategy: union_with_dedup
    - Include ALL unique objectives from both sources
    - Deduplicate semantically similar objectives
    - Preserve source attribution
    """
    merged = []

    # Add all exam objectives (primary)
    for obj in exam_objectives:
        merged.append({
            **obj,
            "source": "exam_research",
            "priority": "primary"
        })

    # Add textbook objectives with deduplication
    for obj in textbook_objectives:
        if not is_semantically_duplicate(obj, merged):
            merged.append({
                **obj,
                "source": "textbook_analysis",
                "priority": "complementary"
            })
        else:
            # Enhance existing objective with textbook context
            enhance_existing_objective(merged, obj)

    return merged
```

#### Semantic Deduplication
```python
def is_semantically_duplicate(new_obj, existing_objs):
    """
    Check if an objective is semantically similar to existing ones.

    Criteria:
    - Same Bloom's verb
    - Similar key concepts (>70% overlap)
    - Similar statement structure
    """
    for existing in existing_objs:
        # Check verb match
        if new_obj['bloomVerb'] == existing['bloomVerb']:
            # Check concept overlap
            new_concepts = set(new_obj.get('keyConcepts', []))
            existing_concepts = set(existing.get('keyConcepts', []))

            if new_concepts and existing_concepts:
                overlap = len(new_concepts & existing_concepts) / len(new_concepts | existing_concepts)
                if overlap > 0.7:
                    return True

    return False
```

### **Workflow Integration**

#### Textbook-First Workflow
When starting with a textbook (no exam objectives):
```
textbook-ingestor → objective-synthesizer → course-outliner → content-generator
```

#### Hybrid Workflow
When both textbook and exam objectives are available:
```
                exam-research ────────┐
                                      ├──► course-outliner → content-generator
textbook-ingestor → objective-synthesizer ─┘
```

#### Fallback Behavior
- If only exam objectives: Use existing workflow unchanged
- If only textbook objectives: Build structure from textbook hierarchy
- If both: Merge with exam objectives as primary, textbook as complementary

### **Structure Mapping from Textbook Objectives**

When building course structure from textbook objectives:

```python
def map_textbook_to_course_structure(textbook_objectives):
    """
    Map textbook chapter/section hierarchy to course weeks/modules.
    """
    structure = {
        "weeks": [],
        "learning_objectives": {
            "course_level": textbook_objectives.get("courseObjectives", []),
            "week_level": [],
            "module_level": []
        }
    }

    # Map chapters to weeks
    for i, chapter in enumerate(textbook_objectives.get("chapters", [])):
        week = {
            "week_number": i + 1,
            "title": chapter["chapterTitle"],
            "objectives": chapter["chapterObjectives"],
            "modules": []
        }

        # Map sections to modules
        for j, section in enumerate(chapter.get("sections", [])):
            module = {
                "module_number": j + 1,
                "title": section["sectionTitle"],
                "objectives": section["sectionObjectives"]
            }
            week["modules"].append(module)

        structure["weeks"].append(week)

    return structure
```

### **Output Enhancement**

The template-enhanced course structure output now includes textbook integration metadata:

```json
{
  "template_integration": { ... },
  "course_structure": { ... },
  "objective_sources": {
    "exam_research": {
      "included": true,
      "objective_count": 45,
      "domains_covered": ["Domain 1", "Domain 2", "Domain 3"]
    },
    "textbook_analysis": {
      "included": true,
      "objective_count": 87,
      "chapters_covered": 12,
      "merge_strategy": "union_with_dedup",
      "deduplicated_count": 23
    }
  },
  "merged_objectives_summary": {
    "total_unique": 109,
    "by_bloom_level": {
      "remember": 15,
      "understand": 22,
      "apply": 38,
      "analyze": 20,
      "evaluate": 10,
      "create": 4
    }
  }
}
```

## Validation and Quality Gates

### **Template Compliance Validation**
- **Feature Preservation**: Template's OSCQR compliance and technical features maintained (95%+ retention)
- **Customization Integration**: Course-specific requirements implemented without breaking template
- **Performance Standards**: Template's loading speed and accessibility performance preserved
- **Import Compatibility**: Template's proven Brightspace import functionality maintained

### **Educational Quality Validation**
- **Learning Objectives Alignment**: Course and weekly objectives support template's assessment framework
- **Content Structure**: Course module structure works within template navigation framework
- **Assessment Integration**: Template's QTI and D2L tools support course assessment requirements
- **UDL Implementation**: Template's accessibility features enhanced for course-specific needs

This enhanced template-aware course outlining ensures optimal development foundation while maximizing the value of institutional OSCQR template investments and dramatically reducing course development timelines.