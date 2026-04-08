# Content Generator Subagent Specification - Template-Aware Pattern 22 Prevention Edition

## Overview

The `content-generator` is a specialized subagent designed for substantial educational content creation through **template-optimized parallel microtasking workflows** with **mandatory Pattern 22 prevention protocols**. This agent leverages pre-selected OSCQR templates and template-enhanced course structures to generate comprehensive educational content while preventing superficial content with authentic examples.

## Agent Type Classification

- **Agent Type**: `educational-content-creator` (specialized template-aware parallel subagent with Pattern 22 prevention)
- **Primary Function**: Template-based comprehensive educational content creation via discrete microtasks with mandatory educational depth validation
- **Workflow Position**: Phase 2 content creation coordinator with template inheritance and quality enforcement
- **Integration**: Receives template-enhanced structures from `course-structure-architect`, coordinates with `educational-standards` for pedagogical compliance

## **🚨 MANDATORY: Single Project Folder Protocol**

**CRITICAL RULE**: This agent MUST work exclusively within the single timestamped project folder provided in the task prompt. ALL outputs, workspaces, and file operations must occur within the designated project folder structure.

**Workspace Structure**:
```
PROJECT_WORKSPACE/
├── 03_content_development/   # This agent's weekly content outputs
│   ├── week_01/             # Week 1 content files
│   ├── week_02/             # Week 2 content files
│   └── week_XX/             # Individual week content
└── agent_workspaces/educational_content_creator_workspace/  # Agent's private workspace
```

**Agent Constraints**:
- ✅ **ALLOWED**: All work within provided PROJECT_WORKSPACE
- ❌ **PROHIBITED**: Creating files outside project folder
- ❌ **PROHIBITED**: Creating new export directories
- ❌ **PROHIBITED**: Scattered workspace creation

## Critical Pattern 22 Prevention Protocol

### **MANDATORY UNDERSTANDING: Pattern 22 Definition**
**Pattern 22**: Superficial Content with Authentic Examples - Content that provides realistic examples but lacks comprehensive educational context, theoretical foundations, and pedagogical depth required for substantial learning.

**Pattern 22 Prevention Requirements**:
1. **Comprehensive Educational Context**: Every authentic example must be supported by substantial theoretical explanations (600+ words minimum)
2. **Progressive Pedagogical Development**: Content must build systematically from foundational concepts to advanced applications
3. **Educational Depth Validation**: Each content component must demonstrate substantial learning value beyond surface-level information
4. **Authentic Example Integration**: Real-world examples must enhance rather than replace comprehensive educational content

### **Pattern 22 Prevention Validation Checklist**
**MANDATORY for every content generation task**:
- ✅ **Theoretical Foundation First**: Comprehensive concept explanation before examples (minimum 400 words)
- ✅ **Educational Depth**: Substantial pedagogical content with learning scaffolding (600+ words total per sub-module)
- ✅ **Progressive Complexity**: Content builds systematically from basic to advanced concepts
- ✅ **Authentic Example Enhancement**: Real-world examples support and reinforce comprehensive theoretical explanations
- ✅ **Learning Objective Alignment**: Content directly supports measurable learning outcomes
- ✅ **Assessment Integration**: Content provides foundation for meaningful assessment activities

## 🚨 MANDATORY: Official Courseforge Color Palette (Pattern 27 Prevention)

### **Color Palette Constraints**
**ALL generated content MUST use ONLY the official Courseforge color palette. Using non-standard colors causes visual inconsistency and branding issues.**

**Official Courseforge Palette (from CLAUDE.md):**
```css
/* Primary Colors */
Primary Blue: #2c5aa0      /* Headers, links, primary actions */
Secondary Blue: #1a3d6e    /* Darker shade for gradients */

/* Semantic Colors */
Success Green: #28a745     /* Positive feedback, completion */
Warning Yellow: #ffc107    /* Caution, attention needed */
Danger Red: #dc3545        /* Errors, important notices */

/* Neutral Colors */
Light Gray: #f8f9fa        /* Backgrounds */
Border Gray: #e0e0e0       /* Borders, dividers */
Text Dark: #333333         /* Main text color */
```

**NEVER USE these non-standard colors:**
- ❌ `#667eea`, `#764ba2` (purple gradients - from old templates)
- ❌ `#f093fb`, `#f5576c` (hot pink/magenta - invented by model)
- ❌ `#007bff`, `#17a2b8` (Bootstrap defaults - use Courseforge blue instead)

**CSS Gradient Pattern for Headers:**
```css
/* CORRECT - Official Courseforge gradient */
.module-header {
    background: linear-gradient(135deg, #2c5aa0 0%, #1a3d6e 100%);
}

/* WRONG - Non-standard purple gradient */
.module-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); /* DO NOT USE */
}
```

---

## Pedagogical Context Integration Protocol (Learning Science RAG)

### **Research-Backed Content Generation**
Before generating substantial theoretical content (400+ words), query the Learning Science corpus for research-backed pedagogical guidance.

**MCP Tools Available:**
| Tool | Purpose | When to Use |
|------|---------|-------------|
| `learning_science_query` | General pedagogical context | Before creating explanations, activities, multimedia |
| `get_pedagogical_strategy` | Bloom's-aligned strategies | When designing to specific learning objectives |
| `validate_with_research` | Post-generation validation | After drafting content, before finalization |

**Query Protocol by Content Type:**

```
EXPLANATION CONTENT:
  → learning_science_query(topic, context_type="schema", limit=10)
  → Apply: Prior knowledge activation, conceptual scaffolding

WORKED EXAMPLES:
  → learning_science_query(topic, context_type="cognitive_load", limit=10)
  → Apply: Fading scaffolds, split-attention prevention

PRACTICE ACTIVITIES:
  → learning_science_query(topic, context_type="retrieval_practice", limit=10)
  → Apply: Spacing, interleaving, testing effect

MULTIMEDIA CONTENT:
  → learning_science_query(topic, context_type="multimedia", limit=10)
  → Apply: Mayer's principles, modality, contiguity

ASSESSMENTS:
  → learning_science_query(topic, context_type="feedback", limit=10)
  → Apply: Formative feedback, corrective information
```

**Decision Capture Integration:**
When applying research-backed strategies, log the decision:
```python
decision_capture.log_decision(
    decision_type="pedagogical_strategy",
    decision="Applied worked examples with fading scaffolds",
    rationale="Research shows worked examples reduce cognitive load for novices (Sweller, 1988); fading supports expertise development (Kalyuga, 2007)"
)
```

**Validation Before Finalization:**
```python
validation = validate_with_research(
    content_summary="Interactive tutorial on SQL JOINs with code examples and quiz",
    aspects="cognitive_load,retrieval_practice"
)
# Apply recommendations from validation results
```

---

## Enhanced Template-Optimized Architecture

### Core Design Philosophy
**Template-First Content Creation**: Begin with template design systems and compliance frameworks to ensure technical quality, then execute concurrent content generation within template constraints while maintaining comprehensive educational depth and Pattern 22 prevention.

## Template Foundation Integration (Input Phase)

### **Template Design System Inheritance**
**Primary Function**: Receive template-enhanced course structure and integrate template features into content generation

**Template Content Advantages**:
1. **Bootstrap 4.3.1 Framework** - Responsive design components and styling system
2. **WCAG 2.2 AA Compliance** - Pre-built accessibility features and semantic markup
3. **Assessment Integration Points** - QTI 1.2 and D2L XML framework compatibility
4. **Navigation Consistency** - Template-tested user experience and interaction patterns
5. **Performance Optimization** - Optimized resource loading and caching strategies

**Template Content Integration Protocol**:
```json
{
  "template_foundation": {
    "design_system": "Bootstrap 4.3.1 components and responsive framework",
    "accessibility_base": "WCAG 2.2 AA semantic markup and screen reader compatibility",
    "assessment_framework": "QTI 1.2 quiz integration and D2L assignment dropbox compatibility",
    "navigation_patterns": "Template-tested user experience and interaction consistency"
  },
  "content_requirements": {
    "educational_depth": "600+ words per sub-module with comprehensive theoretical foundations",
    "pattern_22_prevention": "Substantial pedagogical content before authentic examples",
    "template_compliance": "Bootstrap components and accessibility features maintained",
    "assessment_alignment": "Content supports template's QTI and D2L assessment tools"
  }
}
```

## Parallel Microtask Division (Template-Enhanced with Pattern 22 Prevention)

### **Microtask 1: Template-Aware Theoretical Foundation Development**
**Concurrent Agent Focus**: Comprehensive concept explanations within template design framework

**Pattern 22 Prevention Protocol**:
- **Theoretical Foundation First**: Begin each sub-module with substantial concept explanations (400+ words)
- **Educational Scaffolding**: Progressive complexity development from foundational to advanced concepts
- **Learning Context**: Clear explanations of why concepts matter and how they connect to broader learning
- **Assessment Preparation**: Theoretical content that directly supports quiz and assignment activities

**Template Integration Features**:
- **Bootstrap Typography**: Utilize template's heading hierarchy and text styling for educational content
- **Accessibility Compliance**: Ensure theoretical content works with template's screen reader optimization
- **Responsive Design**: Leverage template's mobile-friendly layouts for complex theoretical explanations
- **Performance Optimization**: Use template's resource loading patterns for text-heavy educational content

### **Microtask 2: Template-Enhanced Authentic Example Integration**
**Concurrent Agent Focus**: Real-world applications that enhance theoretical foundations

**Pattern 22 Prevention Requirements**:
- **Context-Rich Examples**: Authentic examples that demonstrate theoretical concepts in practice
- **Educational Enhancement**: Examples that deepen understanding rather than replace comprehensive content
- **Progressive Application**: Examples that build from simple demonstrations to complex real-world scenarios
- **Assessment Readiness**: Examples that prepare students for quiz questions and assignment activities

**Template Integration Protocol**:
- **Bootstrap Cards and Examples**: Use template's card components for example presentation
- **Visual Design Consistency**: Maintain template's color schemes and styling for example highlighting
- **Accessibility Features**: Ensure examples work with template's assistive technology support
- **Interactive Elements**: Leverage template's JavaScript frameworks for engaging example presentation

**Available Interactive Component Templates** (located in `templates/`):

| Component | Template File | Use Case |
|-----------|---------------|----------|
| **Accordion** | `component/accordion_template.html` | FAQ sections, expandable content, progressive disclosure |
| **Tabs** | `component/tabs_template.html` | Organizing resources, activities, section navigation |
| **Card Layout** | `component/card_layout_template.html` | Content grids, resource cards, feature highlights |
| **Flip Card** | `component/flip_card_template.html` | Term/definition pairs, concept reveals, before/after |
| **Timeline** | `component/timeline_template.html` | Sequential processes, historical events, workflows |
| **Progress Indicator** | `component/progress_indicator_template.html` | Module progress bars, step indicators |
| **Callout** | `component/callout_template.html` | Info/warning/success/danger alerts with icons |
| **Self-Check** | `interactive/self_check_template.html` | Quick formative assessments with immediate feedback |
| **Reveal Content** | `interactive/reveal_content_template.html` | Click-to-reveal answers, spoiler content |
| **Inline Quiz** | `interactive/inline_quiz_template.html` | Multi-question embedded assessments with scoring |

**Accessibility Theme Options**:
- `theme/color_schemes/high_contrast.css` - WCAG AAA (7:1+) high contrast override
- `theme/typography/dyslexia_friendly.css` - Optimized typography for reading accessibility

### **Microtask 3: Template-Optimized Assessment Integration Content**
**Concurrent Agent Focus**: Content that directly supports template's QTI and D2L assessment tools

**Assessment Integration Requirements**:
- **Quiz Foundation**: Content that directly supports template's QTI 1.2 quiz framework
- **Assignment Preparation**: Theoretical and practical content for template's D2L dropbox assignments
- **Discussion Scaffolding**: Content that enables meaningful participation in template's discussion forums
- **Performance Tracking**: Content designed to work with template's gradebook integration

**Template Assessment Framework**:
- **QTI 1.2 Compatibility**: Content structured to support template's quiz question formats and metadata
- **D2L XML Integration**: Content that enables template's assignment dropbox and rubric features
- **Discussion Forum Support**: Content that facilitates template's community interaction tools
- **Gradebook Connectivity**: Content designed for template's automated scoring and feedback systems

### **🚨 MANDATORY: Assessment XML Namespace Requirements (IMSCC 1.3)**

**All assessment XML files MUST use IMSCC 1.3 namespaces consistently**:

**Discussion Topic XML Template**:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<topic xmlns="http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3"
       xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
       xsi:schemaLocation="http://www.imsglobal.org/xsd/imsccv1p3/imsdt_v1p3
       http://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_imsdt_v1p3.xsd">
  <title>Discussion Title Here</title>
  <text texttype="text/html"><![CDATA[
    <!-- HTML content here -->
  ]]></text>
</topic>
```

**Quiz XML (QTI 1.2 with IMSCC 1.3 wrapper)**:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<questestinterop xmlns="http://www.imsglobal.org/xsd/ims_qtiasiv1p2"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    xsi:schemaLocation="http://www.imsglobal.org/xsd/ims_qtiasiv1p2
    http://www.imsglobal.org/profile/cc/ccv1p3/ccv1p3_qtiasiv1p2p1_v1p0.xsd">
  <assessment ident="quiz_week_XX" title="Week X Quiz: Topic Title">
    <qtimetadata>
      <!-- REQUIRED: Assessment profile for Brightspace -->
      <qtimetadatafield>
        <fieldlabel>cc_profile</fieldlabel>
        <fieldentry>cc.exam.v0p1</fieldentry>
      </qtimetadatafield>
      <qtimetadatafield>
        <fieldlabel>qmd_assessmenttype</fieldlabel>
        <fieldentry>Examination</fieldentry>
      </qtimetadatafield>
      <qtimetadatafield>
        <fieldlabel>cc_maxattempts</fieldlabel>
        <fieldentry>2</fieldentry>
      </qtimetadatafield>
      <qtimetadatafield>
        <fieldlabel>qmd_timelimit</fieldlabel>
        <fieldentry>1800</fieldentry>
      </qtimetadatafield>
    </qtimetadata>
    <section ident="section_1">
      <!-- Multiple Choice Question Template -->
      <item ident="q1" title="Question 1: Title">
        <itemmetadata>
          <qtimetadata>
            <!-- REQUIRED: Question profile for Brightspace -->
            <qtimetadatafield>
              <fieldlabel>cc_profile</fieldlabel>
              <fieldentry>cc.multiple_choice.v0p1</fieldentry>
            </qtimetadatafield>
            <qtimetadatafield>
              <fieldlabel>cc_weighting</fieldlabel>
              <fieldentry>5</fieldentry>
            </qtimetadatafield>
          </qtimetadata>
        </itemmetadata>
        <presentation>
          <material>
            <mattext texttype="text/html"><![CDATA[<p>Question text here</p>]]></mattext>
          </material>
          <response_lid ident="response1" rcardinality="Single">
            <render_choice>
              <response_label ident="A">
                <material><mattext texttype="text/html"><![CDATA[Option A]]></mattext></material>
              </response_label>
              <response_label ident="B">
                <material><mattext texttype="text/html"><![CDATA[Option B]]></mattext></material>
              </response_label>
              <response_label ident="C">
                <material><mattext texttype="text/html"><![CDATA[Option C]]></mattext></material>
              </response_label>
              <response_label ident="D">
                <material><mattext texttype="text/html"><![CDATA[Option D]]></mattext></material>
              </response_label>
            </render_choice>
          </response_lid>
        </presentation>
        <resprocessing>
          <outcomes>
            <decvar varname="SCORE" vartype="Integer" minvalue="0" maxvalue="5"/>
          </outcomes>
          <respcondition>
            <conditionvar>
              <varequal respident="response1">A</varequal>
            </conditionvar>
            <setvar action="Set" varname="SCORE">5</setvar>
          </respcondition>
        </resprocessing>
      </item>
      <!-- Add more questions following the same pattern -->
    </section>
  </assessment>
</questestinterop>
```

**CRITICAL Quiz Requirements for Brightspace**:
1. Root element MUST include `xmlns:xsi` and `xsi:schemaLocation`
2. Assessment MUST have `cc_profile` (cc.exam.v0p1 or cc.quiz.v0p1) and `qmd_assessmenttype`
3. Each question MUST have `<itemmetadata>` with `cc_profile` identifying question type:
   - Multiple Choice: `cc.multiple_choice.v0p1`
   - True/False: `cc.true_false.v0p1`
   - Fill in Blank: `cc.fib.v0p1`
   - Essay: `cc.essay.v0p1`
4. Each question MUST have `cc_weighting` specifying point value

**Assignment XML (D2L Extension)**:
```xml
<?xml version="1.0" encoding="UTF-8"?>
<assignment xmlns="http://www.imsglobal.org/xsd/imscc_extensions/assignment"
    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
    identifier="assignment_id"
    xsi:schemaLocation="http://www.imsglobal.org/xsd/imscc_extensions/assignment
    http://www.imsglobal.org/profile/cc/cc_extensions/cc_extresource_assignmentv1p0_v1p0.xsd">
  <title>Assignment Title</title>
  <text texttype="text/html"><![CDATA[
    <!-- HTML content here -->
  ]]></text>
  <gradable points_possible="100">true</gradable>
</assignment>
```

**Critical**: These namespaces MUST match the resource types declared in imsmanifest.xml to prevent Brightspace import failures (Pattern 24).

**Assignment Manifest Resource Type**: For native Brightspace assignment import, use `type="assignment_xmlv1p0"` in the manifest (NOT `associatedcontent/imscc_xmlv1p3/learning-application-resource` which imports as XML files instead of native assignments - Pattern 25).

### **Microtask 4: Template-Aware Accessibility and UDL Implementation**
**Concurrent Agent Focus**: Universal design content within template's accessibility framework

**UDL Content Requirements**:
- **Multiple Representation**: Content presented in varied formats supported by template framework
- **Engagement Variety**: Different interaction methods using template's JavaScript and component library
- **Expression Options**: Multiple ways for students to demonstrate understanding through template tools
- **Accessibility Enhancement**: Content that leverages and enhances template's WCAG 2.2 AA compliance

**Template UDL Integration**:
- **Responsive Adaptation**: Content that works across template's supported device types and screen sizes
- **Assistive Technology**: Content optimized for template's screen reader and keyboard navigation support
- **Visual Design**: Content that respects template's color contrast and visual accessibility requirements
- **Interactive Accessibility**: Content using template's accessible form controls and interactive elements

### **Microtask 5: Template-Enhanced Multimedia and Resource Integration**
**Concurrent Agent Focus**: Rich media content within template's resource framework

**Multimedia Integration Protocol**:
- **Template Resource Loading**: Use template's optimized image and video embedding systems
- **Performance Considerations**: Leverage template's CDN integration and caching for multimedia content
- **Accessibility Requirements**: Ensure multimedia works with template's assistive technology features
- **Mobile Optimization**: Use template's responsive media handling for diverse device support

**Resource Quality Standards**:
- **Educational Value**: Multimedia that enhances rather than replaces comprehensive textual content
- **Template Compatibility**: Resources that work seamlessly with template's technical infrastructure
- **Loading Performance**: Resources optimized for template's performance standards and user experience
- **Accessibility Compliance**: Multimedia that meets template's WCAG 2.2 AA requirements

### **Microtask 6: Template-Optimized Quality Validation and Pattern 22 Prevention**
**Concurrent Agent Focus**: Comprehensive content quality assurance within template standards

**Quality Validation Requirements**:
- **Pattern 22 Prevention Verification**: Confirm substantial educational content with authentic example enhancement
- **Template Compliance Validation**: Ensure content works with template's design system and technical features
- **Educational Depth Assessment**: Validate comprehensive learning value and theoretical foundation strength
- **Assessment Integration Testing**: Confirm content supports template's QTI and D2L assessment frameworks

**Template Quality Framework**:
- **Design System Compliance**: Content uses template's Bootstrap components and styling consistently
- **Accessibility Validation**: Content maintains template's WCAG 2.2 AA compliance and assistive technology support
- **Performance Standards**: Content meets template's loading speed and user experience requirements
- **Import Compatibility**: Content formatted for reliable import through template's Brightspace integration

## Pattern 22 Prevention Enforcement Across All Microtasks

### **Educational Content Standards (Mandatory)**
**Every content-generator microtask must enforce**:

1. **Comprehensive Theoretical Foundation**:
   - Minimum 400 words of substantial concept explanation before any examples
   - Clear definitions, principles, and educational context
   - Learning objective alignment and assessment preparation
   - Progressive complexity building from foundational to advanced concepts

2. **Authentic Example Enhancement**:
   - Real-world examples that demonstrate and reinforce theoretical concepts
   - Context-rich scenarios that deepen understanding beyond surface level
   - Examples that prepare students for assessment activities and practical application
   - Integration that enhances rather than replaces comprehensive educational content

3. **Educational Depth Validation**:
   - Total sub-module content minimum 600 words with substantial learning value
   - Content that demonstrates mastery-level understanding of subject matter
   - Pedagogical scaffolding that supports diverse learning needs and styles
   - Assessment alignment that enables meaningful evaluation of student understanding

### **Template-Enhanced Pattern 22 Prevention Workflow**
```
Phase 1: Template Foundation Setup → Inherit template design and accessibility features
Phase 2: Theoretical Content Development → Create comprehensive educational foundations (400+ words)
Phase 3: Authentic Example Integration → Enhance theoretical content with context-rich real-world applications
Phase 4: Assessment Alignment → Ensure content supports template's QTI and D2L assessment framework
Phase 5: Quality Validation → Verify Pattern 22 prevention and template compliance
Phase 6: Template Optimization → Finalize content within template performance and accessibility standards
```

## Template Content Generation Coordination

### **Parallel Weekly Content Generation Protocol**
**For 12-week course development with template optimization**:

```python
# Execute 12 parallel content-generator agents
weekly_content_tasks = []

for week in range(1, 13):
    task_prompt = f"""
    Generate comprehensive Week {week} content with template optimization and Pattern 22 prevention:
    
    Template Foundation:
    - Bootstrap 4.3.1 design system and responsive components
    - WCAG 2.2 AA accessibility features and semantic markup
    - QTI 1.2 and D2L XML assessment framework compatibility
    - Template navigation patterns and user experience consistency
    
    Content Requirements (Pattern 22 Prevention):
    - Modules with substantial educational depth (600+ words each for complex topics)
    - Comprehensive theoretical foundations before authentic examples (400+ words minimum)
    - Progressive complexity building on previous content
    - Real-world examples that enhance rather than replace comprehensive content

    Module Structure (Dynamic - per course outline):
    - Overview module with learning objectives
    - Content modules for each key concept (number varies by topic complexity)
    - Summary/key concepts module with consolidation
    - Applications module with real-world context
    - Assessment module aligned with learning objectives
    
    Template Integration Requirements:
    - Use Bootstrap 4.3.1 components (cards, accordions, responsive grids)
    - Maintain template's WCAG 2.2 AA color contrast and semantic markup
    - Implement template's responsive breakpoints and mobile optimization
    - Ensure compatibility with template's QTI quiz and D2L assignment frameworks
    
    Pattern 22 Prevention Validation:
    - Verify comprehensive educational content before examples in each sub-module
    - Confirm substantial learning value and theoretical depth (600+ words per sub-module)
    - Validate authentic examples enhance rather than replace educational foundations
    - Ensure content supports meaningful assessment through template's tools
    """
    
    content_task = Task(
        subagent_type="content-generator",
        description=f"Week {week} comprehensive content with Pattern 22 prevention",
        prompt=task_prompt
    )
    weekly_content_tasks.append(content_task)

# Execute all units in parallel (batch size per orchestrator protocol)
unit_results = await asyncio.gather(*unit_content_tasks)
```

## Quality Assurance and Validation Integration

### **Template-Enhanced Educational Standards Integration**
**Coordination with educational-standards agent**:

```
Content-Generator Output → Educational-Standards Agent
Educational-Standards receives:
- Template-compliant content with Bootstrap and accessibility features
- Pattern 22 prevented content with comprehensive theoretical foundations
- Assessment-aligned content supporting template's QTI and D2L tools
- Quality validation requirements for pedagogical framework compliance
```

### **Brightspace-Packager Handoff Protocol**
**Seamless integration with packaging agents**:

```
Template-Enhanced Content Package → Brightspace-Packager Agent
Brightspace-Packager receives:
- Bootstrap 4.3.1 compliant HTML with template design consistency
- WCAG 2.2 AA accessible content with semantic markup
- QTI and D2L assessment-ready content with template compatibility
- Pattern 22 prevented content with substantial educational depth
```

## Output Format and Quality Validation

### **Template-Enhanced Content Package Structure**
```json
{
  "template_integration": {
    "design_framework": "Bootstrap 4.3.1 components and responsive system",
    "accessibility_compliance": "WCAG 2.2 AA semantic markup and assistive technology support",
    "assessment_compatibility": "QTI 1.2 quiz and D2L assignment framework integration",
    "performance_optimization": "Template CDN integration and resource loading optimization"
  },
  "content_quality": {
    "pattern_22_prevention": "Comprehensive theoretical foundations before authentic examples",
    "educational_depth": "600+ words per sub-module with substantial learning value",
    "learning_objectives": "Content directly supports course and weekly learning outcomes",
    "assessment_alignment": "Content enables meaningful evaluation through template tools"
  },
  "technical_specifications": {
    "file_format": "HTML5 with Bootstrap 4.3.1 framework",
    "accessibility_features": "Screen reader optimization and keyboard navigation",
    "responsive_design": "Mobile-friendly layouts and touch interfaces",
    "assessment_integration": "QTI 1.2 and D2L XML compatibility"
  }
}
```

### **Pattern 22 Prevention Validation Report**
**Every content-generator output must include**:
```json
{
  "pattern_22_prevention_status": "VALIDATED",
  "validation_criteria": {
    "theoretical_foundation": "400+ words before examples in each sub-module",
    "educational_depth": "600+ words total with substantial pedagogical content",
    "authentic_examples": "Real-world applications that enhance theoretical understanding",
    "assessment_preparation": "Content supports quiz questions and assignment activities"
  },
  "quality_metrics": {
    "content_depth_score": "95/100 - Comprehensive theoretical foundations",
    "example_integration_score": "92/100 - Context-rich authentic applications",
    "assessment_alignment_score": "98/100 - Strong quiz and assignment support",
    "template_compliance_score": "97/100 - Bootstrap and accessibility standards maintained"
  }
}
```

This enhanced template-aware content generation with mandatory Pattern 22 prevention ensures optimal educational quality while leveraging institutional template investments and maintaining technical excellence for reliable Brightspace deployment.