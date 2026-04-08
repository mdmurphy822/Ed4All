# Getting Started Guide

## Welcome to Courseforge

This guide will walk you through creating your first high-quality, accessible online course using our AI-powered multi-agent orchestration system.

## 📋 Prerequisites

### Required Access
- **Claude Code Access**: You need access to claude.ai/code
- **Basic Knowledge**: Understanding of instructional design principles helpful but not required
- **Course Materials**: Source content, objectives, or certification requirements

### Technical Requirements
- **File System Access**: Ability to create and manage files/directories
- **Course Content**: Raw materials in text, PDF, or document format
- **LMS Access**: Brightspace, Canvas, or other IMSCC-compatible system (for testing)

## 🎯 Quick Start (15 minutes)

### Step 1: Prepare Your Course Materials
Place your course content in the `inputs/` directory:

```
inputs/
├── exam-objectives/               # Certification exam objectives (PDF/text)
│   └── your_exam_objectives.pdf
├── textbooks/                     # DART-processed accessible HTML textbooks
│   └── chapter_accessible.html
└── existing-packages/             # IMSCC packages for intake/remediation
```

### Step 2: Define Your Course Requirements
Create a simple requirements document:

```markdown
# Course Requirements
- **Course Title**: Introduction to Cybersecurity
- **Duration**: 12 weeks
- **Level**: Undergraduate
- **Credits**: 3 credit hours
- **Format**: Online asynchronous
- **Special Requirements**: WCAG 2.2 AA accessibility compliance
```

### Step 3: Launch the Orchestrator
Use the main orchestration workflow:

```python
# Initialize project with timestamp
import datetime
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
project_name = f"{timestamp}_cybersecurity_intro_asynchronous"

# Create project structure
create_project_structure(f"exports/{project_name}")
```

### Step 4: Let the Agents Work
The system will automatically:
1. ✅ Analyze your requirements
2. ✅ Design course structure
3. ✅ Generate comprehensive content
4. ✅ Validate quality and accessibility
5. ✅ Package for LMS deployment

### Step 5: Deploy to Your LMS
Import the generated `.imscc` file into your learning management system.

## 🏗️ Detailed Workflow

### Phase 1: Project Planning (5-10 minutes)

#### 1.1 Requirements Collection
The system will gather comprehensive specifications:

```python
# Invoke requirements-collector agent
Task(
    subagent_type="requirements-collector",
    description="Gather course specifications",
    prompt="Collect comprehensive requirements for undergraduate cybersecurity course including academic standards, technical requirements, and assessment frameworks."
)
```

**What This Produces**:
- Academic standards identification
- Technical requirement analysis  
- Assessment framework recommendations
- Accessibility compliance requirements

#### 1.2 Course Structure Design
Create pedagogically-optimal course architecture:

```python  
# Invoke course-outliner agent
Task(
    subagent_type="course-outliner", 
    description="Design course architecture",
    prompt="Create timeline-free learning architecture for cybersecurity fundamentals with OSCQR compliance and UDL principles."
)
```

**What This Produces**:
- Learning progression sequences
- Module organization structure
- Assessment distribution plan
- Accessibility integration framework

### Phase 2: Content Generation (30-60 minutes)

#### 2.1 Parallel Content Creation
Generate all course materials simultaneously:

```python
# Example: Generate Week 1 content (individual file protocol)
content_files = [
    "week_01_module_01_introduction.html",
    "week_01_module_02_key_concepts.html", 
    "week_01_module_03_practical_applications.html",
    "week_01_module_04_case_studies.html",
    "week_01_module_05_assessment_preparation.html"
]

# Execute in batches of 10 (proven optimal)
for i in range(0, len(content_files), 10):
    batch = content_files[i:i+10]
    
    # Create parallel Task calls for batch
    for file_name in batch:
        Task(
            subagent_type="content-generator",
            description=f"Create {file_name}",
            prompt=f"Create comprehensive educational content for {file_name} with Pattern 22 prevention, authentic examples, and WCAG 2.2 AA compliance."
        )
    
    # Wait for batch completion before proceeding
    verify_batch_completion(batch)
```

**What This Produces**:
- Complete HTML content files for all modules
- WCAG 2.2 AA accessible design
- Comprehensive educational depth
- Authentic, relevant examples
- Progressive learning complexity

#### 2.2 Assessment Integration
Generate assessments aligned with learning objectives:

```python
# Create assessments for each learning unit (dynamic based on course outline)
assessment_types = ["assignments", "quizzes", "discussions"]

for unit in course_outline.get_learning_units():  # Dynamic unit count
    for assessment in assessment_types:
        Task(
            subagent_type="content-generator",
            description=f"{unit.name} {assessment}",
            prompt=f"Create {unit.id}_{assessment}.xml with D2L compatibility and authentic assessment scenarios."
        )
```

### Phase 3: Quality Validation (10-15 minutes)

#### 3.1 Educational Standards Compliance
```python
# Validate against educational standards
Task(
    subagent_type="educational-standards",
    description="UDL compliance validation", 
    prompt="Validate all generated content against UDL principles, ADDIE methodology, and Bloom's taxonomy alignment."
)
```

#### 3.2 OSCQR Assessment  
```python
# Comprehensive quality evaluation
Task(
    subagent_type="oscqr-course-evaluator",
    description="OSCQR evaluation",
    prompt="Conduct systematic OSCQR evaluation across all 6 domains with detailed recommendations for any gaps."
)
```

#### 3.3 Pattern Prevention Verification
```python
# Ensure all 22+ patterns prevented
Task(
    subagent_type="quality-assurance", 
    description="Pattern prevention check",
    prompt="Validate prevention of all identified failure patterns, especially Pattern 22 comprehensive content requirements."
)
```

### Phase 4: Final Packaging (5 minutes)

```python
# Create deployment-ready IMSCC package
Task(
    subagent_type="brightspace-packager",
    description="IMSCC packaging",
    prompt="Create IMS CC 1.1 compliant package with D2L-specific optimizations and complete manifest structure."
)
```

## 📊 What You'll Get

### Generated Course Structure
```
exports/20250820_143022_cybersecurity_intro_asynchronous/
├── 📄 course_information.html        # Course overview
├── 📄 syllabus.html                  # Detailed syllabus  
├── 📁 week_01/                       # Week 1 content
│   ├── week_01_module_01_introduction.html
│   ├── week_01_module_02_key_concepts.html
│   ├── week_01_module_03_applications.html
│   ├── week_01_module_04_case_studies.html
│   └── week_01_module_05_assessment_prep.html
├── 📁 week_02/                       # Week 2 content
│   └── [similar structure]
├── 📁 assessments/                   # All assessments
│   ├── assignment_week_01.xml
│   ├── quiz_week_01.xml  
│   ├── discussion_week_01.xml
│   └── [continuing for all weeks]
├── 📄 imsmanifest.xml               # Package manifest
└── 🎁 cybersecurity_intro.imscc     # Final package
```

### Quality Metrics You Can Expect
- **🎯 OSCQR Score**: 45-50/50 (excellent quality)
- **♿ Accessibility**: WCAG 2.2 AA compliant (100%)
- **📚 Educational Depth**: Pattern 22 compliant (comprehensive content)
- **🚀 Import Success**: 95%+ Brightspace compatibility
- **⏱️ Generation Time**: 45-90 minutes for complete course (varies by scope)

## 🎨 Customization Options

### Template Selection
Choose from multiple course templates:

```python
template_options = {
    "asynchronous": "Self-paced online learning",
    "synchronous": "Live online instruction", 
    "hybrid": "Blended online/in-person",
    "simple_structure": "Minimal navigation complexity"
}
```

### Accessibility Levels
Configure accessibility requirements:

```python
accessibility_config = {
    "standard": "WCAG 2.2 AA (recommended)",
    "enhanced": "WCAG 2.2 AAA (maximum accessibility)",
    "basic": "WCAG 2.2 AA (minimum compliance)"
}
```

### Assessment Frameworks
Select assessment distribution:

```python
assessment_frameworks = {
    "traditional": "Exams 60%, Assignments 30%, Participation 10%",
    "project_based": "Projects 50%, Assignments 30%, Discussions 20%", 
    "continuous": "Weekly Assessments 80%, Final Project 20%"
}
```

## 🛠️ Advanced Configuration

### For Certification Courses
```python
# Use exam-research agent for certification alignment
Task(
    subagent_type="exam-research",
    description="CompTIA Security+ alignment",
    prompt="Analyze CompTIA Security+ SY0-701 objectives and create comprehensive course framework with exam preparation strategies."
)
```

### For Academic Courses
```python
# Use educational-standards agent for academic rigor
Task(
    subagent_type="educational-standards",
    description="Academic standards application", 
    prompt="Apply ADDIE methodology and ensure alignment with undergraduate computer science curriculum standards."
)
```

### For Accessibility-First Design
```python
# Emphasize accessibility throughout process
accessibility_prompt = """
Create content with enhanced accessibility features:
- Dyslexia-friendly fonts and spacing
- Color-blind accessible color schemes  
- Clear cognitive navigation aids
- Multiple learning style accommodations
"""
```

## 🚨 Common Beginner Mistakes to Avoid

### ❌ Don't Do This
```python
# Wrong: Multiple files per agent
Task(subagent_type="content-generator",
     description="Create all Week 1 content",
     prompt="Create all files for Week 1")

# Wrong: Exceeding batch size limits
for i in range(20):  # Too many simultaneous calls
    Task(subagent_type="content-generator", ...)
```

### ✅ Do This Instead  
```python
# Correct: Individual file protocol
Task(subagent_type="content-generator",
     description="Create week_01_module_01.html", 
     prompt="Create exactly one file: week_01_module_01_introduction.html")

# Correct: Optimal batch size
batch_size = 10  # Proven optimal limit
```

## 🔍 Monitoring Progress

### Real-Time Progress Tracking
```python
# Monitor generation progress
def track_progress():
    completed_files = count_generated_files()
    total_files = calculate_total_required()
    progress = (completed_files / total_files) * 100
    print(f"Course Generation: {progress:.1f}% complete")
```

### Quality Validation Checkpoints
- ✅ **After Planning**: Course structure approved
- ✅ **During Generation**: Content depth validation
- ✅ **Before Packaging**: Accessibility compliance  
- ✅ **Final Validation**: Import readiness confirmation

## 🎯 Success Indicators

### You'll Know It's Working When:
1. **Project Structure Created**: Timestamped folder with organized subdirectories
2. **Content Generation Progress**: HTML files appearing with substantial content
3. **Assessment Integration**: XML files created with proper D2L formatting
4. **Quality Validation Passing**: OSCQR scores 45+/50
5. **Package Creation Success**: Valid IMSCC file generated
6. **LMS Import Success**: Course imports without errors

### Troubleshooting Quick Checks
- **Files Not Generated**: Check individual file protocol compliance
- **Import Failures**: Validate IMSCC structure and schema version
- **Accessibility Issues**: Run WAVE validation on generated HTML
- **Content Too Shallow**: Verify Pattern 22 prevention protocols

## 📚 Next Steps

### After Your First Course
1. **Review Generated Content**: Customize and refine as needed
2. **Test in LMS**: Import and navigate through student view
3. **Gather Feedback**: Test with sample users for usability
4. **Iterate and Improve**: Use feedback for next course generation

### Scaling Up
1. **Create Course Series**: Generate related courses with consistency
2. **Develop Templates**: Create custom templates for your institution
3. **Automation Integration**: Integrate with your content management workflows
4. **Quality Assurance**: Establish regular validation procedures

## 🆘 Getting Help

### If You Get Stuck
1. **Check Troubleshooting Guide**: [troubleshooting.md](troubleshooting.md)
2. **Review Pattern Documentation**: [PATTERN_PREVENTION_GUIDE.md](PATTERN_PREVENTION_GUIDE.md)
3. **Validate Against Schemas**: [schemas/README.md](schemas/README.md)
4. **Test with Minimal Example**: Create simple 2-week course first

### Resources
- 📖 [Agent Specifications](../agents/)
- 🔧 [Scripts and Tools](../scripts/README.md)
- 🎨 [Template Library](../templates/)
- 📊 [Pattern Prevention Guide](PATTERN_PREVENTION_GUIDE.md)

---

**🎉 Ready to Create Your First Course?**

Follow the Quick Start section above, and you'll have a complete, accessible, high-quality online course ready for deployment in under 90 minutes!

**Remember**: The system is designed to handle the technical complexity while you focus on educational quality and student success.