# Parallel Workflow Orchestrator

This directory contains the parallel course generation and IMSCC packaging system that significantly reduces total project time by running multiple Claude agents concurrently.

## Overview

The parallel workflow orchestrator replaces the sequential course generation process with a multi-agent approach:

- **Phase 1**: Parallel content generation (one agent per week)
- **Phase 2**: Parallel IMSCC packaging (brightspace-packager agents)
- **Phase 3**: Final manifest generation (after all content complete)
- **Phase 4**: Final package creation

## Performance Benefits

- **Time Reduction**: ~75% faster than sequential processing
- **Concurrent Processing**: Up to 12 agents running simultaneously
- **Efficient Resource Usage**: Coordinated agent management
- **Early Error Detection**: Per-week validation and error handling

## Files

### `parallel_orchestrator.py`
Main entry point for the parallel workflow. Coordinates all phases of the process.

**Usage:**
```bash
cd scripts/parallel-workflow-orchestrator/
python parallel_orchestrator.py
```

### `agent_interface.py`
Interface layer between the orchestrator and Claude Code's agent system. Handles agent launching, monitoring, and coordination.

### `parallel_course_generator.py` 
Legacy implementation - replaced by the modular approach in `parallel_orchestrator.py` and `agent_interface.py`.

## Workflow Phases

### Phase 1: Parallel Content Generation

**Agents Used**: `general-purpose` (one per week)

**Tasks per Agent**:
- Generate 7 HTML sub-modules for assigned week
- Create 1 D2L XML assignment
- Ensure Pattern 22 comprehensive educational content
- Validate mathematical authenticity and theoretical depth

**Expected Output per Week**:
```
week_XX_overview.html (600+ words)
week_XX_concept1.html (800+ words)
week_XX_concept2.html (800+ words)
week_XX_key_concepts.html (accordion format)
week_XX_visual_display.html (mathematical displays)
week_XX_applications.html (real-world applications)
week_XX_study_questions.html (reflection questions)
week_XX_assignment.xml (D2L format)
```

### Phase 2: Parallel IMSCC Packaging

**Agents Used**: `brightspace-packager` (one per week)

**Tasks per Agent**:
- Convert HTML content to IMSCC-compatible format
- Validate D2L XML compliance
- Generate QTI 1.2 assessment components
- Create resource metadata for manifest compilation
- Ensure IMS Common Cartridge 1.2.0 standards

**Output**: IMSCC-ready files (not zipped, ready for manifest)

### Phase 3: Final Manifest Generation

**Agent Used**: `general-purpose` (single agent)

**Tasks**:
- Compile all resource metadata
- Generate complete imsmanifest.xml
- Ensure IMS CC 1.2.0 schema compliance
- Create hierarchical organization structure
- Validate Pattern 17/18/19 prevention

### Phase 4: Final Package Creation

**Process**: Orchestrator (no agent)

**Tasks**:
- Collect all IMSCC-ready files
- Add manifest to ZIP archive
- Validate package integrity
- Generate performance metrics

## Configuration

### Course Requirements

The orchestrator looks for configuration in:
`scripts/course-requirements/current_requirements.json`

**Example configuration**:
```json
{
  "duration_weeks": 12,
  "credit_hours": 3,
  "course_level": "undergraduate",
  "subject": "Linear Algebra",
  "course_title": "Introduction to Linear Algebra",
  "assessment_types": ["assignments", "quizzes", "discussions"],
  "pattern_prevention": {
    "pattern_19": true,
    "pattern_21": true,
    "pattern_22": true
  }
}
```

### Agent Limits

Maximum concurrent agents: **12** (configurable in `agent_interface.py`)

This prevents system overload while maintaining parallel processing benefits.

## Error Handling

### Content Generation Failures
- Individual week failures don't stop the entire process
- Detailed error reporting per week
- Automatic retry capability (future enhancement)

### Packaging Failures
- Week-by-week validation
- Resource metadata verification
- IMSCC compliance checking

### Critical Failures
- Complete workflow termination on critical errors
- Cleanup procedures for temporary files
- Error state logging for debugging

## Pattern Prevention Integration

The parallel workflow maintains all existing pattern prevention protocols:

- **Pattern 19**: Educational structure preservation (per course outline)
- **Pattern 21**: Complete content generation (all weeks substantial)
- **Pattern 22**: Comprehensive educational content (theory + examples)
- **Patterns 1-18**: Technical compliance maintained

## Output Structure

### Working Directory
```
YYYYMMDD_HHMMSS_parallel_firstdraft/
├── week_01/
│   ├── week_01_overview.html
│   ├── week_01_concept1.html
│   ├── ... (8 files total)
├── week_02/
│   └── ... (8 files)
├── ...
└── week_12/
    └── ... (8 files)
```

### Export Directory
```
exports/YYYYMMDD_HHMMSS/
├── imsmanifest.xml
├── week_01/
│   └── (IMSCC-ready files)
├── ...
├── week_12/
│   └── (IMSCC-ready files)
└── linear_algebra_parallel_YYYYMMDD_HHMMSS.imscc
```

## Performance Metrics

The orchestrator provides detailed performance analysis:

- **Total Processing Time**: Actual parallel execution time
- **Phase Breakdown**: Time spent in each phase
- **Sequential Comparison**: Estimated time for sequential processing
- **Efficiency Gain**: Percentage improvement over sequential approach
- **File Counts**: Validation of expected content generation

## Usage Examples

### Basic Execution
```bash
python parallel_orchestrator.py
```

### With Custom Requirements
1. Create requirements file:
```bash
# Edit course requirements
nano scripts/course-requirements/current_requirements.json
```

2. Run orchestrator:
```bash
python parallel_orchestrator.py
```

### Integration with Existing Workflow

The parallel orchestrator can be integrated into the existing course generation workflow:

```python
from parallel_orchestrator import ParallelWorkflowOrchestrator

# Create orchestrator with requirements
orchestrator = ParallelWorkflowOrchestrator(course_requirements)

# Execute parallel workflow
package_path = await orchestrator.execute_parallel_workflow()
```

## Future Enhancements

1. **Adaptive Agent Scaling**: Adjust concurrent agents based on system resources
2. **Resume Capability**: Resume interrupted workflows from last successful phase
3. **Real-time Progress Monitoring**: Web interface for monitoring agent progress
4. **Quality Validation**: Enhanced content quality checking during generation
5. **Template Customization**: Support for different course templates and structures

## Troubleshooting

### Common Issues

**"No content was generated successfully"**
- Check agent interface configuration
- Verify Claude Code agent availability
- Review individual week error messages

**"Package size below expected threshold"**
- Validate content generation completion
- Check for empty or placeholder files
- Review Pattern 21/22 prevention protocols

**"Timeout: agents still running"**
- Increase timeout in `agent_interface.py`
- Check system resources and agent capacity
- Review agent task complexity

### Debug Mode

Enable detailed logging by modifying the orchestrator:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Integration with CLAUDE.md

This parallel workflow system is fully integrated with the project's CLAUDE.md guidance:

- **Pattern Prevention**: All historical patterns (1-22) prevented
- **Quality Standards**: OSCQR compliance maintained
- **File Organization**: Proper `/scripts/` directory structure
- **Documentation**: Comprehensive README and usage examples

The parallel approach significantly reduces project time while maintaining all quality and compliance standards established in the repository.