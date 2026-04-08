# Agent Protocols and Coordination

This file contains detailed protocols for agent behavior, scratchpad usage, and coordination strategies.

## **📝 MANDATORY AGENT SCRATCHPAD PROTOCOL**

**All agents MUST use dedicated scratchpads for architectural and design decisions to avoid preset constraints and enable optimal educational design.**

### **Scratchpad Usage Requirements**
```
WHEN AGENTS SHOULD CREATE SCRATCHPADS:
1. **Educational Architecture Decisions** - course duration, module structure, content organization
2. **Pedagogical Framework Selection** - learning progression, assessment strategy, content types
3. **Content Structure Planning** - optimal number of learning units, cognitive load distribution
4. **Assessment Design** - formative/summative placement, certification preparation strategy

SCRATCHPAD ORGANIZATION PROTOCOL:
- **ONE scratchpad per agent**: `agent_workspaces/{agent_type}_scratchpad.md`
- **All agent work contained within single file**: analysis, decisions, todos, rationale
- **Structured sections**: ## Analysis, ## Decisions, ## Updated Todos, ## Rationale
- **No file proliferation**: Everything for that agent goes in their dedicated scratchpad
```

## **Agent Autonomy and Responsibilities**

### **Agent-to-Orchestrator Todo Integration**
```
AGENT TODO LIST PROTOCOL:
  → All planning agents must provide detailed todo lists
  → Todo items must be specific, actionable tasks
  → Orchestrator loads agent todo lists into TodoWrite
  → Orchestrator executes todo lists using appropriate agents
```

### **Enhanced Agent Autonomy**
```
AGENTS NOW RESPONSIBLE FOR:
  ✅ Analyzing user requirements and determining optimal approaches
  ✅ Creating comprehensive todo lists for orchestrator execution
  ✅ Determining pedagogical frameworks and content structures
  ✅ Recommending parallel batching strategies
  ✅ Providing specific, actionable tasks for orchestrator coordination
```

## **Agent Coordination Strategies**

### **High Volume Coordination (12+ weeks, 84+ files)**
```
  → ORCHESTRATOR uses individual file agents (ONE AGENT PER FILE)
  → PARALLEL execution of multiple agents simultaneously
  → Each agent creates exactly ONE file
  → Progress tracking via TodoWrite
  → Quality validation at integration points
```

### **Medium Volume Coordination (6-12 weeks, 40-80 files)**
```
  → ORCHESTRATOR uses module-based agents
  → Mixed sequential/parallel execution
  → Module-level quality gates
  → Integrated final validation
```

### **Low Volume Coordination (<6 weeks, <40 files)**
```
  → ORCHESTRATOR uses comprehensive agents
  → Sequential execution with parallel validation
  → End-to-end quality assurance
  → Streamlined packaging process
```

## **Individual File Batching Protocol**

### **CRITICAL EXECUTION PROTOCOLS**

**BATCH SIZE LIMITATIONS:** ✅ **TESTED & CONFIRMED**
- Maximum 5-10 simultaneous Task calls per execution block (SYSTEM LIMIT CONFIRMED)
- For 84 files: Execute in 8-17 batches of 5-10 agents each
- Wait for batch completion before starting next batch (MANDATORY)
- This prevents system rejection while maintaining optimal parallelism

**PARALLEL EXECUTION PATTERN:** ✅ **TESTED WORKFLOW**
```
# BATCH 1 (5-10 agents simultaneously - TESTED SAFE LIMIT):
Task(content-generator, "File 1", "Create week_01_file_1.html")
Task(content-generator, "File 2", "Create week_01_file_2.html") 
Task(content-generator, "File 3", "Create week_01_file_3.html")
Task(content-generator, "File 4", "Create week_01_file_4.html")
Task(content-generator, "File 5", "Create week_01_file_5.html")
# OPTIONAL: Add up to 5 more (max 10 total per batch)

# MANDATORY: WAIT for batch completion verification
# Check: file system monitoring, count completion, verify timestamps
# Update TodoWrite with completed tasks
# THEN execute BATCH 2 (next 5-10 agents)
# Continue until all files created
```

### **CRITICAL ANTI-PATTERN ENFORCEMENT**
```
❌ NEVER assign multiple files to one agent (e.g., "create all 7 Week 7 files")
❌ NEVER use prompts like "create week_XX modules 1-7" 
✅ ALWAYS use individual file assignments (e.g., "create week_07_module_04_scenario_analysis.html")
✅ ALWAYS verify each Task call specifies exactly ONE file creation
```

## **Agent Workspace Containment Protocol**

**MANDATORY for ALL specialized agents:**

1. **Single Project Folder**: All agents MUST receive project folder path as primary workspace
2. **Agent Subdirectories**: Each agent creates subdirectory within project folder (never outside)
3. **No Scattered Files**: Agent outputs ONLY within assigned project folder structure
4. **Folder Inheritance**: All agent workspaces contained within single timestamped project folder

## **Individual File Agent Benefits**
- **Single File Focus**: Each agent handles exactly one specific file creation task
- **Reduced Context Load**: Each agent works with minimal, focused requirements  
- **Parallel Execution**: All agents execute simultaneously for maximum efficiency
- **Efficient Coordination**: File-level task distribution prevents context overflow
- **Quality Maintenance**: Each file includes comprehensive educational content
- **No Dependencies**: Each file self-contained with full pedagogical context

## **Enhanced Template Integration for Content Agents**

### **Template Resource Access**
All content-generator agents must utilize templates from the `templates/` directory:
```
templates/
├── lesson/                               # Lesson templates
├── activity/                             # Activity templates
├── assessment/                           # Assessment templates
├── accessibility/                        # Accessibility templates
└── examples/                             # Example implementations
```

### **Required Interactive Components**
Content agents MUST incorporate these elements where pedagogically appropriate:
- **Flip Cards**: For concept reveals and key takeaways
- **Knowledge Checks**: Self-assessment questions with hidden answers
- **Progress Indicators**: Visual progress bars and completion tracking
- **Call-out Boxes**: Info, warning, success, danger variants
- **Tabbed Content**: For organizing module sections
- **Working Accordions**: With proper Bootstrap collapse attributes
- **Activity Cards**: Visual representation of learning activities
- **Timeline Layouts**: For sequential content presentation

### **Component Usage Guidelines**
```
WHEN TO USE INTERACTIVE COMPONENTS:
- Flip Cards → Key concepts, definitions, before/after scenarios
- Knowledge Checks → End of section reviews, concept reinforcement
- Progress Bars → Module/course completion tracking
- Call-out Boxes → Important notices, tips, warnings, achievements
- Tabs → Organizing resources, activities, assessments
- Accordions → FAQ sections, expandable content, progressive disclosure
- Activity Cards → Visual activity overviews with metadata
- Timelines → Sequential processes, course schedules, workflows
```