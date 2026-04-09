# Code Review Pipeline: Ed4All

Summary of systematic code review across 8 rounds covering the full
DART -> Courseforge -> Trainforge -> LibV2 pipeline.

## Round 1: Security & Foundations
- Content store hash validation, TOCTOU symlink race fixes
- State manager temp file collision prevention (mkstemp)
- WriteFacade append(), MCP error disclosure hardening
- Executor silent failure fixes, config validator path checking

## Round 2: Security Hardening & Lint Compliance
- Validator module import allowlist (validation_gates.py)
- PDF path validation in pipeline_tools.py
- Audit logger recursion depth limit
- MD5 -> SHA256 in analysis_tools.py
- Ruff compliance: 112 safe auto-fixes, UP/E501/C901 ignores
- 5 new WriteFacade.append() tests

## Round 3: Regression Fix, Encoding, Schemas
- Fixed broken TestAppend fixture (round 2 regression)
- File leak in quality_feedback.py (open without context manager)
- encoding='utf-8' on 6 open() calls in streaming/decision capture
- Schema fixes: added textbook_to_course to enum, relaxed run_id pattern
- Replaced 12 stderr prints with logger calls in decision_capture.py
- Null JSON guard in status_tracker.py, sanitize_path_component in checkpoint.py
- Path validation for courseforge_tools project_id
- Narrowed content_store.py exception to (OSError, ValueError)
- Created missing __init__.py for MCP/tests and LibV2/tests

## Round 4: Runtime Crash, Data Integrity, Error Handling
- Fixed VerificationResult attribute mismatch (.length -> .total_events)
- Fixed sequence_manager locking (was locking temp file, not target)
- Atomic write cleanup in lockfile.py and checkpoint.py
- Remaining MD5 -> SHA256 in trainforge_tools.py
- CLI export early return hiding success when warnings exist
- Error logging in replay_engine hash chain verification
- Narrowed exception handling in trainforge_tools export and run_summarizer

## Round 5: Pipeline Handoffs & Standalone Operation
- Added imscc_path parameter to generate_assessments() for direct IMSCC input
- IMSCC packages stored in LibV2 after Courseforge packaging
- imsmanifest.xml validation in analyze_imscc_content()
- HTML structure validation on staged DART outputs
- IMSCC pre-validation gate in textbook_to_course workflow
- Updated tool_schemas.py with imscc_path and course_slug params

## Round 6: Decision Capture System
- Made DARTDecisionCapture.save() atomic (temp+rename+fsync)
- Replaced final 7 stderr prints in streaming_capture.py
- Added quality_gate_passed/reason/validation_issues to decision schema
- Completed OPERATION_MAP with 11 missing decision type entries
- Extracted RELAXED_DECISION_TYPES to shared constant

## Round 7: Security, Encoding, Reliability
- DARTDecisionCapture.save() missing self.close() and silent fsync
- Symlink traversal bypass fix (string prefix -> Path.relative_to)
- Symlink creation race window error handling in content_store.py
- Transaction log: serialize before lock, log corrupt entries
- encoding='utf-8' on 15 open() calls across hash_chain, config, libv2_fsck

## Round 8: Pipeline Fidelity, Encoding, Documentation
- Trainforge _parse_assessments(): expanded to match quiz/assignment/discussion XML
- _detect_lms(): fixed iterparse to use io.StringIO (was passing string)
- Trainforge README: updated MultiRetriever -> TrainforgeRAG
- libv2_storage.py: encoding='utf-8' on 7 open() calls + fcntl locking on append
- MCP/server.py: encoding on 5 open() calls
- cli/validators/run_validator.py: encoding on 5 open() calls
- pytest.ini: importmode=importlib for monorepo test collection
