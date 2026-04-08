"""
Trainforge MCP Tools

Tools for assessment-based RAG training on IMSCC packages.
"""

import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path for imports
_MCP_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = _MCP_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from lib.libv2_storage import LibV2Storage  # noqa: E402
from lib.paths import TRAINFORGE_PATH, TRAINING_DIR  # noqa: E402
from lib.secure_paths import sanitize_path_component, validate_path_within_root  # noqa: E402

# Import RAG bridge and assessment components
try:
    from Trainforge.rag.libv2_bridge import TrainforgeRAG, get_rag_for_course  # noqa: F401
    HAS_RAG_BRIDGE = True
except ImportError:
    HAS_RAG_BRIDGE = False

try:
    from Trainforge.generators.assessment_generator import (  # noqa: F401
        AssessmentGenerator,
        generate_assessment,
    )
    HAS_ASSESSMENT_GENERATOR = True
except ImportError:
    HAS_ASSESSMENT_GENERATOR = False

try:
    from lib.trainforge_capture import QuestionData, create_trainforge_capture  # noqa: F401
    HAS_LEGACY_CAPTURE = True
except ImportError:
    HAS_LEGACY_CAPTURE = False

# Import new telemetry system
try:
    from LibV2.telemetry import ArtifactRef, CaptureSession, InputRef  # noqa: F401
    HAS_TELEMETRY = True
except ImportError:
    HAS_TELEMETRY = False

logger = logging.getLogger(__name__)


def _create_trainforge_session(course_code: str, imscc_source: str, phase: str):
    """Create a capture session for Trainforge operations."""
    if HAS_TELEMETRY:
        try:
            return CaptureSession.start_run(
                tool_id="trainforge",
                component=phase,
                meta={"course_code": course_code, "imscc_source": imscc_source},
                course_code=course_code,
                phase=phase,
            )
        except Exception as e:
            logger.warning(f"Failed to create telemetry session: {e}")
    # Fallback to legacy capture
    if HAS_LEGACY_CAPTURE:
        try:
            capture = create_trainforge_capture(
                course_code=course_code,
                imscc_source=imscc_source,
                phase=phase
            )
            capture.__enter__()
            return capture
        except Exception as e:
            logger.warning(f"Failed to create legacy capture: {e}")
    return None


def _log_chunk_retrieval(capture, query: str, chunks_retrieved: list, chunks_used: list, latency_ms: float):
    """Log RAG chunk retrieval event."""
    if capture is None:
        return

    if HAS_TELEMETRY and isinstance(capture, CaptureSession):
        capture.emit(
            "retrieval.selected_chunks",
            payload={
                "query": query,
                "chunks_retrieved": chunks_retrieved,
                "chunks_used": chunks_used,
            },
            metrics={
                "rag_k": len(chunks_retrieved),
                "chunks_selected": len(chunks_used),
                "latency_ms": latency_ms
            },
            phase="retrieve"
        )
    elif hasattr(capture, 'log_chunk_retrieval'):
        capture.log_chunk_retrieval(query, chunks_retrieved, chunks_used, latency_ms)


def _log_question_generation(capture, question_data, source_chunks: list, rationale: str, latency_ms: float):
    """Log question generation event."""
    if capture is None:
        return

    if HAS_TELEMETRY and isinstance(capture, CaptureSession):
        # Convert question data to dict if needed
        q_dict = question_data if isinstance(question_data, dict) else {
            "question_id": getattr(question_data, 'question_id', ''),
            "question_type": getattr(question_data, 'question_type', ''),
            "bloom_level": getattr(question_data, 'bloom_level', ''),
            "difficulty": getattr(question_data, 'difficulty', ''),
        }
        capture.emit(
            "generation.completed",
            payload={
                "task": "mcq",
                "question": q_dict,
                "source_chunks": source_chunks,
                "rationale": rationale,
            },
            metrics={"latency_ms": latency_ms},
            phase="generate"
        )
    elif hasattr(capture, 'log_question_generation'):
        capture.log_question_generation(question_data, source_chunks, rationale, latency_ms)


def _log_assessment_assembly(capture, assessment_id: str, question_ids: list, total_points: int,
                             time_limit: int, rationale: str):
    """Log assessment assembly event."""
    if capture is None:
        return

    if HAS_TELEMETRY and isinstance(capture, CaptureSession):
        capture.emit(
            "export.completed",
            payload={
                "assessment_id": assessment_id,
                "question_ids": question_ids,
                "total_points": total_points,
                "time_limit_minutes": time_limit,
                "rationale": rationale,
            },
            phase="export"
        )
    elif hasattr(capture, 'log_assessment_assembly'):
        capture.log_assessment_assembly(assessment_id, question_ids, total_points, time_limit, rationale)


def _finalize_capture(capture, status: str = "success"):
    """Finalize a capture session."""
    if capture is None:
        return {}

    summary = {}
    if HAS_TELEMETRY and isinstance(capture, CaptureSession):
        capture.finish_run(status)
    elif hasattr(capture, '__exit__'):
        capture.__exit__(None, None, None)
        if hasattr(capture, 'get_session_summary'):
            summary = capture.get_session_summary()
    return summary

# Derived paths
TRAINING_OUTPUT = TRAINING_DIR / "trainforge"


def _validate_trainforge_paths():
    """Validate Trainforge paths at module load."""
    if not TRAINFORGE_PATH.exists():
        logger.warning(f"Trainforge installation not found: {TRAINFORGE_PATH}")
    else:
        logger.info(f"Trainforge installation validated: {TRAINFORGE_PATH}")

    if not TRAINING_OUTPUT.exists():
        TRAINING_OUTPUT.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created training output directory: {TRAINING_OUTPUT}")


_validate_trainforge_paths()


def register_trainforge_tools(mcp):
    """Register Trainforge tools with the MCP server."""

    @mcp.tool()
    async def analyze_imscc_content(imscc_path: str) -> str:
        """
        Analyze IMSCC package content for assessment generation.

        Args:
            imscc_path: Path to IMSCC package

        Returns:
            Content analysis with learning objectives, modules, and assessment opportunities
        """
        try:
            import zipfile

            imscc = Path(imscc_path)
            if not imscc.exists():
                return json.dumps({"error": f"IMSCC not found: {imscc_path}"})

            analysis = {
                "source": str(imscc),
                "analyzed_at": datetime.now().isoformat(),
                "content": {
                    "html_modules": 0,
                    "existing_assessments": 0,
                    "total_word_count": 0
                },
                "learning_objectives": [],
                "assessment_opportunities": []
            }

            # Open and analyze
            with zipfile.ZipFile(imscc, 'r') as z:
                for name in z.namelist():
                    if name.endswith('.html'):
                        analysis["content"]["html_modules"] += 1

                        # Read content
                        content = z.read(name).decode('utf-8', errors='ignore')
                        word_count = len(content.split())
                        analysis["content"]["total_word_count"] += word_count

                        # Extract learning objectives (simplified)
                        if 'objective' in content.lower():
                            analysis["learning_objectives"].append({
                                "source_file": name,
                                "detected": True
                            })

                    elif name.endswith('.xml') and 'assessment' in name.lower():
                        analysis["content"]["existing_assessments"] += 1

            # Suggest assessment opportunities
            if analysis["content"]["html_modules"] > 0:
                analysis["assessment_opportunities"] = [
                    {
                        "type": "quiz",
                        "coverage": "per_module",
                        "estimated_questions": analysis["content"]["html_modules"] * 5
                    },
                    {
                        "type": "exam",
                        "coverage": "comprehensive",
                        "estimated_questions": min(50, analysis["content"]["html_modules"] * 3)
                    }
                ]

            return json.dumps(analysis)

        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def generate_assessments(
        course_id: str,
        objective_ids: str,
        bloom_levels: str,
        question_count: int = 10,
        course_slug: str = ""
    ) -> str:
        """
        Generate assessments from course content using RAG retrieval.

        Args:
            course_id: Course identifier (e.g., INT_101)
            objective_ids: Comma-separated learning objective IDs to assess
            bloom_levels: Comma-separated Bloom's levels to target
                         (remember, understand, apply, analyze, evaluate, create)
            question_count: Number of questions to generate (default: 10)
            course_slug: LibV2 course slug for RAG retrieval (optional)

        Returns:
            Generated assessment data with question IDs and RAG-retrieved content
        """
        import time
        start_time = time.time()

        try:
            objectives = [o.strip() for o in objective_ids.split(",")]
            levels = [l.strip() for l in bloom_levels.split(",")]

            # Sanitize course_id to prevent path traversal
            safe_course_id = sanitize_path_component(course_id)

            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            assessment_id = f"ASM-{safe_course_id}-{session_id}"

            # Create output directory with path validation
            output_dir = validate_path_within_root(
                TRAINING_OUTPUT / safe_course_id / f"assessment_{session_id}",
                TRAINING_OUTPUT
            )
            output_dir.mkdir(parents=True, exist_ok=True)

            # Initialize capture session
            capture = _create_trainforge_session(
                course_code=safe_course_id,
                imscc_source=f"libv2://{course_slug or safe_course_id}",
                phase="question-generation"
            )

            # Try to find course in LibV2 for RAG retrieval
            rag = None
            corpus_stats = None
            if HAS_RAG_BRIDGE and course_slug:
                try:
                    rag = get_rag_for_course(course_slug)
                    if rag.has_corpus:
                        corpus_stats = rag.get_corpus_stats()
                        logger.info(f"RAG initialized for {course_slug}: {corpus_stats.get('chunk_count', 0)} chunks")
                except Exception as e:
                    logger.warning(f"Could not initialize RAG for {course_slug}: {e}")

            # Generate assessment using RAG-retrieved chunks
            questions = []
            rag_metrics = {
                "total_chunks_retrieved": 0,
                "total_chunks_used": 0,
                "avg_retrieval_latency_ms": 0.0,
                "retrieval_count": 0
            }

            questions_per_combo = max(1, question_count // (len(objectives) * len(levels)))

            for obj_id in objectives:
                for bloom_level in levels:
                    if len(questions) >= question_count:
                        break

                    # Set LO context in legacy capture (new telemetry includes this in emit)
                    if capture and hasattr(capture, 'set_learning_objective_context'):
                        capture.set_learning_objective_context(
                            lo_id=obj_id,
                            bloom_target=bloom_level
                        )

                    # Retrieve relevant chunks for this objective
                    source_chunks = []
                    if rag:
                        try:
                            chunks, metrics = rag.retrieve_for_objective(
                                objective_text=obj_id,
                                bloom_level=bloom_level,
                                top_k=5
                            )
                            source_chunks = [c.to_dict() for c in chunks]

                            # Log chunk retrieval
                            if chunks:
                                _log_chunk_retrieval(
                                    capture,
                                    query=obj_id,
                                    chunks_retrieved=[{"chunk_id": c.chunk_id, "relevance_score": c.score, "token_count": c.tokens_estimate} for c in chunks],
                                    chunks_used=[{"chunk_id": c.chunk_id, "relevance_score": c.score, "token_count": c.tokens_estimate} for c in chunks[:3]],
                                    latency_ms=metrics.retrieval_latency_ms
                                )

                            # Update metrics
                            rag_metrics["total_chunks_retrieved"] += metrics.chunks_retrieved
                            rag_metrics["total_chunks_used"] += min(3, metrics.chunks_retrieved)
                            rag_metrics["avg_retrieval_latency_ms"] += metrics.retrieval_latency_ms
                            rag_metrics["retrieval_count"] += 1
                        except Exception as e:
                            logger.warning(f"RAG retrieval failed for {obj_id}: {e}")

                    # Generate questions for this objective/level combo
                    for _ in range(questions_per_combo):
                        if len(questions) >= question_count:
                            break

                        question_id = f"Q-{str(uuid.uuid4())[:8]}"
                        question_gen_start = time.time()

                        # Build question with content from chunks if available
                        question_stem = f"Question about {obj_id}"
                        correct_answer = "Correct answer based on content"

                        if source_chunks and len(source_chunks) > 0:
                            # Use chunk content to build more specific question
                            first_chunk = source_chunks[0]
                            chunk_text = first_chunk.get("text", "")[:500]
                            question_stem = f"Based on the following content, {bloom_level} the key concepts:\n\n{chunk_text}"

                        question = {
                            "question_id": question_id,
                            "objective_id": obj_id,
                            "bloom_level": bloom_level,
                            "question_type": "multiple_choice" if bloom_level in ["remember", "understand"] else "short_answer",
                            "stem": question_stem,
                            "correct_answer": correct_answer,
                            "source_chunks": [c.get("chunk_id", "") for c in source_chunks[:3]],
                            "status": "generated",
                            "generation_latency_ms": (time.time() - question_gen_start) * 1000
                        }
                        questions.append(question)

                        # Log question generation
                        q_data = {
                            "question_id": question_id,
                            "question_type": question["question_type"],
                            "question_stem": question_stem[:200],
                            "correct_answer": correct_answer,
                            "difficulty": "medium",
                            "bloom_level": bloom_level
                        }
                        _log_question_generation(
                            capture,
                            question_data=q_data,
                            source_chunks=[c.get("chunk_id", "") for c in source_chunks[:3]],
                            rationale=f"Generated {question['question_type']} targeting {bloom_level} level for objective {obj_id}",
                            latency_ms=question["generation_latency_ms"]
                        )

                if len(questions) >= question_count:
                    break

            # Calculate average retrieval latency
            if rag_metrics["retrieval_count"] > 0:
                rag_metrics["avg_retrieval_latency_ms"] /= rag_metrics["retrieval_count"]

            # Build assessment
            assessment = {
                "assessment_id": assessment_id,
                "course_id": course_id,
                "course_slug": course_slug,
                "created_at": datetime.now().isoformat(),
                "objectives_targeted": objectives,
                "bloom_levels": levels,
                "requested_count": question_count,
                "actual_count": len(questions),
                "questions": questions,
                "rag_metrics": rag_metrics,
                "corpus_stats": corpus_stats,
                "status": "generated",
                "total_generation_time_ms": (time.time() - start_time) * 1000
            }

            # Log assessment assembly
            _log_assessment_assembly(
                capture,
                assessment_id=assessment_id,
                question_ids=[q["question_id"] for q in questions],
                total_points=len(questions) * 2,
                time_limit=len(questions) * 2,
                rationale=f"Assembled {len(questions)} questions targeting {len(objectives)} objectives at {len(levels)} Bloom levels"
            )

            # Save assessment
            assessment_path = output_dir / f"{assessment_id}.json"
            with open(assessment_path, 'w') as f:
                json.dump(assessment, f, indent=2)

            # Close capture
            session_summary = _finalize_capture(capture)

            return json.dumps({
                "success": True,
                "assessment_id": assessment_id,
                "question_count": len(questions),
                "output_path": str(assessment_path),
                "rag_enabled": rag is not None,
                "decision_capture_enabled": capture is not None,
                "rag_metrics": rag_metrics,
                "session_summary": session_summary,
                "status": "generated"
            })

        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def validate_assessment(assessment_id: str) -> str:
        """
        Validate generated assessment for quality and alignment.

        Args:
            assessment_id: Assessment identifier to validate

        Returns:
            Validation report with scores and issues
        """
        try:
            # Validate training output directory exists
            if not TRAINING_OUTPUT.exists():
                return json.dumps({"error": f"Training output directory not found: {TRAINING_OUTPUT}"})

            # Search for assessment file
            for course_dir in TRAINING_OUTPUT.iterdir():
                if not course_dir.is_dir():
                    continue
                try:
                    for session_dir in course_dir.iterdir():
                        if not session_dir.is_dir():
                            continue
                        assessment_file = session_dir / f"{assessment_id}.json"
                        if assessment_file.exists():
                            try:
                                with open(assessment_file) as f:
                                    assessment = json.load(f)
                            except json.JSONDecodeError as e:
                                return json.dumps({"error": f"Invalid assessment JSON: {e}"})

                            # Run validation
                            validation = {
                                "assessment_id": assessment_id,
                                "validated_at": datetime.now().isoformat(),
                                "scores": {
                                    "objective_coverage": 0.0,
                                    "bloom_alignment": 0.0,
                                    "question_quality": 0.0,
                                    "overall": 0.0
                                },
                                "issues": [],
                                "passed": False
                            }

                            questions = assessment.get("questions", [])
                            objectives = assessment.get("objectives_targeted", [])

                            # Calculate coverage
                            if objectives:
                                covered = set(q.get("objective_id") for q in questions if q.get("objective_id"))
                                validation["scores"]["objective_coverage"] = (
                                    len(covered & set(objectives)) / len(objectives)
                                )

                            # Calculate bloom alignment
                            bloom_targets = assessment.get("bloom_levels", [])
                            if bloom_targets and questions:
                                aligned = sum(
                                    1 for q in questions
                                    if q.get("bloom_level") in bloom_targets
                                )
                                validation["scores"]["bloom_alignment"] = aligned / len(questions)

                            # Mock quality score
                            validation["scores"]["question_quality"] = 0.75

                            # Overall score
                            scores = validation["scores"]
                            validation["scores"]["overall"] = (
                                scores["objective_coverage"] * 0.4 +
                                scores["bloom_alignment"] * 0.3 +
                                scores["question_quality"] * 0.3
                            )

                            validation["passed"] = validation["scores"]["overall"] >= 0.7

                            if not validation["passed"]:
                                if scores["objective_coverage"] < 0.8:
                                    validation["issues"].append("Insufficient objective coverage")
                                if scores["bloom_alignment"] < 0.8:
                                    validation["issues"].append("Bloom level misalignment")

                            return json.dumps(validation)
                except OSError:
                    continue  # Skip inaccessible directories

            return json.dumps({"error": f"Assessment not found: {assessment_id}"})

        except Exception as e:
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def export_training_data(
        format_type: str = "jsonl",
        date_range: Optional[str] = None,
        min_quality: str = "developing",
        min_confidence: float = 0.0,
        require_accepted: bool = False,
        deduplicate: bool = True,
        decision_types: Optional[str] = None
    ) -> str:
        """
        Export captured training data in specified format with quality filtering.

        Args:
            format_type: Output format ("jsonl", "alpaca", "openai", "dpo")
            date_range: Optional "YYYYMMDD-YYYYMMDD" date range filter
            min_quality: Minimum quality level ("inadequate", "developing", "proficient", "exemplary")
            min_confidence: Minimum confidence score (0.0-1.0)
            require_accepted: Only include records with outcome.accepted=True
            deduplicate: Remove duplicate decisions based on content hash
            decision_types: Comma-separated list of decision types to include (optional)

        Returns:
            Export path, statistics, and manifest
        """
        import hashlib
        from collections import defaultdict

        # Quality level ordering
        QUALITY_ORDER = {"inadequate": 0, "developing": 1, "proficient": 2, "exemplary": 3}

        def get_quality_level(record):
            return record.get("metadata", {}).get("quality_level", "unknown")

        def content_hash(record):
            key = json.dumps({
                "decision_type": record.get("decision_type", ""),
                "decision": record.get("decision", ""),
                "rationale": record.get("rationale", "")[:100],
            }, sort_keys=True)
            return hashlib.md5(key.encode()).hexdigest()[:12]

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_id = f"EXP-{timestamp}-{str(uuid.uuid4())[:8]}"

            # Collect all decision files
            decision_files = list(TRAINING_OUTPUT.rglob("decisions_*.jsonl"))

            # Also check courseforge and dart directories
            courseforge_output = TRAINING_DIR / "courseforge"
            dart_output = TRAINING_DIR / "dart"
            for alt_dir in [courseforge_output, dart_output]:
                if alt_dir.exists():
                    decision_files.extend(alt_dir.rglob("decisions_*.jsonl"))

            # Determine course code
            course_code = "aggregated"
            if decision_files:
                first_file = decision_files[0]
                parts = first_file.parts
                for i, part in enumerate(parts):
                    if part in ["trainforge", "courseforge", "dart"] and i + 1 < len(parts):
                        course_code = parts[i + 1]
                        break

            # Use LibV2Storage for proper path
            storage = LibV2Storage(course_code)
            output_dir = storage.get_training_capture_path("trainforge", "export")
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / f"training_export_{format_type}_{timestamp}.jsonl"
            manifest_file = output_dir / f"manifest_{timestamp}.json"

            # Load all records
            all_records = []
            for df in decision_files:
                try:
                    with open(df) as f:
                        for line in f:
                            if line.strip():
                                record = json.loads(line)
                                record["_source_file"] = str(df)
                                all_records.append(record)
                except Exception:
                    continue

            # Initialize filter statistics
            filter_stats = {
                "total_scanned": len(all_records),
                "passed_date_range": 0,
                "passed_quality": 0,
                "passed_confidence": 0,
                "passed_accepted": 0,
                "passed_decision_type": 0,
                "after_deduplication": 0,
            }

            records = all_records

            # Filter by date range
            if date_range:
                start, end = date_range.split("-")
                records = [
                    r for r in records
                    if start <= r.get("timestamp", "")[:10].replace("-", "") <= end
                ]
            filter_stats["passed_date_range"] = len(records)

            # Filter by quality level
            min_order = QUALITY_ORDER.get(min_quality, 1)
            records = [
                r for r in records
                if QUALITY_ORDER.get(get_quality_level(r), 0) >= min_order
            ]
            filter_stats["passed_quality"] = len(records)

            # Filter by confidence
            if min_confidence > 0:
                records = [
                    r for r in records
                    if r.get("confidence", 0.5) >= min_confidence
                ]
            filter_stats["passed_confidence"] = len(records)

            # Filter by accepted outcome
            if require_accepted:
                records = [
                    r for r in records
                    if r.get("outcome", {}).get("accepted", False)
                ]
            filter_stats["passed_accepted"] = len(records)

            # Filter by decision types
            if decision_types:
                allowed_types = [t.strip() for t in decision_types.split(",")]
                records = [
                    r for r in records
                    if r.get("decision_type", "") in allowed_types
                ]
            filter_stats["passed_decision_type"] = len(records)

            # Deduplicate
            if deduplicate:
                seen = set()
                deduped = []
                for r in records:
                    h = content_hash(r)
                    if h not in seen:
                        seen.add(h)
                        deduped.append(r)
                records = deduped
            filter_stats["after_deduplication"] = len(records)

            # Export based on format
            exported = []
            quality_distribution = defaultdict(int)

            with open(output_file, 'w') as f:
                if format_type == "dpo":
                    # DPO format: pair records by (course_id, lo_id, bloom_level)
                    # Chosen = higher quality, Rejected = lower quality
                    groups = defaultdict(list)
                    for r in records:
                        # Group by learning objective context
                        lo_id = r.get("metadata", {}).get("lo_id", "unknown")
                        bloom = r.get("metadata", {}).get("bloom_target", "unknown")
                        course = r.get("course_id", "unknown")
                        key = (course, lo_id, bloom)
                        groups[key].append(r)

                    # Create DPO pairs from each group
                    for key, group_records in groups.items():
                        if len(group_records) < 2:
                            continue

                        # Sort by quality (higher quality first)
                        sorted_recs = sorted(
                            group_records,
                            key=lambda x: (
                                QUALITY_ORDER.get(get_quality_level(x), 0),
                                x.get("confidence", 0.5)
                            ),
                            reverse=True
                        )

                        # Pair adjacent records (best with second-best, etc.)
                        for i in range(0, len(sorted_recs) - 1, 2):
                            chosen = sorted_recs[i]
                            rejected = sorted_recs[i + 1]

                            # Only pair if there's a quality difference
                            chosen_q = QUALITY_ORDER.get(get_quality_level(chosen), 0)
                            rejected_q = QUALITY_ORDER.get(get_quality_level(rejected), 0)

                            if chosen_q > rejected_q or chosen.get("confidence", 0) > rejected.get("confidence", 0):
                                dpo_pair = {
                                    "prompt": f"Generate a {key[2]} level assessment for learning objective: {key[1]}",
                                    "chosen": chosen.get("decision", ""),
                                    "chosen_rationale": chosen.get("rationale", ""),
                                    "rejected": rejected.get("decision", ""),
                                    "rejected_rationale": rejected.get("rationale", ""),
                                    "chosen_quality": get_quality_level(chosen),
                                    "rejected_quality": get_quality_level(rejected),
                                    "chosen_confidence": chosen.get("confidence", 0.5),
                                    "rejected_confidence": rejected.get("confidence", 0.5),
                                    "learning_objective_id": key[1],
                                    "bloom_level": key[2],
                                    "course_id": key[0],
                                }
                                f.write(json.dumps(dpo_pair) + '\n')
                                exported.append(dpo_pair)
                                quality_distribution["dpo_pairs"] += 1

                else:
                    for record in records:
                        quality = get_quality_level(record)
                        quality_distribution[quality] += 1

                        if format_type == "jsonl":
                            # Remove internal fields before export
                            export_record = {k: v for k, v in record.items() if not k.startswith("_")}
                            f.write(json.dumps(export_record) + '\n')
                            exported.append(export_record)

                        elif format_type == "alpaca":
                            alpaca = {
                                "instruction": record.get("operation", ""),
                                "input": record.get("context", ""),
                                "output": record.get("decision", ""),
                                "metadata": {
                                    "course_id": record.get("course_id"),
                                    "decision_type": record.get("decision_type"),
                                    "quality_level": quality,
                                    "confidence": record.get("confidence", 0.5),
                                    "rationale": record.get("rationale", ""),
                                    "bloom_level": record.get("metadata", {}).get("bloom_target"),
                                    "lo_id": record.get("metadata", {}).get("lo_id"),
                                }
                            }
                            f.write(json.dumps(alpaca) + '\n')
                            exported.append(alpaca)

                        elif format_type == "openai":
                            # Enhanced OpenAI format with quality context in system prompt
                            system_content = f"You are an educational content assistant specialized in {record.get('decision_type', 'assessment generation')}."
                            if quality in ["proficient", "exemplary"]:
                                system_content += " Provide high-quality, pedagogically-grounded responses."

                            user_content = record.get("context", "")
                            if record.get("metadata", {}).get("lo_id"):
                                user_content = f"Learning Objective: {record['metadata']['lo_id']}\n\n{user_content}"

                            openai_rec = {
                                "messages": [
                                    {"role": "system", "content": system_content},
                                    {"role": "user", "content": user_content},
                                    {"role": "assistant", "content": record.get("decision", "")}
                                ],
                                "metadata": {
                                    "quality_level": quality,
                                    "rationale": record.get("rationale", ""),
                                }
                            }
                            f.write(json.dumps(openai_rec) + '\n')
                            exported.append(openai_rec)

            # Generate manifest
            manifest = {
                "export_id": export_id,
                "timestamp": datetime.now().isoformat(),
                "format": format_type,
                "filters_applied": {
                    "date_range": date_range,
                    "min_quality": min_quality,
                    "min_confidence": min_confidence,
                    "require_accepted": require_accepted,
                    "deduplicate": deduplicate,
                    "decision_types": decision_types,
                },
                "filter_stages": filter_stats,
                "output_stats": {
                    "records_exported": len(exported),
                    "quality_distribution": dict(quality_distribution),
                },
                "source_stats": {
                    "source_files": len(decision_files),
                    "total_records_scanned": filter_stats["total_scanned"],
                },
                "output_file": str(output_file),
            }

            # Write manifest
            with open(manifest_file, 'w') as f:
                json.dump(manifest, f, indent=2)

            return json.dumps({
                "success": True,
                "export_id": export_id,
                "format": format_type,
                "records_exported": len(exported),
                "output_path": str(output_file),
                "manifest_path": str(manifest_file),
                "filter_stats": filter_stats,
                "quality_distribution": dict(quality_distribution),
            })

        except Exception as e:
            logger.exception("Error exporting training data")
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def get_trainforge_status() -> str:
        """
        Get Trainforge installation status and training data statistics.

        Returns:
            Installation status and data summary
        """
        try:
            status = {
                "installed": TRAINFORGE_PATH.exists(),
                "path": str(TRAINFORGE_PATH),
                "training_output": str(TRAINING_OUTPUT),
                "statistics": {
                    "total_courses": 0,
                    "total_assessments": 0,
                    "total_decisions": 0,
                    "total_questions": 0
                }
            }

            # Count statistics
            if TRAINING_OUTPUT.exists():
                status["statistics"]["total_courses"] = sum(
                    1 for d in TRAINING_OUTPUT.iterdir() if d.is_dir()
                )

                for course_dir in TRAINING_OUTPUT.iterdir():
                    if course_dir.is_dir():
                        assessment_files = list(course_dir.rglob("ASM-*.json"))
                        status["statistics"]["total_assessments"] += len(assessment_files)

                        decision_files = list(course_dir.rglob("decisions_*.jsonl"))
                        for df in decision_files:
                            with open(df) as f:
                                status["statistics"]["total_decisions"] += sum(1 for _ in f)

            return json.dumps(status)

        except Exception as e:
            return json.dumps({"error": str(e)})
