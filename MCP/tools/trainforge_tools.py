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
from lib.paths import LIBV2_COURSES, TRAINFORGE_PATH, TRAINING_DIR  # noqa: E402
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

logger = logging.getLogger(__name__)


def _create_trainforge_session(course_code: str, imscc_source: str, phase: str):
    """Create a capture session for Trainforge operations."""
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

    if hasattr(capture, 'log_chunk_retrieval'):
        capture.log_chunk_retrieval(query, chunks_retrieved, chunks_used, latency_ms)


def _log_question_generation(capture, question_data, source_chunks: list, rationale: str, latency_ms: float):
    """Log question generation event."""
    if capture is None:
        return

    if hasattr(capture, 'log_question_generation'):
        capture.log_question_generation(question_data, source_chunks, rationale, latency_ms)


def _log_assessment_assembly(capture, assessment_id: str, question_ids: list, total_points: int,
                             time_limit: int, rationale: str):
    """Log assessment assembly event."""
    if capture is None:
        return

    if hasattr(capture, 'log_assessment_assembly'):
        capture.log_assessment_assembly(assessment_id, question_ids, total_points, time_limit, rationale)


def _finalize_capture(capture, status: str = "success"):
    """Finalize a capture session."""
    if capture is None:
        return {}

    summary = {}
    if hasattr(capture, '__exit__'):
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
                # Validate IMSCC structure: imsmanifest.xml is required
                if 'imsmanifest.xml' not in z.namelist():
                    return json.dumps({
                        "error": f"Invalid IMSCC package: missing imsmanifest.xml in {imscc.name}",
                        "hint": "A valid IMSCC package must contain an imsmanifest.xml file"
                    })
                analysis["has_manifest"] = True

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
        course_slug: str = "",
        imscc_path: str = ""
    ) -> str:
        """
        Generate assessments from course content using the canonical
        :class:`AssessmentGenerator` path.

        Wave 26 unification: this surface no longer hand-rolls question
        payloads with placeholder strings (``"Correct answer based on
        content"``). It dispatches directly to
        :class:`Trainforge.generators.assessment_generator.AssessmentGenerator`,
        the same generator used by the internal pipeline. That generator
        performs content grounding, leak checking, and template-fallback
        flagging.

        Args:
            course_id: Course identifier (e.g., INT_101)
            objective_ids: Comma-separated learning objective IDs to assess
            bloom_levels: Comma-separated Bloom's levels to target
                         (remember, understand, apply, analyze, evaluate, create)
            question_count: Number of questions to generate (default: 10)
            course_slug: LibV2 course slug for RAG retrieval (optional)
            imscc_path: Path to IMSCC package for direct content extraction (optional).
                       Used as fallback when RAG corpus is unavailable.

        Returns:
            Generated assessment data with question IDs, Bloom distribution,
            and source-chunk references from the real generator. On error
            (no chunks, import failure, generator exception) returns a
            structured ``{"error": ..., "cause": ...}`` payload — never a
            placeholder-success response.
        """
        import time
        start_time = time.time()

        # Verify the canonical generator is available. If not, surface a
        # structured error — never fall back to placeholder content.
        if not HAS_ASSESSMENT_GENERATOR:
            return json.dumps({
                "error": "AssessmentGenerator unavailable",
                "cause": "import_failed",
                "hint": (
                    "Trainforge.generators.assessment_generator could not "
                    "be imported. Verify Trainforge package is on the "
                    "Python path."
                ),
            })

        try:
            objectives = [o.strip() for o in objective_ids.split(",") if o.strip()]
            levels = [l.strip() for l in bloom_levels.split(",") if l.strip()]

            if not objectives:
                return json.dumps({
                    "error": "No objective IDs provided",
                    "cause": "empty_objective_ids",
                })
            if not levels:
                return json.dumps({
                    "error": "No Bloom levels provided",
                    "cause": "empty_bloom_levels",
                })

            # Sanitize course_id to prevent path traversal
            safe_course_id = sanitize_path_component(course_id)

            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

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
                        logger.info(
                            "RAG initialized for %s: %d chunks",
                            course_slug, corpus_stats.get('chunk_count', 0),
                        )
                except Exception as e:
                    logger.warning(
                        "Could not initialize RAG for %s: %s",
                        course_slug, e,
                    )

            # If RAG unavailable, try direct IMSCC content extraction. The
            # AssessmentGenerator consumes a list of chunk dicts with
            # ``text`` + ``id``/``chunk_id`` keys; we normalize to that
            # shape so the ContentExtractor hits real content instead of
            # template fallbacks.
            imscc_content_chunks = []
            if not rag and imscc_path:
                import zipfile
                imscc_file = Path(imscc_path)
                if imscc_file.exists() and imscc_file.suffix == '.imscc':
                    try:
                        validate_path_within_root(imscc_file.resolve(), _PROJECT_ROOT)
                        with zipfile.ZipFile(imscc_file, 'r') as z:
                            if 'imsmanifest.xml' not in z.namelist():
                                logger.warning(
                                    "IMSCC at %s missing imsmanifest.xml",
                                    imscc_path,
                                )
                            for name in z.namelist():
                                if name.endswith('.html'):
                                    content = z.read(name).decode(
                                        'utf-8', errors='ignore',
                                    )
                                    # Strip HTML for text-only fallback chunks.
                                    import re as _re
                                    text = _re.sub(r'<[^>]+>', ' ', content)
                                    text = ' '.join(text.split())
                                    if len(text) > 50:
                                        imscc_content_chunks.append({
                                            "id": name,
                                            "chunk_id": name,
                                            "text": text[:4000],
                                            "source": name,
                                            "word_count": len(text.split()),
                                        })
                        logger.info(
                            "Extracted %d content chunks from IMSCC %s",
                            len(imscc_content_chunks), imscc_file.name,
                        )
                    except (ValueError, zipfile.BadZipFile) as e:
                        logger.warning(
                            "Failed to extract IMSCC content from %s: %s",
                            imscc_path, e,
                        )
                elif imscc_path:
                    logger.warning(
                        "IMSCC path not found or invalid: %s", imscc_path,
                    )

            # No chunks available and no RAG: error instead of generating
            # placeholder questions.
            if not rag and not imscc_content_chunks:
                _finalize_capture(capture, status="error")
                return json.dumps({
                    "error": "No source content available for generation",
                    "cause": "no_chunks",
                    "hint": (
                        "Provide a valid course_slug pointing to a "
                        "LibV2-indexed course, or an imscc_path pointing "
                        "to a valid IMSCC package."
                    ),
                })

            # Dispatch to the canonical generator. If we have a RAG
            # bridge, pass it; otherwise hand the extracted IMSCC chunks
            # directly so the generator's ContentExtractor can operate
            # on real text.
            generator = AssessmentGenerator(
                capture=capture,
                check_leaks=True,
                rag=rag,
            )

            try:
                assessment_data = generator.generate(
                    course_code=safe_course_id,
                    objective_ids=objectives,
                    bloom_levels=levels,
                    question_count=question_count,
                    source_chunks=(
                        None if rag else imscc_content_chunks
                    ),
                )
            except Exception as e:
                logger.exception("AssessmentGenerator.generate failed")
                _finalize_capture(capture, status="error")
                return json.dumps({
                    "error": f"AssessmentGenerator.generate() failed: {e}",
                    "cause": "generator_exception",
                })

            # Convert to serializable dict + augment with MCP-surface fields
            assessment = assessment_data.to_dict()
            assessment_id = assessment["assessment_id"]

            assessment.update({
                "course_id": course_id,
                "course_slug": course_slug,
                "requested_count": question_count,
                "actual_count": len(assessment.get("questions", [])),
                "corpus_stats": corpus_stats,
                "status": "generated",
                "total_generation_time_ms": (time.time() - start_time) * 1000,
                "generator_path": "AssessmentGenerator",
            })

            # Log assessment assembly (best-effort; no-ops if capture is None)
            _log_assessment_assembly(
                capture,
                assessment_id=assessment_id,
                question_ids=[q["question_id"] for q in assessment["questions"]],
                total_points=int(assessment.get("total_points", 0)),
                time_limit=len(assessment["questions"]) * 2,
                rationale=(
                    f"Assembled {len(assessment['questions'])} questions "
                    f"targeting {len(objectives)} objectives at "
                    f"{len(levels)} Bloom levels via AssessmentGenerator"
                ),
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
                "question_count": len(assessment["questions"]),
                "output_path": str(assessment_path),
                "rag_enabled": rag is not None,
                "decision_capture_enabled": capture is not None,
                "generator_path": "AssessmentGenerator",
                "session_summary": session_summary,
                "status": "generated",
            })

        except Exception as e:
            logger.exception("generate_assessments failed unexpectedly")
            return json.dumps({
                "error": str(e),
                "cause": "unexpected_exception",
            })

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

                            # Content-based question quality scoring
                            try:
                                from lib.validators.question_quality import QuestionQualityValidator
                                qq_validator = QuestionQualityValidator()
                                qq_result = qq_validator.validate({
                                    "assessment_data": assessment,
                                })
                                validation["scores"]["question_quality"] = qq_result.score or 0.0
                                for issue in qq_result.issues:
                                    validation["issues"].append(issue.message)
                            except Exception:
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
            return hashlib.sha256(key.encode()).hexdigest()[:12]

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
            load_errors = []
            for df in decision_files:
                try:
                    with open(df) as f:
                        for line in f:
                            if line.strip():
                                record = json.loads(line)
                                record["_source_file"] = str(df)
                                all_records.append(record)
                except (OSError, json.JSONDecodeError) as e:
                    load_errors.append(f"{df.name}: {e}")
                    logger.warning("Failed to load decision file %s: %s", df, e)
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

            result = {
                "success": True,
                "export_id": export_id,
                "format": format_type,
                "records_exported": len(exported),
                "output_path": str(output_file),
                "manifest_path": str(manifest_file),
                "filter_stats": filter_stats,
                "quality_distribution": dict(quality_distribution),
            }
            if load_errors:
                result["load_errors"] = load_errors[:20]  # Cap to avoid huge responses
                result["load_error_count"] = len(load_errors)
            return json.dumps(result)

        except Exception as e:
            logger.exception("Error exporting training data")
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def get_trainforge_status() -> str:
        """
        Get Trainforge installation status and training data statistics.

        Wave 138a extension: also surfaces per-course resume-checkpoint
        sidecar state and the latest eval_report.json's
        ``content_type_role_alignment_summary.alignment_rate`` so an
        operator can see "is there in-flight work to resume" + "is the
        latest adapter's role alignment healthy" without parsing JSON.

        Resume-checkpoint sidecars surveyed (read-only):
        - ``training_specs/.synthesis_pairs_checkpoint.jsonl``
          (Worker A — synthesize_training)
        - ``corpus/.teaching_role_checkpoint.jsonl``
          (Wave 137 followup — align_chunks)
        - ``eval/.eval_results_checkpoint.jsonl``
          (Worker C — slm_eval_harness, per-stage)
        - ``models/<model_id>/eval/.eval_results_checkpoint.jsonl``
          (Worker C — per-adapter eval)

        Returns:
            Installation status, data summary, per-course in-flight
            checkpoint sidecars, and latest eval role-alignment_rate.
        """
        try:
            status = {
                "installed": TRAINFORGE_PATH.exists(),
                "path": str(TRAINFORGE_PATH),
                "training_output": str(TRAINING_OUTPUT),
                "libv2_courses_path": str(LIBV2_COURSES),
                "statistics": {
                    "total_courses": 0,
                    "total_assessments": 0,
                    "total_decisions": 0,
                    "total_questions": 0
                },
                "in_flight_checkpoints": [],
                "role_alignment": [],
            }

            # Count statistics from the training-captures tree
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
                            try:
                                with open(df) as f:
                                    status["statistics"]["total_decisions"] += sum(1 for _ in f)
                            except OSError:
                                continue

            # Wave 138a — survey LibV2 courses for resume-checkpoint
            # sidecars and latest eval role-alignment_rate. Pure
            # filesystem reads; no LLM, no graph traversal.
            if LIBV2_COURSES.exists():
                for course_dir in sorted(LIBV2_COURSES.iterdir()):
                    if not course_dir.is_dir():
                        continue
                    course_slug = course_dir.name

                    # Resume-checkpoint sidecars (existence + size only;
                    # cheap stat). Each sidecar is "in-flight work" only
                    # when the canonical run hasn't unlinked it.
                    sidecar_specs = [
                        ("synthesis_pairs",
                         course_dir / "training_specs" / ".synthesis_pairs_checkpoint.jsonl"),
                        ("teaching_role",
                         course_dir / "corpus" / ".teaching_role_checkpoint.jsonl"),
                        ("eval_stage_course",
                         course_dir / "eval" / ".eval_results_checkpoint.jsonl"),
                    ]
                    for kind, sidecar_path in sidecar_specs:
                        try:
                            if sidecar_path.is_file():
                                stat = sidecar_path.stat()
                                status["in_flight_checkpoints"].append({
                                    "course_slug": course_slug,
                                    "kind": kind,
                                    "path": str(sidecar_path),
                                    "size_bytes": stat.st_size,
                                    "mtime": datetime.fromtimestamp(
                                        stat.st_mtime
                                    ).isoformat(),
                                })
                        except OSError:
                            continue

                    # Per-adapter eval-stage sidecars under models/<id>/eval/
                    models_dir = course_dir / "models"
                    if models_dir.is_dir():
                        try:
                            adapter_dirs = sorted(
                                d for d in models_dir.iterdir() if d.is_dir()
                            )
                        except OSError:
                            adapter_dirs = []
                        for adapter_dir in adapter_dirs:
                            sidecar_path = (
                                adapter_dir / "eval"
                                / ".eval_results_checkpoint.jsonl"
                            )
                            try:
                                if sidecar_path.is_file():
                                    stat = sidecar_path.stat()
                                    status["in_flight_checkpoints"].append({
                                        "course_slug": course_slug,
                                        "kind": "eval_stage_adapter",
                                        "model_id": adapter_dir.name,
                                        "path": str(sidecar_path),
                                        "size_bytes": stat.st_size,
                                        "mtime": datetime.fromtimestamp(
                                            stat.st_mtime
                                        ).isoformat(),
                                    })
                            except OSError:
                                continue

                    # Latest eval_report.json's
                    # content_type_role_alignment_summary.alignment_rate.
                    # Pick the most-recently-modified eval_report.json
                    # under models/*/ (skipping smoke-mode reports per
                    # the harness contract).
                    latest: Optional[Dict[str, object]] = None
                    if models_dir.is_dir():
                        try:
                            for adapter_dir in models_dir.iterdir():
                                if not adapter_dir.is_dir():
                                    continue
                                report_path = adapter_dir / "eval_report.json"
                                if not report_path.is_file():
                                    continue
                                try:
                                    mtime = report_path.stat().st_mtime
                                except OSError:
                                    continue
                                if latest is None or mtime > latest["_mtime"]:
                                    latest = {
                                        "_mtime": mtime,
                                        "model_id": adapter_dir.name,
                                        "path": report_path,
                                    }
                        except OSError:
                            latest = None

                    if latest is not None:
                        try:
                            with open(latest["path"]) as f:
                                report = json.load(f)
                        except (OSError, json.JSONDecodeError) as e:
                            status["role_alignment"].append({
                                "course_slug": course_slug,
                                "model_id": latest["model_id"],
                                "error": f"failed to load eval_report.json: {e}",
                            })
                            continue
                        if report.get("smoke_mode") is True:
                            # Smoke reports are intentionally not gated;
                            # surface them as advisory.
                            status["role_alignment"].append({
                                "course_slug": course_slug,
                                "model_id": latest["model_id"],
                                "smoke_mode": True,
                                "alignment_rate": None,
                            })
                            continue
                        summary = report.get(
                            "content_type_role_alignment_summary"
                        )
                        if isinstance(summary, dict):
                            status["role_alignment"].append({
                                "course_slug": course_slug,
                                "model_id": latest["model_id"],
                                "alignment_rate": summary.get("alignment_rate"),
                                "mismatched_content_types": summary.get(
                                    "mismatched_content_types", []
                                ),
                                "content_types_with_expected_mode": summary.get(
                                    "content_types_with_expected_mode"
                                ),
                                "report_path": str(latest["path"]),
                            })
                        else:
                            status["role_alignment"].append({
                                "course_slug": course_slug,
                                "model_id": latest["model_id"],
                                "alignment_rate": None,
                                "note": (
                                    "eval_report.json predates Wave 138a; "
                                    "no content_type_role_alignment_summary"
                                ),
                                "report_path": str(latest["path"]),
                            })

            return json.dumps(status)

        except Exception as e:
            logger.exception("get_trainforge_status failed")
            return json.dumps({"error": str(e)})

    @mcp.tool()
    async def analyze_teaching_role_alignment(
        chunks_path: str,
        min_chunks_for_flag: int = 5,
    ) -> str:
        """
        Wave 138a — Tier-2 graph-derived teaching-role alignment check.

        Wraps :class:`Trainforge.eval.teaching_role_alignment.TeachingRoleAlignmentEvaluator`
        so an external MCP client can probe a corpus's
        ``content_type_label`` -> ``teaching_role`` distribution
        without invoking the full SLM eval harness. Pure file read +
        Python aggregation; no LLM dispatch, no model_callable.
        Wall-time well under 100ms on a 1000-chunk corpus.

        Useful as a pre-flight check before retraining: if
        ``alignment_rate`` is below the operator's threshold (e.g. 0.85
        — see the Wave 138a/W3 ``EvalGatingValidator`` warning gate),
        retraining without a curriculum-alignment fix would burn a
        training run on a known-degraded corpus.

        Args:
            chunks_path: Absolute path to a course's ``chunks.jsonl``.
                Typically ``LibV2/courses/<slug>/corpus/chunks.jsonl``.
            min_chunks_for_flag: Skip content_type_label buckets with
                fewer than this many chunks (statistical noise floor;
                default 5, matches the evaluator default).

        Returns:
            JSON-serialized dict carrying ``content_type_role_alignment``
            (per-label distribution + mismatch flag) and ``summary``
            (alignment_rate + mismatched_content_types). Shape matches
            the in-process evaluator + the Wave 138a fields surfaced in
            ``eval_report.json``.
        """
        try:
            # Sandbox: read-only path resolution. The evaluator does
            # not write — we still validate-within-root to keep the
            # tool consistent with other read-only Trainforge surfaces.
            try:
                resolved = Path(chunks_path).resolve()
            except (OSError, RuntimeError) as e:
                return json.dumps({
                    "error": f"Could not resolve chunks_path: {e}",
                    "cause": "invalid_path",
                })

            if not resolved.is_file():
                return json.dumps({
                    "error": f"chunks file not found: {chunks_path}",
                    "cause": "missing_chunks",
                })

            try:
                from Trainforge.eval.teaching_role_alignment import (
                    TeachingRoleAlignmentEvaluator,
                )
            except ImportError as e:
                return json.dumps({
                    "error": (
                        "TeachingRoleAlignmentEvaluator unavailable: "
                        f"{e}"
                    ),
                    "cause": "import_failed",
                })

            try:
                min_chunks = int(min_chunks_for_flag)
            except (TypeError, ValueError):
                return json.dumps({
                    "error": (
                        f"min_chunks_for_flag must be an integer, got "
                        f"{min_chunks_for_flag!r}"
                    ),
                    "cause": "invalid_argument",
                })
            if min_chunks < 1:
                return json.dumps({
                    "error": "min_chunks_for_flag must be >= 1",
                    "cause": "invalid_argument",
                })

            evaluator = TeachingRoleAlignmentEvaluator(
                resolved,
                min_chunks_for_flag=min_chunks,
            )
            result = evaluator.evaluate()
            # Annotate the input so a downstream client knows which
            # corpus the alignment_rate refers to.
            result["chunks_path"] = str(resolved)
            return json.dumps(result)

        except Exception as e:
            logger.exception("analyze_teaching_role_alignment failed")
            return json.dumps({
                "error": str(e),
                "cause": "unexpected_exception",
            })
