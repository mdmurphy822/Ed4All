#!/usr/bin/env python3
"""
Generic Course Corpus Pipeline

Processes any Courseforge IMSCC package into a Sourceforge-compatible
RAG corpus for LibV2 import.

Usage:
    python -m Trainforge.process_course \
        --imscc path/to/course.imscc \
        --course-code DIGPED_101 \
        --division ARTS --domain education --subdomain instructional-design \
        --output Trainforge/output/digped_101

    # With objectives file for Bloom's-based difficulty mapping:
    python -m Trainforge.process_course \
        --imscc path/to/course.imscc \
        --objectives path/to/objectives.json \
        --course-code DIGPED_101 \
        --division ARTS --domain education \
        --output Trainforge/output/digped_101 \
        --import-to-libv2
"""

import argparse
import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from Trainforge.parsers.html_content_parser import HTMLContentParser, HTMLTextExtractor
from lib.decision_capture import DecisionCapture


# ---------------------------------------------------------------------------
# Bloom's → difficulty mapping
# ---------------------------------------------------------------------------

BLOOM_TO_DIFFICULTY = {
    "remember": "foundational",
    "understand": "foundational",
    "apply": "intermediate",
    "analyze": "intermediate",
    "evaluate": "advanced",
    "create": "advanced",
}

# Numeric weights for median-based difficulty calculation
BLOOM_WEIGHT = {
    "remember": 1, "understand": 2, "apply": 3,
    "analyze": 4, "evaluate": 5, "create": 6,
}

# Resource types that cap difficulty one level below week max
# (overviews and summaries are inherently introductory)
INTRODUCTORY_RESOURCE_TYPES = {"overview", "summary"}

# ---------------------------------------------------------------------------
# Resource type classification patterns
# ---------------------------------------------------------------------------

RESOURCE_TYPE_PATTERNS = [
    # quiz / self-check / assessment
    (re.compile(r"self[_-]?check|quiz|assessment", re.I), "quiz"),
    # overview / introduction
    (re.compile(r"overview|introduction", re.I), "overview"),
    # summary / recap
    (re.compile(r"summary|recap", re.I), "summary"),
    # discussion
    (re.compile(r"discussion", re.I), "discussion"),
    # application / activity
    (re.compile(r"application|activity", re.I), "application"),
]


def classify_resource(path: str) -> Tuple[str, str, str]:
    """
    Classify an HTML resource and extract module info from its path.

    Returns:
        (resource_type, module_id, module_title)
    """
    stem = Path(path).stem
    path_lower = path.lower()

    # Determine resource type
    resource_type = "page"  # default
    for pattern, rtype in RESOURCE_TYPE_PATTERNS:
        if pattern.search(path_lower):
            resource_type = rtype
            break

    module_id = stem
    # Build a human-readable title from the stem
    # Strip leading week_XX_ or section_XX_ prefix, then the content_XX_ prefix
    title = stem
    title = re.sub(r"^(?:week|section)_\d+_", "", title)
    title = re.sub(r"^(?:content|module)_\d+_", "", title)
    module_title = title.replace("_", " ").strip().title() or stem.replace("_", " ").title()

    return resource_type, module_id, module_title


def extract_week_number(path: str) -> int:
    """Extract week/section number from path. Returns 0 if not found."""
    m = re.search(r"(?:week|section)[_-]?(\d+)", path, re.I)
    return int(m.group(1)) if m else 0


# ---------------------------------------------------------------------------
# Objectives loader
# ---------------------------------------------------------------------------

def load_objectives(objectives_path: Path) -> Dict[str, Any]:
    """
    Load objectives JSON and build week→bloom mapping.

    Returns dict with keys:
        terminal_objectives: list
        chapter_objectives: list
        week_bloom_map: {week_num: [bloom_levels]}
        bloom_distribution: {level: count}
        description: str
    """
    with open(objectives_path) as f:
        data = json.load(f)

    week_bloom: Dict[int, List[str]] = defaultdict(list)

    for chapter in data.get("chapter_objectives", []):
        # Parse week range from chapter name like "Week 1-2: ..."
        chapter_name = chapter.get("chapter", "")
        week_match = re.search(r"[Ww]eek\s+(\d+)(?:\s*-\s*(\d+))?", chapter_name)
        if week_match:
            start = int(week_match.group(1))
            end = int(week_match.group(2)) if week_match.group(2) else start
            weeks = list(range(start, end + 1))
        else:
            weeks = []

        for obj in chapter.get("objectives", []):
            bloom = obj.get("bloomLevel", "").lower()
            if bloom:
                for w in weeks:
                    week_bloom[w].append(bloom)

    return {
        "terminal_objectives": data.get("terminal_objectives", []),
        "chapter_objectives": data.get("chapter_objectives", []),
        "week_bloom_map": dict(week_bloom),
        "bloom_distribution": data.get("bloom_distribution", {}),
        "description": data.get("description", ""),
        "course_title": data.get("course_title", ""),
    }


# ---------------------------------------------------------------------------
# Concept tag normalization
# ---------------------------------------------------------------------------

def normalize_tag(raw: str) -> str:
    """Normalize a concept string to lowercase-hyphenated tag."""
    tag = raw.lower().strip()
    tag = re.sub(r"[^a-z0-9\s-]", "", tag)
    tag = re.sub(r"\s+", "-", tag)
    tag = tag.strip("-")
    # Limit to 4 words
    parts = tag.split("-")
    if len(parts) > 4:
        tag = "-".join(parts[:4])
    # Tags must start with a letter (LibV2 lowercase-hyphenated format)
    if tag and not tag[0].isalpha():
        return ""
    return tag


# ---------------------------------------------------------------------------
# CourseProcessor
# ---------------------------------------------------------------------------

class CourseProcessor:
    """Generic processor that turns a Courseforge IMSCC into a Trainforge corpus."""

    TARGET_CHUNK_SIZE = 500
    MIN_CHUNK_SIZE = 100  # Courseforge pages can be short (overviews, summaries)
    MAX_CHUNK_SIZE = 800

    def __init__(
        self,
        imscc_path: str,
        output_dir: str,
        course_code: str,
        division: str = "STEM",
        domain: str = "",
        subdomains: Optional[List[str]] = None,
        secondary_domains: Optional[List[str]] = None,
        topics: Optional[List[str]] = None,
        objectives_path: Optional[str] = None,
    ):
        self.imscc_path = Path(imscc_path)
        self.output_dir = Path(output_dir)
        self.course_code = course_code
        self.division = division
        self.domain = domain
        self.subdomains = subdomains or []
        self.secondary_domains = secondary_domains or []
        self.topics = topics or []

        # Sub-directories
        self.corpus_dir = self.output_dir / "corpus"
        self.graph_dir = self.output_dir / "graph"
        self.training_specs_dir = self.output_dir / "training_specs"
        self.pedagogy_dir = self.output_dir / "pedagogy"
        self.quality_dir = self.output_dir / "quality"

        # Objectives (optional)
        self.objectives: Optional[Dict[str, Any]] = None
        if objectives_path:
            self.objectives = load_objectives(Path(objectives_path))

        # Decision capture
        self.capture = DecisionCapture(
            course_code=course_code,
            phase="content_extraction",
            tool="trainforge",
            streaming=True,
        )

        # HTML parser
        self.html_parser = HTMLContentParser()

        # Stats
        self.stats: Dict[str, Any] = {
            "total_chunks": 0,
            "total_words": 0,
            "total_tokens_estimate": 0,
            "chunk_types": defaultdict(int),
            "difficulty_distribution": defaultdict(int),
            "sections_processed": 0,
            "modules_processed": 0,
            "quizzes_processed": 0,
        }
        self._all_concept_tags: set = set()

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(self) -> Dict[str, Any]:
        """Run the full 6-stage pipeline. Returns summary dict."""
        print(f"[Trainforge] Processing {self.imscc_path.name} → {self.output_dir}")

        self._create_directories()

        # Stage 1
        print("[1/6] Extracting IMSCC package...")
        title, html_files = self._extract_imscc()

        # Stage 2
        print("[2/6] Parsing HTML content...")
        parsed_items = self._parse_html(html_files)

        # Stage 3
        print("[3/6] Chunking content into pedagogical units...")
        chunks = self._chunk_content(parsed_items)

        # Stage 4
        print("[4/6] Writing chunks...")
        self._write_chunks(chunks)

        # Stage 5
        print("[5/6] Generating metadata...")
        concept_graph = self._generate_concept_graph(chunks)
        manifest = self._generate_manifest(title, concept_graph=concept_graph)
        corpus_stats = self._generate_corpus_stats()
        quality_report = self._generate_quality_report(chunks)

        # Stage 6
        print("[6/6] Writing metadata files...")
        self._write_metadata(manifest, corpus_stats, concept_graph, quality_report)

        summary = {
            "status": "success",
            "output_dir": str(self.output_dir),
            "course_code": self.course_code,
            "title": title,
            "stats": {k: (dict(v) if isinstance(v, defaultdict) else v) for k, v in self.stats.items()},
        }

        print(f"\n[SUCCESS] Generated {self.stats['total_chunks']} chunks")
        print(f"  Total words: {self.stats['total_words']:,}")
        print(f"  Total tokens (est): {self.stats['total_tokens_estimate']:,}")
        print(f"  Output: {self.output_dir}")

        return summary

    # ------------------------------------------------------------------
    # Stage 1: Extract IMSCC
    # ------------------------------------------------------------------

    def _extract_imscc(self) -> Tuple[str, List[Dict[str, Any]]]:
        if not self.imscc_path.exists():
            raise FileNotFoundError(f"IMSCC not found: {self.imscc_path}")

        self.capture.log_decision(
            decision_type="imscc_extraction",
            decision=f"Extract {self.imscc_path.name}",
            rationale="Parse IMSCC manifest and HTML resources to build RAG corpus for LibV2 import",
        )

        html_files: List[Dict[str, Any]] = []
        title = self.course_code

        with zipfile.ZipFile(self.imscc_path, "r") as z:
            # Try to get title from manifest
            try:
                manifest_xml = z.read("imsmanifest.xml").decode("utf-8")
                root = ET.fromstring(manifest_xml)
                # Search for title across common namespaces
                for ns_uri in [
                    "http://ltsc.ieee.org/xsd/imsccv1p3/LOM/manifest",
                    "http://ltsc.ieee.org/xsd/imsccv1p1/LOM/manifest",
                    "http://www.imsglobal.org/xsd/imsmd_v1p2",
                ]:
                    elem = root.find(f".//{{{ns_uri}}}title/{{{ns_uri}}}string")
                    if elem is not None and elem.text:
                        title = elem.text.strip()
                        break
                # Fallback: try unnamespaced
                if title == self.course_code:
                    for elem in root.iter():
                        if elem.tag.endswith("}string") or elem.tag == "string":
                            if elem.text and len(elem.text.strip()) > 5:
                                title = elem.text.strip()
                                break
            except Exception:
                pass

            # If we have an objectives file with a title, prefer that
            if self.objectives and self.objectives.get("course_title"):
                title = self.objectives["course_title"]

            # Extract HTML files
            for name in z.namelist():
                if name.endswith(".html") or name.endswith(".htm"):
                    try:
                        content = z.read(name).decode("utf-8", errors="ignore")
                        html_files.append({"path": name, "content": content, "id": Path(name).stem})
                    except Exception as e:
                        print(f"  Warning: Failed to read {name}: {e}")

        print(f"  Course title: {title}")
        print(f"  HTML files: {len(html_files)}")

        return title, html_files

    # ------------------------------------------------------------------
    # Stage 2: Parse HTML
    # ------------------------------------------------------------------

    def _parse_html(self, html_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        parsed_items = []

        for item in html_files:
            if not item.get("content"):
                continue

            content = item["content"]
            resource_type, module_id, module_title = classify_resource(item["path"])

            # Strip assessment feedback from quiz HTML BEFORE parsing
            # so sections don't contain answer feedback text
            if resource_type == "quiz":
                content = self._strip_assessment_feedback(content)

            parsed = self.html_parser.parse(content)
            week_num = extract_week_number(item["path"])

            parsed_items.append({
                "item_id": item["id"],
                "item_path": item["path"],
                "title": parsed.title,
                "resource_type": resource_type,
                "module_id": module_id,
                "module_title": module_title,
                "week_num": week_num,
                "word_count": parsed.word_count,
                "sections": parsed.sections,
                "learning_objectives": parsed.learning_objectives,
                "key_concepts": parsed.key_concepts,
                "interactive_components": parsed.interactive_components,
                "raw_html": content,
                # New: metadata from JSON-LD / data-cf-* attributes
                "page_id": parsed.page_id,
                "misconceptions": parsed.misconceptions,
                "suggested_assessment_types": parsed.suggested_assessment_types,
                "courseforge_metadata": parsed.metadata.get("courseforge"),
            })

        self.stats["modules_processed"] = len([p for p in parsed_items if p["resource_type"] == "page"])
        self.stats["quizzes_processed"] = len([p for p in parsed_items if p["resource_type"] == "quiz"])

        # Count unique weeks/sections
        weeks = {p["week_num"] for p in parsed_items if p["week_num"] > 0}
        self.stats["sections_processed"] = len(weeks)

        print(f"  Parsed {len(parsed_items)} items (modules={self.stats['modules_processed']}, quizzes={self.stats['quizzes_processed']}, weeks={len(weeks)})")
        return parsed_items

    # ------------------------------------------------------------------
    # Stage 3: Chunk content
    # ------------------------------------------------------------------

    def _chunk_content(self, parsed_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []
        chunk_counter = 1
        prefix = f"{self.course_code.lower()}_chunk_"
        prev_chunk_id: Optional[str] = None
        current_module_id: Optional[str] = None
        position_in_module = 0

        for item in parsed_items:
            # Reset position counter when module changes
            if item["module_id"] != current_module_id:
                current_module_id = item["module_id"]
                position_in_module = 0

            # Strip assessment feedback from quiz/self-check content
            raw_html = item["raw_html"]
            if item["resource_type"] == "quiz":
                raw_html = self._strip_assessment_feedback(raw_html)

            if not item["sections"]:
                # No sections — chunk the whole item as one piece
                text = self._extract_plain_text(raw_html)
                if item["resource_type"] == "quiz":
                    text = self._strip_feedback_from_text(text)
                if text.strip():
                    item_chunks = self._chunk_text_block(
                        text=text,
                        html=raw_html,
                        item=item,
                        heading=item["title"],
                        chunk_type=self._type_from_resource(item["resource_type"]),
                        prefix=prefix,
                        start_id=chunk_counter,
                        follows_chunk_id=prev_chunk_id,
                        position_in_module=position_in_module,
                    )
                    chunks.extend(item_chunks)
                    chunk_counter += len(item_chunks)
                    if item_chunks:
                        prev_chunk_id = item_chunks[-1]["id"]
                        position_in_module += len(item_chunks)
                continue

            # Merge adjacent small sections into larger pedagogical units
            merged = self._merge_small_sections(item["sections"])

            for heading, text, chunk_type in merged:
                if not text.strip():
                    continue
                # Strip feedback from quiz section text (sections were parsed before HTML stripping)
                if item["resource_type"] == "quiz":
                    text = self._strip_feedback_from_text(text)
                html_block = self._extract_section_html(raw_html, heading)
                item_chunks = self._chunk_text_block(
                    text=text,
                    html=html_block,
                    item=item,
                    heading=heading,
                    chunk_type=chunk_type,
                    prefix=prefix,
                    start_id=chunk_counter,
                    follows_chunk_id=prev_chunk_id,
                    position_in_module=position_in_module,
                )
                chunks.extend(item_chunks)
                chunk_counter += len(item_chunks)
                if item_chunks:
                    prev_chunk_id = item_chunks[-1]["id"]
                    position_in_module += len(item_chunks)

        self.stats["total_chunks"] = len(chunks)
        print(f"  Generated {len(chunks)} chunks")
        return chunks

    def _merge_small_sections(self, sections) -> List[Tuple[str, str, str]]:
        """
        Merge adjacent sections that are below MIN_CHUNK_SIZE into combined blocks.

        Returns list of (heading, combined_text, chunk_type) tuples.
        """
        merged: List[Tuple[str, str, str]] = []
        buffer_heading = ""
        buffer_text = ""
        buffer_wc = 0
        buffer_type = "explanation"

        for section in sections:
            section_type = self._type_from_heading(section.heading)

            if buffer_wc == 0:
                # Start a new buffer
                buffer_heading = section.heading
                buffer_text = section.content
                buffer_wc = section.word_count
                buffer_type = section_type
            elif buffer_wc + section.word_count <= self.MAX_CHUNK_SIZE:
                # Merge into buffer
                buffer_text += "\n\n" + section.content
                buffer_wc += section.word_count
                # Keep the first heading but prefer non-trivial types
                if buffer_type == "explanation" and section_type != "explanation":
                    buffer_type = section_type
            else:
                # Flush buffer and start new
                merged.append((buffer_heading, buffer_text, buffer_type))
                buffer_heading = section.heading
                buffer_text = section.content
                buffer_wc = section.word_count
                buffer_type = section_type

        # Flush remaining
        if buffer_text.strip():
            merged.append((buffer_heading, buffer_text, buffer_type))

        return merged

    def _chunk_text_block(
        self, text: str, html: str, item: Dict[str, Any],
        heading: str, chunk_type: str, prefix: str, start_id: int,
        follows_chunk_id: Optional[str] = None,
        position_in_module: int = 0,
    ) -> List[Dict[str, Any]]:
        """Split a text block into chunks of appropriate size."""
        word_count = len(text.split())
        chunks = []

        if word_count <= self.MAX_CHUNK_SIZE:
            # Fits in one chunk
            chunks.append(self._create_chunk(
                chunk_id=f"{prefix}{start_id:05d}",
                text=text, html=html, item=item,
                section_heading=heading, chunk_type=chunk_type,
                follows_chunk_id=follows_chunk_id,
                position_in_module=position_in_module,
            ))
        else:
            # Split by sentences
            sub_texts = self._split_by_sentences(text, self.TARGET_CHUNK_SIZE)
            for i, sub_text in enumerate(sub_texts):
                part_heading = f"{heading} (part {i + 1})" if len(sub_texts) > 1 else heading
                prev_id = follows_chunk_id if i == 0 else f"{prefix}{start_id + i - 1:05d}"
                chunks.append(self._create_chunk(
                    chunk_id=f"{prefix}{start_id + i:05d}",
                    text=sub_text, html="" if i > 0 else html, item=item,
                    section_heading=part_heading, chunk_type=chunk_type,
                    follows_chunk_id=prev_id,
                    position_in_module=position_in_module + i,
                ))

        return chunks

    def _create_chunk(
        self, chunk_id: str, text: str, html: str, item: Dict[str, Any],
        section_heading: str, chunk_type: str,
        follows_chunk_id: Optional[str] = None,
        position_in_module: int = 0,
    ) -> Dict[str, Any]:
        words = text.split()
        word_count = len(words)
        tokens_estimate = int(word_count * 1.3)

        concept_tags = self._extract_concept_tags(text, item)
        difficulty = self._determine_difficulty(text, item)

        chunk: Dict[str, Any] = {
            "id": chunk_id,
            "chunk_type": chunk_type,
            "text": text,
            "html": html,
            "follows_chunk": follows_chunk_id,
            "source": {
                "course_id": self.course_code,
                "module_id": item["module_id"],
                "module_title": item["module_title"],
                "lesson_id": item["item_id"],
                "lesson_title": item["title"],
                "resource_type": item["resource_type"],
                "section_heading": section_heading,
                "position_in_module": position_in_module,
            },
            "concept_tags": concept_tags,
            "learning_outcome_refs": self._extract_objective_refs(item),
            "difficulty": difficulty,
            "tokens_estimate": tokens_estimate,
            "word_count": word_count,
        }

        # Enrich from Courseforge metadata (JSON-LD / data-cf-*)
        bloom_level, content_type_label, key_terms = self._extract_section_metadata(
            item, section_heading
        )
        # Fallback: if section metadata didn't provide bloom_level,
        # derive from page-level JSON-LD objectives or parsed objectives
        if not bloom_level:
            cf_meta = item.get("courseforge_metadata")
            if cf_meta and cf_meta.get("learningObjectives"):
                for lo in cf_meta["learningObjectives"]:
                    if lo.get("bloomLevel"):
                        bloom_level = lo["bloomLevel"]
                        break
        if not bloom_level:
            for lo in item.get("learning_objectives", []):
                bl = lo.bloom_level if hasattr(lo, "bloom_level") else lo.get("bloom_level")
                if bl:
                    bloom_level = bl
                    break

        if bloom_level:
            chunk["bloom_level"] = bloom_level
        if content_type_label:
            chunk["content_type_label"] = content_type_label
        if key_terms:
            chunk["key_terms"] = key_terms

        # Page-level metadata
        misconceptions = item.get("misconceptions", [])
        if misconceptions:
            chunk["misconceptions"] = misconceptions

        self.stats["total_words"] += word_count
        self.stats["total_tokens_estimate"] += tokens_estimate
        self.stats["chunk_types"][chunk_type] += 1
        self.stats["difficulty_distribution"][difficulty] += 1
        self._all_concept_tags.update(concept_tags)

        return chunk

    def _extract_section_metadata(
        self, item: Dict[str, Any], section_heading: str
    ) -> Tuple[Optional[str], Optional[str], List[Dict[str, str]]]:
        """Extract bloom_level, content_type_label, and key_terms for a section.

        Checks JSON-LD sections metadata first, then falls back to
        ContentSection data-cf-* attributes.
        """
        bloom_level: Optional[str] = None
        content_type_label: Optional[str] = None
        key_terms: List[Dict[str, str]] = []

        # Normalize heading: strip "(part N)" suffix added by _chunk_text_block
        # so multi-part chunks still match their JSON-LD / data-cf-* metadata.
        chunk_heading = re.sub(r'\s*\(part\s+\d+\)\s*$', '', section_heading).lower()

        # Try JSON-LD sections metadata
        cf_meta = item.get("courseforge_metadata")
        if cf_meta and cf_meta.get("sections"):
            for sec in cf_meta["sections"]:
                if sec.get("heading", "").lower() == chunk_heading:
                    content_type_label = sec.get("contentType")
                    bloom_range = sec.get("bloomRange", [])
                    if bloom_range:
                        bloom_level = bloom_range[0] if isinstance(bloom_range, list) else bloom_range
                    for kt in sec.get("keyTerms", []):
                        if isinstance(kt, dict) and kt.get("term"):
                            key_terms.append({"term": kt["term"], "definition": kt.get("definition", "")})
                    break

        # Fallback: data-cf-* attributes from parsed sections
        if not content_type_label:
            for section in item.get("sections", []):
                if section.heading.lower() == chunk_heading:
                    content_type_label = section.content_type
                    if section.key_terms:
                        key_terms = [{"term": t, "definition": ""} for t in section.key_terms]
                    break

        # Fallback: derive bloom_level from learning objectives
        if not bloom_level and item.get("learning_objectives"):
            for lo in item["learning_objectives"]:
                if lo.bloom_level:
                    bloom_level = lo.bloom_level
                    break

        return bloom_level, content_type_label, key_terms

    # ------------------------------------------------------------------
    # Chunk helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _type_from_resource(resource_type: str) -> str:
        mapping = {
            "quiz": "assessment_item",
            "overview": "overview",
            "summary": "summary",
            "discussion": "exercise",
            "application": "exercise",
        }
        return mapping.get(resource_type, "explanation")

    @staticmethod
    def _type_from_heading(heading: str) -> str:
        h = heading.lower()
        if any(kw in h for kw in ("example", "case study", "scenario")):
            return "example"
        if any(kw in h for kw in ("exercise", "activity", "practice", "application")):
            return "exercise"
        if any(kw in h for kw in ("summary", "recap", "key takeaway", "conclusion")):
            return "summary"
        if any(kw in h for kw in ("overview", "introduction", "welcome")):
            return "overview"
        if any(kw in h for kw in ("self-check", "self check", "knowledge check", "quiz", "check your")):
            return "assessment_item"
        if any(kw in h for kw in ("discussion", "reflection")):
            return "exercise"
        return "explanation"

    # Common educational concept patterns for text-based extraction
    CONCEPT_PATTERNS: Dict[str, List[str]] = {
        "learning-theory": ["learning theory", "learning theories"],
        "behaviorism": ["behaviorism", "behaviorist"],
        "cognitivism": ["cognitivism", "cognitivist", "information processing"],
        "constructivism": ["constructivism", "constructivist"],
        "connectivism": ["connectivism", "connectivist", "networked learning"],
        "instructional-design": ["instructional design"],
        "addie": ["addie model", "addie"],
        "backward-design": ["backward design", "understanding by design"],
        "cognitive-load": ["cognitive load", "intrinsic load", "extraneous load", "germane load"],
        "multimedia-learning": ["multimedia learning", "multimedia principle", "mayer"],
        "blooms-taxonomy": ["bloom's taxonomy", "blooms taxonomy", "bloom's", "higher-order thinking"],
        "assessment": ["assessment", "formative assessment", "summative assessment"],
        "rubric": ["rubric"],
        "accessibility": ["accessibility", "accessible", "wcag"],
        "udl": ["universal design for learning", "udl"],
        "oscqr": ["oscqr", "course quality"],
        "blended-learning": ["blended learning", "hybrid learning"],
        "online-learning": ["online learning", "online instruction", "distance learning"],
        "synchronous": ["synchronous"],
        "asynchronous": ["asynchronous"],
        "scaffolding": ["scaffolding", "zone of proximal development"],
        "engagement": ["student engagement", "learner engagement"],
        "community-of-inquiry": ["community of inquiry", "coi framework"],
        "alignment": ["constructive alignment", "learning objectives", "learning outcomes"],
        "feedback": ["feedback", "timely feedback"],
    }

    # Pattern for course/terminal/learning objective codes (CO-01, TO-08, LO-003, etc.)
    OBJECTIVE_CODE_RE = re.compile(r'^[a-z]{2}-\d{2,3}$')
    # Week prefix pattern (w01-, w02-) used by Courseforge JSON-LD but absent in course.json
    WEEK_PREFIX_RE = re.compile(r'^w\d{2}-', re.IGNORECASE)

    # Non-concept tags to filter out (generic metadata, not knowledge concepts)
    NON_CONCEPT_TAGS = {
        "estimated-time", "time", "minutes", "hours",
        # Bloom verbs (pedagogical intent, not domain concepts)
        "define", "list", "recall", "identify", "name", "state",
        "explain", "describe", "summarize", "interpret", "paraphrase",
        "apply", "demonstrate", "implement", "solve", "use", "execute",
        "analyze", "differentiate", "examine", "compare", "contrast", "organize",
        "evaluate", "assess", "critique", "judge", "justify", "argue",
        "create", "design", "develop", "construct", "produce", "formulate",
        # Course logistics
        "initial-post", "replies", "due", "guidelines",
        "correct", "incorrect", "submit", "deadline", "grading",
        "readings", "resources", "learning-objectives",
    }

    def _extract_concept_tags(self, text: str, item: Dict[str, Any]) -> List[str]:
        tags: List[str] = []

        # Key concepts from HTML parser (bold terms, definitions)
        for concept in item.get("key_concepts", []):
            tag = normalize_tag(concept)
            if not tag or len(tag) < 3 or tag in tags:
                continue
            # Skip objective codes (co-01, to-01, w01-co-01) and non-concept tags
            if (self.OBJECTIVE_CODE_RE.match(tag)
                    or self.WEEK_PREFIX_RE.match(tag)
                    or tag in self.NON_CONCEPT_TAGS):
                continue
            tags.append(tag)

        # Text-based concept detection
        text_lower = text.lower()
        for tag, patterns in self.CONCEPT_PATTERNS.items():
            if tag not in tags and any(p in text_lower for p in patterns):
                tags.append(tag)

        return tags[:10]

    def _extract_objective_refs(self, item: Dict[str, Any]) -> List[str]:
        """Extract learning objective reference codes from item.

        Prefers structured IDs from JSON-LD or parsed LearningObjective.id,
        falls back to regex CO/TO code extraction from key_concepts.
        """
        refs: List[str] = []

        # Prefer structured objective IDs from parser (JSON-LD or data-cf-*)
        for lo in item.get("learning_objectives", []):
            obj_id = lo.id if hasattr(lo, "id") else lo.get("id")
            if obj_id:
                normalized = obj_id.lower().strip()
                # Strip week prefix (w01-, w02-) to align with course.json format
                normalized = self.WEEK_PREFIX_RE.sub('', normalized)
                if normalized and normalized not in refs:
                    refs.append(normalized)
        if refs:
            return refs

        # Fallback: regex extraction from key_concepts
        for concept in item.get("key_concepts", []):
            tag = normalize_tag(concept)
            if tag and self.OBJECTIVE_CODE_RE.match(tag) and tag not in refs:
                refs.append(tag)
        return refs

    @staticmethod
    def _median_difficulty(blooms: List[str]) -> str:
        """Compute difficulty from median Bloom's weight of a week's objectives."""
        weights = sorted(BLOOM_WEIGHT[b] for b in blooms if b in BLOOM_WEIGHT)
        if not weights:
            return "intermediate"
        median = weights[len(weights) // 2]
        if median <= 2:
            return "foundational"
        if median <= 4:
            return "intermediate"
        return "advanced"

    @staticmethod
    def _cap_difficulty(difficulty: str) -> str:
        """Lower difficulty by one level (for introductory resource types)."""
        if difficulty == "advanced":
            return "intermediate"
        if difficulty == "intermediate":
            return "foundational"
        return "foundational"

    def _determine_difficulty(self, text: str, item: Dict[str, Any]) -> str:
        difficulty = None

        # First: check JSON-LD metadata for authoritative Bloom's levels
        cf_meta = item.get("courseforge_metadata")
        if cf_meta and cf_meta.get("learningObjectives"):
            for lo in cf_meta["learningObjectives"]:
                bl = lo.get("bloomLevel")
                if bl and bl in BLOOM_TO_DIFFICULTY:
                    difficulty = BLOOM_TO_DIFFICULTY[bl]
                    break

        # Second: use objectives file if we have week→bloom mapping
        if difficulty is None and self.objectives:
            week = item.get("week_num", 0)
            blooms = self.objectives.get("week_bloom_map", {}).get(week, [])
            if blooms:
                difficulty = self._median_difficulty(blooms)

        # Third: check learning objectives extracted from HTML
        if difficulty is None and item.get("learning_objectives"):
            for lo in item["learning_objectives"]:
                if lo.bloom_level and lo.bloom_level in BLOOM_TO_DIFFICULTY:
                    difficulty = BLOOM_TO_DIFFICULTY[lo.bloom_level]
                    break

        # Fallback: keyword heuristics
        if difficulty is None:
            text_lower = text.lower()
            if any(kw in text_lower for kw in ("basic", "introduction", "overview", "what is", "define")):
                difficulty = "foundational"
            elif any(kw in text_lower for kw in ("evaluate", "create", "design", "critique", "justify")):
                difficulty = "advanced"
            else:
                difficulty = "intermediate"

        # Cap difficulty for introductory resource types
        if item.get("resource_type") in INTRODUCTORY_RESOURCE_TYPES:
            difficulty = self._cap_difficulty(difficulty)

        return difficulty

    @staticmethod
    def _extract_plain_text(html: str) -> str:
        extractor = HTMLTextExtractor()
        extractor.feed(html)
        return extractor.get_text()

    @staticmethod
    def _extract_section_html(html: str, heading: str) -> str:
        if not heading:
            return ""
        # Match from the opening <hN> tag containing this heading text
        # through to the next <hN> tag or end of string
        pattern = r"<h[1-6][^>]*>\s*" + re.escape(heading) + r".*?(?=<h[1-6]|$)"
        match = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
        return match.group(0) if match else ""

    @staticmethod
    def _strip_assessment_feedback(html: str) -> str:
        """Remove answer feedback from quiz/self-check HTML before text extraction.

        Courseforge quizzes embed correct/incorrect feedback in
        <div class="sc-feedback"> blocks and data-correct attributes on labels.
        This strips that content so assessment chunks contain only question stems
        and answer options without revealing correctness.
        """
        # Remove feedback divs (Courseforge self-check pattern)
        cleaned = re.sub(
            r'<div\s+class="sc-feedback"[^>]*>.*?</div>',
            '', html, flags=re.DOTALL | re.IGNORECASE,
        )
        # Remove data-correct attributes from labels
        cleaned = re.sub(
            r'\s+data-correct="[^"]*"',
            '', cleaned,
        )
        return cleaned

    @staticmethod
    def _strip_feedback_from_text(text: str) -> str:
        """Remove residual feedback markers from plain text extraction.

        Handles both line-level and inline feedback patterns since the text
        extractor often concatenates feedback inline with answer options.
        """
        # Remove inline feedback: "Correct. <explanation>" or "Incorrect. <explanation>"
        # These appear after answer option text, running to the next answer option or end
        text = re.sub(
            r'\s*(?:Correct|Incorrect)\.\s+[^.]*(?:\.[^A-Z])*\.?',
            '', text,
        )
        # Also remove standalone lines
        lines = text.split('\n')
        filtered = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("Correct.") or stripped.startswith("Incorrect."):
                continue
            filtered.append(line)
        return '\n'.join(filtered)

    @staticmethod
    def _split_by_sentences(text: str, target_words: int) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: List[str] = []
        current: List[str] = []
        current_wc = 0

        for sentence in sentences:
            swc = len(sentence.split())
            if current_wc + swc > target_words and current:
                chunks.append(" ".join(current))
                current = [sentence]
                current_wc = swc
            else:
                current.append(sentence)
                current_wc += swc

        if current:
            chunks.append(" ".join(current))
        return chunks

    # ------------------------------------------------------------------
    # Stage 4: Write chunks
    # ------------------------------------------------------------------

    def _write_chunks(self, chunks: List[Dict[str, Any]]):
        jsonl_path = self.corpus_dir / "chunks.jsonl"
        json_path = self.corpus_dir / "chunks.json"

        with open(jsonl_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2, ensure_ascii=False)

        self.capture.log_decision(
            decision_type="chunk_serialization",
            decision=f"Write {len(chunks)} chunks to JSONL and JSON",
            rationale="JSONL format required for LibV2 streaming retrieval; JSON array for debugging and validation",
        )

    # ------------------------------------------------------------------
    # Stage 5: Generate metadata
    # ------------------------------------------------------------------

    def _auto_extract_topics(self) -> List[str]:
        """Extract topic tags from objectives chapter titles."""
        if not self.objectives:
            return []
        topics = []
        for ch in self.objectives.get("chapter_objectives", []):
            title = ch.get("chapter", "")
            # Strip "Week X-Y: " prefix and trailing preposition phrases
            cleaned = re.sub(r"^Week\s+\d+[-–]\d+:\s*", "", title)
            # Remove trailing prepositional phrases ("in Digital Environments", etc.)
            cleaned = re.sub(r"\s+(?:in|for)\s+.*$", "", cleaned, flags=re.IGNORECASE)
            if cleaned:
                tag = normalize_tag(cleaned)
                if tag and tag not in topics:
                    topics.append(tag)
        return topics

    def _auto_extract_subtopics(self, concept_graph: Dict[str, Any],
                                 exclude: List[str] = None, limit: int = 10) -> List[str]:
        """Extract subtopics from top concept graph nodes by frequency."""
        exclude_set = set(exclude or [])
        nodes = sorted(
            concept_graph.get("nodes", []),
            key=lambda n: n.get("frequency", 0),
            reverse=True,
        )
        subtopics = []
        for node in nodes:
            tag = node["id"]
            if tag not in exclude_set and tag not in subtopics:
                subtopics.append(tag)
            if len(subtopics) >= limit:
                break
        return subtopics

    def _generate_manifest(self, title: str,
                           concept_graph: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        description = ""
        if self.objectives:
            description = self.objectives.get("description", "")
        if not description:
            description = f"{title} - processed by Trainforge"

        # Detect section structure
        sections: List[str] = []
        if self.objectives:
            for ch in self.objectives.get("chapter_objectives", []):
                sections.append(ch.get("chapter", ""))

        # Auto-extract topics from objectives, subtopics from concept graph
        topics = self.topics or self._auto_extract_topics()
        subtopics = self._auto_extract_subtopics(
            concept_graph or {}, exclude=topics,
        )

        return {
            "course_id": self.course_code,
            "title": title,
            "description": description,
            "course_title": title,
            "sourceforge_version": "1.0",
            "export_timestamp": datetime.now().isoformat(),
            "source": {
                "type": "imscc",
                "path": str(self.imscc_path),
                "lms": "courseforge",
                "version": "1.3",
            },
            "classification": {
                "division": self.division,
                "primary_domain": self.domain,
                "secondary_domains": self.secondary_domains,
                "subdomains": self.subdomains,
                "topics": topics,
                "subtopics": subtopics,
            },
            "structure": {
                "total_sections": self.stats["sections_processed"],
                "sections": sections,
                "items_per_section": (
                    self.stats["modules_processed"] // max(self.stats["sections_processed"], 1)
                ),
            },
            "pedagogy": self._build_pedagogy_summary(),
            "processing": {
                "pipeline": "trainforge",
                "version": "1.0",
                "processed_date": datetime.now().isoformat(),
                "chunk_strategy": "pedagogical-units",
                "target_chunk_size": self.TARGET_CHUNK_SIZE,
            },
            "statistics": {
                "chunks": self.stats["total_chunks"],
                "total_words": self.stats["total_words"],
                "total_tokens": self.stats["total_tokens_estimate"],
                "concepts": len(self._all_concept_tags),
            },
        }

    def _generate_corpus_stats(self) -> Dict[str, Any]:
        total = self.stats["total_chunks"]
        return {
            "total_chunks": total,
            "total_words": self.stats["total_words"],
            "total_tokens_estimate": self.stats["total_tokens_estimate"],
            "avg_words_per_chunk": self.stats["total_words"] / total if total else 0,
            "chunk_type_distribution": dict(self.stats["chunk_types"]),
            "difficulty_distribution": dict(self.stats["difficulty_distribution"]),
            "modules_processed": self.stats["modules_processed"],
            "quizzes_processed": self.stats["quizzes_processed"],
            "sections_processed": self.stats["sections_processed"],
            "generated_at": datetime.now().isoformat(),
        }

    def _generate_concept_graph(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Build concept graph from tag co-occurrence."""
        tag_frequency: Dict[str, int] = defaultdict(int)
        co_occurrence: Dict[Tuple[str, str], int] = defaultdict(int)

        for chunk in chunks:
            tags = chunk.get("concept_tags", [])
            for tag in tags:
                tag_frequency[tag] += 1
            # Co-occurrence edges
            for i, a in enumerate(tags):
                for b in tags[i + 1:]:
                    key = tuple(sorted([a, b]))
                    co_occurrence[key] += 1

        # Build nodes: include all concepts appearing 2+ times
        # (single-occurrence tags are likely noise or overly specific)
        sorted_tags = sorted(tag_frequency.items(), key=lambda x: -x[1])
        nodes = [
            {"id": tag, "label": tag.replace("-", " ").title(), "frequency": freq}
            for tag, freq in sorted_tags
            if freq >= 2
        ]
        node_ids = {n["id"] for n in nodes}

        # Build edges (between nodes with any co-occurrence)
        edges = []
        for (a, b), weight in co_occurrence.items():
            if a in node_ids and b in node_ids:
                edges.append({"source": a, "target": b, "weight": weight})

        return {
            "nodes": nodes,
            "edges": edges,
            "generated_at": datetime.now().isoformat(),
        }

    def _generate_quality_report(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        total = len(chunks) or 1

        in_range = sum(1 for c in chunks if self.MIN_CHUNK_SIZE <= c["word_count"] <= self.MAX_CHUNK_SIZE)
        size_compliance = in_range / total

        with_tags = sum(1 for c in chunks if len(c.get("concept_tags", [])) >= 2)
        tag_coverage = with_tags / total

        with_html = sum(1 for c in chunks if c.get("html", "").strip())
        html_preservation = with_html / total

        # Bloom's level coverage
        with_bloom = sum(1 for c in chunks if c.get("bloom_level"))
        bloom_coverage = with_bloom / total

        # Learning outcome reference coverage
        with_lo_refs = sum(1 for c in chunks if c.get("learning_outcome_refs"))
        lo_coverage = with_lo_refs / total

        overall = (size_compliance * 0.25 + tag_coverage * 0.2 +
                   html_preservation * 0.2 + bloom_coverage * 0.2 + lo_coverage * 0.15)

        issues: List[str] = []
        recommendations: List[str] = []

        if size_compliance < 0.8:
            issues.append("Chunk size compliance below 80%")
            recommendations.append("Review chunking thresholds")
        if tag_coverage < 0.7:
            issues.append("Concept tag coverage below 70%")
            recommendations.append("Enhance concept extraction")
        if bloom_coverage < 0.9:
            issues.append(f"Bloom level coverage {bloom_coverage:.0%} — below 90% threshold")
        if lo_coverage < 0.8:
            issues.append(f"Learning outcome coverage {lo_coverage:.0%} — below 80% threshold")
        if not issues:
            recommendations.append("Corpus meets all quality thresholds")

        return {
            "overall_quality_score": round(overall, 3),
            "metrics": {
                "chunk_size_compliance": round(size_compliance, 3),
                "concept_tag_coverage": round(tag_coverage, 3),
                "html_preservation_rate": round(html_preservation, 3),
                "bloom_level_coverage": round(bloom_coverage, 3),
                "learning_outcome_coverage": round(lo_coverage, 3),
                "avg_chunk_size_words": round(self.stats["total_words"] / total, 1),
            },
            "validation": {"passed": overall >= 0.75, "issues": issues},
            "recommendations": recommendations,
        }

    def _build_pedagogy_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "instructional_approach": "competency-based",
            "learning_theory": "constructivism",
            "engagement_patterns": ["interactive-scenarios", "formative-assessment"],
        }
        if self.objectives and self.objectives.get("bloom_distribution"):
            summary["bloom_coverage"] = self.objectives["bloom_distribution"]
        return summary

    def _build_course_json(self, manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Build course.json with structured learning outcomes for LibV2."""
        outcomes = []

        for to in self.objectives.get("terminal_objectives", []):
            outcomes.append({
                "id": to["id"].lower(),
                "statement": to["statement"],
                "bloom_level": to.get("bloomLevel", "understand"),
                "hierarchy_level": "terminal",
            })

        for ch in self.objectives.get("chapter_objectives", []):
            for obj in ch.get("objectives", []):
                outcomes.append({
                    "id": obj["id"].lower(),
                    "statement": obj["statement"],
                    "bloom_level": obj.get("bloomLevel", "understand"),
                    "hierarchy_level": "chapter",
                })

        return {
            "course_code": self.course_code,
            "title": manifest.get("title", ""),
            "learning_outcomes": outcomes,
        }

    # ------------------------------------------------------------------
    # Stage 6: Write metadata
    # ------------------------------------------------------------------

    def _write_metadata(
        self,
        manifest: Dict[str, Any],
        corpus_stats: Dict[str, Any],
        concept_graph: Dict[str, Any],
        quality_report: Dict[str, Any],
    ):
        def _write(path: Path, data: Dict[str, Any]):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        _write(self.output_dir / "manifest.json", manifest)

        # course.json — structured learning outcomes for LibV2 validator
        if self.objectives:
            course_data = self._build_course_json(manifest)
            _write(self.output_dir / "course.json", course_data)

        _write(self.corpus_dir / "corpus_stats.json", corpus_stats)
        _write(self.graph_dir / "concept_graph.json", concept_graph)
        _write(self.quality_dir / "quality_report.json", quality_report)

        # Pedagogy model
        pedagogy = self._build_pedagogy_summary()
        _write(self.pedagogy_dir / "pedagogy_model.json", pedagogy)

        # Training specs
        training_specs = {
            "format": "instruction-following",
            "target_models": ["claude-opus-4-6", "claude-sonnet-4-6"],
            "training_objectives": [
                f"{self.domain}_instruction",
                f"{self.domain}_reasoning",
            ],
            "statistics": {
                "total_tokens": self.stats["total_tokens_estimate"],
            },
        }
        _write(self.training_specs_dir / "dataset_config.json", training_specs)

        # IMPORT_SUMMARY.md
        self._write_import_summary(manifest, corpus_stats, quality_report)

    def _write_import_summary(
        self, manifest: Dict[str, Any], stats: Dict[str, Any], quality: Dict[str, Any]
    ):
        lines = [
            f"# Import Summary: {manifest['title']}",
            "",
            f"**Course Code:** {self.course_code}",
            f"**Processed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"**Division:** {self.division} | **Domain:** {self.domain}",
            "",
            "## Corpus Statistics",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Chunks | {stats['total_chunks']} |",
            f"| Total Words | {stats['total_words']:,} |",
            f"| Total Tokens (est) | {stats['total_tokens_estimate']:,} |",
            f"| Avg Words/Chunk | {stats['avg_words_per_chunk']:.1f} |",
            f"| Sections | {stats['sections_processed']} |",
            f"| Modules | {stats['modules_processed']} |",
            f"| Quizzes | {stats['quizzes_processed']} |",
            "",
            "## Chunk Type Distribution",
            "",
        ]
        for ctype, count in sorted(stats.get("chunk_type_distribution", {}).items()):
            lines.append(f"- **{ctype}**: {count}")

        lines.extend([
            "",
            "## Difficulty Distribution",
            "",
        ])
        for diff, count in sorted(stats.get("difficulty_distribution", {}).items()):
            lines.append(f"- **{diff}**: {count}")

        lines.extend([
            "",
            "## Quality",
            "",
            f"- Overall Score: **{quality['overall_quality_score']:.3f}**",
            f"- Passed: **{quality['validation']['passed']}**",
            "",
            "Ready for LibV2 import.",
        ])

        with open(self.output_dir / "IMPORT_SUMMARY.md", "w") as f:
            f.write("\n".join(lines) + "\n")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _create_directories(self):
        for d in [self.corpus_dir, self.graph_dir, self.training_specs_dir,
                  self.pedagogy_dir, self.quality_dir]:
            d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Process a Courseforge IMSCC into a Trainforge RAG corpus",
    )
    p.add_argument("--imscc", required=True, help="Path to .imscc file")
    p.add_argument("--course-code", required=True, help="Course code (e.g. DIGPED_101)")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--objectives", help="Path to objectives JSON (optional)")
    p.add_argument("--division", default="STEM", choices=["STEM", "ARTS"], help="Division")
    p.add_argument("--domain", required=True, help="Primary domain")
    p.add_argument("--subdomain", action="append", default=[], help="Subdomain (repeatable)")
    p.add_argument("--secondary-domain", action="append", default=[], help="Secondary domain (repeatable)")
    p.add_argument("--topic", action="append", default=[], help="Topic (repeatable)")
    p.add_argument("--align", action="store_true",
                   help="Run alignment stage after processing (prereq_concepts, teaching_role, learning_outcome_refs)")
    p.add_argument("--llm-provider", default="mock", choices=["mock", "anthropic"],
                   help="LLM provider for alignment stage (default: mock)")
    p.add_argument("--import-to-libv2", action="store_true", help="Import into LibV2 after processing")
    return p


def main():
    args = build_parser().parse_args()

    processor = CourseProcessor(
        imscc_path=args.imscc,
        output_dir=args.output,
        course_code=args.course_code,
        division=args.division,
        domain=args.domain,
        subdomains=args.subdomain,
        secondary_domains=args.secondary_domain,
        topics=args.topic,
        objectives_path=args.objectives,
    )

    result = processor.process()

    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"Course: {result['title']}")
    print(f"Output: {result['output_dir']}")
    print(f"Chunks: {result['stats']['total_chunks']}")
    print(f"Words:  {result['stats']['total_words']:,}")
    print(f"Tokens: {result['stats']['total_tokens_estimate']:,}")
    print(f"\nChunk types:")
    for ct, count in result["stats"]["chunk_types"].items():
        print(f"  {ct}: {count}")
    print(f"\nDifficulty:")
    for d, count in result["stats"]["difficulty_distribution"].items():
        print(f"  {d}: {count}")

    # Optional alignment stage
    if args.align:
        print("\n[Alignment] Running alignment stage...")
        from Trainforge.align_chunks import main as align_main
        align_args = argparse.Namespace(
            corpus=args.output,
            objectives=args.objectives,
            fields="prereq_concepts,teaching_role,learning_outcome_refs",
            llm_provider=args.llm_provider,
            llm_model="claude-haiku-4-5-20251001",
            dry_run=False,
            verbose=False,
        )
        align_main(align_args)

    # Optional LibV2 import
    if args.import_to_libv2:
        print("\n[LibV2] Importing into LibV2...")
        try:
            from LibV2.tools.libv2.importer import import_course as do_import

            slug = do_import(
                source_dir=Path(args.output),
                repo_root=PROJECT_ROOT / "LibV2",
                division=args.division,
                domain=args.domain,
                subdomains=args.subdomain if args.subdomain else None,
                topics=args.topic if args.topic else None,
                secondary_domains=args.secondary_domain if args.secondary_domain else None,
                imscc_path=Path(args.imscc),
                strict_validation=False,
            )
            print(f"[LibV2] Imported as: {slug}")
            print(f"[LibV2] Location: LibV2/courses/{slug}/")
        except Exception as e:
            print(f"[LibV2] Import failed: {e}")
            print("[LibV2] You can import manually later with:")
            print(f"  python -m LibV2.tools.libv2.cli import {args.output} --domain {args.domain} --division {args.division}")

    print("\nDone!")
    return result


if __name__ == "__main__":
    main()
