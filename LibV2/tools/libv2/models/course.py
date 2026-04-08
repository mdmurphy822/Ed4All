"""Course data models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Classification:
    """Course classification in the taxonomy."""

    division: str  # STEM or ARTS
    primary_domain: str
    secondary_domains: list[str] = field(default_factory=list)
    subdomains: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    subtopics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "division": self.division,
            "primary_domain": self.primary_domain,
            "secondary_domains": self.secondary_domains,
            "subdomains": self.subdomains,
            "topics": self.topics,
            "subtopics": self.subtopics,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Classification":
        return cls(
            division=data["division"],
            primary_domain=data["primary_domain"],
            secondary_domains=data.get("secondary_domains", []),
            subdomains=data.get("subdomains", []),
            topics=data.get("topics", []),
            subtopics=data.get("subtopics", []),
        )


@dataclass
class ContentProfile:
    """Content statistics for a course."""

    total_chunks: int
    total_tokens: int
    total_concepts: int = 0
    language: str = "en"
    difficulty_distribution: dict[str, int] = field(default_factory=dict)
    chunk_type_distribution: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_chunks": self.total_chunks,
            "total_tokens": self.total_tokens,
            "total_concepts": self.total_concepts,
            "language": self.language,
            "difficulty_distribution": self.difficulty_distribution,
            "chunk_type_distribution": self.chunk_type_distribution,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContentProfile":
        return cls(
            total_chunks=data["total_chunks"],
            total_tokens=data["total_tokens"],
            total_concepts=data.get("total_concepts", 0),
            language=data.get("language", "en"),
            difficulty_distribution=data.get("difficulty_distribution", {}),
            chunk_type_distribution=data.get("chunk_type_distribution", {}),
        )


@dataclass
class SourceforgeManifest:
    """Original manifest from Sourceforge."""

    sourceforge_version: str
    export_timestamp: str
    course_id: str
    course_title: str

    def to_dict(self) -> dict:
        return {
            "sourceforge_version": self.sourceforge_version,
            "export_timestamp": self.export_timestamp,
            "course_id": self.course_id,
            "course_title": self.course_title,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SourceforgeManifest":
        return cls(
            sourceforge_version=data["sourceforge_version"],
            export_timestamp=data["export_timestamp"],
            course_id=data["course_id"],
            course_title=data["course_title"],
        )


@dataclass
class OntologyMapping:
    """Mapping to external ontologies."""

    acm_ccs: list[dict] = field(default_factory=list)
    lcsh: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "acm_ccs": self.acm_ccs,
            "lcsh": self.lcsh,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OntologyMapping":
        return cls(
            acm_ccs=data.get("acm_ccs", []),
            lcsh=data.get("lcsh", []),
        )


@dataclass
class SLMProcessing:
    """Track SLM processing history for self-improvement loop."""

    slm_version: Optional[str] = None  # Version of SLM used to process
    processing_timestamp: Optional[str] = None
    specialists_used: list[str] = field(default_factory=list)  # Which SLM specialists processed this
    generation: int = 0  # Which iteration of the improvement loop
    parent_version: Optional[str] = None  # Previous version this was derived from

    def to_dict(self) -> dict:
        return {
            "slm_version": self.slm_version,
            "processing_timestamp": self.processing_timestamp,
            "specialists_used": self.specialists_used,
            "generation": self.generation,
            "parent_version": self.parent_version,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SLMProcessing":
        return cls(
            slm_version=data.get("slm_version"),
            processing_timestamp=data.get("processing_timestamp"),
            specialists_used=data.get("specialists_used", []),
            generation=data.get("generation", 0),
            parent_version=data.get("parent_version"),
        )


@dataclass
class SourceArtifact:
    """Metadata for a single source artifact (PDF, HTML, or IMSCC)."""

    filename: str
    checksum: str  # sha256 hash
    file_size: int  # bytes
    added_timestamp: str  # ISO 8601

    def to_dict(self) -> dict:
        return {
            "filename": self.filename,
            "checksum": self.checksum,
            "file_size": self.file_size,
            "added_timestamp": self.added_timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SourceArtifact":
        return cls(
            filename=data["filename"],
            checksum=data["checksum"],
            file_size=data["file_size"],
            added_timestamp=data["added_timestamp"],
        )


@dataclass
class ArxivMetadata:
    """Metadata from Arxiv for research papers."""

    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    categories: list[str]
    primary_category: str
    published_date: str
    updated_date: Optional[str] = None

    def to_dict(self) -> dict:
        result = {
            "arxiv_id": self.arxiv_id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "categories": self.categories,
            "primary_category": self.primary_category,
            "published_date": self.published_date,
        }
        if self.updated_date:
            result["updated_date"] = self.updated_date
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "ArxivMetadata":
        return cls(
            arxiv_id=data["arxiv_id"],
            title=data["title"],
            authors=data["authors"],
            abstract=data["abstract"],
            categories=data["categories"],
            primary_category=data["primary_category"],
            published_date=data["published_date"],
            updated_date=data.get("updated_date"),
        )


@dataclass
class SourceArtifacts:
    """Collection of source artifacts preserved with the course."""

    pdf: Optional[SourceArtifact] = None
    html: Optional[SourceArtifact] = None
    imscc: Optional[SourceArtifact] = None

    def to_dict(self) -> dict:
        result = {}
        if self.pdf:
            result["pdf"] = self.pdf.to_dict()
        if self.html:
            result["html"] = self.html.to_dict()
        if self.imscc:
            result["imscc"] = self.imscc.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "SourceArtifacts":
        pdf = SourceArtifact.from_dict(data["pdf"]) if "pdf" in data else None
        html = SourceArtifact.from_dict(data["html"]) if "html" in data else None
        imscc = SourceArtifact.from_dict(data["imscc"]) if "imscc" in data else None
        return cls(pdf=pdf, html=html, imscc=imscc)


@dataclass
class CourseManifest:
    """Extended manifest for a course in LibV2."""

    libv2_version: str
    slug: str
    import_timestamp: str
    sourceforge_manifest: SourceforgeManifest
    classification: Classification
    content_profile: ContentProfile
    ontology_mappings: Optional[OntologyMapping] = None
    relationships: dict = field(default_factory=dict)
    quality_metadata: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)
    slm_processing: Optional[SLMProcessing] = None  # SLM version tracking
    source_package: Optional[str] = None  # Filename of source IMSCC in source/ dir (legacy)
    source_artifacts: Optional[SourceArtifacts] = None  # Enhanced source tracking
    arxiv_metadata: Optional[ArxivMetadata] = None  # Arxiv paper metadata

    def to_dict(self) -> dict:
        result = {
            "libv2_version": self.libv2_version,
            "slug": self.slug,
            "import_timestamp": self.import_timestamp,
            "sourceforge_manifest": self.sourceforge_manifest.to_dict(),
            "classification": self.classification.to_dict(),
            "content_profile": self.content_profile.to_dict(),
        }
        if self.ontology_mappings:
            result["ontology_mappings"] = self.ontology_mappings.to_dict()
        if self.relationships:
            result["relationships"] = self.relationships
        if self.quality_metadata:
            result["quality_metadata"] = self.quality_metadata
        if self.provenance:
            result["provenance"] = self.provenance
        if self.slm_processing:
            result["slm_processing"] = self.slm_processing.to_dict()
        if self.source_package:
            result["source_package"] = self.source_package
        if self.source_artifacts:
            result["source_artifacts"] = self.source_artifacts.to_dict()
        if self.arxiv_metadata:
            result["arxiv_metadata"] = self.arxiv_metadata.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "CourseManifest":
        ontology_mappings = None
        if "ontology_mappings" in data:
            ontology_mappings = OntologyMapping.from_dict(data["ontology_mappings"])

        slm_processing = None
        if "slm_processing" in data:
            slm_processing = SLMProcessing.from_dict(data["slm_processing"])

        source_artifacts = None
        if "source_artifacts" in data:
            source_artifacts = SourceArtifacts.from_dict(data["source_artifacts"])

        arxiv_metadata = None
        if "arxiv_metadata" in data:
            arxiv_metadata = ArxivMetadata.from_dict(data["arxiv_metadata"])

        return cls(
            libv2_version=data["libv2_version"],
            slug=data["slug"],
            import_timestamp=data["import_timestamp"],
            sourceforge_manifest=SourceforgeManifest.from_dict(data["sourceforge_manifest"]),
            classification=Classification.from_dict(data["classification"]),
            content_profile=ContentProfile.from_dict(data["content_profile"]),
            ontology_mappings=ontology_mappings,
            relationships=data.get("relationships", {}),
            quality_metadata=data.get("quality_metadata", {}),
            provenance=data.get("provenance", {}),
            slm_processing=slm_processing,
            source_package=data.get("source_package"),
            source_artifacts=source_artifacts,
            arxiv_metadata=arxiv_metadata,
        )

    @property
    def title(self) -> str:
        return self.sourceforge_manifest.course_title
