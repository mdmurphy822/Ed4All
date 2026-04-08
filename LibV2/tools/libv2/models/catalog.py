"""Catalog data models."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CatalogEntry:
    """A course entry in the catalog."""

    slug: str
    title: str
    division: str
    primary_domain: str
    secondary_domains: list[str] = field(default_factory=list)
    subdomains: list[str] = field(default_factory=list)
    chunk_count: int = 0
    concept_count: int = 0
    token_count: int = 0
    difficulty_primary: str = "mixed"
    language: str = "en"
    validation_status: str = "pending"

    def to_dict(self) -> dict:
        return {
            "slug": self.slug,
            "title": self.title,
            "division": self.division,
            "primary_domain": self.primary_domain,
            "secondary_domains": self.secondary_domains,
            "subdomains": self.subdomains,
            "chunk_count": self.chunk_count,
            "concept_count": self.concept_count,
            "token_count": self.token_count,
            "difficulty_primary": self.difficulty_primary,
            "language": self.language,
            "validation_status": self.validation_status,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CatalogEntry":
        return cls(
            slug=data["slug"],
            title=data["title"],
            division=data["division"],
            primary_domain=data["primary_domain"],
            secondary_domains=data.get("secondary_domains", []),
            subdomains=data.get("subdomains", []),
            chunk_count=data.get("chunk_count", 0),
            concept_count=data.get("concept_count", 0),
            token_count=data.get("token_count", 0),
            difficulty_primary=data.get("difficulty_primary", "mixed"),
            language=data.get("language", "en"),
            validation_status=data.get("validation_status", "pending"),
        )

    @classmethod
    def from_manifest(cls, manifest: "CourseManifest") -> "CatalogEntry":
        """Create catalog entry from a course manifest."""
        from .course import CourseManifest

        # Determine primary difficulty
        dist = manifest.content_profile.difficulty_distribution
        if dist:
            max_difficulty = max(dist, key=dist.get)
            difficulty_primary = max_difficulty
        else:
            difficulty_primary = "mixed"

        return cls(
            slug=manifest.slug,
            title=manifest.title,
            division=manifest.classification.division,
            primary_domain=manifest.classification.primary_domain,
            secondary_domains=manifest.classification.secondary_domains,
            subdomains=manifest.classification.subdomains,
            chunk_count=manifest.content_profile.total_chunks,
            concept_count=manifest.content_profile.total_concepts,
            token_count=manifest.content_profile.total_tokens,
            difficulty_primary=difficulty_primary,
            language=manifest.content_profile.language,
            validation_status=manifest.quality_metadata.get("validation_status", "pending"),
        )


@dataclass
class MasterCatalog:
    """The master catalog containing all courses."""

    version: str
    generated_at: str
    total_courses: int
    courses: list[CatalogEntry]

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "generated_at": self.generated_at,
            "total_courses": self.total_courses,
            "courses": [c.to_dict() for c in self.courses],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MasterCatalog":
        return cls(
            version=data["version"],
            generated_at=data["generated_at"],
            total_courses=data["total_courses"],
            courses=[CatalogEntry.from_dict(c) for c in data["courses"]],
        )

    def find_by_slug(self, slug: str) -> Optional[CatalogEntry]:
        """Find a course by slug."""
        for course in self.courses:
            if course.slug == slug:
                return course
        return None

    def filter_by_domain(self, domain: str) -> list[CatalogEntry]:
        """Filter courses by domain (primary or secondary)."""
        return [
            c for c in self.courses
            if c.primary_domain == domain or domain in c.secondary_domains
        ]

    def filter_by_division(self, division: str) -> list[CatalogEntry]:
        """Filter courses by division."""
        return [c for c in self.courses if c.division == division]
