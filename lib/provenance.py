"""
Provenance - Hash-Based Provenance for Inputs and Outputs

Provides utilities for creating input references and output pointers
with cryptographic hashes for integrity verification.

Phase 0 Hardening: Requirement 4 (Decision Capture Integrity)
"""

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union


# ============================================================================
# CONSTANTS
# ============================================================================

HASH_ALGORITHMS = Literal["sha256", "sha512", "blake3"]
DEFAULT_ALGORITHM = "sha256"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ByteRange:
    """Byte range within a file."""
    start: int
    end: int

    def to_dict(self) -> Dict[str, int]:
        return {"start": self.start, "end": self.end}


@dataclass
class InputRef:
    """
    Reference to an input source with provenance.

    Used to track what inputs were used for a decision, with cryptographic
    verification of content integrity.
    """
    source_type: str  # "textbook", "pdf", "imscc", "web_search", "prompt_template", etc.
    path_or_id: str   # File path, URL, or identifier
    content_hash: str = ""  # Hash of content
    hash_algorithm: str = DEFAULT_ALGORITHM
    size_bytes: int = 0
    byte_range: Optional[ByteRange] = None
    excerpt_range: str = ""  # Human-readable range like "lines:100-200"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "source_type": self.source_type,
            "path_or_id": self.path_or_id,
        }
        if self.content_hash:
            result["content_hash"] = self.content_hash
            result["hash_algorithm"] = self.hash_algorithm
        if self.size_bytes:
            result["size_bytes"] = self.size_bytes
        if self.byte_range:
            result["byte_range"] = self.byte_range.to_dict()
        if self.excerpt_range:
            result["excerpt_range"] = self.excerpt_range
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InputRef":
        """Create from dictionary."""
        byte_range = None
        if "byte_range" in data:
            byte_range = ByteRange(**data["byte_range"])
        return cls(
            source_type=data["source_type"],
            path_or_id=data["path_or_id"],
            content_hash=data.get("content_hash", ""),
            hash_algorithm=data.get("hash_algorithm", DEFAULT_ALGORITHM),
            size_bytes=data.get("size_bytes", 0),
            byte_range=byte_range,
            excerpt_range=data.get("excerpt_range", ""),
            metadata=data.get("metadata", {}),
        )


@dataclass
class OutputRef:
    """
    Reference to an output artifact (pointer, not blob).

    Used to track what outputs were produced by a decision, with
    cryptographic verification for integrity.
    """
    artifact_type: str  # "html", "imscc", "assessment", "chunk", etc.
    path: str  # File path relative to run artifacts
    content_hash: str = ""  # Hash of content
    hash_algorithm: str = DEFAULT_ALGORITHM
    size_bytes: int = 0
    byte_range: Optional[ByteRange] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "artifact_type": self.artifact_type,
            "path": self.path,
        }
        if self.content_hash:
            result["content_hash"] = self.content_hash
            result["hash_algorithm"] = self.hash_algorithm
        if self.size_bytes:
            result["size_bytes"] = self.size_bytes
        if self.byte_range:
            result["byte_range"] = self.byte_range.to_dict()
        if self.metadata:
            result["metadata"] = self.metadata
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OutputRef":
        """Create from dictionary."""
        byte_range = None
        if "byte_range" in data:
            byte_range = ByteRange(**data["byte_range"])
        return cls(
            artifact_type=data["artifact_type"],
            path=data["path"],
            content_hash=data.get("content_hash", ""),
            hash_algorithm=data.get("hash_algorithm", DEFAULT_ALGORITHM),
            size_bytes=data.get("size_bytes", 0),
            byte_range=byte_range,
            metadata=data.get("metadata", {}),
        )


# ============================================================================
# HASHING FUNCTIONS
# ============================================================================

def get_hasher(algorithm: str = DEFAULT_ALGORITHM):
    """
    Get a hasher for the specified algorithm.

    Args:
        algorithm: Hash algorithm (sha256, sha512, blake3)

    Returns:
        Hasher object

    Raises:
        ValueError: If algorithm not supported
    """
    if algorithm == "sha256":
        return hashlib.sha256()
    elif algorithm == "sha512":
        return hashlib.sha512()
    elif algorithm == "blake3":
        try:
            import blake3
            return blake3.blake3()
        except ImportError:
            raise ValueError("blake3 not installed. Install with: pip install blake3")
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")


def hash_content(
    content: Union[bytes, str],
    algorithm: str = DEFAULT_ALGORITHM
) -> str:
    """
    Hash content bytes or string.

    Args:
        content: Content to hash (bytes or string)
        algorithm: Hash algorithm

    Returns:
        Hex digest of hash
    """
    if isinstance(content, str):
        content = content.encode('utf-8')

    hasher = get_hasher(algorithm)
    hasher.update(content)
    return hasher.hexdigest()


def hash_file(
    path: Union[str, Path],
    algorithm: str = DEFAULT_ALGORITHM,
    chunk_size: int = 8192
) -> str:
    """
    Hash a file's contents.

    Args:
        path: Path to file
        algorithm: Hash algorithm
        chunk_size: Size of chunks to read

    Returns:
        Hex digest of hash

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    hasher = get_hasher(algorithm)
    with open(path, 'rb') as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)

    return hasher.hexdigest()


def hash_file_range(
    path: Union[str, Path],
    start: int,
    end: int,
    algorithm: str = DEFAULT_ALGORITHM
) -> str:
    """
    Hash a byte range within a file.

    Args:
        path: Path to file
        start: Start byte offset
        end: End byte offset
        algorithm: Hash algorithm

    Returns:
        Hex digest of hash
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    hasher = get_hasher(algorithm)
    with open(path, 'rb') as f:
        f.seek(start)
        to_read = end - start
        while to_read > 0:
            chunk = f.read(min(8192, to_read))
            if not chunk:
                break
            hasher.update(chunk)
            to_read -= len(chunk)

    return hasher.hexdigest()


# ============================================================================
# INPUT/OUTPUT REF CREATION
# ============================================================================

def create_input_ref(
    source_type: str,
    path_or_id: str,
    compute_hash: bool = True,
    algorithm: str = DEFAULT_ALGORITHM,
    excerpt_range: str = "",
    byte_range: Optional[tuple[int, int]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> InputRef:
    """
    Create an InputRef with automatic hash computation.

    Args:
        source_type: Type of source (textbook, pdf, imscc, etc.)
        path_or_id: Path to file or identifier
        compute_hash: Whether to compute content hash (default True)
        algorithm: Hash algorithm
        excerpt_range: Human-readable range
        byte_range: Optional (start, end) tuple for partial content
        metadata: Additional metadata

    Returns:
        InputRef with provenance
    """
    path = Path(path_or_id)
    content_hash = ""
    size_bytes = 0

    if compute_hash and path.exists():
        if byte_range:
            content_hash = hash_file_range(path, byte_range[0], byte_range[1], algorithm)
            size_bytes = byte_range[1] - byte_range[0]
        else:
            content_hash = hash_file(path, algorithm)
            size_bytes = path.stat().st_size

    br = ByteRange(byte_range[0], byte_range[1]) if byte_range else None

    return InputRef(
        source_type=source_type,
        path_or_id=str(path_or_id),
        content_hash=content_hash,
        hash_algorithm=algorithm if content_hash else DEFAULT_ALGORITHM,
        size_bytes=size_bytes,
        byte_range=br,
        excerpt_range=excerpt_range,
        metadata=metadata or {},
    )


def create_output_ref(
    artifact_type: str,
    path: Union[str, Path],
    compute_hash: bool = True,
    algorithm: str = DEFAULT_ALGORITHM,
    metadata: Optional[Dict[str, Any]] = None,
) -> OutputRef:
    """
    Create an OutputRef with automatic hash computation.

    Args:
        artifact_type: Type of artifact (html, imscc, assessment, etc.)
        path: Path to output file
        compute_hash: Whether to compute content hash (default True)
        algorithm: Hash algorithm
        metadata: Additional metadata

    Returns:
        OutputRef with provenance
    """
    path = Path(path)
    content_hash = ""
    size_bytes = 0

    if compute_hash and path.exists():
        content_hash = hash_file(path, algorithm)
        size_bytes = path.stat().st_size

    return OutputRef(
        artifact_type=artifact_type,
        path=str(path),
        content_hash=content_hash,
        hash_algorithm=algorithm if content_hash else DEFAULT_ALGORITHM,
        size_bytes=size_bytes,
        metadata=metadata or {},
    )


def create_input_ref_from_content(
    source_type: str,
    identifier: str,
    content: Union[bytes, str],
    algorithm: str = DEFAULT_ALGORITHM,
    excerpt_range: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> InputRef:
    """
    Create an InputRef from in-memory content.

    Args:
        source_type: Type of source
        identifier: Identifier for the content
        content: Content bytes or string
        algorithm: Hash algorithm
        excerpt_range: Human-readable range
        metadata: Additional metadata

    Returns:
        InputRef with provenance
    """
    if isinstance(content, str):
        content_bytes = content.encode('utf-8')
    else:
        content_bytes = content

    content_hash = hash_content(content_bytes, algorithm)

    return InputRef(
        source_type=source_type,
        path_or_id=identifier,
        content_hash=content_hash,
        hash_algorithm=algorithm,
        size_bytes=len(content_bytes),
        excerpt_range=excerpt_range,
        metadata=metadata or {},
    )


# ============================================================================
# VERIFICATION
# ============================================================================

def verify_artifact(
    path: Union[str, Path],
    expected_hash: str,
    algorithm: str = DEFAULT_ALGORITHM
) -> bool:
    """
    Verify artifact integrity by comparing hash.

    Args:
        path: Path to artifact
        expected_hash: Expected hash value
        algorithm: Hash algorithm used

    Returns:
        True if hash matches, False otherwise
    """
    try:
        actual_hash = hash_file(path, algorithm)
        return actual_hash == expected_hash
    except FileNotFoundError:
        return False


def verify_input_ref(input_ref: InputRef) -> bool:
    """
    Verify an InputRef's integrity.

    Args:
        input_ref: InputRef to verify

    Returns:
        True if hash matches (or no hash to verify), False otherwise
    """
    if not input_ref.content_hash:
        return True  # No hash to verify

    path = Path(input_ref.path_or_id)
    if not path.exists():
        return False

    if input_ref.byte_range:
        actual_hash = hash_file_range(
            path,
            input_ref.byte_range.start,
            input_ref.byte_range.end,
            input_ref.hash_algorithm
        )
    else:
        actual_hash = hash_file(path, input_ref.hash_algorithm)

    return actual_hash == input_ref.content_hash


def verify_output_ref(output_ref: OutputRef) -> bool:
    """
    Verify an OutputRef's integrity.

    Args:
        output_ref: OutputRef to verify

    Returns:
        True if hash matches (or no hash to verify), False otherwise
    """
    if not output_ref.content_hash:
        return True  # No hash to verify

    return verify_artifact(
        output_ref.path,
        output_ref.content_hash,
        output_ref.hash_algorithm
    )


# ============================================================================
# MERKLE ROOT
# ============================================================================

def compute_merkle_root(hashes: List[str], algorithm: str = DEFAULT_ALGORITHM) -> str:
    """
    Compute Merkle root from a list of hashes.

    Args:
        hashes: List of hex digest hashes
        algorithm: Hash algorithm

    Returns:
        Merkle root as hex digest
    """
    if not hashes:
        return hash_content(b"", algorithm)

    if len(hashes) == 1:
        return hashes[0]

    # Pad to even length
    if len(hashes) % 2 == 1:
        hashes = hashes + [hashes[-1]]

    # Combine pairs
    combined = []
    for i in range(0, len(hashes), 2):
        pair_content = hashes[i] + hashes[i + 1]
        combined.append(hash_content(pair_content, algorithm))

    return compute_merkle_root(combined, algorithm)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    # Constants
    "DEFAULT_ALGORITHM",
    # Data classes
    "ByteRange",
    "InputRef",
    "OutputRef",
    # Hashing functions
    "get_hasher",
    "hash_content",
    "hash_file",
    "hash_file_range",
    # Ref creation
    "create_input_ref",
    "create_output_ref",
    "create_input_ref_from_content",
    # Verification
    "verify_artifact",
    "verify_input_ref",
    "verify_output_ref",
    # Merkle
    "compute_merkle_root",
]
