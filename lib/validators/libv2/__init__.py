"""LibV2-archive validators.

Cluster of three validators that operate on LibV2 archive integrity:
* :class:`LibV2ManifestValidator` — manifest JSON + scaffold + hash agreement.
* :class:`LibV2ModelValidator` — model_card.json schema + weights agreement.
* :class:`PacketIntegrityValidator` — full packet structural / referential integrity.

Wave W-D10 T10.1: this is the canonical import path. The legacy flat
modules (``lib/validators/libv2_manifest.py``,
``lib/validators/libv2_model.py``,
``lib/validators/libv2_packet_integrity.py``) re-export from here via
shim files for one minor-version cycle.
"""

from lib.validators.libv2.course_completeness import CourseCompletenessValidator
from lib.validators.libv2.manifest import LibV2ManifestValidator
from lib.validators.libv2.model import LibV2ModelValidator
from lib.validators.libv2.packet_integrity import PacketIntegrityValidator

__all__ = [
    "CourseCompletenessValidator",
    "LibV2ManifestValidator",
    "LibV2ModelValidator",
    "PacketIntegrityValidator",
]
