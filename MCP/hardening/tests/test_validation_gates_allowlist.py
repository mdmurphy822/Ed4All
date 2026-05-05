"""Regression sentinel for ValidationGateManager.ALLOWED_VALIDATOR_PREFIXES.

W9 surfaced that the four `Courseforge.router.inter_tier_gates.Block*Validator`
classes wired in `config/workflows.yaml` failed to load through the executor's
gate-manager path because their module prefix wasn't in the allowlist. Gates
fired `passed=False` for the wrong reason (VALIDATOR_ERROR) instead of running
the actual validation logic. The fix added `"Courseforge.router."` to the
allowlist tuple. This test pins that contract so removal regresses loudly.
"""

from MCP.hardening.validation_gates import ValidationGateManager


REQUIRED_PREFIXES = (
    "lib.validators.",
    "lib.leak_checker",
    "DART.pdf_converter.",
    "Courseforge.router.",
)


def test_allowlist_includes_required_prefixes():
    actual = ValidationGateManager.ALLOWED_VALIDATOR_PREFIXES
    for prefix in REQUIRED_PREFIXES:
        assert prefix in actual, (
            f"Required validator-module prefix {prefix!r} is missing from "
            f"ValidationGateManager.ALLOWED_VALIDATOR_PREFIXES "
            f"({actual!r}). Removing this prefix causes gates wired at "
            f"that module path to fail with ImportError → VALIDATOR_ERROR "
            f"instead of running their validation logic."
        )


def test_courseforge_router_block_validators_load_through_manager():
    """The four Block*Validators wired in config/workflows.yaml must
    actually instantiate via load_validator — not raise allowlist
    rejection. Verifies the W9 bug is closed at the integration boundary
    where the executor's gate-manager path lives.
    """
    manager = ValidationGateManager()
    for dotted_path in (
        "Courseforge.router.inter_tier_gates.BlockCurieAnchoringValidator",
        "Courseforge.router.inter_tier_gates.BlockContentTypeValidator",
        "Courseforge.router.inter_tier_gates.BlockPageObjectivesValidator",
        "Courseforge.router.inter_tier_gates.BlockSourceRefValidator",
    ):
        validator = manager.load_validator(dotted_path)
        assert validator is not None, dotted_path
        assert validator.__class__.__module__.startswith(
            "Courseforge.router."
        ), validator.__class__.__module__
