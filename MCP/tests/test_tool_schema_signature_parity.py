"""Wave 28e — ``TOOL_SCHEMAS`` optional-kwarg parity with ``@mcp.tool()`` sigs.

Uses ``inspect.signature`` to verify that every optional keyword
argument on a registered ``@mcp.tool()`` function also appears in
its ``TOOL_SCHEMAS`` ``optional`` list (or as a target in
``param_mapping``). This catches the Wave 22 F4 ``figures_dir``
regression class: signature added, schema never updated, external
callers' kwarg silently dropped by the strict param-mapping layer.

The test is scoped to ``extract_and_convert_pdf`` today (the
Wave 28e fix site) with a generic walker so future ``@mcp.tool()``
signature additions are caught automatically.
"""

from __future__ import annotations

import inspect
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from MCP.core.tool_schemas import TOOL_SCHEMAS  # noqa: E402
from MCP.tools import dart_tools  # noqa: E402


class _MCPStub:
    """Minimal MCP stub that captures ``@mcp.tool()`` registrations."""

    def __init__(self):
        self.registered = {}

    def tool(self, *args, **kwargs):
        def decorator(fn):
            self.registered[fn.__name__] = fn
            return fn
        return decorator


def _collect_dart_tools():
    mcp = _MCPStub()
    dart_tools.register_dart_tools(mcp)
    return mcp.registered


def _signature_optional_kwargs(fn) -> set[str]:
    """Return the set of parameter names with defaults (optional kwargs).

    Excludes ``self`` + positional-only / VAR_POSITIONAL / VAR_KEYWORD.
    A parameter is considered optional when it has a default value.
    """
    sig = inspect.signature(fn)
    out: set[str] = set()
    for name, param in sig.parameters.items():
        if name in ("self",):
            continue
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        if param.default is not inspect.Parameter.empty:
            out.add(name)
    return out


def _schema_known_names(tool_name: str) -> set[str]:
    """Return the set of names the schema can map or accept directly."""
    schema = TOOL_SCHEMAS.get(tool_name, {})
    known: set[str] = set()
    known.update(schema.get("required", []))
    known.update(schema.get("optional", []))
    # param_mapping targets are also recognized.
    known.update(schema.get("param_mapping", {}).values())
    return known


class TestToolSchemaSignatureParity:
    def test_extract_and_convert_pdf_includes_figures_dir(self):
        """Wave 28e fix site: ``figures_dir`` is in the schema."""
        schema = TOOL_SCHEMAS["extract_and_convert_pdf"]
        assert "figures_dir" in schema["optional"]
        assert "figures_dir" in schema["defaults"]
        assert schema["defaults"]["figures_dir"] is None
        # ``figures`` alias maps to ``figures_dir``.
        assert schema["param_mapping"].get("figures") == "figures_dir"

    def test_extract_and_convert_pdf_signature_matches_schema(self):
        """Every optional kwarg on the @mcp.tool() signature is in the schema."""
        tools = _collect_dart_tools()
        fn = tools["extract_and_convert_pdf"]
        sig_optional = _signature_optional_kwargs(fn)
        schema_known = _schema_known_names("extract_and_convert_pdf")

        missing = sig_optional - schema_known
        assert not missing, (
            f"extract_and_convert_pdf optional kwargs {missing} absent "
            f"from TOOL_SCHEMAS — strict param mapping will drop them. "
            f"schema_known={schema_known}, sig_optional={sig_optional}"
        )

    @pytest.mark.parametrize(
        "tool_name",
        [
            "extract_and_convert_pdf",
            # Add more DART tools here as their schemas gain parity
            # coverage. Listed tools must have TOOL_SCHEMAS entries.
        ],
    )
    def test_dart_tool_optional_kwargs_in_schema(self, tool_name):
        """Generic parity walker: every DART @mcp.tool() optional kwarg
        appears in its TOOL_SCHEMAS entry.
        """
        tools = _collect_dart_tools()
        if tool_name not in tools:
            pytest.skip(f"{tool_name} not registered in dart_tools")
        if tool_name not in TOOL_SCHEMAS:
            pytest.skip(f"{tool_name} not in TOOL_SCHEMAS")

        fn = tools[tool_name]
        sig_optional = _signature_optional_kwargs(fn)
        schema_known = _schema_known_names(tool_name)

        missing = sig_optional - schema_known
        assert not missing, (
            f"{tool_name} optional kwargs {missing} absent from "
            f"TOOL_SCHEMAS — strict param mapping will drop them."
        )
